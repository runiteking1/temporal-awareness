"""
Steer Kirby MCQ-27 with CAA vectors under Chain-of-Thought prompting.

Tests whether the CAA steering vector from notebook 08 can shift the
model's temporal discounting when it reasons before answering. Two
conditions are run:

  1. Zero-shot CoT: model reasons from scratch — steering may alter
     the reasoning itself or be "argued away" by the chain of thought.
  2. Few-shot CoT: few-shot examples include terse now/later answers
     (known from prior work to suppress the thinking mechanism).

Compares steered k values against the human benchmarks:
  - Human controls:      k ≈ 0.013
  - Human heroin users:  k ≈ 0.025
"""

import json
import math
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Inner repo for ModelRunner ────────────────────────────────────────────
INNER_REPO = Path(__file__).resolve().parent / "temporal-awareness"
if str(INNER_REPO) not in sys.path:
    sys.path.insert(0, str(INNER_REPO))

from src.inference.model_runner import ModelRunner
from src.inference.backends import ModelBackend
from src.inference.backends.transformerlens import TransformerLensBackend

# ── Constants ─────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
CAA_VECTOR_PATH = (
    INNER_REPO / "out" / "steering_from_scratch" / "probing"
    / "temporal_caa_vector_norm.pt"
)
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

STEER_LAYER = 22  # best layer from notebook 08 extended sweep
ALPHAS = [0, 10, 20, 30, 40, 50]
NEG_ALPHAS = [-10, -20, -30]

# ── Kirby MCQ-27 ─────────────────────────────────────────────────────────
QUESTIONS_TXT = """1. Would you prefer $54 today, or $55 in 117 days?
2. Would you prefer $55 today, or $75 in 61 days?
3. Would you prefer $19 today, or $25 in 53 days?
4. Would you prefer $31 today, or $85 in 7 days?
5. Would you prefer $14 today, or $25 in 19 days?
6. Would you prefer $47 today, or $50 in 160 days?
7. Would you prefer $15 today, or $35 in 13 days?
8. Would you prefer $25 today, or $60 in 14 days?
9. Would you prefer $78 today, or $80 in 162 days?
10. Would you prefer $40 today, or $55 in 62 days?
11. Would you prefer $11 today, or $30 in 7 days?
12. Would you prefer $67 today, or $75 in 119 days?
13. Would you prefer $34 today, or $35 in 186 days?
14. Would you prefer $27 today, or $50 in 21 days?
15. Would you prefer $69 today, or $85 in 91 days?
16. Would you prefer $49 today, or $60 in 89 days?
17. Would you prefer $80 today, or $85 in 157 days?
18. Would you prefer $24 today, or $35 in 29 days?
19. Would you prefer $33 today, or $80 in 14 days?
20. Would you prefer $28 today, or $30 in 179 days?
21. Would you prefer $34 today, or $50 in 30 days?
22. Would you prefer $25 today, or $30 in 80 days?
23. Would you prefer $41 today, or $75 in 20 days?
24. Would you prefer $54 today, or $60 in 111 days?
25. Would you prefer $54 today, or $80 in 30 days?
26. Would you prefer $22 today, or $25 in 136 days?
27. Would you prefer $20 today, or $55 in 7 days?"""

# ── Prompts ──────────────────────────────────────────────────────────────

# Zero-shot CoT: model reasons from scratch
COT_SYSTEM_PROMPT = (
    "You are a 35-year-old adult with a stable job and average finances. "
    "You are completing a psychology questionnaire about monetary preferences. "
    "Both options are guaranteed — you will get the money. "
    "For each question, briefly reason about the tradeoff, "
    "then on a new line write your final answer as exactly one word: NOW or LATER."
)

# Few-shot CoT: terse few-shot examples (from discount_factor_best.py)
# Prior work showed these suppress the thinking mechanism entirely.
BASE_INSTRUCTION = (
    "You are completing a psychology questionnaire about monetary preferences. "
    "Both options are guaranteed real payments. "
    "People differ on these choices — there are no right or wrong answers. "
    "Briefly reason about the tradeoff, then on a new line write your "
    "final answer as exactly one word: NOW or LATER."
)

FEWSHOT_COT = [
    {"role": "system", "content":
        "You are a financially comfortable 45-year-old professional. "
        + BASE_INSTRUCTION},
    {"role": "user", "content": "Would you prefer $90 today, or $95 in 120 days?"},
    {"role": "assistant", "content": "now"},
    {"role": "user", "content": "Would you prefer $50 today, or $70 in 60 days?"},
    {"role": "assistant", "content": "now"},
    {"role": "user", "content": "Would you prefer $24 today, or $35 in 29 days?"},
    {"role": "assistant", "content": "later"},
    {"role": "user", "content": "Would you prefer $20 today, or $55 in 7 days?"},
    {"role": "assistant", "content": "later"},
]


# ── Kirby scoring ────────────────────────────────────────────────────────
_Q_RE = re.compile(
    r"(\d+)\. Would you prefer \$(\d+) today, or \$(\d+) in (\d+) days\?"
)


def sigfigs(x, n=2):
    if x == 0:
        return 0
    return round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1))


def parse_trials():
    trials = []
    for line in QUESTIONS_TXT.strip().splitlines():
        m = _Q_RE.match(line.strip())
        if m:
            order, sir, ldr, delay = int(m[1]), int(m[2]), int(m[3]), int(m[4])
            k_indiff = sigfigs((ldr / sir - 1) / delay)
            trials.append(dict(order=order, sir=sir, ldr=ldr, delay=delay,
                               k_indiff=k_indiff))
    return pd.DataFrame(trials)


def magnitude(ldr):
    if ldr <= 35:
        return "small"
    elif ldr <= 60:
        return "medium"
    return "large"


def estimate_k(df):
    sorted_ks = sorted(df["k_indiff"].unique())
    candidates = [sorted_ks[0]]
    for i in range(len(sorted_ks) - 1):
        candidates.append(math.sqrt(sorted_ks[i] * sorted_ks[i + 1]))
    candidates.append(sorted_ks[-1])

    def consistent(k_val):
        n = 0
        for _, row in df.iterrows():
            if row["k_indiff"] > k_val and row["chose_delayed"]:
                n += 1
            elif row["k_indiff"] < k_val and not row["chose_delayed"]:
                n += 1
            elif abs(row["k_indiff"] - k_val) < 1e-10:
                n += 1
        return n

    scored = [(k, consistent(k)) for k in candidates]
    max_n = max(s[1] for s in scored)
    best = [s[0] for s in scored if s[1] == max_n]
    k = math.exp(sum(math.log(b) for b in best) / len(best))
    return dict(k=k, consistency=max_n / len(df),
                n_consistent=max_n, n_trials=len(df))


def classify_cot(reply):
    """Parse 'now' or 'later' from a CoT response.

    Strips <think>...</think> blocks, ignores echoed 'now or later'
    instruction text, then takes the last occurrence.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()
    if not cleaned:
        cleaned = reply
    cleaned_lower = cleaned.lower()
    # Remove instruction echoes
    cleaned_lower = re.sub(r"\bnow or later\b", "___", cleaned_lower)

    last_now = cleaned_lower.rfind("now")
    last_later = cleaned_lower.rfind("later")

    if last_now == -1 and last_later == -1:
        return cleaned_lower.strip()
    if last_later > last_now:
        return "delayed"
    return "now"


# ── Arch mapping for TransformerLens ─────────────────────────────────────
TL_ARCH_MAP = {
    "Qwen/Qwen3-4B-Instruct-2507": "Qwen/Qwen3-4B",
}


def _patched_init_transformerlens(self) -> None:
    arch_ref = TL_ARCH_MAP.get(self.model_name, self.model_name)
    in_registry = (arch_ref == self.model_name)
    print(f"Loading {self.model_name} on {self.device} (TransformerLens)...")
    if in_registry:
        self._model = HookedTransformer.from_pretrained_no_processing(
            self.model_name, device=self.device, dtype=self.dtype
        )
    else:
        print(f"  arch ref → {arch_ref}")
        hf_model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=self.dtype, device_map="cpu"
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = HookedTransformer.from_pretrained(
            arch_ref,
            hf_model=hf_model,
            tokenizer=tokenizer,
            dtype=self.dtype,
            move_to_device=True,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
        )
        del hf_model
    self._model.eval()
    self._backend = TransformerLensBackend(self)


ModelRunner._init_transformerlens = _patched_init_transformerlens


# ── Generation with steering ────────────────────────────────────────────
@torch.no_grad()
def generate_steered(runner, messages, caa_vec, layer, alpha,
                     max_new_tokens=200):
    """Generate with CAA steering applied at a given layer."""
    tokenizer = runner._model.tokenizer
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = runner.encode(formatted)
    prompt_len = input_ids.shape[1]

    with torch.autocast("cuda", dtype=torch.float16):
        if alpha == 0:
            output_ids = runner._model.generate(
                input_ids, max_new_tokens=max_new_tokens,
                do_sample=False, stop_at_eos=True, verbose=False,
                use_past_kv_cache=True,
            )
        else:
            hook_name = f"blocks.{layer}.hook_resid_post"
            hook_fn = (lambda value, hook, a=alpha, v=caa_vec:
                       value + a * v.to(value.device))
            with runner._model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
                output_ids = runner._model.generate(
                    input_ids, max_new_tokens=max_new_tokens,
                    do_sample=False, stop_at_eos=True, verbose=False,
                    use_past_kv_cache=False,
                )

    raw = runner._model.tokenizer.decode(
        output_ids[0, prompt_len:], skip_special_tokens=True
    ).strip()
    torch.cuda.empty_cache()
    return raw


# ── Run one condition ───────────────────────────────���────────────────────
def run_condition(runner, messages_prefix, trials_df, caa_vec, layer, alpha,
                  label, max_new_tokens=200):
    """Run all 27 Kirby questions with CoT + steering."""
    print(f"\n{'=' * 70}")
    print(f"  {label}  (layer={layer}, α={alpha:+d}, max_tokens={max_new_tokens})")
    print(f"{'=' * 70}\n")

    df = trials_df.copy()
    responses = []
    raw_replies = []
    t0 = time.time()

    for i, (_, trial) in enumerate(df.iterrows()):
        sir, ldr, delay = int(trial.sir), int(trial.ldr), int(trial.delay)
        q = f"Would you prefer ${sir} today, or ${ldr} in {delay} days?"
        msgs = messages_prefix + [{"role": "user", "content": q}]
        reply = generate_steered(runner, msgs, caa_vec, layer, alpha,
                                 max_new_tokens=max_new_tokens)
        ans = classify_cot(reply)
        responses.append(ans)
        raw_replies.append(reply)

        # Show first 3 questions' full reasoning, then just the answer
        if i < 3:
            reasoning_preview = reply.replace("\n", " ")
            if len(reasoning_preview) > 200:
                reasoning_preview = reasoning_preview[:197] + "..."
            print(f"  Q{int(trial.order):>2d}: ${sir:>3d} now vs ${ldr:>3d} "
                  f"in {delay:>3d}d (k={trial.k_indiff:.4f}) => {ans:>7s}")
            print(f"       [{reasoning_preview}]")
        else:
            print(f"  Q{int(trial.order):>2d}: ${sir:>3d} now vs ${ldr:>3d} "
                  f"in {delay:>3d}d (k={trial.k_indiff:.4f}) => {ans:>7s}")
        sys.stdout.flush()

    df["response"] = responses
    df["chose_delayed"] = df["response"].apply(lambda r: r == "delayed")
    df["magnitude"] = df["ldr"].apply(magnitude)

    result = estimate_k(df)
    result["alpha"] = alpha
    result["layer"] = layer
    result["label"] = label
    result["responses"] = responses
    result["raw_replies"] = raw_replies

    n_delayed = sum(1 for r in responses if r == "delayed")
    elapsed = time.time() - t0
    print(f"\n  k = {result['k']:.6f}   "
          f"Consistency: {result['n_consistent']}/{result['n_trials']} "
          f"({result['consistency']:.1%})")
    print(f"  Delayed choices: {n_delayed}/27 ({n_delayed / 27:.0%})")
    print(f"  Time: {elapsed:.0f}s ({elapsed / 27:.1f}s/question)")
    sys.stdout.flush()

    return result


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Steer Kirby MCQ-27 under Chain-of-Thought prompting"
    )
    parser.add_argument("--layer", type=int, default=STEER_LAYER,
                        help=f"Residual-stream layer to steer (default {STEER_LAYER})")
    parser.add_argument("--max-tokens", type=int, default=200,
                        help="Max new tokens for CoT generation (default 200)")
    args = parser.parse_args()

    steer_layer = args.layer
    max_tokens = args.max_tokens

    trials_df = parse_trials()
    print(f"Parsed {len(trials_df)} Kirby MCQ-27 trials")

    # Load model
    runner = ModelRunner(MODEL_NAME, backend=ModelBackend.TRANSFORMERLENS,
                         dtype=torch.float16)
    print(f"Model loaded: {runner.n_layers} layers, d_model={runner.d_model}")

    # Load CAA vector
    caa_vec = torch.load(CAA_VECTOR_PATH, map_location=runner.device)
    print(f"CAA vector loaded: shape={caa_vec.shape}, "
          f"norm={torch.norm(caa_vec):.4f}")

    all_results = []

    # ── Part A: Zero-shot CoT + steering ─────────────────────────────────
    print(f"\n{'#' * 70}")
    print(f"# PART A: Zero-Shot CoT + CAA Steering")
    print(f"{'#' * 70}")

    zs_prefix = [{"role": "system", "content": COT_SYSTEM_PROMPT}]

    for alpha in ALPHAS + NEG_ALPHAS:
        sign = "+" if alpha > 0 else ""
        if alpha == 0:
            label = "Zero-shot CoT baseline"
        else:
            label = f"Zero-shot CoT α={sign}{alpha}"
        result = run_condition(runner, zs_prefix, trials_df, caa_vec,
                               steer_layer, alpha, label,
                               max_new_tokens=max_tokens)
        result["condition"] = "zero_shot_cot"
        all_results.append(result)

    # ── Part B: Few-shot CoT + steering ──────────────────────────────────
    print(f"\n{'#' * 70}")
    print(f"# PART B: Few-Shot CoT + CAA Steering")
    print(f"{'#' * 70}")

    for alpha in ALPHAS + NEG_ALPHAS:
        sign = "+" if alpha > 0 else ""
        if alpha == 0:
            label = "Few-shot CoT baseline"
        else:
            label = f"Few-shot CoT α={sign}{alpha}"
        result = run_condition(runner, FEWSHOT_COT, trials_df, caa_vec,
                               steer_layer, alpha, label,
                               max_new_tokens=max_tokens)
        result["condition"] = "few_shot_cot"
        all_results.append(result)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"  SUMMARY: CAA Steering + CoT on Kirby MCQ-27 (layer {steer_layer})")
    print(f"{'=' * 80}\n")

    for condition_label, condition_key in [
        ("ZERO-SHOT CoT", "zero_shot_cot"),
        ("FEW-SHOT CoT", "few_shot_cot"),
    ]:
        subset = [r for r in all_results if r["condition"] == condition_key]
        print(f"  ── {condition_label} ──")
        print(f"  {'Label':<35s}  {'α':>5s}  {'k':>10s}  "
              f"{'Consist.':>10s}  {'Delayed':>8s}")
        print(f"  {'-' * 35}  {'-' * 5}  {'-' * 10}  "
              f"{'-' * 10}  {'-' * 8}")
        for r in subset:
            n_del = sum(1 for x in r["responses"] if x == "delayed")
            print(f"  {r['label']:<35s}  {r['alpha']:>+5d}  "
                  f"{r['k']:>10.6f}  {r['consistency']:>10.1%}  "
                  f"{n_del:>3d}/27")
        print()

    print(f"  Human controls:  k ≈ 0.013")
    print(f"  Human heroin:    k ≈ 0.025")

    # ── Reasoning quality: show one example at each alpha ────────────────
    print(f"\n{'=' * 80}")
    print(f"  SAMPLE REASONING: Q4 ($31 now vs $85 in 7d, k_indiff=0.25)")
    print(f"{'=' * 80}")

    q4_idx = trials_df[trials_df["order"] == 4].index[0]
    for r in all_results:
        raw = r["raw_replies"][q4_idx]
        ans = r["responses"][q4_idx]
        preview = raw.replace("\n", " ")
        if len(preview) > 300:
            preview = preview[:297] + "..."
        print(f"\n  [{r['label']}] => {ans}")
        print(f"  {preview}")

    # ── Save results ─────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "steered_kirby_cot.json"
    save_data = []
    for r in all_results:
        save_data.append({
            "label": r["label"],
            "condition": r["condition"],
            "alpha": r["alpha"],
            "layer": r["layer"],
            "k": r["k"],
            "consistency": r["consistency"],
            "n_consistent": r["n_consistent"],
            "n_trials": r["n_trials"],
            "responses": r["responses"],
            "raw_replies": r["raw_replies"],
        })
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved → {out_path}")
