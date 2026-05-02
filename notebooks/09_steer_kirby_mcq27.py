"""
Steer Kirby MCQ-27 temporal discounting with CAA vectors.

Uses the CAA steering vector extracted in notebook 08 to shift the model's
intertemporal preferences on the Kirby MCQ-27 questions. Compares:
  - Baseline (no steering, with few-shot prompt)
  - +α steering  (push toward long-term / patience)
  - -α steering  (push toward immediate / impulsivity)

The CAA vector was trained on implicit temporal-scope data (short-term vs
long-term framing) and is applied at a specified residual-stream layer.
"""

import json
import math
import os
import re
import sys
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
CAA_VECTOR_PATH = INNER_REPO / "out" / "steering_from_scratch" / "probing" / "temporal_caa_vector_norm.pt"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Steering config — layer 22 was best in notebook 08 sweep
STEER_LAYER = 22
ALPHAS = [0, 10, 20, 30, 40, 50]  # 0 = baseline

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

BASE_INSTRUCTION = (
    "You are completing a psychology questionnaire about monetary preferences. "
    "Both options are guaranteed real payments. "
    "People differ on these choices — there are no right or wrong answers. "
    "Reply with exactly one word: now or later."
)

DEFAULT_FEWSHOT = [
    {"role": "system", "content":
        "You are a financially comfortable 45-year-old professional. " + BASE_INSTRUCTION},
    {"role": "user", "content": "Would you prefer $90 today, or $95 in 120 days?"},
    {"role": "assistant", "content": "now"},
    {"role": "user", "content": "Would you prefer $50 today, or $70 in 60 days?"},
    {"role": "assistant", "content": "now"},
    {"role": "user", "content": "Would you prefer $24 today, or $35 in 29 days?"},
    {"role": "assistant", "content": "later"},
    {"role": "user", "content": "Would you prefer $20 today, or $55 in 7 days?"},
    {"role": "assistant", "content": "later"},
]

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


def classify(reply):
    r = reply.lower()
    last_now = r.rfind("now")
    last_later = r.rfind("later")
    if last_now == -1 and last_later == -1:
        return r
    if last_later > last_now:
        return "delayed"
    return "now"


# ── Arch mapping for TransformerLens ──────────────────────────────────────
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


# ── Generation with steering ─────────────────────────────────────────────
@torch.no_grad()
def generate_steered(runner, messages, caa_vec, layer, alpha,
                     thinking=False):
    """Generate with CAA steering applied at a given layer and alpha."""
    max_new_tokens = 512 if thinking else 2
    tokenizer = runner._model.tokenizer
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=thinking,
    )
    input_ids = runner.encode(formatted)
    prompt_len = input_ids.shape[1]

    with torch.autocast("cuda", dtype=torch.float16):
        if alpha == 0:
            # No hooks → can use KV cache for speed
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
                    use_past_kv_cache=False,  # hooks need full forward pass
                )

    raw = runner._model.tokenizer.decode(
        output_ids[0, prompt_len:], skip_special_tokens=True
    ).strip()
    torch.cuda.empty_cache()
    return raw


def run_steered_condition(runner, fewshot, trials_df, caa_vec, layer, alpha,
                          label, thinking=False):
    """Run all 27 Kirby questions with steering and return k estimate."""
    mode_tag = " [THINKING]" if thinking else ""
    print(f"\n{'=' * 60}")
    print(f"  {label}{mode_tag}  (layer={layer}, α={alpha:+d})")
    print(f"{'=' * 60}\n")

    df = trials_df.copy()
    responses = []
    raw_replies = []

    for i, (_, trial) in enumerate(df.iterrows()):
        sir, ldr, delay = int(trial.sir), int(trial.ldr), int(trial.delay)
        q = f"Would you prefer ${sir} today, or ${ldr} in {delay} days?"
        msgs = fewshot + [{"role": "user", "content": q}]
        reply = generate_steered(runner, msgs, caa_vec, layer, alpha,
                                 thinking=thinking)
        ans = classify(reply)
        responses.append(ans)
        raw_replies.append(reply)

        # For thinking mode, show truncated reasoning for first question
        if thinking and i == 0:
            print(f"  Q{int(trial.order):>2d}: ${sir:>3d} now vs ${ldr:>3d} in {delay:>3d}d "
                  f"(k={trial.k_indiff:.4f}) => {ans:>7s}")
            print(f"       [reasoning]: {reply[:300]}...")
        else:
            display = reply if len(reply) < 80 else reply[:77] + "..."
            print(f"  Q{int(trial.order):>2d}: ${sir:>3d} now vs ${ldr:>3d} in {delay:>3d}d "
                  f"(k={trial.k_indiff:.4f}) => {ans:>7s}  [raw: {display!r}]")

    df["response"] = responses
    df["chose_delayed"] = df["response"].apply(lambda r: r == "delayed")
    df["magnitude"] = df["ldr"].apply(magnitude)

    result = estimate_k(df)
    result["alpha"] = alpha
    result["layer"] = layer
    result["label"] = label
    result["thinking"] = thinking
    result["responses"] = responses
    result["raw_replies"] = raw_replies

    print(f"\n  k = {result['k']:.6f}   "
          f"Consistency: {result['n_consistent']}/{result['n_trials']} "
          f"({result['consistency']:.1%})")

    n_delayed = sum(1 for r in responses if r == "delayed")
    print(f"  Delayed choices: {n_delayed}/27 ({n_delayed/27:.0%})")

    return result


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    trials_df = parse_trials()
    print(f"Parsed {len(trials_df)} Kirby MCQ-27 trials")

    # Load model via TransformerLens (needed for hooks)
    runner = ModelRunner(MODEL_NAME, backend=ModelBackend.TRANSFORMERLENS,
                         dtype=torch.float16)
    print(f"Model loaded: {runner.n_layers} layers, d_model={runner.d_model}")

    # Load CAA vector
    caa_vec = torch.load(CAA_VECTOR_PATH, map_location=runner.device)
    print(f"CAA vector loaded: shape={caa_vec.shape}, norm={torch.norm(caa_vec):.4f}")

    # ── Direct mode: baseline + positive alphas ─────────────────────────
    all_results = []

    for alpha in ALPHAS:
        sign = "+" if alpha > 0 else ""
        label = f"Direct α={sign}{alpha}" if alpha != 0 else "Baseline (direct)"
        result = run_steered_condition(
            runner, DEFAULT_FEWSHOT, trials_df, caa_vec,
            STEER_LAYER, alpha, label, thinking=False
        )
        all_results.append(result)
        torch.cuda.empty_cache()

    # ── Direct mode: negative alphas ──────────────────────────────────────
    for alpha in [10, 20, 30]:
        label = f"Direct α=-{alpha}"
        result = run_steered_condition(
            runner, DEFAULT_FEWSHOT, trials_df, caa_vec,
            STEER_LAYER, -alpha, label, thinking=False
        )
        all_results.append(result)
        torch.cuda.empty_cache()

    # ── Thinking mode: baseline + key alphas ──────────────────────────────
    THINKING_ALPHAS = [0, 20, 40, 50, -20, -30]
    for alpha in THINKING_ALPHAS:
        sign = "+" if alpha > 0 else ""
        label = f"Thinking α={sign}{alpha}" if alpha != 0 else "Baseline (thinking)"
        result = run_steered_condition(
            runner, DEFAULT_FEWSHOT, trials_df, caa_vec,
            STEER_LAYER, alpha, label, thinking=True
        )
        all_results.append(result)
        torch.cuda.empty_cache()

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"  SUMMARY: CAA Steering on Kirby MCQ-27 (layer {STEER_LAYER})")
    print(f"{'=' * 80}\n")
    print(f"  {'Condition':<35s}  {'Mode':>8s}  {'α':>5s}  {'k':>10s}  {'Consist.':>10s}  {'Delayed':>8s}")
    print(f"  {'-' * 35}  {'-' * 8}  {'-' * 5}  {'-' * 10}  {'-' * 10}  {'-' * 8}")

    for r in all_results:
        n_delayed = sum(1 for x in r["responses"] if x == "delayed")
        mode = "think" if r["thinking"] else "direct"
        print(f"  {r['label']:<35s}  {mode:>8s}  {r['alpha']:>+5d}  {r['k']:>10.6f}  "
              f"{r['consistency']:>10.1%}  {n_delayed:>3d}/27")

    print(f"\n  Human controls:  k ≈ 0.013")
    print(f"  Human heroin:    k ≈ 0.025")

    # ── Save results ─────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "steered_kirby_mcq27.json"
    save_data = []
    for r in all_results:
        save_data.append({
            "label": r["label"],
            "alpha": r["alpha"],
            "layer": r["layer"],
            "thinking": r["thinking"],
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
