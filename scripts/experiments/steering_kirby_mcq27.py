#!/usr/bin/env python3
"""
Steering Vectors Replacing Few-Shot for Kirby MCQ-27.

Can a steering vector applied to zero-shot prompts reproduce the discount
rates achieved by calibrated few-shot prompting?

Approach:
  1. Run all 27 Kirby MCQ-27 questions through Gemma-2-2B under three
     conditions: zero-shot, default few-shot (k≈0.013), heroin few-shot (k≈0.025)
  2. Extract last-token residual stream activations at multiple layers
  3. Compute DiffMean steering vectors:  sv = mean(fewshot_acts) - mean(zero_acts)
  4. Apply steering to zero-shot prompts at varying alpha
  5. Measure P("now") and P("later") from steered logits
  6. Compute k at each alpha; find alpha that best matches few-shot k

Since Gemma-2-2B is a base model (no chat template), we use a simple
Q/A completion format rather than chat messages.

Usage:
    python steering_kirby_mcq27.py
    python steering_kirby_mcq27.py --layers 6 13 20 25
    python steering_kirby_mcq27.py --device cuda
"""

import sys
import math
import json
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm

# ============================================================
# Kirby MCQ-27 (same questions as discount_factor_best.py)
# ============================================================

KIRBY_QUESTIONS = [
    dict(order=1,  sir=54, ldr=55,  delay=117),
    dict(order=2,  sir=55, ldr=75,  delay=61),
    dict(order=3,  sir=19, ldr=25,  delay=53),
    dict(order=4,  sir=31, ldr=85,  delay=7),
    dict(order=5,  sir=14, ldr=25,  delay=19),
    dict(order=6,  sir=47, ldr=50,  delay=160),
    dict(order=7,  sir=15, ldr=35,  delay=13),
    dict(order=8,  sir=25, ldr=60,  delay=14),
    dict(order=9,  sir=78, ldr=80,  delay=162),
    dict(order=10, sir=40, ldr=55,  delay=62),
    dict(order=11, sir=11, ldr=30,  delay=7),
    dict(order=12, sir=67, ldr=75,  delay=119),
    dict(order=13, sir=34, ldr=35,  delay=186),
    dict(order=14, sir=27, ldr=50,  delay=21),
    dict(order=15, sir=69, ldr=85,  delay=91),
    dict(order=16, sir=49, ldr=60,  delay=89),
    dict(order=17, sir=80, ldr=85,  delay=157),
    dict(order=18, sir=24, ldr=35,  delay=29),
    dict(order=19, sir=33, ldr=80,  delay=14),
    dict(order=20, sir=28, ldr=30,  delay=179),
    dict(order=21, sir=34, ldr=50,  delay=30),
    dict(order=22, sir=25, ldr=30,  delay=80),
    dict(order=23, sir=41, ldr=75,  delay=20),
    dict(order=24, sir=54, ldr=60,  delay=111),
    dict(order=25, sir=54, ldr=80,  delay=30),
    dict(order=26, sir=22, ldr=25,  delay=136),
    dict(order=27, sir=20, ldr=55,  delay=7),
]


def sigfigs(x, n=2):
    if x == 0:
        return 0
    return round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1))


def add_k_indiff(questions):
    """Add k_indiff to each question dict."""
    for q in questions:
        q["k_indiff"] = sigfigs((q["ldr"] / q["sir"] - 1) / q["delay"])
    return questions


def magnitude(ldr):
    if ldr <= 35:
        return "small"
    elif ldr <= 60:
        return "medium"
    return "large"


def estimate_k(chose_delayed_list, k_indiffs):
    """Estimate k using Kirby maximum-consistency method."""
    sorted_ks = sorted(set(k_indiffs))
    candidates = [sorted_ks[0]]
    for i in range(len(sorted_ks) - 1):
        candidates.append(math.sqrt(sorted_ks[i] * sorted_ks[i + 1]))
    candidates.append(sorted_ks[-1])

    def consistent(k_val):
        n = 0
        for ki, chose_d in zip(k_indiffs, chose_delayed_list):
            if ki > k_val and chose_d:
                n += 1
            elif ki < k_val and not chose_d:
                n += 1
            elif abs(ki - k_val) < 1e-10:
                n += 1
        return n

    scored = [(k, consistent(k)) for k in candidates]
    max_n = max(s[1] for s in scored)
    best = [s[0] for s in scored if s[1] == max_n]
    k = math.exp(sum(math.log(b) for b in best) / len(best))
    return dict(k=k, consistency=max_n / len(chose_delayed_list),
                n_consistent=max_n, n_trials=len(chose_delayed_list))


# ============================================================
# Prompt formatting (completion-style for base Gemma-2-2B)
# ============================================================

def format_question(q):
    return f"Would you prefer ${q['sir']} today, or ${q['ldr']} in {q['delay']} days?"


# Few-shot examples matching discount_factor_best.py flip points
DEFAULT_FEWSHOT_EXAMPLES = [
    ("Would you prefer $90 today, or $95 in 120 days?", "now"),
    ("Would you prefer $50 today, or $70 in 60 days?", "now"),
    ("Would you prefer $24 today, or $35 in 29 days?", "later"),
    ("Would you prefer $20 today, or $55 in 7 days?", "later"),
]

HEROIN_FEWSHOT_EXAMPLES = [
    ("Would you prefer $90 today, or $95 in 120 days?", "now"),
    ("Would you prefer $50 today, or $70 in 60 days?", "now"),
    ("Would you prefer $25 today, or $60 in 14 days?", "later"),
    ("Would you prefer $20 today, or $55 in 7 days?", "later"),
]

INSTRUCTION = (
    "You are completing a psychology questionnaire about monetary preferences. "
    "Both options are guaranteed real payments. "
    "Reply with exactly one word: now or later.\n\n"
)


def make_prompt(question_text, fewshot_examples=None):
    """Build a completion prompt with optional few-shot examples."""
    parts = [INSTRUCTION]
    if fewshot_examples:
        for q_text, answer in fewshot_examples:
            parts.append(f"Q: {q_text}\nA: {answer}\n\n")
    parts.append(f"Q: {question_text}\nA:")
    return "".join(parts)


# ============================================================
# Core experiment
# ============================================================

@dataclass
class Config:
    model_name: str = "gemma-2-2b"
    device: str = "cpu"
    layers: List[int] = field(default_factory=lambda: [6, 13, 20, 25])
    alphas: List[float] = field(default_factory=lambda: [0, 1, 2, 5, 10, 25, 50, 100])
    batch_size: int = 8
    output_dir: str = "steering_kirby"


def extract_activations_and_logits(model, prompts, layers, batch_size=8):
    """Extract last-token activations at multiple layers + logits.

    Returns:
        activations: dict[layer] -> np.ndarray (n_prompts, d_model)
        logits: np.ndarray (n_prompts, vocab_size) — logits at last token
    """
    activations = {layer: [] for layer in layers}
    all_logits = []

    hook_names = [f"blocks.{l}.hook_resid_post" for l in layers]

    for i in tqdm(range(0, len(prompts), batch_size), desc="Extracting"):
        batch = prompts[i:i + batch_size]

        with torch.no_grad():
            logits, cache = model.run_with_cache(
                batch,
                names_filter=hook_names,
                return_type="logits",
            )

        # Last-token logits
        # For batched inputs with padding, we need the actual last token
        # TransformerLens pads on the left, so last position is always valid
        all_logits.append(logits[:, -1, :].detach().float().cpu().numpy())

        for layer in layers:
            hook = f"blocks.{layer}.hook_resid_post"
            acts = cache[hook][:, -1, :].detach().float().cpu().numpy()
            activations[layer].append(acts)

    for layer in layers:
        activations[layer] = np.concatenate(activations[layer], axis=0)
    all_logits = np.concatenate(all_logits, axis=0)

    return activations, all_logits


def get_token_ids(model, tokens):
    """Get token IDs for a list of token strings."""
    ids = {}
    for t in tokens:
        toks = model.to_tokens(t, prepend_bos=False)
        ids[t] = toks[0, 0].item()
    return ids


def logits_to_choice(logits_row, now_id, later_id):
    """Convert logit row to P(now), P(later), and choice."""
    p_now = logits_row[now_id]
    p_later = logits_row[later_id]
    # softmax over just these two tokens
    max_val = max(p_now, p_later)
    exp_now = math.exp(p_now - max_val)
    exp_later = math.exp(p_later - max_val)
    total = exp_now + exp_later
    return exp_now / total, exp_later / total


def run_experiment(config: Config):
    from transformer_lens import HookedTransformer

    print(f"Steering Kirby MCQ-27 Experiment")
    print(f"Model: {config.model_name}")
    print(f"Layers: {config.layers}")
    print(f"Alphas: {config.alphas}")
    print(f"Device: {config.device}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    # Setup
    questions = add_k_indiff(KIRBY_QUESTIONS)
    k_indiffs = [q["k_indiff"] for q in questions]

    results_dir = Path(__file__).parent.parent.parent / "results" / config.output_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading model...")
    model = HookedTransformer.from_pretrained(config.model_name, device=config.device)
    print(f"  d_model = {model.cfg.d_model}, n_layers = {model.cfg.n_layers}\n")

    # Get token IDs for "now" and "later"
    # Try common tokenizations
    token_ids = get_token_ids(model, [" now", " later", "now", "later"])
    now_id = token_ids.get(" now", token_ids.get("now"))
    later_id = token_ids.get(" later", token_ids.get("later"))
    print(f"  Token IDs: now={now_id}, later={later_id}")

    # ============================================================
    # Phase 1: Extract activations under three conditions
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  Phase 1: Extract activations")
    print(f"{'='*60}\n")

    conditions = {
        "zero_shot": None,
        "default_fewshot": DEFAULT_FEWSHOT_EXAMPLES,
        "heroin_fewshot": HEROIN_FEWSHOT_EXAMPLES,
    }

    condition_acts = {}   # condition -> {layer -> (27, d_model)}
    condition_logits = {} # condition -> (27, vocab)
    condition_choices = {}

    for cond_name, fewshot in conditions.items():
        print(f"\n  --- {cond_name} ---")
        prompts = [make_prompt(format_question(q), fewshot) for q in questions]

        # Show first prompt for verification
        if cond_name == "zero_shot":
            print(f"  Sample prompt:\n{prompts[0][:300]}...\n")

        acts, logits = extract_activations_and_logits(
            model, prompts, config.layers, config.batch_size)

        condition_acts[cond_name] = acts
        condition_logits[cond_name] = logits

        # Score this condition
        choices = []
        for j in range(len(questions)):
            p_now, p_later = logits_to_choice(logits[j], now_id, later_id)
            chose_delayed = p_later > p_now
            choices.append(chose_delayed)
            q = questions[j]
            label = "later" if chose_delayed else "now"
            print(f"    Q{q['order']:>2d}: ${q['sir']:>2d} vs ${q['ldr']:>2d} in {q['delay']:>3d}d "
                  f"(k={q['k_indiff']:.4f}) => {label}  "
                  f"[P(now)={p_now:.3f}, P(later)={p_later:.3f}]")

        condition_choices[cond_name] = choices
        result = estimate_k(choices, k_indiffs)
        print(f"\n    k = {result['k']:.6f}   Consistency: {result['consistency']:.1%}")

    # ============================================================
    # Phase 2: Compute steering vectors
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  Phase 2: Compute steering vectors")
    print(f"{'='*60}\n")

    steering_vectors = {}  # (direction, layer) -> vector

    for layer in config.layers:
        zero_acts = condition_acts["zero_shot"][layer]
        default_acts = condition_acts["default_fewshot"][layer]
        heroin_acts = condition_acts["heroin_fewshot"][layer]

        # DiffMean vectors
        sv_default = default_acts.mean(axis=0) - zero_acts.mean(axis=0)
        sv_heroin = heroin_acts.mean(axis=0) - zero_acts.mean(axis=0)
        sv_heroin_vs_default = heroin_acts.mean(axis=0) - default_acts.mean(axis=0)

        # Normalize
        for name, sv in [("default", sv_default), ("heroin", sv_heroin),
                         ("heroin_vs_default", sv_heroin_vs_default)]:
            norm = np.linalg.norm(sv)
            steering_vectors[(name, layer)] = sv / norm if norm > 0 else sv
            print(f"  Layer {layer:>2d}, {name:<20s}: ||sv|| = {norm:.4f}")

        # Cosine similarity between default and heroin directions
        cos_sim = np.dot(sv_default, sv_heroin) / (
            np.linalg.norm(sv_default) * np.linalg.norm(sv_heroin) + 1e-10)
        print(f"  Layer {layer:>2d}, cos(default, heroin) = {cos_sim:.4f}")

    # ============================================================
    # Phase 3: Apply steering to zero-shot prompts
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  Phase 3: Steer zero-shot prompts")
    print(f"{'='*60}\n")

    zero_prompts = [make_prompt(format_question(q), None) for q in questions]
    steering_results = {}  # (direction, layer, alpha) -> {k, consistency, choices}

    for direction in ["default", "heroin"]:
        for layer in config.layers:
            sv = steering_vectors[(direction, layer)]

            for alpha in config.alphas:
                hook_name = f"blocks.{layer}.hook_resid_post"

                def steering_hook(activation, hook, sv=sv, alpha=alpha):
                    activation[:, -1, :] += torch.tensor(
                        sv * alpha, dtype=activation.dtype, device=activation.device)
                    return activation

                # Run with hook
                choices = []
                for j, prompt in enumerate(zero_prompts):
                    with torch.no_grad():
                        logits = model.run_with_hooks(
                            prompt,
                            fwd_hooks=[(hook_name, steering_hook)],
                            return_type="logits",
                        )
                    last_logits = logits[0, -1, :].float().cpu().numpy()
                    p_now, p_later = logits_to_choice(last_logits, now_id, later_id)
                    choices.append(p_later > p_now)

                result = estimate_k(choices, k_indiffs)
                n_later = sum(choices)
                steering_results[(direction, layer, alpha)] = {
                    "k": result["k"],
                    "consistency": result["consistency"],
                    "n_later": n_later,
                    "choices": choices,
                }

                if alpha in [0, 10, 50, 100]:
                    print(f"  {direction:>7s} | layer {layer:>2d} | α={alpha:>5.0f} | "
                          f"k={result['k']:.6f} | {n_later:>2d}/27 later | "
                          f"consist={result['consistency']:.0%}")

    # ============================================================
    # Phase 4: Summary
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}\n")

    # Baselines
    zero_k = estimate_k(condition_choices["zero_shot"], k_indiffs)
    default_k = estimate_k(condition_choices["default_fewshot"], k_indiffs)
    heroin_k = estimate_k(condition_choices["heroin_fewshot"], k_indiffs)

    print(f"  Baselines:")
    print(f"    Zero-shot k     = {zero_k['k']:.6f}")
    print(f"    Default few-shot k = {default_k['k']:.6f}  (target: 0.013)")
    print(f"    Heroin few-shot k  = {heroin_k['k']:.6f}  (target: 0.025)")

    # Best steering results per direction
    print(f"\n  Best steering match per layer:")
    print(f"  {'Direction':<10s}  {'Layer':>5s}  {'Alpha':>6s}  {'Steered k':>10s}  "
          f"{'Target k':>9s}  {'Error':>8s}")
    print(f"  {'-'*10}  {'-'*5}  {'-'*6}  {'-'*10}  {'-'*9}  {'-'*8}")

    for direction, target_k in [("default", 0.013), ("heroin", 0.025)]:
        for layer in config.layers:
            best_alpha = None
            best_error = float("inf")
            best_k = None
            for alpha in config.alphas:
                r = steering_results[(direction, layer, alpha)]
                error = abs(math.log(r["k"] + 1e-10) - math.log(target_k))
                if error < best_error:
                    best_error = error
                    best_alpha = alpha
                    best_k = r["k"]
            print(f"  {direction:<10s}  {layer:>5d}  {best_alpha:>6.0f}  "
                  f"{best_k:>10.6f}  {target_k:>9.3f}  "
                  f"{abs(best_k - target_k):>8.4f}")

    # ============================================================
    # Save results
    # ============================================================
    save_data = {
        "config": asdict(config),
        "baselines": {
            "zero_shot": {"k": zero_k["k"], "consistency": zero_k["consistency"]},
            "default_fewshot": {"k": default_k["k"], "consistency": default_k["consistency"]},
            "heroin_fewshot": {"k": heroin_k["k"], "consistency": heroin_k["consistency"]},
        },
        "steering_results": {
            f"{d}__layer{l}__alpha{a}": {
                "k": r["k"], "consistency": r["consistency"],
                "n_later": r["n_later"],
            }
            for (d, l, a), r in steering_results.items()
        },
        "steering_vectors": {
            f"{name}__layer{layer}": {
                "norm": float(np.linalg.norm(sv)),
            }
            for (name, layer), sv in steering_vectors.items()
        },
        "timestamp": datetime.now().isoformat(),
    }

    outpath = results_dir / "steering_kirby_results.json"
    with open(outpath, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to {outpath}")

    # Save steering vectors as .npy for reuse
    for (name, layer), sv in steering_vectors.items():
        np.save(results_dir / f"sv_{name}_layer{layer}.npy", sv)
    print(f"  Steering vectors saved to {results_dir}/sv_*.npy")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Steering vectors for Kirby MCQ-27")
    parser.add_argument("--layers", type=int, nargs="+", default=[6, 13, 20, 25])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[0, 1, 2, 5, 10, 25, 50, 100])
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    config = Config(
        layers=args.layers,
        device=args.device,
        alphas=args.alphas,
        batch_size=args.batch_size,
    )
    run_experiment(config)
