#!/usr/bin/env python3
"""
UMAP + Probing of Kirby MCQ-27 Trials Under Different Flip Points.

Runs all 27 Kirby questions through Gemma-2-2B with 4 few-shot configurations,
extracts activations at the decision token, and produces:

  1. Linear probes: predict now/later from activations (5-fold CV accuracy)
  2. UMAP visualizations:
     - Color by model choice (now=red, later=blue)
     - Color by condition (zero-shot / default / heroin / ...)
     - Color by k_indiff (continuous, log scale)
  3. Boundary analysis: how does the probe decision boundary shift across
     few-shot configurations?

This script shares the same model (Gemma-2-2B) and prompt format as
steering_kirby_mcq27.py, so results are directly comparable.

Usage:
    python 07_kirby_umap_probing.py
    python 07_kirby_umap_probing.py --layers 6 13 20 25
    python 07_kirby_umap_probing.py --device cuda
"""

import sys
import math
import json
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from tqdm import tqdm

import umap
from sklearn.decomposition import PCA

# Add src to path for probe import
_src = str(Path(__file__).parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from probes.probe import LinearProbe

# ============================================================
# Kirby MCQ-27 (shared with steering_kirby_mcq27.py)
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
    for q in questions:
        q["k_indiff"] = sigfigs((q["ldr"] / q["sir"] - 1) / q["delay"])
    return questions


def estimate_k(chose_delayed_list, k_indiffs):
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
# Prompt formatting
# ============================================================

INSTRUCTION = (
    "You are completing a psychology questionnaire about monetary preferences. "
    "Both options are guaranteed real payments. "
    "Reply with exactly one word: now or later.\n\n"
)

DEFAULT_FEWSHOT = [
    ("Would you prefer $90 today, or $95 in 120 days?", "now"),
    ("Would you prefer $50 today, or $70 in 60 days?", "now"),
    ("Would you prefer $24 today, or $35 in 29 days?", "later"),
    ("Would you prefer $20 today, or $55 in 7 days?", "later"),
]

HEROIN_FEWSHOT = [
    ("Would you prefer $90 today, or $95 in 120 days?", "now"),
    ("Would you prefer $50 today, or $70 in 60 days?", "now"),
    ("Would you prefer $25 today, or $60 in 14 days?", "later"),
    ("Would you prefer $20 today, or $55 in 7 days?", "later"),
]

# Extra condition: very patient (flip at very low k)
PATIENT_FEWSHOT = [
    ("Would you prefer $90 today, or $95 in 120 days?", "now"),
    ("Would you prefer $50 today, or $70 in 60 days?", "later"),
    ("Would you prefer $24 today, or $35 in 29 days?", "later"),
    ("Would you prefer $20 today, or $55 in 7 days?", "later"),
]


def format_question(q):
    return f"Would you prefer ${q['sir']} today, or ${q['ldr']} in {q['delay']} days?"


def make_prompt(question_text, fewshot=None):
    parts = [INSTRUCTION]
    if fewshot:
        for q_text, answer in fewshot:
            parts.append(f"Q: {q_text}\nA: {answer}\n\n")
    parts.append(f"Q: {question_text}\nA:")
    return "".join(parts)


# ============================================================
# Activation extraction
# ============================================================

def extract_activations_and_logits(model, prompts, layers, batch_size=8):
    """Extract last-token activations at multiple layers + logits."""
    activations = {layer: [] for layer in layers}
    all_logits = []
    hook_names = [f"blocks.{l}.hook_resid_post" for l in layers]

    for i in tqdm(range(0, len(prompts), batch_size), desc="  Extracting"):
        batch = prompts[i:i + batch_size]
        with torch.no_grad():
            logits, cache = model.run_with_cache(
                batch, names_filter=hook_names, return_type="logits")

        all_logits.append(logits[:, -1, :].detach().float().cpu().numpy())
        for layer in layers:
            hook = f"blocks.{layer}.hook_resid_post"
            acts = cache[hook][:, -1, :].detach().float().cpu().numpy()
            activations[layer].append(acts)

    for layer in layers:
        activations[layer] = np.concatenate(activations[layer], axis=0)
    return activations, np.concatenate(all_logits, axis=0)


def logits_to_choice(logits_row, now_id, later_id):
    p_now = logits_row[now_id]
    p_later = logits_row[later_id]
    max_val = max(p_now, p_later)
    exp_now = math.exp(p_now - max_val)
    exp_later = math.exp(p_later - max_val)
    total = exp_now + exp_later
    return exp_now / total, exp_later / total


# ============================================================
# Main
# ============================================================

def main(layers, device, batch_size):
    questions = add_k_indiff(KIRBY_QUESTIONS)
    k_indiffs = [q["k_indiff"] for q in questions]
    k_indiffs_arr = np.array(k_indiffs)

    results_dir = Path(__file__).parent.parent / "results" / "kirby_umap_probing"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Kirby MCQ-27 UMAP + Probing Experiment")
    print(f"Layers: {layers}, Device: {device}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    # Load model
    from transformer_lens import HookedTransformer
    print("Loading Gemma-2-2B...")
    model = HookedTransformer.from_pretrained("gemma-2-2b", device=device)
    d_model = model.cfg.d_model
    print(f"  d_model = {d_model}\n")

    # Token IDs
    token_ids = {}
    for t in [" now", " later", "now", "later"]:
        toks = model.to_tokens(t, prepend_bos=False)
        token_ids[t] = toks[0, 0].item()
    now_id = token_ids.get(" now", token_ids.get("now"))
    later_id = token_ids.get(" later", token_ids.get("later"))
    print(f"  Token IDs: now={now_id}, later={later_id}\n")

    # ============================================================
    # Run all conditions
    # ============================================================
    CONDITIONS = {
        "zero_shot": None,
        "patient": PATIENT_FEWSHOT,
        "default": DEFAULT_FEWSHOT,
        "heroin": HEROIN_FEWSHOT,
    }
    COND_COLORS = {
        "zero_shot": "#888888",
        "patient": "#2196F3",
        "default": "#4CAF50",
        "heroin": "#FF9800",
    }

    all_acts = {layer: [] for layer in layers}
    all_choices = []       # bool: True=later
    all_conditions = []    # str: condition name
    all_k_indiffs = []     # float: k_indiff of each trial
    all_orders = []        # int: question order
    condition_results = {} # condition -> {k, consistency}

    for cond_name, fewshot in CONDITIONS.items():
        print(f"\n--- {cond_name} ---")
        prompts = [make_prompt(format_question(q), fewshot) for q in questions]
        acts, logits = extract_activations_and_logits(model, prompts, layers, batch_size)

        choices = []
        for j in range(len(questions)):
            p_now, p_later = logits_to_choice(logits[j], now_id, later_id)
            choices.append(p_later > p_now)

        result = estimate_k(choices, k_indiffs)
        condition_results[cond_name] = result
        n_later = sum(choices)
        print(f"    k = {result['k']:.6f}  Consistency: {result['consistency']:.0%}  "
              f"({n_later}/27 later)")

        for layer in layers:
            all_acts[layer].append(acts[layer])
        all_choices.extend(choices)
        all_conditions.extend([cond_name] * 27)
        all_k_indiffs.extend(k_indiffs)
        all_orders.extend([q["order"] for q in questions])

    # Stack everything
    for layer in layers:
        all_acts[layer] = np.concatenate(all_acts[layer], axis=0)
    all_choices = np.array(all_choices, dtype=int)  # 0=now, 1=later
    all_k_indiffs = np.array(all_k_indiffs)
    n_total = len(all_choices)
    n_conditions = len(CONDITIONS)

    print(f"\n  Total samples: {n_total} ({n_conditions} conditions × 27 questions)")

    # ============================================================
    # Linear probing
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  Linear Probing: predict now/later from activations")
    print(f"{'='*60}\n")

    probe_results = {}
    probe_vectors = {}

    for layer in layers:
        X = all_acts[layer]
        y = all_choices

        probe = LinearProbe(regularization_C=1.0)
        cv_mean, cv_std, train_acc = probe.train(X, y, n_cv_folds=5)
        sv = probe.get_steering_vector(normalize=True)
        bias = probe.get_bias()

        # Probe logit (signed distance from boundary) for each sample
        probe_logits = X @ sv + bias

        probe_results[layer] = {
            "cv_accuracy": f"{cv_mean:.3f} ± {cv_std:.3f}",
            "train_accuracy": train_acc,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
        }
        probe_vectors[layer] = sv

        print(f"  Layer {layer:>2d}:  CV = {cv_mean:.1%} ± {cv_std:.1%}  "
              f"Train = {train_acc:.1%}")

        # Per-condition probe logit means (shows boundary shift)
        print(f"           Mean probe logit by condition:")
        for cond_name in CONDITIONS:
            mask = np.array([c == cond_name for c in all_conditions])
            mean_logit = probe_logits[mask].mean()
            std_logit = probe_logits[mask].std()
            print(f"             {cond_name:<12s}: {mean_logit:>+6.3f} ± {std_logit:.3f}")

    # ============================================================
    # Compare probe vectors with steering vectors (if available)
    # ============================================================
    sv_dir = Path(__file__).parent.parent / "results" / "steering_kirby"
    if sv_dir.exists():
        print(f"\n  Comparing probe vectors with steering vectors from Experiment 2:")
        for layer in layers:
            for direction in ["default", "heroin"]:
                sv_path = sv_dir / f"sv_{direction}_layer{layer}.npy"
                if sv_path.exists():
                    steer_sv = np.load(sv_path)
                    probe_sv = probe_vectors[layer]
                    cos_sim = np.dot(steer_sv, probe_sv) / (
                        np.linalg.norm(steer_sv) * np.linalg.norm(probe_sv) + 1e-10)
                    print(f"    Layer {layer}, {direction} steer vs probe: cos = {cos_sim:.4f}")

    # ============================================================
    # UMAP + PCA visualizations
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  Generating UMAP + PCA visualizations")
    print(f"{'='*60}\n")

    for layer in layers:
        X = all_acts[layer]

        # --- PCA (more stable for small n) ---
        pca = PCA(n_components=2)
        pca_emb = pca.fit_transform(X)

        # --- UMAP ---
        n_neighbors = min(15, n_total // 4)
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
        umap_emb = reducer.fit_transform(X)

        for method_name, emb in [("PCA", pca_emb), ("UMAP", umap_emb)]:
            fig, axes = plt.subplots(1, 3, figsize=(20, 6))

            # Plot 1: Color by choice (now/later)
            ax = axes[0]
            colors = ["#E53935" if c == 0 else "#1E88E5" for c in all_choices]
            ax.scatter(emb[:, 0], emb[:, 1], c=colors, alpha=0.7, s=40, edgecolors="white", linewidths=0.3)
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#E53935', markersize=8, label='now'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#1E88E5', markersize=8, label='later'),
            ]
            ax.legend(handles=legend_elements, loc="best")
            ax.set_title(f"Model Choice (now/later)")
            ax.set_xlabel(f"{method_name} 1")
            ax.set_ylabel(f"{method_name} 2")

            # Plot 2: Color by condition
            ax = axes[1]
            for cond_name in CONDITIONS:
                mask = [c == cond_name for c in all_conditions]
                mask_idx = np.where(mask)[0]
                ax.scatter(emb[mask_idx, 0], emb[mask_idx, 1],
                          c=COND_COLORS[cond_name], alpha=0.7, s=40,
                          label=cond_name, edgecolors="white", linewidths=0.3)
            ax.legend(loc="best")
            ax.set_title(f"Condition")
            ax.set_xlabel(f"{method_name} 1")
            ax.set_ylabel(f"{method_name} 2")

            # Plot 3: Color by k_indiff (log scale)
            ax = axes[2]
            log_k = np.log10(all_k_indiffs + 1e-6)
            scatter = ax.scatter(emb[:, 0], emb[:, 1],
                                c=log_k, cmap="viridis", alpha=0.7, s=40,
                                edgecolors="white", linewidths=0.3)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("log₁₀(k_indiff)")
            ax.set_title(f"Indifference k (log scale)")
            ax.set_xlabel(f"{method_name} 1")
            ax.set_ylabel(f"{method_name} 2")

            fig.suptitle(f"Layer {layer} — {method_name} of Kirby MCQ-27 activations "
                        f"({n_conditions} conditions × 27 questions)", fontsize=13)
            plt.tight_layout()
            outpath = results_dir / f"kirby_{method_name.lower()}_layer{layer}.png"
            plt.savefig(outpath, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved {outpath.name}")

    # ============================================================
    # Boundary shift analysis plot
    # ============================================================
    print(f"\n  Generating boundary shift plot...")

    fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 5))
    if len(layers) == 1:
        axes = [axes]

    for ax, layer in zip(axes, layers):
        X = all_acts[layer]
        probe = LinearProbe(regularization_C=1.0)
        probe.train(X, all_choices, n_cv_folds=3)
        sv = probe.get_steering_vector(normalize=True)
        bias = probe.get_bias()
        probe_logits = X @ sv + bias

        cond_names = list(CONDITIONS.keys())
        means = []
        stds = []
        for cond_name in cond_names:
            mask = np.array([c == cond_name for c in all_conditions])
            means.append(probe_logits[mask].mean())
            stds.append(probe_logits[mask].std())

        colors = [COND_COLORS[c] for c in cond_names]
        bars = ax.bar(cond_names, means, yerr=stds, color=colors,
                      capsize=5, alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_ylabel("Mean probe logit\n(>0 = later, <0 = now)")
        ax.set_title(f"Layer {layer}")
        ax.tick_params(axis='x', rotation=30)

    fig.suptitle("Decision boundary shift across few-shot conditions", fontsize=13)
    plt.tight_layout()
    outpath = results_dir / "boundary_shift.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {outpath.name}")

    # ============================================================
    # Save numerical results
    # ============================================================
    save_data = {
        "conditions": {
            name: {
                "k": r["k"],
                "consistency": r["consistency"],
                "n_consistent": r["n_consistent"],
            }
            for name, r in condition_results.items()
        },
        "probes": {
            str(layer): probe_results[layer]
            for layer in layers
        },
        "layers": layers,
        "n_questions": 27,
        "n_conditions": n_conditions,
        "model": "gemma-2-2b",
        "timestamp": datetime.now().isoformat(),
    }
    outpath = results_dir / "kirby_umap_probing_results.json"
    with open(outpath, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to {outpath}")
    print(f"  All plots saved to {results_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UMAP + probing of Kirby MCQ-27")
    parser.add_argument("--layers", type=int, nargs="+", default=[6, 13, 20, 25])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    main(layers=args.layers, device=args.device, batch_size=args.batch_size)
