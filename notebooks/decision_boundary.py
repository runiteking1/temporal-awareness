"""
Decision boundary finder for LLM temporal discounting.

For each Kirby trial, uses binary search to find the exact delayed-reward
amount where the LLM flips its decision (now <-> later). This reveals
the LLM's true indifference point and implied discount rate.
"""

import os
import sys
import json
import math
from datetime import datetime

import pandas as pd
import torch

# Import shared infrastructure from the Kirby experiment
sys.path.insert(0, os.path.dirname(__file__))
from discount_factor_llm import (
    parse_trials, magnitude, sigfigs,
    load_model, _generate, parse_cot_answer,
    SYSTEM_PROMPT, HEROIN_SYSTEM_PROMPT,
    COT_SYSTEM_PROMPT, HEROIN_COT_SYSTEM_PROMPT, HF_TOKEN,
)

PROMPTS = {
    "default": SYSTEM_PROMPT,
    "heroin": HEROIN_SYSTEM_PROMPT,
    "default_cot": COT_SYSTEM_PROMPT,
    "heroin_cot": HEROIN_COT_SYSTEM_PROMPT,
}


def classify_response(reply_text):
    """Classify a raw reply as 'now' or 'later'."""
    reply = reply_text.lower().strip()
    if "now" in reply or "asap" in reply:
        return "now"
    elif "later" in reply:
        return "later"
    return reply  # unrecognized


def ask_binary(sir, ldr, delay, tokenizer, mdl, system_prompt=SYSTEM_PROMPT, cot=False):
    """Ask a single now-vs-later question and return (choice, reasoning).

    choice is 'now' or 'later'; reasoning is the full CoT text or None.
    """
    q = f"Would you prefer ${sir} today, or ${ldr} in {delay} days?"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": q},
    ]
    if cot:
        reply = _generate(tokenizer, mdl, messages, max_new_tokens=200)
        choice, _ = parse_cot_answer(reply)
        choice = "later" if choice == "delayed" else choice
        return choice, reply
    else:
        reply = _generate(tokenizer, mdl, messages, max_new_tokens=2).lower()
        return classify_response(reply), None


def find_boundary(sir, ldr_original, delay, tokenizer, mdl, system_prompt=SYSTEM_PROMPT, max_steps=20, cot=False):
    """
    Binary search for the delayed-reward value where the LLM flips.

    Returns a dict with:
      - original_choice: what the LLM chose at the original LDR
      - boundary_ldr: the LDR value at the decision boundary
      - boundary_k: implied discount rate at that boundary
      - search_log: list of (ldr_tested, choice, reasoning) tuples
    """
    original_choice, reasoning = ask_binary(sir, ldr_original, delay, tokenizer, mdl, system_prompt, cot=cot)
    search_log = [(ldr_original, original_choice, reasoning)]

    if original_choice == "now":
        # LLM wants money now — increase LDR until it flips to "later"
        # Search range: [ldr_original, sir * 20] (up to 20x the immediate reward)
        lo, hi = float(ldr_original), float(sir * 20)
        target_flip = "later"
        # First verify the upper bound actually flips
        hi_choice, hi_reason = ask_binary(sir, int(hi), delay, tokenizer, mdl, system_prompt, cot=cot)
        search_log.append((int(hi), hi_choice, hi_reason))
        if hi_choice != target_flip:
            # Even at 20x, LLM still says "now" — extreme present bias
            return dict(
                original_choice=original_choice,
                boundary_ldr=None,
                boundary_k=None,
                flipped=False,
                search_log=search_log,
            )
    elif original_choice == "later":
        # LLM wants later — decrease LDR toward SIR until it flips to "now"
        # Search range: [sir, ldr_original]
        lo, hi = float(sir), float(ldr_original)
        target_flip = "now"
        # Verify lower bound flips
        lo_choice, lo_reason = ask_binary(sir, int(lo), delay, tokenizer, mdl, system_prompt, cot=cot)
        search_log.append((int(lo), lo_choice, lo_reason))
        if lo_choice != target_flip:
            # Even at LDR == SIR, LLM still says "later"
            return dict(
                original_choice=original_choice,
                boundary_ldr=float(sir),
                boundary_k=0.0,
                flipped=False,
                search_log=search_log,
            )
    else:
        return dict(
            original_choice=original_choice,
            boundary_ldr=None,
            boundary_k=None,
            flipped=False,
            search_log=search_log,
        )

    # Binary search
    for step in range(max_steps):
        mid = round((lo + hi) / 2)
        if mid == lo or mid == hi:
            break
        choice, reason = ask_binary(sir, int(mid), delay, tokenizer, mdl, system_prompt, cot=cot)
        search_log.append((int(mid), choice, reason))

        if original_choice == "now":
            # Searching upward: if still "now", raise lo; if "later", lower hi
            if choice == "now":
                lo = mid
            else:
                hi = mid
        else:
            # Searching downward: if still "later", lower hi; if "now", raise lo
            if choice == "later":
                hi = mid
            else:
                lo = mid

    # Boundary is at the midpoint of the final bracket
    boundary_ldr = round((lo + hi) / 2)
    # Implied k at boundary: k = (LDR/SIR - 1) / delay
    if boundary_ldr > sir and delay > 0:
        boundary_k = (boundary_ldr / sir - 1) / delay
    else:
        boundary_k = 0.0

    return dict(
        original_choice=original_choice,
        boundary_ldr=boundary_ldr,
        boundary_k=round(boundary_k, 6),
        flipped=True,
        search_log=search_log,
    )


def run_boundary_search(model_id, trials_df, persona="default"):
    """Run boundary search for all 27 Kirby trials on a given model."""
    system_prompt = PROMPTS[persona]
    cot = persona.endswith("_cot")
    tokenizer, mdl = load_model(model_id)
    results = []

    for _, trial in trials_df.iterrows():
        sir, ldr, delay = int(trial.sir), int(trial.ldr), int(trial.delay)
        q_num = int(trial.order)
        print(f"\n{'='*60}")
        print(f"Q{q_num}: ${sir} now vs ${ldr} in {delay} days  "
              f"(k_indiff={trial.k_indiff:.4f}, {magnitude(ldr)})")
        print(f"{'='*60}")

        result = find_boundary(sir, ldr, delay, tokenizer, mdl, system_prompt, cot=cot)
        result["question"] = q_num
        result["sir"] = sir
        result["ldr_original"] = ldr
        result["delay"] = delay
        result["k_indiff"] = trial.k_indiff
        result["magnitude"] = magnitude(ldr)

        if result["flipped"]:
            print(f"  Original: {result['original_choice']} at LDR=${ldr}")
            print(f"  Boundary: LDR=${result['boundary_ldr']}  "
                  f"(implied k={result['boundary_k']:.6f})")
        else:
            print(f"  Original: {result['original_choice']} at LDR=${ldr}")
            print(f"  NO FLIP found in search range")

        print(f"  Search steps: {len(result['search_log'])}")
        for entry in result['search_log']:
            ldr_val, ch, reason = entry
            print(f"    LDR=${ldr_val:>6d} => {ch}")
            if reason:
                print(f"      CoT: {reason.replace(chr(10), ' ')}")

        results.append(result)

    del mdl, tokenizer
    torch.cuda.empty_cache()
    return results


def summarize(results, model_id):
    """Print a summary table of all boundary results."""
    print(f"\n\n{'='*80}")
    print(f"DECISION BOUNDARY SUMMARY — {model_id}")
    print(f"{'='*80}\n")

    header = (f"  {'Q#':>3s}  {'SIR':>5s}  {'LDR':>5s}  {'Delay':>5s}  "
              f"{'Choice':>6s}  {'Boundary':>8s}  {'k_indiff':>8s}  "
              f"{'k_boundary':>10s}  {'Mag':>6s}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in sorted(results, key=lambda x: x["k_indiff"]):
        bnd = f"${r['boundary_ldr']}" if r["boundary_ldr"] is not None else "N/A"
        kb = f"{r['boundary_k']:.6f}" if r["boundary_k"] is not None else "N/A"
        print(f"  {r['question']:>3d}  ${r['sir']:>4d}  ${r['ldr_original']:>4d}  "
              f"{r['delay']:>5d}  {r['original_choice']:>6s}  {bnd:>8s}  "
              f"{r['k_indiff']:>8.4f}  {kb:>10s}  {r['magnitude']:>6s}")

    # Stats
    flipped = [r for r in results if r["flipped"] and r["boundary_k"] is not None]
    if flipped:
        ks = [r["boundary_k"] for r in flipped]
        print(f"\n  Boundaries found: {len(flipped)}/{len(results)}")
        print(f"  Mean boundary k:   {sum(ks)/len(ks):.6f}")
        print(f"  Median boundary k: {sorted(ks)[len(ks)//2]:.6f}")
        print(f"  Min boundary k:    {min(ks):.6f}")
        print(f"  Max boundary k:    {max(ks):.6f}")

        # Compare to Kirby indifference k
        print(f"\n  Comparison to Kirby indifference k:")
        for r in sorted(flipped, key=lambda x: x["k_indiff"]):
            ratio = r["boundary_k"] / r["k_indiff"] if r["k_indiff"] > 0 else float("inf")
            print(f"    Q{r['question']:>2d}: k_indiff={r['k_indiff']:.4f}  "
                  f"k_boundary={r['boundary_k']:.6f}  ratio={ratio:.2f}x")


def save_results(results, model, persona, output_path=None):
    """Save results to JSON."""
    if output_path is None:
        safe_name = model.replace("/", "_")
        output_path = f"results/decision_boundary_{safe_name}_{persona}.json"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    save_data = {
        "model": model,
        "persona": persona,
        "timestamp": datetime.now().isoformat(),
        "results": [
            {k: v for k, v in r.items() if k != "search_log"}
            | {"search_log": [
                {"ldr": ldr, "choice": ch, "reasoning": reason}
                for ldr, ch, reason in r["search_log"]
            ]}
            for r in results
        ],
    }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Find LLM decision boundaries on Kirby MCQ-27")
    parser.add_argument("--model", default=None,
                        help="HuggingFace model ID (default: run all)")
    parser.add_argument("--persona", default=None,
                        choices=["default", "heroin", "default_cot", "heroin_cot"],
                        help="Persona to use (default: run all)")
    parser.add_argument("--output", default=None,
                        help="Output JSON file path (only used with single model+persona)")
    args = parser.parse_args()

    MODELS = ["Qwen/Qwen3-4B", "Qwen/Qwen3-8B"]
    PERSONAS = ["default", "heroin", "default_cot", "heroin_cot"]

    models = [args.model] if args.model else MODELS
    personas = [args.persona] if args.persona else PERSONAS

    trials_df = parse_trials()
    print(f"Parsed {len(trials_df)} trials\n")

    for model_id in models:
        for persona in personas:
            label = f"{model_id} ({persona})"
            print(f"\n{'#'*70}")
            print(f"# {label}")
            print(f"{'#'*70}")

            results = run_boundary_search(model_id, trials_df, persona=persona)
            summarize(results, label)
            save_results(results, model_id, persona, output_path=args.output)
