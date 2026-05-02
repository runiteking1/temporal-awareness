"""
Holt-Laury Multiple Price List (MPL) applied to LLMs.

Tests whether distributional flip-point calibration — the technique that
worked for temporal discounting (Kirby MCQ-27) — transfers to risk aversion.

The Holt-Laury (2002) instrument presents 10 binary choices between a "safe"
lottery A and a "risky" lottery B.  As the probability of the high payoff
increases from 0.1 to 1.0, risk-neutral agents switch from A→B at row 5;
risk-averse agents switch later; risk-seeking agents switch earlier.

We use the same 3×2 factorial design as discount_factor_best.py:
  3 personas  ×  2 few-shot flip-point configs  ×  2 modes (direct / thinking)

Wang et al. (2025) reported that random ICL examples are "fundamentally
ineffective" for risk modulation (flat accuracy from 0–40 shots).  We test
whether *ordered distributional* examples with a controlled flip point
succeed where random ICL failed.

Reference payoffs (Holt & Laury 2002, Table 1, 1× scale):
  Option A: {p, $2.00; 1-p, $1.60}
  Option B: {p, $3.85; 1-p, $0.10}
  p ∈ {0.1, 0.2, ..., 1.0}

CRRA switch-point mapping (Holt & Laury 2002, Table 3):
  Switch row → CRRA range → midpoint r
"""

import os
import sys
import math
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# Holt-Laury 10-item lottery instrument
# ============================================================

# Standard payoffs (Holt & Laury 2002, 1× scale)
A_HIGH, A_LOW = 2.00, 1.60
B_HIGH, B_LOW = 3.85, 0.10

ROWS = []
for i in range(1, 11):
    p = i / 10.0
    ev_a = p * A_HIGH + (1 - p) * A_LOW
    ev_b = p * B_HIGH + (1 - p) * B_LOW
    ROWS.append(dict(
        row=i, p=p,
        a_high=A_HIGH, a_low=A_LOW,
        b_high=B_HIGH, b_low=B_LOW,
        ev_a=round(ev_a, 2),
        ev_b=round(ev_b, 2),
        ev_diff=round(ev_a - ev_b, 2),
    ))

# CRRA coefficient ranges for each switch point (Holt & Laury 2002, Table 3)
# Switch at row N means chose A for rows 1..N-1 and B for rows N..10
# "Switch at row 1" means always B (risk-seeking), "switch at 11" means always A
CRRA_RANGES = {
    1:  (-1.71, -0.95),    # always B: very risk-seeking
    2:  (-0.95, -0.49),
    3:  (-0.49, -0.15),
    4:  (-0.15,  0.15),    # ~risk-neutral
    5:  (0.15,  0.41),     # mildly risk-averse
    6:  (0.41,  0.68),     # risk-averse (modal human response)
    7:  (0.68,  0.97),
    8:  (0.97,  1.37),
    9:  (1.37,  "inf"),    # highly risk-averse
    10: (1.37,  "inf"),    # always A (same range as 9 in HL2002)
    11: (1.37,  "inf"),    # never switch
}

def crra_midpoint(switch_row):
    """Return CRRA midpoint for a given switch row."""
    lo, hi = CRRA_RANGES.get(switch_row, (1.37, 2.0))
    if hi == "inf":
        return lo + 0.3  # convention for the open-ended interval
    return (lo + hi) / 2.0


def find_switch_point(choices):
    """Find switch row from A→B in a list of 10 choices.

    Returns the first row where the agent chose B.
    If agent always chose A, returns 11.
    If choices are inconsistent (multiple switches), returns the
    first B and reports consistency.
    """
    first_b = None
    last_a = 0
    switches = 0

    for i, c in enumerate(choices):
        row = i + 1
        if c == "A" and first_b is not None:
            switches += 1  # inconsistent: A after B
        if c == "B" and first_b is None:
            first_b = row
        if c == "A":
            last_a = row

    if first_b is None:
        return 11, 10  # always A, perfect consistency

    # Consistency: count how many choices are consistent with switch at first_b
    n_consistent = 0
    for i, c in enumerate(choices):
        row = i + 1
        expected = "A" if row < first_b else "B"
        if c == expected:
            n_consistent += 1

    return first_b, n_consistent


# ============================================================
# Few-shot prompts (calibrated flip points)
# ============================================================

BASE_INSTRUCTION = (
    "You are completing a psychology questionnaire about lottery preferences. "
    "Both lotteries are real — you will play the one you choose. "
    "People differ on these choices — there are no right or wrong answers. "
    "Reply with exactly one word: A or B."
)

def format_lottery_question(row_dict):
    """Format a single Holt-Laury row as a natural-language question."""
    p = row_dict["p"]
    p_pct = int(p * 100)
    q_pct = 100 - p_pct
    return (
        f"Option A gives you a {p_pct}% chance of $2.00 and a {q_pct}% chance of $1.60. "
        f"Option B gives you a {p_pct}% chance of $3.85 and a {q_pct}% chance of $0.10. "
        f"Which do you prefer?"
    )

# --- Risk-averse flip (targeting human-modal switch at row 6, CRRA ~0.55) ---
# Pattern: A/A/B/B with flip between rows 4-5 and 7-8
RISK_AVERSE_FEWSHOT = [
    {"role": "system", "content":
        "You are a financially comfortable 45-year-old professional. " + BASE_INSTRUCTION},
    # Row 2 (p=0.2): safe choice, clear
    {"role": "user", "content":
        "Option A gives you a 20% chance of $2.00 and a 80% chance of $1.60. "
        "Option B gives you a 20% chance of $3.85 and a 80% chance of $0.10. "
        "Which do you prefer?"},
    {"role": "assistant", "content": "A"},
    # Row 4 (p=0.4): still safe
    {"role": "user", "content":
        "Option A gives you a 40% chance of $2.00 and a 60% chance of $1.60. "
        "Option B gives you a 40% chance of $3.85 and a 60% chance of $0.10. "
        "Which do you prefer?"},
    {"role": "assistant", "content": "A"},
    # Row 7 (p=0.7): switch to risky
    {"role": "user", "content":
        "Option A gives you a 70% chance of $2.00 and a 30% chance of $1.60. "
        "Option B gives you a 70% chance of $3.85 and a 30% chance of $0.10. "
        "Which do you prefer?"},
    {"role": "assistant", "content": "B"},
    # Row 9 (p=0.9): clearly risky
    {"role": "user", "content":
        "Option A gives you a 90% chance of $2.00 and a 10% chance of $1.60. "
        "Option B gives you a 90% chance of $3.85 and a 10% chance of $0.10. "
        "Which do you prefer?"},
    {"role": "assistant", "content": "B"},
]

# --- Risk-neutral flip (targeting switch at row 4-5, CRRA ~0.0) ---
# Flip between rows 3-4 and 5-6
RISK_NEUTRAL_FEWSHOT = [
    {"role": "system", "content":
        "You are a financially comfortable 45-year-old professional. " + BASE_INSTRUCTION},
    # Row 1 (p=0.1): safe
    {"role": "user", "content":
        "Option A gives you a 10% chance of $2.00 and a 90% chance of $1.60. "
        "Option B gives you a 10% chance of $3.85 and a 90% chance of $0.10. "
        "Which do you prefer?"},
    {"role": "assistant", "content": "A"},
    # Row 3 (p=0.3): still safe
    {"role": "user", "content":
        "Option A gives you a 30% chance of $2.00 and a 70% chance of $1.60. "
        "Option B gives you a 30% chance of $3.85 and a 70% chance of $0.10. "
        "Which do you prefer?"},
    {"role": "assistant", "content": "A"},
    # Row 5 (p=0.5): switch to risky
    {"role": "user", "content":
        "Option A gives you a 50% chance of $2.00 and a 50% chance of $1.60. "
        "Option B gives you a 50% chance of $3.85 and a 50% chance of $0.10. "
        "Which do you prefer?"},
    {"role": "assistant", "content": "B"},
    # Row 8 (p=0.8): risky
    {"role": "user", "content":
        "Option A gives you a 80% chance of $2.00 and a 20% chance of $1.60. "
        "Option B gives you a 80% chance of $3.85 and a 20% chance of $0.10. "
        "Which do you prefer?"},
    {"role": "assistant", "content": "B"},
]

# --- Cross conditions for factorial ---
CROSS_CAUTIOUS_PERSONA_NEUTRAL_SHOTS = [
    {"role": "system", "content":
        "You are a 36-year-old in a tough financial situation. " + BASE_INSTRUCTION},
    # Same shots as risk-neutral
    {"role": "user", "content":
        "Option A gives you a 10% chance of $2.00 and a 90% chance of $1.60. "
        "Option B gives you a 10% chance of $3.85 and a 90% chance of $0.10. "
        "Which do you prefer?"},
    {"role": "assistant", "content": "A"},
    {"role": "user", "content":
        "Option A gives you a 30% chance of $2.00 and a 70% chance of $1.60. "
        "Option B gives you a 30% chance of $3.85 and a 70% chance of $0.10. "
        "Which do you prefer?"},
    {"role": "assistant", "content": "A"},
    {"role": "user", "content":
        "Option A gives you a 50% chance of $2.00 and a 50% chance of $1.60. "
        "Option B gives you a 50% chance of $3.85 and a 50% chance of $0.10. "
        "Which do you prefer?"},
    {"role": "assistant", "content": "B"},
    {"role": "user", "content":
        "Option A gives you a 80% chance of $2.00 and a 20% chance of $1.60. "
        "Option B gives you a 80% chance of $3.85 and a 20% chance of $0.10. "
        "Which do you prefer?"},
    {"role": "assistant", "content": "B"},
]

CROSS_TOUGH_PERSONA_AVERSE_SHOTS = [
    {"role": "system", "content":
        "You are a 36-year-old in a tough financial situation. " + BASE_INSTRUCTION},
    # Same shots as risk-averse
    {"role": "user", "content":
        "Option A gives you a 20% chance of $2.00 and a 80% chance of $1.60. "
        "Option B gives you a 20% chance of $3.85 and a 80% chance of $0.10. "
        "Which do you prefer?"},
    {"role": "assistant", "content": "A"},
    {"role": "user", "content":
        "Option A gives you a 40% chance of $2.00 and a 60% chance of $1.60. "
        "Option B gives you a 40% chance of $3.85 and a 60% chance of $0.10. "
        "Which do you prefer?"},
    {"role": "assistant", "content": "A"},
    {"role": "user", "content":
        "Option A gives you a 70% chance of $2.00 and a 30% chance of $1.60. "
        "Option B gives you a 70% chance of $3.85 and a 30% chance of $0.10. "
        "Which do you prefer?"},
    {"role": "assistant", "content": "B"},
    {"role": "user", "content":
        "Option A gives you a 90% chance of $2.00 and a 10% chance of $1.60. "
        "Option B gives you a 90% chance of $3.85 and a 10% chance of $0.10. "
        "Which do you prefer?"},
    {"role": "assistant", "content": "B"},
]

NEUTRAL_AVERSE_SHOTS = [
    {"role": "system", "content":
        "You are an adult completing a survey. " + BASE_INSTRUCTION},
    # Same shots as risk-averse
    {"role": "user", "content":
        "Option A gives you a 20% chance of $2.00 and a 80% chance of $1.60. "
        "Option B gives you a 20% chance of $3.85 and a 80% chance of $0.10. "
        "Which do you prefer?"},
    {"role": "assistant", "content": "A"},
    {"role": "user", "content":
        "Option A gives you a 40% chance of $2.00 and a 60% chance of $1.60. "
        "Option B gives you a 40% chance of $3.85 and a 60% chance of $0.10. "
        "Which do you prefer?"},
    {"role": "assistant", "content": "A"},
    {"role": "user", "content":
        "Option A gives you a 70% chance of $2.00 and a 30% chance of $1.60. "
        "Option B gives you a 70% chance of $3.85 and a 30% chance of $0.10. "
        "Which do you prefer?"},
    {"role": "assistant", "content": "B"},
    {"role": "user", "content":
        "Option A gives you a 90% chance of $2.00 and a 10% chance of $1.60. "
        "Option B gives you a 90% chance of $3.85 and a 10% chance of $0.10. "
        "Which do you prefer?"},
    {"role": "assistant", "content": "B"},
]

NEUTRAL_NEUTRAL_SHOTS = [
    {"role": "system", "content":
        "You are an adult completing a survey. " + BASE_INSTRUCTION},
    # Same shots as risk-neutral
    {"role": "user", "content":
        "Option A gives you a 10% chance of $2.00 and a 90% chance of $1.60. "
        "Option B gives you a 10% chance of $3.85 and a 90% chance of $0.10. "
        "Which do you prefer?"},
    {"role": "assistant", "content": "A"},
    {"role": "user", "content":
        "Option A gives you a 30% chance of $2.00 and a 70% chance of $1.60. "
        "Option B gives you a 30% chance of $3.85 and a 70% chance of $0.10. "
        "Which do you prefer?"},
    {"role": "assistant", "content": "A"},
    {"role": "user", "content":
        "Option A gives you a 50% chance of $2.00 and a 50% chance of $1.60. "
        "Option B gives you a 50% chance of $3.85 and a 50% chance of $0.10. "
        "Which do you prefer?"},
    {"role": "assistant", "content": "B"},
    {"role": "user", "content":
        "Option A gives you a 80% chance of $2.00 and a 20% chance of $1.60. "
        "Option B gives you a 80% chance of $3.85 and a 20% chance of $0.10. "
        "Which do you prefer?"},
    {"role": "assistant", "content": "B"},
]


# ============================================================
# Model
# ============================================================

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
HF_TOKEN = os.environ.get("HF_TOKEN", "")


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, padding_side="left")
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16, token=HF_TOKEN)
    print(f"Loaded {MODEL_ID}")
    return tok, mdl


def generate(tok, mdl, messages, thinking=False):
    max_tokens = 512 if thinking else 2
    prompt = tok.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
        enable_thinking=thinking)
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(mdl.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = mdl.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    raw = tok.decode(out[0][inputs["input_ids"].shape[-1]:],
                     skip_special_tokens=True).strip()
    return raw


def classify(reply):
    """Extract A or B from model reply."""
    r = reply.lower()
    last_a = r.rfind("a")
    last_b = r.rfind("b")
    if last_a == -1 and last_b == -1:
        return r
    if last_b > last_a:
        return "B"
    return "A"


# ============================================================
# Run one condition
# ============================================================

def run_condition(tok, mdl, fewshot, label, thinking=False):
    print(f"\n{'='*60}")
    print(f"  {label}{' [THINKING]' if thinking else ''}")
    print(f"{'='*60}\n")

    choices = []
    for row_dict in ROWS:
        q = format_lottery_question(row_dict)
        msgs = fewshot + [{"role": "user", "content": q}]
        reply = generate(tok, mdl, msgs, thinking=thinking)
        ans = classify(reply)
        choices.append(ans)
        print(f"  Row {row_dict['row']:>2d}: p={row_dict['p']:.1f}  "
              f"EV(A)={row_dict['ev_a']:.2f}  EV(B)={row_dict['ev_b']:.2f}  "
              f"=> {ans}")
        if thinking and row_dict['row'] == 1:
            print(f"  [Sample reasoning]: {reply[:300]}...")

    switch_row, n_consistent = find_switch_point(choices)
    r_mid = crra_midpoint(switch_row)
    lo, hi = CRRA_RANGES.get(switch_row, (1.37, "inf"))

    print(f"\n  Switch point: row {switch_row}")
    print(f"  CRRA range: [{lo}, {hi}]")
    print(f"  CRRA midpoint: {r_mid:.2f}")
    print(f"  Consistency: {n_consistent}/10 ({n_consistent/10:.0%})")
    print(f"  Choices: {' '.join(choices)}")

    n_safe = sum(1 for c in choices if c == "A")
    result = dict(
        label=label,
        thinking=thinking,
        choices=choices,
        switch_row=switch_row,
        crra_midpoint=r_mid,
        crra_range=[lo, str(hi)],
        n_safe=n_safe,
        n_consistent=n_consistent,
        consistency=n_consistent / 10,
    )
    return result


# ============================================================
# Main: 3×2 factorial + thinking mode
# ============================================================

if __name__ == "__main__":
    print("Holt-Laury Multiple Price List — Flip-Point Calibration Experiment")
    print(f"Model: {MODEL_ID}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()

    # Print the instrument
    print(f"{'Row':>3s}  {'p':>4s}  {'EV(A)':>6s}  {'EV(B)':>6s}  {'EV(A)-EV(B)':>11s}")
    print(f"{'-'*3}  {'-'*4}  {'-'*6}  {'-'*6}  {'-'*11}")
    for r in ROWS:
        print(f"{r['row']:>3d}  {r['p']:>4.1f}  {r['ev_a']:>6.2f}  {r['ev_b']:>6.2f}  {r['ev_diff']:>+11.2f}")
    print()

    tok, mdl = load_model()

    all_results = {}

    # --- Direct mode (no thinking) ---
    conditions = [
        ("default_averse",  RISK_AVERSE_FEWSHOT,                "Default persona + risk-averse shots"),
        ("default_neutral", RISK_NEUTRAL_FEWSHOT,               "Default persona + risk-neutral shots"),
        ("tough_neutral",   CROSS_CAUTIOUS_PERSONA_NEUTRAL_SHOTS, "Tough-situation persona + risk-neutral shots"),
        ("tough_averse",    CROSS_TOUGH_PERSONA_AVERSE_SHOTS,   "Tough-situation persona + risk-averse shots"),
        ("neutral_averse",  NEUTRAL_AVERSE_SHOTS,               "Neutral persona + risk-averse shots"),
        ("neutral_neutral", NEUTRAL_NEUTRAL_SHOTS,              "Neutral persona + risk-neutral shots"),
    ]

    for key, fewshot, label in conditions:
        result = run_condition(tok, mdl, fewshot, label, thinking=False)
        all_results[f"direct_{key}"] = result

    # --- Thinking mode ---
    for key, fewshot, label in conditions:
        result = run_condition(tok, mdl, fewshot, label, thinking=True)
        all_results[f"thinking_{key}"] = result

    del mdl, tok
    torch.cuda.empty_cache()

    # ============================================================
    # Final comparison
    # ============================================================
    print(f"\n{'='*72}")
    print(f"  FINAL COMPARISON — Holt-Laury Flip-Point Calibration")
    print(f"{'='*72}\n")

    print(f"  {'Condition':<45s}  {'Switch':>6s}  {'CRRA':>6s}  {'#Safe':>5s}  {'Consist':>7s}")
    print(f"  {'-'*45}  {'-'*6}  {'-'*6}  {'-'*5}  {'-'*7}")

    # Group by mode
    for mode in ["direct", "thinking"]:
        mode_label = "Direct" if mode == "direct" else "Thinking"
        print(f"  — {mode_label} mode —")
        for key, _, label in conditions:
            rkey = f"{mode}_{key}"
            if rkey in all_results:
                r = all_results[rkey]
                print(f"    {label:<43s}  {r['switch_row']:>6d}  "
                      f"{r['crra_midpoint']:>6.2f}  {r['n_safe']:>5d}  "
                      f"{r['consistency']:>6.0%}")

    print(f"\n  — Human benchmarks (Holt & Laury 2002) —")
    print(f"    {'Modal response (1× payoffs)':<43s}  {'6':>6s}  {'0.55':>6s}  {'6':>5s}  {'—':>7s}")
    print(f"    {'Mean # safe choices':<43s}  {'—':>6s}  {'—':>6s}  {'5.9':>5s}  {'—':>7s}")

    # Key test: does persona text matter?
    print(f"\n  KEY TEST: Persona effect (holding few-shot constant)")
    print(f"  {'-'*60}")
    for shots_label, keys in [
        ("Risk-averse shots", ["direct_default_averse", "direct_tough_averse", "direct_neutral_averse"]),
        ("Risk-neutral shots", ["direct_default_neutral", "direct_tough_neutral", "direct_neutral_neutral"]),
    ]:
        switches = [all_results[k]["switch_row"] for k in keys if k in all_results]
        crras = [all_results[k]["crra_midpoint"] for k in keys if k in all_results]
        all_same = len(set(switches)) == 1
        print(f"  {shots_label}: switch rows = {switches}  "
              f"CRRAs = {[f'{c:.2f}' for c in crras]}  "
              f"{'IDENTICAL ✓' if all_same else 'DIFFER ✗'}")

    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    outpath = results_dir / f"holt_laury_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {outpath}")
