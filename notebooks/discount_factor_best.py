"""
Best-case Kirby MCQ-27 for Qwen3-4B-Instruct-2507.

Uses calibrated few-shot prompting to produce human-like discount rates.
  - Default persona:  k ≈ 0.010  (human controls: k = 0.013)
  - Heroin equivalent: k ≈ 0.026  (heroin patients: k = 0.025)
"""

import os
import sys
import math
import re

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# Kirby MCQ-27 questions
# ============================================================

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

# ============================================================
# Few-shot prompts (calibrated to match human k values)
# ============================================================

BASE_INSTRUCTION = (
    "You are completing a psychology questionnaire about monetary preferences. "
    "Both options are guaranteed real payments. "
    "People differ on these choices — there are no right or wrong answers. "
    "Reply with exactly one word: now or later."
)

# All few-shot configs: 4 examples, pattern now/now/later/later.
# "Default shots" = patient flip (k ≈ 0.0067 → 0.016), targeting human controls k≈0.013
# "Heroin shots"  = impulsive flip (k ≈ 0.006 → 0.10)

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

HEROIN_FEWSHOT = [
    {"role": "system", "content":
        "You are a 36-year-old in a tough financial situation. " + BASE_INSTRUCTION},
    {"role": "user", "content": "Would you prefer $90 today, or $95 in 120 days?"},
    {"role": "assistant", "content": "now"},
    {"role": "user", "content": "Would you prefer $50 today, or $70 in 60 days?"},
    {"role": "assistant", "content": "now"},
    {"role": "user", "content": "Would you prefer $25 today, or $60 in 14 days?"},
    {"role": "assistant", "content": "later"},
    {"role": "user", "content": "Would you prefer $20 today, or $55 in 7 days?"},
    {"role": "assistant", "content": "later"},
]

CROSS_DEFAULT_PERSONA_HEROIN_SHOTS = [
    {"role": "system", "content":
        "You are a financially comfortable 45-year-old professional. " + BASE_INSTRUCTION},
    {"role": "user", "content": "Would you prefer $90 today, or $95 in 120 days?"},
    {"role": "assistant", "content": "now"},
    {"role": "user", "content": "Would you prefer $50 today, or $70 in 60 days?"},
    {"role": "assistant", "content": "now"},
    {"role": "user", "content": "Would you prefer $25 today, or $60 in 14 days?"},
    {"role": "assistant", "content": "later"},
    {"role": "user", "content": "Would you prefer $20 today, or $55 in 7 days?"},
    {"role": "assistant", "content": "later"},
]

CROSS_HEROIN_PERSONA_DEFAULT_SHOTS = [
    {"role": "system", "content":
        "You are a 36-year-old in a tough financial situation. " + BASE_INSTRUCTION},
    {"role": "user", "content": "Would you prefer $90 today, or $95 in 120 days?"},
    {"role": "assistant", "content": "now"},
    {"role": "user", "content": "Would you prefer $50 today, or $70 in 60 days?"},
    {"role": "assistant", "content": "now"},
    {"role": "user", "content": "Would you prefer $24 today, or $35 in 29 days?"},
    {"role": "assistant", "content": "later"},
    {"role": "user", "content": "Would you prefer $20 today, or $55 in 7 days?"},
    {"role": "assistant", "content": "later"},
]

NEUTRAL_DEFAULT_SHOTS = [
    {"role": "system", "content":
        "You are an adult completing a survey. " + BASE_INSTRUCTION},
    {"role": "user", "content": "Would you prefer $90 today, or $95 in 120 days?"},
    {"role": "assistant", "content": "now"},
    {"role": "user", "content": "Would you prefer $50 today, or $70 in 60 days?"},
    {"role": "assistant", "content": "now"},
    {"role": "user", "content": "Would you prefer $24 today, or $35 in 29 days?"},
    {"role": "assistant", "content": "later"},
    {"role": "user", "content": "Would you prefer $20 today, or $55 in 7 days?"},
    {"role": "assistant", "content": "later"},
]

NEUTRAL_HEROIN_SHOTS = [
    {"role": "system", "content":
        "You are an adult completing a survey. " + BASE_INSTRUCTION},
    {"role": "user", "content": "Would you prefer $90 today, or $95 in 120 days?"},
    {"role": "assistant", "content": "now"},
    {"role": "user", "content": "Would you prefer $50 today, or $70 in 60 days?"},
    {"role": "assistant", "content": "now"},
    {"role": "user", "content": "Would you prefer $25 today, or $60 in 14 days?"},
    {"role": "assistant", "content": "later"},
    {"role": "user", "content": "Would you prefer $20 today, or $55 in 7 days?"},
    {"role": "assistant", "content": "later"},
]

# ============================================================
# Parsing and scoring (Kirby et al. 1999)
# ============================================================

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
    # For thinking mode, the final answer comes after </think> — take the last
    # occurrence of "now" or "later" as the answer
    r = reply.lower()
    # Find last occurrence of each
    last_now = r.rfind("now")
    last_later = r.rfind("later")
    if last_now == -1 and last_later == -1:
        return r
    if last_later > last_now:
        return "delayed"
    return "now"


# ============================================================
# Run one condition
# ============================================================

def run_condition(tok, mdl, fewshot, trials_df, label, thinking=False):
    print(f"\n{'='*60}")
    print(f"  {label}{' [THINKING]' if thinking else ''}")
    print(f"{'='*60}\n")

    df = trials_df.copy()
    responses = []
    for i, (_, trial) in enumerate(df.iterrows()):
        sir, ldr, delay = int(trial.sir), int(trial.ldr), int(trial.delay)
        q = f"Would you prefer ${sir} today, or ${ldr} in {delay} days?"
        msgs = fewshot + [{"role": "user", "content": q}]
        reply = generate(tok, mdl, msgs, thinking=thinking)
        ans = classify(reply)
        responses.append(ans)
        print(f"  Q{int(trial.order):>2d}: ${sir:>3d} now vs ${ldr:>3d} in {delay:>3d}d "
              f"(k={trial.k_indiff:.4f}) => {ans}")
        if thinking and i == 0:
            print(f"  [Sample reasoning Q1]: {reply[:300]}...")

    df["response"] = responses
    df["chose_delayed"] = df["response"].apply(lambda r: r == "delayed")
    df["magnitude"] = df["ldr"].apply(magnitude)

    result = estimate_k(df)
    print(f"\n  k = {result['k']:.6f}   "
          f"Consistency: {result['n_consistent']}/{result['n_trials']} "
          f"({result['consistency']:.1%})")

    print(f"\n  By magnitude:")
    for mag in ["small", "medium", "large"]:
        sub = df[df["magnitude"] == mag]
        r = estimate_k(sub)
        print(f"    {mag:>6s}: k = {r['k']:.6f}  ({r['consistency']:.0%})")

    return result


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    trials_df = parse_trials()
    print(f"Parsed {len(trials_df)} Kirby MCQ-27 trials\n")

    tok, mdl = load_model()

    # --- Direct mode (no thinking) ---
    r_default = run_condition(tok, mdl, DEFAULT_FEWSHOT, trials_df,
                              "Default persona + default shots")
    r_heroin = run_condition(tok, mdl, HEROIN_FEWSHOT, trials_df,
                             "Heroin persona + heroin shots")
    r_cross1 = run_condition(tok, mdl, CROSS_DEFAULT_PERSONA_HEROIN_SHOTS, trials_df,
                              "Default persona + heroin shots")
    r_cross2 = run_condition(tok, mdl, CROSS_HEROIN_PERSONA_DEFAULT_SHOTS, trials_df,
                              "Heroin persona + default shots")
    r_neutral1 = run_condition(tok, mdl, NEUTRAL_DEFAULT_SHOTS, trials_df,
                                "Neutral persona + default shots")
    r_neutral2 = run_condition(tok, mdl, NEUTRAL_HEROIN_SHOTS, trials_df,
                                "Neutral persona + heroin shots")

    # --- Thinking mode (reasoning enabled) ---
    t_default = run_condition(tok, mdl, DEFAULT_FEWSHOT, trials_df,
                              "Default persona + default shots", thinking=True)
    t_heroin = run_condition(tok, mdl, HEROIN_FEWSHOT, trials_df,
                             "Heroin persona + heroin shots", thinking=True)
    t_cross1 = run_condition(tok, mdl, CROSS_DEFAULT_PERSONA_HEROIN_SHOTS, trials_df,
                              "Default persona + heroin shots", thinking=True)
    t_cross2 = run_condition(tok, mdl, CROSS_HEROIN_PERSONA_DEFAULT_SHOTS, trials_df,
                              "Heroin persona + default shots", thinking=True)
    t_neutral1 = run_condition(tok, mdl, NEUTRAL_DEFAULT_SHOTS, trials_df,
                                "Neutral persona + default shots", thinking=True)
    t_neutral2 = run_condition(tok, mdl, NEUTRAL_HEROIN_SHOTS, trials_df,
                                "Neutral persona + heroin shots", thinking=True)

    del mdl, tok
    torch.cuda.empty_cache()

    # Final comparison
    print(f"\n{'='*72}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*72}\n")
    print(f"  {'Condition':<40s}  {'Direct':>8s}  {'Thinking':>8s}  {'Human k':>8s}")
    print(f"  {'-'*40}  {'-'*8}  {'-'*8}  {'-'*8}")
    print(f"  {'Default persona + default shots':<40s}  {r_default['k']:>8.4f}  "
          f"{t_default['k']:>8.4f}  {'0.013':>8s}")
    print(f"  {'Heroin persona + heroin shots':<40s}  {r_heroin['k']:>8.4f}  "
          f"{t_heroin['k']:>8.4f}  {'0.025':>8s}")
    print(f"  {'Default persona + heroin shots':<40s}  {r_cross1['k']:>8.4f}  "
          f"{t_cross1['k']:>8.4f}")
    print(f"  {'Heroin persona + default shots':<40s}  {r_cross2['k']:>8.4f}  "
          f"{t_cross2['k']:>8.4f}")
    print(f"  {'Neutral persona + default shots':<40s}  {r_neutral1['k']:>8.4f}  "
          f"{t_neutral1['k']:>8.4f}")
    print(f"  {'Neutral persona + heroin shots':<40s}  {r_neutral2['k']:>8.4f}  "
          f"{t_neutral2['k']:>8.4f}")
    print(f"  {'-'*40}  {'-'*8}  {'-'*8}  {'-'*8}")
    print(f"  {'Human controls':<40s}  {'0.013':>8s}  {'':>8s}  {'0.013':>8s}")
    print(f"  {'Human heroin patients':<40s}  {'0.025':>8s}  {'':>8s}  {'0.025':>8s}")
