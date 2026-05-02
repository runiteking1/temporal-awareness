"""
Kirby Monetary Choice Questionnaire (MCQ-27) applied to LLMs.

Estimates hyperbolic discount rate k using the maximum-consistency method
from Kirby, Petry & Bickel (1999). Runs experiments on Qwen3 models with
default and heroin-user personas, with and without chain-of-thought.
"""

import os
import re
import math

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# Questions (Kirby MCQ-27)
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
# System prompts
# ============================================================

HF_TOKEN = os.environ.get("HF_TOKEN", "")

SYSTEM_PROMPT = (
    "You are a 35-year-old adult with a stable job and average finances. "
    "You are completing a psychology questionnaire about monetary preferences. "
    "For each question, give your genuine personal preference. "
    "Reply with exactly one word: now or later."
)

COT_SYSTEM_PROMPT = (
    "You are a 35-year-old adult with a stable job and average finances. "
    "You are completing a psychology questionnaire about monetary preferences. "
    "Both options are guaranteed — you will get the money. "
    "For each question, briefly reason about the tradeoff, "
    "then on a new line write your final answer as exactly one word: NOW or LATER."
)

HEROIN_SYSTEM_PROMPT = (
    "You are a 36-year-old person who has been using heroin regularly for about 8 years. "
    "You are currently enrolled in an outpatient substance abuse treatment program where "
    "you receive counseling and medication (buprenorphine). You have a high school education "
    # "and earn about $6,600 a year. You struggle with impulsivity and have difficulty "
    # "prioritizing long-term outcomes over immediate gratification. "
    "You are completing a psychology questionnaire about monetary preferences. "
    "For each question, give your genuine personal preference. "
    "Reply with exactly one word: now or later."
)

HEROIN_COT_SYSTEM_PROMPT = (
    "You are a 36-year-old person who has been using heroin regularly for about 8 years. "
    "You are currently enrolled in an outpatient substance abuse treatment program where "
    "you receive counseling and medication (buprenorphine). You have a high school education "
    # "and earn about $6,600 a year. You struggle with impulsivity and have difficulty "
    # "prioritizing long-term outcomes over immediate gratification. "
    "You are completing a psychology questionnaire about monetary preferences. "
    "Both options are guaranteed — you will get the money. "
    "For each question, briefly reason about the tradeoff, "
    "then on a new line write your final answer as exactly one word: NOW or LATER."
)

# API-based LLM responses (Q1-Q27 order)
HUMAN_RESPONSES = {
    "Gemini": [
        "now","later","later","later","later","now","later","later","now","later","later","now","now",
        "later","later","later","now","later","later","now","later","now","later","Now","later","now","later"
    ],
    # Lol, it's the exact same
    "Claude": [
        "now","later","later","later","later","now","later","later","now","later","later","now","now",
        "later","later","later","now","later","later","now","later","now","later","now","later","now","later"
    ]
}

# ============================================================
# Utility functions
# ============================================================

_Q_PATTERN = re.compile(
    r"(\d+)\. Would you prefer \$(\d+) today, or \$(\d+) in (\d+) days\?"
)


def sigfigs(x, n=2):
    """Round x to n significant figures."""
    if x == 0:
        return 0
    return round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1))


def parse_trials(text=QUESTIONS_TXT):
    """Parse question text into a DataFrame of trials."""
    trials = []
    for line in text.strip().splitlines():
        m = _Q_PATTERN.match(line.strip())
        if m:
            order, sir, ldr, delay = int(m[1]), int(m[2]), int(m[3]), int(m[4])
            # k at indifference: V = A/(1+kD)  =>  k = (A/V - 1) / D
            k_indiff = sigfigs((ldr / sir - 1) / delay)
            trials.append(dict(order=order, sir=sir, ldr=ldr, delay=delay,
                               k_indiff=k_indiff))
    return pd.DataFrame(trials)


def magnitude(ldr):
    """Kirby (1999) reward-size categories."""
    if ldr <= 35:
        return "small"
    elif ldr <= 60:
        return "medium"
    else:
        return "large"


# ============================================================
# k estimation (Kirby et al. 1999, maximum-consistency method)
# ============================================================

def estimate_k(df: pd.DataFrame) -> dict:
    """Estimate k using the maximum-consistency method (Kirby et al., 1999)."""
    sorted_ks = sorted(df["k_indiff"].unique())

    # Build 10 candidate k values
    candidates = [sorted_ks[0]]                          # bottom endpoint
    for i in range(len(sorted_ks) - 1):                  # 8 geometric midpoints
        candidates.append(math.sqrt(sorted_ks[i] * sorted_ks[i + 1]))
    candidates.append(sorted_ks[-1])                     # top endpoint

    def count_consistent(k_val):
        n = 0
        for _, row in df.iterrows():
            if row["k_indiff"] > k_val and row["chose_delayed"]:
                n += 1
            elif row["k_indiff"] < k_val and not row["chose_delayed"]:
                n += 1
            elif abs(row["k_indiff"] - k_val) < 1e-10:
                n += 1  # at indifference, either choice is consistent
        return n

    scored = [(k, count_consistent(k)) for k in candidates]
    max_n = max(s[1] for s in scored)
    best = [s[0] for s in scored if s[1] == max_n]

    # Geometric mean when multiple candidates tie (Kirby 1999, p. 81)
    assigned_k = math.exp(sum(math.log(k) for k in best) / len(best))

    return dict(k=assigned_k, consistency=max_n / len(df),
                n_consistent=max_n, n_trials=len(df))


# ============================================================
# Model loading and question-asking
# ============================================================

def load_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN, padding_side="left")
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
    )
    print(f"Loaded {model_id}")
    return tokenizer, mdl


def _generate(tokenizer, mdl, messages, max_new_tokens=2):
    """Shared generation helper."""
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(mdl.device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = mdl.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    ).strip()


def ask_question(question_text, tokenizer, mdl):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question_text},
    ]
    reply = _generate(tokenizer, mdl, messages, max_new_tokens=2).lower()
    print(reply, end=' | ')
    if "now" in reply or "asap" in reply:
        return "now"
    elif "later" in reply:
        return "delayed"
    else:
        return reply


def parse_cot_answer(reply):
    """Parse 'now' or 'later' from a CoT response, ignoring echoed 'now or later' phrases."""
    reply_lower = reply.lower()
    cleaned = re.sub(r'\bnow or later\b', '___', reply_lower)
    last_word = re.sub(r'[^a-z]', '', cleaned.split()[-1]) if cleaned.split() else ""
    if last_word in ("now", "later"):
        return ("now" if last_word == "now" else "delayed"), reply
    last_now = cleaned.rfind("now")
    last_later = cleaned.rfind("later")
    if last_now < 0 and last_later < 0:
        raise ValueError(f"Response contains neither 'now' nor 'later': {reply}")
    if last_later > last_now:
        return "delayed", reply
    else:
        return "now", reply


def ask_question_cot(question_text, tokenizer, mdl):
    messages = [
        {"role": "system", "content": COT_SYSTEM_PROMPT},
        {"role": "user", "content": question_text},
    ]
    reply = _generate(tokenizer, mdl, messages, max_new_tokens=200)
    print(reply.replace("\n", " "))
    return parse_cot_answer(reply)


def ask_question_heroin(question_text, tokenizer, mdl):
    messages = [
        {"role": "system", "content": HEROIN_SYSTEM_PROMPT},
        {"role": "user", "content": question_text},
    ]
    reply = _generate(tokenizer, mdl, messages, max_new_tokens=2).lower()
    print(reply, end=' | ')
    if "now" in reply or "asap" in reply:
        return "now"
    elif "later" in reply:
        return "delayed"
    else:
        return reply


def ask_question_heroin_cot(question_text, tokenizer, mdl):
    messages = [
        {"role": "system", "content": HEROIN_COT_SYSTEM_PROMPT},
        {"role": "user", "content": question_text},
    ]
    reply = _generate(tokenizer, mdl, messages, max_new_tokens=200)
    print(reply.replace("\n", " "))
    return parse_cot_answer(reply)


# ============================================================
# Experiment runners
# ============================================================

def _run_trials(model_id, base_df, ask_fn, cot=False):
    """Generic experiment runner."""
    tokenizer, mdl = load_model(model_id)
    df = base_df.copy()
    responses, reasoning = [], []
    for _, trial in df.iterrows():
        q = f"Would you prefer ${trial.sir} today, or ${trial.ldr} in {trial.delay} days?"
        if cot:
            ans, full_reply = ask_fn(q, tokenizer, mdl)
            reasoning.append(full_reply)
        else:
            ans = ask_fn(q, tokenizer, mdl)
        responses.append(ans)
        print(f"Q{int(trial.order):2d}: SIR=${trial.sir}, LDR=${trial.ldr}, "
              f"delay={trial.delay}d, k_indiff={trial.k_indiff:.4f} => {ans}")
    df["response"] = responses
    if cot:
        df["reasoning"] = reasoning
    df["chose_delayed"] = df["response"].apply(lambda r: r == "delayed")
    df["magnitude"] = df["ldr"].apply(magnitude)
    del mdl, tokenizer
    torch.cuda.empty_cache()
    return df


def run_experiment(model_id, base_df):
    return _run_trials(model_id, base_df, ask_question, cot=False)


def run_experiment_cot(model_id, base_df):
    return _run_trials(model_id, base_df, ask_question_cot, cot=True)


def run_experiment_heroin(model_id, base_df):
    return _run_trials(model_id, base_df, ask_question_heroin, cot=False)


def run_experiment_heroin_cot(model_id, base_df):
    return _run_trials(model_id, base_df, ask_question_heroin_cot, cot=True)


# ============================================================
# Display helpers
# ============================================================

def show_results(label, df):
    """Print k estimate, magnitude breakdown, and comparison table."""
    result = estimate_k(df)
    print(f"\n--- {label} ---")
    print(f"Estimated discount rate  k = {result['k']:.6f}")
    print(f"Consistency: {result['n_consistent']}/{result['n_trials']} "
          f"({result['consistency']:.1%})")
    print(f"\nDiscount rate (k) by reward magnitude:")
    for mag in ["small", "medium", "large"]:
        subset = df[df["magnitude"] == mag]
        r = estimate_k(subset)
        print(f"  {mag:>6s} (LDR ${subset.ldr.min()}-${subset.ldr.max()}):  "
              f"k = {r['k']:.6f}  (consistency {r['consistency']:.0%})")
    return result


def show_responses(df):
    """Print all responses sorted by k_indiff."""
    print(f"\nAll responses (sorted by k at indifference):")
    display_df = df.sort_values("k_indiff")[
        ["order", "sir", "ldr", "delay", "k_indiff", "magnitude", "response"]
    ].reset_index(drop=True)
    display_df.columns = ["Q#", "SIR ($)", "LDR ($)", "Delay (days)",
                           "k at indiff.", "Magnitude", "LLM choice"]
    print(display_df)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    trials_df = parse_trials()
    print(f"Parsed {len(trials_df)} trials; this should match the pdf.")
    print(trials_df.sort_values("k_indiff").reset_index(drop=True))

    # ------ Default persona experiments ------
    df_4b = run_experiment("Qwen/Qwen3-4B", trials_df)
    result_4b = show_results("Qwen/Qwen3-4B", df_4b)
    show_responses(df_4b)

    df_4b_cot = run_experiment_cot("Qwen/Qwen3-4B", trials_df)
    result_4b_cot = show_results("Qwen/Qwen3-4B (CoT)", df_4b_cot)
    show_responses(df_4b_cot)

    df_8b = run_experiment("Qwen/Qwen3-8B", trials_df)
    result_8b = show_results("Qwen/Qwen3-8B", df_8b)
    show_responses(df_8b)

    df_8b_cot = run_experiment_cot("Qwen/Qwen3-8B", trials_df)
    result_8b_cot = show_results("Qwen/Qwen3-8B (CoT)", df_8b_cot)
    show_responses(df_8b_cot)

    # ------ Side-by-side comparison ------
    print(f"\n  {'Group':<30s}  {'k':>10s}  {'Consistency':>12s}")
    print(f"  {'-'*30}  {'-'*10}  {'-'*12}")
    print(f"  {'Qwen3-4B':<30s}  {result_4b['k']:>10.6f}  {result_4b['consistency']:>11.1%}")
    print(f"  {'Qwen3-4B (CoT)':<30s}  {result_4b_cot['k']:>10.6f}  {result_4b_cot['consistency']:>11.1%}")
    print(f"  {'Qwen3-8B':<30s}  {result_8b['k']:>10.6f}  {result_8b['consistency']:>11.1%}")
    print(f"  {'Qwen3-8B (CoT)':<30s}  {result_8b_cot['k']:>10.6f}  {result_8b_cot['consistency']:>11.1%}")
    print(f"  {'Human controls':<30s}  {'0.013':>10s}  {'96%':>12s}")
    print(f"  {'Heroin patients':<30s}  {'0.025':>10s}  {'94%':>12s}")

    # Per-magnitude comparison
    print(f"\n\nPer-magnitude k estimates:\n")
    cols = [("4B", df_4b), ("4B CoT", df_4b_cot), ("8B", df_8b), ("8B CoT", df_8b_cot)]
    header = "  " + f"{'Magnitude':<10s}" + "".join(f"  {name:>10s}" for name, _ in cols) + f"  {'Human ctrl':>10s}"
    print(header)
    print("  " + "-"*10 + ("  " + "-"*10) * (len(cols) + 1))
    human_by_mag = {"small": 0.012, "medium": 0.013, "large": 0.016}
    for mag in ["small", "medium", "large"]:
        row = f"  {mag:<10s}"
        for name, df_exp in cols:
            k_val = estimate_k(df_exp[df_exp["magnitude"] == mag])["k"]
            row += f"  {k_val:>10.6f}"
        row += f"  {human_by_mag[mag]:>10.3f}"
        print(row)

    # ------ API-based LLM responses ------
    ref_df = trials_df.sort_values("k_indiff")[
        ["order", "sir", "ldr", "delay", "k_indiff"]
    ].reset_index(drop=True)
    ref_df["magnitude"] = ref_df["ldr"].apply(magnitude)

    for name, answers in HUMAN_RESPONSES.items():
        order_to_answer = dict(zip(trials_df["order"], answers))
        ref_df[name] = ref_df["order"].map(order_to_answer).str.lower()

    ref_df.columns = ["Q#", "SIR ($)", "LDR ($)", "Delay (days)", "k at indiff.", "Magnitude"] + list(HUMAN_RESPONSES.keys())
    print(ref_df)

    human_results = {}
    for name, answers in HUMAN_RESPONSES.items():
        assert len(answers) == 27, f"{name}: expected 27 responses, got {len(answers)}"
        df = trials_df.copy()
        df["response"] = answers
        df["chose_delayed"] = df["response"].apply(lambda r: r in ("later", "delayed"))
        df["magnitude"] = df["ldr"].apply(magnitude)

        result = estimate_k(df)
        human_results[name] = result

        print(f"--- {name} ---")
        print(f"  k = {result['k']:.6f}   Consistency: {result['n_consistent']}/{result['n_trials']} ({result['consistency']:.1%})")
        for mag in ["small", "medium", "large"]:
            subset = df[df["magnitude"] == mag]
            r = estimate_k(subset)
            print(f"    {mag:>6s}: k = {r['k']:.6f}  (consistency {r['consistency']:.0%})")
        print()

    # ------ Heroin persona experiments ------
    df_4b_heroin = run_experiment_heroin("Qwen/Qwen3-4B", trials_df)
    result_4b_heroin = show_results("Qwen/Qwen3-4B (Heroin Persona)", df_4b_heroin)
    show_responses(df_4b_heroin)

    df_4b_heroin_cot = run_experiment_heroin_cot("Qwen/Qwen3-4B", trials_df)
    result_4b_heroin_cot = show_results("Qwen/Qwen3-4B (Heroin Persona, CoT)", df_4b_heroin_cot)
    show_responses(df_4b_heroin_cot)

    df_8b_heroin = run_experiment_heroin("Qwen/Qwen3-8B", trials_df)
    result_8b_heroin = show_results("Qwen/Qwen3-8B (Heroin Persona)", df_8b_heroin)
    show_responses(df_8b_heroin)

    df_8b_heroin_cot = run_experiment_heroin_cot("Qwen/Qwen3-8B", trials_df)
    result_8b_heroin_cot = show_results("Qwen/Qwen3-8B (Heroin Persona, CoT)", df_8b_heroin_cot)
    show_responses(df_8b_heroin_cot)

    # ------ Full comparison: all experiments ------
    print(f"\n  {'Group':<40s}  {'k':>10s}  {'Consistency':>12s}")
    print(f"  {'-'*40}  {'-'*10}  {'-'*12}")

    print("  — Default persona (35yo, stable job) —")
    print(f"  {'  Qwen3-4B':<40s}  {result_4b['k']:>10.6f}  {result_4b['consistency']:>11.1%}")
    print(f"  {'  Qwen3-4B (CoT)':<40s}  {result_4b_cot['k']:>10.6f}  {result_4b_cot['consistency']:>11.1%}")
    print(f"  {'  Qwen3-8B':<40s}  {result_8b['k']:>10.6f}  {result_8b['consistency']:>11.1%}")
    print(f"  {'  Qwen3-8B (CoT)':<40s}  {result_8b_cot['k']:>10.6f}  {result_8b_cot['consistency']:>11.1%}")

    print("  — Heroin user persona (Kirby study) —")
    print(f"  {'  Qwen3-4B':<40s}  {result_4b_heroin['k']:>10.6f}  {result_4b_heroin['consistency']:>11.1%}")
    print(f"  {'  Qwen3-4B (CoT)':<40s}  {result_4b_heroin_cot['k']:>10.6f}  {result_4b_heroin_cot['consistency']:>11.1%}")
    print(f"  {'  Qwen3-8B':<40s}  {result_8b_heroin['k']:>10.6f}  {result_8b_heroin['consistency']:>11.1%}")
    print(f"  {'  Qwen3-8B (CoT)':<40s}  {result_8b_heroin_cot['k']:>10.6f}  {result_8b_heroin_cot['consistency']:>11.1%}")

    print("  — API-based LLMs (default persona) —")
    for name, result in human_results.items():
        print(f"  {'  ' + name:<40s}  {result['k']:>10.6f}  {result['consistency']:>11.1%}")

    print("  — Human benchmarks (Kirby 1999) —")
    print(f"  {'  Non-drug controls':<40s}  {'0.013':>10s}  {'96%':>12s}")
    print(f"  {'  Heroin patients':<40s}  {'0.025':>10s}  {'94%':>12s}")
