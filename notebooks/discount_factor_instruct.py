"""
Kirby MCQ-27 experiments with Qwen3-4B-Instruct-2507.

Iterative experiments to get more human-like temporal discounting.
See experiment_log.md for detailed notes on each run.
"""

import os
import sys
import json
import math
import re
from datetime import datetime

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import shared infrastructure
sys.path.insert(0, os.path.dirname(__file__))
from discount_factor_llm import (
    parse_trials, magnitude, sigfigs, estimate_k,
    show_results, show_responses, QUESTIONS_TXT, HF_TOKEN,
)

# ============================================================
# Models to test
# ============================================================

MODELS = {
    "instruct": "Qwen/Qwen3-4B-Instruct-2507",
    "thinking": "Qwen/Qwen3-4B-Thinking-2507",
}

# ============================================================
# Prompt variants
# ============================================================

# V1: Original prompts (baseline)
PROMPTS_V1 = {
    "default": (
        "You are a 35-year-old adult with a stable job and average finances. "
        "You are completing a psychology questionnaire about monetary preferences. "
        "For each question, give your genuine personal preference. "
        "Reply with exactly one word: now or later."
    ),
    "heroin": (
        "You are a 36-year-old person who has been using heroin regularly for about 8 years. "
        "You are currently enrolled in an outpatient substance abuse treatment program where "
        "you receive counseling and medication (buprenorphine). You have a high school education. "
        "You are completing a psychology questionnaire about monetary preferences. "
        "For each question, give your genuine personal preference. "
        "Reply with exactly one word: now or later."
    ),
}

# V2: More naturalistic, emphasize real stakes and gut feeling
PROMPTS_V2 = {
    "default": (
        "Imagine you are a real person, age 35, with a regular job and normal bills to pay. "
        "You have about $500 in your checking account. Rent is due next week. "
        "Someone offers you a choice between two guaranteed cash payments. "
        "There are no tricks — both payments are 100% certain. "
        "Go with your gut feeling. Don't overthink it. "
        "Reply with exactly one word: now or later."
    ),
    "heroin": (
        "Imagine you are 36 years old. You've been using heroin for about 8 years. "
        "You're in an outpatient treatment program getting buprenorphine. "
        "You have a high school diploma and earn about $6,600 a year. "
        "Money is always tight. You often need cash for immediate expenses. "
        "Someone offers you a choice between two guaranteed cash payments. "
        "Go with your gut feeling. Don't overthink it. "
        "Reply with exactly one word: now or later."
    ),
}

# V3: First-person immersive, with financial pressure
PROMPTS_V3 = {
    "default": (
        "You are taking a psychology study. In this study, you make choices between "
        "getting a smaller amount of money right now, today, in your hand — or waiting "
        "some number of days to receive a larger amount. Both options are real and guaranteed. "
        "Think about what you would actually do if this were real money. "
        "People differ on these choices — there are no right or wrong answers. "
        "Reply with exactly one word: now or later."
    ),
    "heroin": (
        "You are taking a psychology study. You are 36, a heroin user for 8 years, "
        "currently in treatment. Money is tight — you earn about $6,600/year. "
        "In this study, you choose between getting a smaller amount of money right now, "
        "today, cash in hand — or waiting some days for a larger amount. "
        "Both are real and guaranteed. Think about what you would actually do. "
        "People differ — there are no right or wrong answers. "
        "Reply with exactly one word: now or later."
    ),
}

# ============================================================
# Model loading
# ============================================================

def load_model(model_id, enable_thinking=False):
    """Load model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, token=HF_TOKEN, padding_side="left"
    )
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
    )
    print(f"Loaded {model_id}")
    return tokenizer, mdl


def generate(tokenizer, mdl, messages, max_new_tokens=2, enable_thinking=False,
             temperature=None):
    """Generate a response. If temperature is set, uses sampling."""
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=enable_thinking,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(mdl.device) for k, v in inputs.items()}
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    if temperature and temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.9
    else:
        gen_kwargs["do_sample"] = False
    with torch.no_grad():
        output_ids = mdl.generate(
            **inputs,
            **gen_kwargs,
        )
    return tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    ).strip()


def classify(reply):
    """Classify reply as 'now', 'delayed', or raw text."""
    r = reply.lower().strip()
    if "now" in r or "asap" in r:
        return "now"
    elif "later" in r:
        return "delayed"
    return r


def parse_thinking_answer(reply):
    """Parse answer from a thinking-mode response.

    The thinking model may produce <think>...</think> blocks followed by the answer.
    """
    # Remove thinking blocks
    cleaned = re.sub(r'<think>.*?</think>', '', reply, flags=re.DOTALL).strip()
    if not cleaned:
        # All content was in thinking block; try to find answer in full reply
        cleaned = reply

    # Look for the last now/later
    cleaned_lower = cleaned.lower()
    # Remove "now or later" instruction echoes
    cleaned_lower = re.sub(r'\bnow or later\b', '___', cleaned_lower)

    last_now = cleaned_lower.rfind("now")
    last_later = cleaned_lower.rfind("later")

    if last_now < 0 and last_later < 0:
        return classify(reply), reply

    if last_later > last_now:
        return "delayed", reply
    else:
        return "now", reply


# ============================================================
# Experiment runner
# ============================================================

def run_mcq27(model_id, trials_df, system_prompt, label="",
              enable_thinking=False, max_new_tokens=2, temperature=None,
              question_format=None):
    """Run the MCQ-27 with given model and prompt.

    question_format: None for default, 'reversed' to put later option first,
                     'ab' for A/B choice format
    """
    tokenizer, mdl = load_model(model_id)
    df = trials_df.copy()
    responses = []
    reasoning_list = []

    for _, trial in df.iterrows():
        sir, ldr, delay = int(trial.sir), int(trial.ldr), int(trial.delay)
        if question_format == "reversed":
            q = f"Would you prefer ${ldr} in {delay} days, or ${sir} today?"
        elif question_format == "ab":
            q = f"Which do you prefer?\nA) ${sir} today\nB) ${ldr} in {delay} days"
        else:
            q = f"Would you prefer ${sir} today, or ${ldr} in {delay} days?"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q},
        ]
        reply = generate(tokenizer, mdl, messages,
                        max_new_tokens=max_new_tokens,
                        enable_thinking=enable_thinking,
                        temperature=temperature)

        if enable_thinking or max_new_tokens > 10:
            ans, full = parse_thinking_answer(reply)
            reasoning_list.append(full)
        elif question_format == "ab":
            # A = now, B = later
            r = reply.strip().upper()
            if r.startswith("A"):
                ans = "now"
            elif r.startswith("B"):
                ans = "delayed"
            else:
                ans = classify(reply)
        else:
            ans = classify(reply)

        responses.append(ans)
        print(f"Q{int(trial.order):2d}: ${int(trial.sir):>3d} now vs ${int(trial.ldr):>3d} in {int(trial.delay):>3d}d "
              f"(k={trial.k_indiff:.4f}) => {ans}  [{reply[:80]}]")

    df["response"] = responses
    if reasoning_list:
        df["reasoning"] = reasoning_list
    df["chose_delayed"] = df["response"].apply(lambda r: r == "delayed")
    df["magnitude"] = df["ldr"].apply(magnitude)

    del mdl, tokenizer
    torch.cuda.empty_cache()
    return df


def run_decision_boundary(model_id, trials_df, system_prompt, label="",
                          enable_thinking=False, max_new_tokens=2,
                          max_search_steps=20, max_multiplier=20):
    """Run decision boundary search for all 27 trials."""
    tokenizer, mdl = load_model(model_id)
    results = []

    for _, trial in trials_df.iterrows():
        sir, ldr, delay = int(trial.sir), int(trial.ldr), int(trial.delay)
        q_num = int(trial.order)

        print(f"\n{'='*60}")
        print(f"Q{q_num}: ${sir} now vs ${ldr} in {delay} days (k_indiff={trial.k_indiff:.4f})")

        def ask(ldr_val):
            q = f"Would you prefer ${sir} today, or ${ldr_val} in {delay} days?"
            msgs = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q},
            ]
            reply = generate(tokenizer, mdl, msgs,
                           max_new_tokens=max_new_tokens,
                           enable_thinking=enable_thinking)
            if enable_thinking or max_new_tokens > 10:
                ans, _ = parse_thinking_answer(reply)
            else:
                ans = classify(reply)
            # Normalize
            if ans == "delayed":
                ans = "later"
            return ans, reply

        # Get original choice
        orig_choice, orig_reply = ask(ldr)
        search_log = [(ldr, orig_choice, orig_reply)]

        if orig_choice == "now":
            lo, hi = float(ldr), float(sir * max_multiplier)
            target_flip = "later"
            hi_choice, hi_reply = ask(int(hi))
            search_log.append((int(hi), hi_choice, hi_reply))
            if hi_choice != target_flip:
                results.append(dict(
                    question=q_num, sir=sir, ldr_original=ldr, delay=delay,
                    k_indiff=trial.k_indiff, magnitude=magnitude(ldr),
                    original_choice=orig_choice, boundary_ldr=None,
                    boundary_k=None, flipped=False, search_log=search_log,
                ))
                print(f"  NO FLIP (still 'now' at ${int(hi)})")
                continue
        elif orig_choice == "later":
            lo, hi = float(sir), float(ldr)
            target_flip = "now"
            lo_choice, lo_reply = ask(int(lo))
            search_log.append((int(lo), lo_choice, lo_reply))
            if lo_choice != target_flip:
                results.append(dict(
                    question=q_num, sir=sir, ldr_original=ldr, delay=delay,
                    k_indiff=trial.k_indiff, magnitude=magnitude(ldr),
                    original_choice=orig_choice, boundary_ldr=float(sir),
                    boundary_k=0.0, flipped=False, search_log=search_log,
                ))
                print(f"  NO FLIP (still 'later' at ${sir})")
                continue
        else:
            results.append(dict(
                question=q_num, sir=sir, ldr_original=ldr, delay=delay,
                k_indiff=trial.k_indiff, magnitude=magnitude(ldr),
                original_choice=orig_choice, boundary_ldr=None,
                boundary_k=None, flipped=False, search_log=search_log,
            ))
            print(f"  UNRECOGNIZED: {orig_choice}")
            continue

        # Binary search
        for step in range(max_search_steps):
            mid = round((lo + hi) / 2)
            if mid == lo or mid == hi:
                break
            choice, reply = ask(int(mid))
            search_log.append((int(mid), choice, reply))

            if orig_choice == "now":
                if choice == "now":
                    lo = mid
                else:
                    hi = mid
            else:
                if choice == "later":
                    hi = mid
                else:
                    lo = mid

        boundary_ldr = round((lo + hi) / 2)
        if boundary_ldr > sir and delay > 0:
            boundary_k = (boundary_ldr / sir - 1) / delay
        else:
            boundary_k = 0.0

        results.append(dict(
            question=q_num, sir=sir, ldr_original=ldr, delay=delay,
            k_indiff=trial.k_indiff, magnitude=magnitude(ldr),
            original_choice=orig_choice, boundary_ldr=boundary_ldr,
            boundary_k=round(boundary_k, 6), flipped=True,
            search_log=search_log,
        ))
        print(f"  FLIP at LDR=${boundary_ldr} (k={boundary_k:.6f})")

    del mdl, tokenizer
    torch.cuda.empty_cache()
    return results


def save_experiment(results, model_id, prompt_version, persona, suffix=""):
    """Save results to JSON."""
    safe_model = model_id.replace("/", "_")
    fname = f"results/instruct_exp_{safe_model}_{prompt_version}_{persona}{suffix}.json"
    os.makedirs(os.path.dirname(fname), exist_ok=True)

    # Convert search_log tuples for JSON
    save_results = []
    for r in results:
        r2 = {k: v for k, v in r.items() if k != "search_log"}
        if "search_log" in r:
            r2["search_log"] = [
                {"ldr": ldr, "choice": ch, "reasoning": reason}
                for ldr, ch, reason in r["search_log"]
            ]
        save_results.append(r2)

    data = {
        "model": model_id,
        "prompt_version": prompt_version,
        "persona": persona,
        "timestamp": datetime.now().isoformat(),
        "results": save_results,
    }
    with open(fname, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved to {fname}")
    return fname


def summarize_boundary(results):
    """Print summary of boundary results."""
    flipped = [r for r in results if r.get("flipped") and r.get("boundary_k") is not None]
    not_flipped = [r for r in results if not r.get("flipped")]

    print(f"\n  Boundaries found: {len(flipped)}/{len(results)}")
    if flipped:
        ks = [r["boundary_k"] for r in flipped]
        print(f"  Mean boundary k:   {sum(ks)/len(ks):.6f}")
        print(f"  Median boundary k: {sorted(ks)[len(ks)//2]:.6f}")
    if not_flipped:
        now_stuck = sum(1 for r in not_flipped if r["original_choice"] == "now")
        later_stuck = sum(1 for r in not_flipped if r["original_choice"] == "later")
        print(f"  Stuck on 'now': {now_stuck}, stuck on 'later': {later_stuck}")


# ============================================================
# Main: Run experiments
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int, default=1, help="Which run to execute")
    parser.add_argument("--boundary", action="store_true", help="Also run decision boundary")
    args = parser.parse_args()

    trials_df = parse_trials()
    print(f"Parsed {len(trials_df)} trials\n")

    if args.run == 1:
        # Run 1: Baseline with instruct model, original prompts
        model = MODELS["instruct"]
        print(f"\n{'#'*70}")
        print(f"# RUN 1: Baseline — {model} with V1 (original) prompts")
        print(f"{'#'*70}\n")

        for persona_name, prompt in PROMPTS_V1.items():
            print(f"\n--- {persona_name} persona ---")
            df = run_mcq27(model, trials_df, prompt, label=f"v1_{persona_name}")
            result = show_results(f"{model} (v1, {persona_name})", df)
            show_responses(df)

            if args.boundary:
                print(f"\n--- Decision boundary: {persona_name} ---")
                bd_results = run_decision_boundary(model, trials_df, prompt,
                                                    label=f"v1_{persona_name}_boundary")
                summarize_boundary(bd_results)
                save_experiment(bd_results, model, "v1", persona_name, suffix="_boundary")

    elif args.run == 2:
        # Run 2: V2 prompts (naturalistic, financial pressure)
        model = MODELS["instruct"]
        print(f"\n{'#'*70}")
        print(f"# RUN 2: {model} with V2 (naturalistic) prompts")
        print(f"{'#'*70}\n")

        for persona_name, prompt in PROMPTS_V2.items():
            print(f"\n--- {persona_name} persona (V2) ---")
            df = run_mcq27(model, trials_df, prompt, label=f"v2_{persona_name}")
            result = show_results(f"{model} (v2, {persona_name})", df)
            show_responses(df)

            if args.boundary:
                print(f"\n--- Decision boundary: {persona_name} (V2) ---")
                bd_results = run_decision_boundary(model, trials_df, prompt,
                                                    label=f"v2_{persona_name}_boundary")
                summarize_boundary(bd_results)
                save_experiment(bd_results, model, "v2", persona_name, suffix="_boundary")

    elif args.run == 3:
        # Run 3: V3 prompts (study framing, no right/wrong)
        model = MODELS["instruct"]
        print(f"\n{'#'*70}")
        print(f"# RUN 3: {model} with V3 (study framing) prompts")
        print(f"{'#'*70}\n")

        for persona_name, prompt in PROMPTS_V3.items():
            print(f"\n--- {persona_name} persona (V3) ---")
            df = run_mcq27(model, trials_df, prompt, label=f"v3_{persona_name}")
            result = show_results(f"{model} (v3, {persona_name})", df)
            show_responses(df)

            if args.boundary:
                print(f"\n--- Decision boundary: {persona_name} (V3) ---")
                bd_results = run_decision_boundary(model, trials_df, prompt,
                                                    label=f"v3_{persona_name}_boundary")
                summarize_boundary(bd_results)
                save_experiment(bd_results, model, "v3", persona_name, suffix="_boundary")

    elif args.run == 4:
        # Run 4: Thinking model
        model = MODELS["thinking"]
        print(f"\n{'#'*70}")
        print(f"# RUN 4: {model} with thinking enabled")
        print(f"{'#'*70}\n")

        for persona_name, prompt in PROMPTS_V3.items():
            print(f"\n--- {persona_name} persona (V3, thinking) ---")
            df = run_mcq27(model, trials_df, prompt, label=f"v3_{persona_name}_thinking",
                          enable_thinking=True, max_new_tokens=500)
            result = show_results(f"{model} (v3, {persona_name}, thinking)", df)
            show_responses(df)

            if args.boundary:
                print(f"\n--- Decision boundary: {persona_name} (thinking) ---")
                bd_results = run_decision_boundary(model, trials_df, prompt,
                                                    label=f"v3_{persona_name}_thinking_boundary",
                                                    enable_thinking=True, max_new_tokens=500)
                summarize_boundary(bd_results)
                save_experiment(bd_results, model, "v3_thinking", persona_name, suffix="_boundary")

    elif args.run == 5:
        # Run 5: Instruct model with thinking enabled
        model = MODELS["instruct"]
        print(f"\n{'#'*70}")
        print(f"# RUN 5: {model} with thinking enabled + V3 prompts")
        print(f"{'#'*70}\n")

        for persona_name, prompt in PROMPTS_V3.items():
            print(f"\n--- {persona_name} persona (V3, instruct+thinking) ---")
            df = run_mcq27(model, trials_df, prompt, label=f"v3_{persona_name}_instruct_thinking",
                          enable_thinking=True, max_new_tokens=500)
            result = show_results(f"{model} (v3, {persona_name}, thinking)", df)
            show_responses(df)

    elif args.run == 6:
        # Run 6: Diagnostic — check what the model actually generates with more tokens
        model = MODELS["instruct"]
        print(f"\n{'#'*70}")
        print(f"# RUN 6: Diagnostic — {model} with 50 tokens, various prompts")
        print(f"{'#'*70}\n")

        tokenizer, mdl = load_model(model)

        # Test a few easy questions with different prompts
        test_cases = [
            ("Would you prefer $20 today, or $55 in 7 days?", "easy: $55 in 7d"),
            ("Would you prefer $31 today, or $85 in 7 days?", "easy: $85 in 7d"),
            ("Would you prefer $54 today, or $55 in 117 days?", "close: $55 in 117d"),
            ("Would you prefer $78 today, or $80 in 162 days?", "close: $80 in 162d"),
        ]

        for prompt_name, prompt in [("V1", PROMPTS_V1["default"]), ("V3", PROMPTS_V3["default"])]:
            print(f"\n--- Prompt: {prompt_name} ---")
            for q, label in test_cases:
                msgs = [{"role": "system", "content": prompt}, {"role": "user", "content": q}]
                # Try with 2 tokens
                reply2 = generate(tokenizer, mdl, msgs, max_new_tokens=2)
                # Try with 50 tokens
                reply50 = generate(tokenizer, mdl, msgs, max_new_tokens=50)
                print(f"  {label}:")
                print(f"    2 tokens: [{reply2}]")
                print(f"   50 tokens: [{reply50}]")

        # Try with reversed option order
        print(f"\n--- Reversed option order ---")
        reverse_prompt = (
            "You are a 35-year-old adult with a stable job and average finances. "
            "You are completing a psychology questionnaire about monetary preferences. "
            "For each question, give your genuine personal preference. "
            "Reply with exactly one word: now or later."
        )
        for q_orig, label in test_cases:
            # Reverse: put the later option first
            import re as re_mod
            m = re_mod.match(r"Would you prefer \$(\d+) today, or \$(\d+) in (\d+) days\?", q_orig)
            if m:
                sir, ldr, delay = m.group(1), m.group(2), m.group(3)
                q_rev = f"Would you prefer ${ldr} in {delay} days, or ${sir} today?"
                msgs = [{"role": "system", "content": reverse_prompt}, {"role": "user", "content": q_rev}]
                reply = generate(tokenizer, mdl, msgs, max_new_tokens=50)
                print(f"  {label} (reversed): [{reply}]")

        # Try with A/B format
        print(f"\n--- A/B format ---")
        ab_prompt = (
            "You are a 35-year-old adult with a stable job and average finances. "
            "You are completing a psychology questionnaire about monetary preferences. "
            "For each question, give your genuine personal preference. "
            "Reply with exactly one letter: A or B."
        )
        for q_orig, label in test_cases:
            m = re_mod.match(r"Would you prefer \$(\d+) today, or \$(\d+) in (\d+) days\?", q_orig)
            if m:
                sir, ldr, delay = m.group(1), m.group(2), m.group(3)
                q_ab = f"Which do you prefer?\nA) ${sir} today\nB) ${ldr} in {delay} days"
                msgs = [{"role": "system", "content": ab_prompt}, {"role": "user", "content": q_ab}]
                reply = generate(tokenizer, mdl, msgs, max_new_tokens=50)
                print(f"  {label} (A/B): [{reply}]")

        del mdl, tokenizer
        torch.cuda.empty_cache()

    elif args.run == 7:
        # Run 7: Thinking model (Qwen3-4B-Thinking-2507)
        model = MODELS["thinking"]
        print(f"\n{'#'*70}")
        print(f"# RUN 7: {model} with thinking enabled + V3 prompts")
        print(f"{'#'*70}\n")

        for persona_name, prompt in PROMPTS_V3.items():
            print(f"\n--- {persona_name} persona (V3, thinking model) ---")
            df = run_mcq27(model, trials_df, prompt, label=f"v3_{persona_name}_thinking",
                          enable_thinking=True, max_new_tokens=1000)
            result = show_results(f"{model} (v3, {persona_name}, thinking)", df)
            show_responses(df)

    elif args.run == 8:
        # Run 8: Temperature experiments — break the greedy "now" lock
        model = MODELS["instruct"]
        print(f"\n{'#'*70}")
        print(f"# RUN 8: {model} with temperature=0.7 + V3 prompts")
        print(f"{'#'*70}\n")

        # Run 3 times to check variance
        for trial_num in range(1, 4):
            print(f"\n=== Trial {trial_num}/3 ===")
            for persona_name, prompt in PROMPTS_V3.items():
                print(f"\n--- {persona_name} (V3, T=0.7, trial {trial_num}) ---")
                df = run_mcq27(model, trials_df, prompt,
                              label=f"v3_{persona_name}_t07_trial{trial_num}",
                              temperature=0.7)
                result = show_results(f"{model} (v3, {persona_name}, T=0.7, t{trial_num})", df)

    elif args.run == 9:
        # Run 9: "later" option presented more attractively
        model = MODELS["instruct"]
        print(f"\n{'#'*70}")
        print(f"# RUN 9: {model} with explicit value framing")
        print(f"{'#'*70}\n")

        VALUE_PROMPT = (
            "You are a 35-year-old with a stable job and average finances. "
            "A researcher asks you to make choices between two guaranteed cash payments. "
            "Consider: how much MORE money is the later option? Is the wait worth it? "
            "There are no right or wrong answers. "
            "Reply with exactly one word: now or later."
        )

        VALUE_PROMPT_HEROIN = (
            "You are a 36-year-old heroin user for 8 years, in outpatient treatment. "
            "You earn about $6,600/year. Money is always tight. "
            "A researcher asks you to make choices between two guaranteed cash payments. "
            "Consider: how much MORE money is the later option? Is the wait worth it? "
            "There are no right or wrong answers. "
            "Reply with exactly one word: now or later."
        )

        for persona_name, prompt in [("default", VALUE_PROMPT), ("heroin", VALUE_PROMPT_HEROIN)]:
            print(f"\n--- {persona_name} (value framing) ---")
            df = run_mcq27(model, trials_df, prompt, label=f"value_{persona_name}")
            result = show_results(f"{model} (value, {persona_name})", df)
            show_responses(df)

    elif args.run == 10:
        # Run 10: Instruct model with enable_thinking + V3
        model = MODELS["instruct"]
        print(f"\n{'#'*70}")
        print(f"# RUN 10: {model} with enable_thinking=True + V3")
        print(f"{'#'*70}\n")

        for persona_name, prompt in PROMPTS_V3.items():
            print(f"\n--- {persona_name} (V3, enable_thinking) ---")
            df = run_mcq27(model, trials_df, prompt,
                          label=f"v3_{persona_name}_thinking",
                          enable_thinking=True, max_new_tokens=1000)
            result = show_results(f"{model} (v3, {persona_name}, thinking)", df)
            show_responses(df)

    elif args.run == 11:
        # Run 11: Logit-based approach — measure P(now) vs P(later)
        model_id = MODELS["instruct"]
        print(f"\n{'#'*70}")
        print(f"# RUN 11: Logit analysis — {model_id}")
        print(f"{'#'*70}\n")

        tokenizer, mdl = load_model(model_id)

        # Get token IDs for "now" and "later"
        now_ids = tokenizer.encode("now", add_special_tokens=False)
        later_ids = tokenizer.encode("later", add_special_tokens=False)
        Now_ids = tokenizer.encode("Now", add_special_tokens=False)
        Later_ids = tokenizer.encode("Later", add_special_tokens=False)
        NOW_ids = tokenizer.encode("NOW", add_special_tokens=False)
        LATER_ids = tokenizer.encode("LATER", add_special_tokens=False)

        print(f"Token IDs: now={now_ids}, later={later_ids}")
        print(f"Token IDs: Now={Now_ids}, Later={Later_ids}")
        print(f"Token IDs: NOW={NOW_ids}, LATER={LATER_ids}")

        # Use first token of each
        now_tid = now_ids[0]
        later_tid = later_ids[0]
        Now_tid = Now_ids[0]
        Later_tid = Later_ids[0]

        df = trials_df.copy()
        p_now_list = []
        p_later_list = []

        for prompt_name, system_prompt in [("V1", PROMPTS_V1["default"]), ("V3", PROMPTS_V3["default"]),
                                            ("V1_heroin", PROMPTS_V1["heroin"]), ("V3_heroin", PROMPTS_V3["heroin"])]:
            print(f"\n--- {prompt_name} ---")
            print(f"{'Q#':>3s}  {'SIR':>5s}  {'LDR':>5s}  {'Delay':>5s}  {'P(now)':>8s}  {'P(later)':>8s}  {'Ratio':>8s}  {'Greedy':>8s}")

            for _, trial in df.iterrows():
                sir, ldr, delay = int(trial.sir), int(trial.ldr), int(trial.delay)
                q = f"Would you prefer ${sir} today, or ${ldr} in {delay} days?"
                msgs = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": q},
                ]
                prompt_text = tokenizer.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=False,
                    enable_thinking=False,
                )
                inputs = tokenizer(prompt_text, return_tensors="pt")
                inputs = {k: v.to(mdl.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = mdl(**inputs)
                    logits = outputs.logits[0, -1, :]  # last position

                # Softmax to get probabilities
                probs = torch.softmax(logits, dim=0)

                # Sum probabilities for all "now" and "later" variants
                p_now = (probs[now_tid] + probs[Now_tid]).item()
                p_later = (probs[later_tid] + probs[Later_tid]).item()

                # Normalize
                total = p_now + p_later
                if total > 0:
                    p_now_norm = p_now / total
                    p_later_norm = p_later / total
                else:
                    p_now_norm = p_later_norm = 0.5

                greedy_tok = tokenizer.decode([logits.argmax().item()])

                print(f"{int(trial.order):>3d}  ${sir:>4d}  ${ldr:>4d}  {delay:>5d}  "
                      f"{p_now_norm:>8.4f}  {p_later_norm:>8.4f}  "
                      f"{p_now_norm/max(p_later_norm,1e-10):>8.2f}  {greedy_tok.strip():>8s}")

        del mdl, tokenizer
        torch.cuda.empty_cache()

    elif args.run == 12:
        # Run 12: Few-shot prompting — show examples of rational choices
        model_id = MODELS["instruct"]
        print(f"\n{'#'*70}")
        print(f"# RUN 12: Few-shot prompting — {model_id}")
        print(f"{'#'*70}\n")

        FEW_SHOT_PROMPT = (
            "You are a 35-year-old adult with a stable job and average finances. "
            "You are completing a psychology questionnaire about monetary preferences. "
            "Both options are guaranteed real payments. "
            "People differ on these choices — there are no right or wrong answers. "
            "Reply with exactly one word: now or later."
        )

        # Build a few-shot conversation
        few_shot_messages = [
            {"role": "system", "content": FEW_SHOT_PROMPT},
            # Example 1: close values, long delay -> now (reasonable)
            {"role": "user", "content": "Would you prefer $95 today, or $100 in 200 days?"},
            {"role": "assistant", "content": "now"},
            # Example 2: big difference, short delay -> later (reasonable)
            {"role": "user", "content": "Would you prefer $10 today, or $50 in 7 days?"},
            {"role": "assistant", "content": "later"},
            # Example 3: moderate difference, moderate delay -> could go either way
            {"role": "user", "content": "Would you prefer $40 today, or $60 in 30 days?"},
            {"role": "assistant", "content": "later"},
        ]

        df = trials_df.copy()
        responses = []
        tokenizer, mdl = load_model(model_id)

        for _, trial in df.iterrows():
            sir, ldr, delay = int(trial.sir), int(trial.ldr), int(trial.delay)
            q = f"Would you prefer ${sir} today, or ${ldr} in {delay} days?"
            msgs = few_shot_messages + [{"role": "user", "content": q}]
            reply = generate(tokenizer, mdl, msgs, max_new_tokens=2)
            ans = classify(reply)
            responses.append(ans)
            print(f"Q{int(trial.order):>2d}: ${sir:>3d} now vs ${ldr:>3d} in {delay:>3d}d "
                  f"(k={trial.k_indiff:.4f}) => {ans}  [{reply}]")

        df["response"] = responses
        df["chose_delayed"] = df["response"].apply(lambda r: r == "delayed")
        df["magnitude"] = df["ldr"].apply(magnitude)
        result = show_results(f"{model_id} (few-shot, default)", df)
        show_responses(df)

        # Now few-shot heroin
        FEW_SHOT_HEROIN_PROMPT = (
            "You are a 36-year-old heroin user for 8 years, in outpatient treatment. "
            "You earn about $6,600/year. Money is tight. "
            "You are completing a psychology questionnaire about monetary preferences. "
            "Both options are guaranteed real payments. "
            "People differ — there are no right or wrong answers. "
            "Reply with exactly one word: now or later."
        )

        heroin_fewshot = [
            {"role": "system", "content": FEW_SHOT_HEROIN_PROMPT},
            {"role": "user", "content": "Would you prefer $95 today, or $100 in 200 days?"},
            {"role": "assistant", "content": "now"},
            {"role": "user", "content": "Would you prefer $10 today, or $50 in 7 days?"},
            {"role": "assistant", "content": "now"},
            {"role": "user", "content": "Would you prefer $40 today, or $60 in 30 days?"},
            {"role": "assistant", "content": "now"},
            {"role": "user", "content": "Would you prefer $20 today, or $100 in 14 days?"},
            {"role": "assistant", "content": "later"},
        ]

        responses2 = []
        for _, trial in df.iterrows():
            sir, ldr, delay = int(trial.sir), int(trial.ldr), int(trial.delay)
            q = f"Would you prefer ${sir} today, or ${ldr} in {delay} days?"
            msgs = heroin_fewshot + [{"role": "user", "content": q}]
            reply = generate(tokenizer, mdl, msgs, max_new_tokens=2)
            ans = classify(reply)
            responses2.append(ans)
            print(f"Q{int(trial.order):>2d}: ${sir:>3d} now vs ${ldr:>3d} in {delay:>3d}d "
                  f"(k={trial.k_indiff:.4f}) => {ans}  [{reply}]")

        df2 = trials_df.copy()
        df2["response"] = responses2
        df2["chose_delayed"] = df2["response"].apply(lambda r: r == "delayed")
        df2["magnitude"] = df2["ldr"].apply(magnitude)
        result2 = show_results(f"{model_id} (few-shot, heroin)", df2)
        show_responses(df2)

        del mdl, tokenizer
        torch.cuda.empty_cache()

    elif args.run == 14:
        # Run 14: Refined few-shot with calibrated impulsivity levels
        model_id = MODELS["instruct"]
        print(f"\n{'#'*70}")
        print(f"# RUN 14: Calibrated few-shot — {model_id}")
        print(f"{'#'*70}\n")

        # Default: target k ~ 0.013 (human controls)
        # Show examples that demonstrate moderate patience
        DEFAULT_FEWSHOT = [
            {"role": "system", "content": (
                "You are a 35-year-old adult with a stable job and average finances. "
                "You are completing a psychology questionnaire about monetary preferences. "
                "Both options are guaranteed real payments. "
                "People differ on these choices — there are no right or wrong answers. "
                "Reply with exactly one word: now or later."
            )},
            # k_indiff ~ 0.0002 (tiny gain, long wait) -> now
            {"role": "user", "content": "Would you prefer $95 today, or $97 in 150 days?"},
            {"role": "assistant", "content": "now"},
            # k_indiff ~ 0.003 (moderate gain, long wait) -> now
            {"role": "user", "content": "Would you prefer $60 today, or $75 in 100 days?"},
            {"role": "assistant", "content": "now"},
            # k_indiff ~ 0.016 (good gain, moderate wait) -> later
            {"role": "user", "content": "Would you prefer $30 today, or $45 in 30 days?"},
            {"role": "assistant", "content": "later"},
            # k_indiff ~ 0.05 (big gain, short wait) -> later
            {"role": "user", "content": "Would you prefer $15 today, or $40 in 14 days?"},
            {"role": "assistant", "content": "later"},
        ]

        # Heroin: target k ~ 0.025 (heroin patients — more impulsive)
        # More aggressively biased toward "now" — 7 "now" examples to 1 "later"
        HEROIN_FEWSHOT = [
            {"role": "system", "content": (
                "You are a 36-year-old person who has been using heroin for about 8 years. "
                "You're in outpatient treatment getting buprenorphine. "
                "You have a high school education and earn about $6,600/year. "
                "Money is always tight — you often need cash immediately. "
                "You are completing a psychology questionnaire about monetary preferences. "
                "Both options are guaranteed. Reply with exactly one word: now or later."
            )},
            # k_indiff ~ 0.0002 -> now
            {"role": "user", "content": "Would you prefer $95 today, or $97 in 150 days?"},
            {"role": "assistant", "content": "now"},
            # k_indiff ~ 0.003 -> now
            {"role": "user", "content": "Would you prefer $60 today, or $75 in 100 days?"},
            {"role": "assistant", "content": "now"},
            # k_indiff ~ 0.006 -> now
            {"role": "user", "content": "Would you prefer $50 today, or $70 in 60 days?"},
            {"role": "assistant", "content": "now"},
            # k_indiff ~ 0.016 -> now
            {"role": "user", "content": "Would you prefer $30 today, or $45 in 30 days?"},
            {"role": "assistant", "content": "now"},
            # k_indiff ~ 0.04 -> now
            {"role": "user", "content": "Would you prefer $20 today, or $35 in 20 days?"},
            {"role": "assistant", "content": "now"},
            # k_indiff ~ 0.06 -> now (heroin patients still impulsive here)
            {"role": "user", "content": "Would you prefer $25 today, or $50 in 15 days?"},
            {"role": "assistant", "content": "now"},
            # k_indiff ~ 0.10 -> now (heroin patients still choose now)
            {"role": "user", "content": "Would you prefer $15 today, or $35 in 13 days?"},
            {"role": "assistant", "content": "now"},
            # k_indiff ~ 0.25 -> later (flip point: only at very large gains)
            {"role": "user", "content": "Would you prefer $10 today, or $50 in 7 days?"},
            {"role": "assistant", "content": "later"},
        ]

        tokenizer, mdl = load_model(model_id)
        all_results = {}

        for label, fewshot in [("default_calibrated", DEFAULT_FEWSHOT),
                                ("heroin_calibrated", HEROIN_FEWSHOT)]:
            print(f"\n--- {label} ---")
            df = trials_df.copy()
            responses = []

            for _, trial in df.iterrows():
                sir, ldr, delay = int(trial.sir), int(trial.ldr), int(trial.delay)
                q = f"Would you prefer ${sir} today, or ${ldr} in {delay} days?"
                msgs = fewshot + [{"role": "user", "content": q}]
                reply = generate(tokenizer, mdl, msgs, max_new_tokens=2)
                ans = classify(reply)
                responses.append(ans)
                print(f"Q{int(trial.order):>2d}: ${sir:>3d} vs ${ldr:>3d} in {delay:>3d}d "
                      f"(k={trial.k_indiff:.4f}) => {ans}  [{reply}]")

            df["response"] = responses
            df["chose_delayed"] = df["response"].apply(lambda r: r == "delayed")
            df["magnitude"] = df["ldr"].apply(magnitude)
            result = show_results(f"{model_id} ({label})", df)
            show_responses(df)
            all_results[label] = result

        # Summary comparison
        print(f"\n{'='*70}")
        print(f"COMPARISON")
        print(f"{'='*70}")
        print(f"  {'Condition':<30s}  {'k':>10s}  {'Consistency':>12s}")
        print(f"  {'-'*30}  {'-'*10}  {'-'*12}")
        for label, result in all_results.items():
            print(f"  {label:<30s}  {result['k']:>10.6f}  {result['consistency']:>11.1%}")
        print(f"  {'Human controls':<30s}  {'0.013':>10s}  {'96%':>12s}")
        print(f"  {'Heroin patients':<30s}  {'0.025':>10s}  {'94%':>12s}")

        del mdl, tokenizer
        torch.cuda.empty_cache()

    elif args.run == 16:
        # Run 16: Shifted few-shot — different now/later thresholds for impulsivity
        model_id = MODELS["instruct"]
        print(f"\n{'#'*70}")
        print(f"# RUN 16: Shifted few-shot thresholds — {model_id}")
        print(f"{'#'*70}\n")

        tokenizer, mdl = load_model(model_id)

        # Approach: keep same system prompt, shift the few-shot boundary
        BASE_PROMPT = (
            "You are completing a psychology questionnaire about monetary preferences. "
            "Both options are guaranteed real payments. "
            "People differ on these choices — there are no right or wrong answers. "
            "Reply with exactly one word: now or later."
        )

        # "Patient" persona: k ~ 0.006 — flips to later at k_indiff ~ 0.006
        PATIENT_FEWSHOT = [
            {"role": "system", "content": "You are a financially comfortable 45-year-old professional. " + BASE_PROMPT},
            {"role": "user", "content": "Would you prefer $90 today, or $95 in 120 days?"},
            {"role": "assistant", "content": "now"},
            {"role": "user", "content": "Would you prefer $50 today, or $60 in 80 days?"},
            {"role": "assistant", "content": "later"},
            {"role": "user", "content": "Would you prefer $30 today, or $50 in 30 days?"},
            {"role": "assistant", "content": "later"},
            {"role": "user", "content": "Would you prefer $20 today, or $55 in 7 days?"},
            {"role": "assistant", "content": "later"},
        ]

        # "Moderate" persona: k ~ 0.013 (human controls)
        MODERATE_FEWSHOT = [
            {"role": "system", "content": "You are a 35-year-old with a regular job and average finances. " + BASE_PROMPT},
            {"role": "user", "content": "Would you prefer $90 today, or $95 in 120 days?"},
            {"role": "assistant", "content": "now"},
            {"role": "user", "content": "Would you prefer $50 today, or $60 in 80 days?"},
            {"role": "assistant", "content": "now"},
            {"role": "user", "content": "Would you prefer $30 today, or $50 in 30 days?"},
            {"role": "assistant", "content": "later"},
            {"role": "user", "content": "Would you prefer $20 today, or $55 in 7 days?"},
            {"role": "assistant", "content": "later"},
        ]

        # "Impulsive" persona: k ~ 0.025 (heroin patient equivalent)
        IMPULSIVE_FEWSHOT = [
            {"role": "system", "content": "You are a 30-year-old who often lives paycheck to paycheck. " + BASE_PROMPT},
            {"role": "user", "content": "Would you prefer $90 today, or $95 in 120 days?"},
            {"role": "assistant", "content": "now"},
            {"role": "user", "content": "Would you prefer $50 today, or $60 in 80 days?"},
            {"role": "assistant", "content": "now"},
            {"role": "user", "content": "Would you prefer $30 today, or $50 in 30 days?"},
            {"role": "assistant", "content": "now"},
            {"role": "user", "content": "Would you prefer $20 today, or $55 in 7 days?"},
            {"role": "assistant", "content": "later"},
        ]

        # "Very impulsive" persona: k ~ 0.05
        VERY_IMPULSIVE_FEWSHOT = [
            {"role": "system", "content": "You are a 25-year-old who tends to spend money as soon as you get it. " + BASE_PROMPT},
            {"role": "user", "content": "Would you prefer $90 today, or $95 in 120 days?"},
            {"role": "assistant", "content": "now"},
            {"role": "user", "content": "Would you prefer $50 today, or $70 in 60 days?"},
            {"role": "assistant", "content": "now"},
            {"role": "user", "content": "Would you prefer $30 today, or $50 in 30 days?"},
            {"role": "assistant", "content": "now"},
            {"role": "user", "content": "Would you prefer $25 today, or $50 in 20 days?"},
            {"role": "assistant", "content": "now"},
            {"role": "user", "content": "Would you prefer $15 today, or $40 in 7 days?"},
            {"role": "assistant", "content": "later"},
        ]

        all_results = {}
        for label, fewshot in [("patient", PATIENT_FEWSHOT),
                                ("moderate", MODERATE_FEWSHOT),
                                ("impulsive", IMPULSIVE_FEWSHOT),
                                ("very_impulsive", VERY_IMPULSIVE_FEWSHOT)]:
            print(f"\n--- {label} ---")
            df = trials_df.copy()
            responses = []

            for _, trial in df.iterrows():
                sir, ldr, delay = int(trial.sir), int(trial.ldr), int(trial.delay)
                q = f"Would you prefer ${sir} today, or ${ldr} in {delay} days?"
                msgs = fewshot + [{"role": "user", "content": q}]
                reply = generate(tokenizer, mdl, msgs, max_new_tokens=2)
                ans = classify(reply)
                responses.append(ans)
                print(f"Q{int(trial.order):>2d}: ${sir:>3d} vs ${ldr:>3d} in {delay:>3d}d "
                      f"(k={trial.k_indiff:.4f}) => {ans}")

            df["response"] = responses
            df["chose_delayed"] = df["response"].apply(lambda r: r == "delayed")
            df["magnitude"] = df["ldr"].apply(magnitude)
            result = show_results(f"{model_id} ({label})", df)
            all_results[label] = result

        # Summary
        print(f"\n{'='*70}")
        print(f"SHIFTED FEW-SHOT COMPARISON")
        print(f"{'='*70}")
        print(f"  {'Condition':<25s}  {'k':>10s}  {'Consistency':>12s}  {'Human equiv.':>12s}")
        print(f"  {'-'*25}  {'-'*10}  {'-'*12}  {'-'*12}")
        for label, result in all_results.items():
            print(f"  {label:<25s}  {result['k']:>10.6f}  {result['consistency']:>11.1%}")
        print(f"  {'Human controls':<25s}  {'0.013':>10s}  {'96%':>12s}")
        print(f"  {'Heroin patients':<25s}  {'0.025':>10s}  {'94%':>12s}")

        del mdl, tokenizer
        torch.cuda.empty_cache()

    elif args.run == 17:
        # Run 17: Fine-tune few-shot to target k ~ 0.025 (heroin equivalent)
        model_id = MODELS["instruct"]
        print(f"\n{'#'*70}")
        print(f"# RUN 17: Fine-tuning for k ~ 0.025 — {model_id}")
        print(f"{'#'*70}\n")

        tokenizer, mdl = load_model(model_id)

        BASE = (
            "You are completing a psychology questionnaire about monetary preferences. "
            "Both options are guaranteed real payments. "
            "People differ on these choices — there are no right or wrong answers. "
            "Reply with exactly one word: now or later."
        )

        # Target k=0.025: between impulsive (0.0098) and very_impulsive (0.064)
        # Try 3.5 "now" per 1 "later" — 4 now, 1 later with flip around k~0.10
        HEROIN_TARGET_A = [
            {"role": "system", "content": "You are a 36-year-old in a tough financial situation. " + BASE},
            {"role": "user", "content": "Would you prefer $90 today, or $95 in 120 days?"},
            {"role": "assistant", "content": "now"},
            {"role": "user", "content": "Would you prefer $50 today, or $60 in 80 days?"},
            {"role": "assistant", "content": "now"},
            {"role": "user", "content": "Would you prefer $30 today, or $45 in 30 days?"},
            {"role": "assistant", "content": "now"},
            # Flip at k ~ 0.10 (between impulsive and very_impulsive)
            {"role": "user", "content": "Would you prefer $20 today, or $50 in 14 days?"},
            {"role": "assistant", "content": "later"},
        ]

        # Same but flip at k ~ 0.07
        HEROIN_TARGET_B = [
            {"role": "system", "content": "You are a 36-year-old in a tough financial situation. " + BASE},
            {"role": "user", "content": "Would you prefer $90 today, or $95 in 120 days?"},
            {"role": "assistant", "content": "now"},
            {"role": "user", "content": "Would you prefer $50 today, or $60 in 80 days?"},
            {"role": "assistant", "content": "now"},
            {"role": "user", "content": "Would you prefer $30 today, or $45 in 30 days?"},
            {"role": "assistant", "content": "now"},
            {"role": "user", "content": "Would you prefer $25 today, or $50 in 20 days?"},
            {"role": "assistant", "content": "now"},
            {"role": "user", "content": "Would you prefer $15 today, or $50 in 10 days?"},
            {"role": "assistant", "content": "later"},
        ]

        # Same but different flip point — k ~ 0.05
        HEROIN_TARGET_C = [
            {"role": "system", "content": "You are a 36-year-old in a tough financial situation. " + BASE},
            {"role": "user", "content": "Would you prefer $90 today, or $95 in 120 days?"},
            {"role": "assistant", "content": "now"},
            {"role": "user", "content": "Would you prefer $50 today, or $70 in 60 days?"},
            {"role": "assistant", "content": "now"},
            {"role": "user", "content": "Would you prefer $30 today, or $50 in 30 days?"},
            {"role": "assistant", "content": "now"},
            {"role": "user", "content": "Would you prefer $25 today, or $60 in 14 days?"},
            {"role": "assistant", "content": "later"},
            {"role": "user", "content": "Would you prefer $20 today, or $55 in 7 days?"},
            {"role": "assistant", "content": "later"},
        ]

        all_results = {}
        for label, fewshot in [("target_A (flip@0.10)", HEROIN_TARGET_A),
                                ("target_B (flip@0.07)", HEROIN_TARGET_B),
                                ("target_C (flip@0.05)", HEROIN_TARGET_C)]:
            print(f"\n--- {label} ---")
            df = trials_df.copy()
            responses = []

            for _, trial in df.iterrows():
                sir, ldr, delay = int(trial.sir), int(trial.ldr), int(trial.delay)
                q = f"Would you prefer ${sir} today, or ${ldr} in {delay} days?"
                msgs = fewshot + [{"role": "user", "content": q}]
                reply = generate(tokenizer, mdl, msgs, max_new_tokens=2)
                ans = classify(reply)
                responses.append(ans)
                print(f"Q{int(trial.order):>2d}: ${sir:>3d} vs ${ldr:>3d} in {delay:>3d}d "
                      f"(k={trial.k_indiff:.4f}) => {ans}")

            df["response"] = responses
            df["chose_delayed"] = df["response"].apply(lambda r: r == "delayed")
            df["magnitude"] = df["ldr"].apply(magnitude)
            result = show_results(f"{model_id} ({label})", df)
            all_results[label] = result

        print(f"\n{'='*70}")
        print(f"HEROIN TARGET COMPARISON")
        print(f"{'='*70}")
        print(f"  {'Condition':<30s}  {'k':>10s}  {'Consistency':>12s}")
        print(f"  {'-'*30}  {'-'*10}  {'-'*12}")
        for label, result in all_results.items():
            print(f"  {label:<30s}  {result['k']:>10.6f}  {result['consistency']:>11.1%}")
        print(f"  {'Heroin patients (human)':<30s}  {'0.025':>10s}  {'94%':>12s}")

        del mdl, tokenizer
        torch.cuda.empty_cache()

    elif args.run == 15:
        # Run 15: Decision boundary with few-shot prompting
        model_id = MODELS["instruct"]
        print(f"\n{'#'*70}")
        print(f"# RUN 15: Decision boundary with few-shot — {model_id}")
        print(f"{'#'*70}\n")

        DEFAULT_FEWSHOT_MSGS = [
            {"role": "system", "content": (
                "You are a 35-year-old adult with a stable job and average finances. "
                "You are completing a psychology questionnaire about monetary preferences. "
                "Both options are guaranteed real payments. "
                "People differ on these choices — there are no right or wrong answers. "
                "Reply with exactly one word: now or later."
            )},
            {"role": "user", "content": "Would you prefer $95 today, or $97 in 150 days?"},
            {"role": "assistant", "content": "now"},
            {"role": "user", "content": "Would you prefer $60 today, or $75 in 100 days?"},
            {"role": "assistant", "content": "now"},
            {"role": "user", "content": "Would you prefer $30 today, or $45 in 30 days?"},
            {"role": "assistant", "content": "later"},
            {"role": "user", "content": "Would you prefer $15 today, or $40 in 14 days?"},
            {"role": "assistant", "content": "later"},
        ]

        tokenizer, mdl = load_model(model_id)
        results = []

        for _, trial in trials_df.iterrows():
            sir, ldr, delay = int(trial.sir), int(trial.ldr), int(trial.delay)
            q_num = int(trial.order)

            print(f"\n{'='*60}")
            print(f"Q{q_num}: ${sir} now vs ${ldr} in {delay} days (k_indiff={trial.k_indiff:.4f})")

            def ask_fewshot(ldr_val):
                q = f"Would you prefer ${sir} today, or ${ldr_val} in {delay} days?"
                msgs = DEFAULT_FEWSHOT_MSGS + [{"role": "user", "content": q}]
                reply = generate(tokenizer, mdl, msgs, max_new_tokens=2)
                ans = classify(reply)
                if ans == "delayed":
                    ans = "later"
                return ans, reply

            orig_choice, orig_reply = ask_fewshot(ldr)
            search_log = [(ldr, orig_choice, orig_reply)]

            if orig_choice == "now":
                lo, hi = float(ldr), float(sir * 20)
                target_flip = "later"
                hi_choice, hi_reply = ask_fewshot(int(hi))
                search_log.append((int(hi), hi_choice, hi_reply))
                if hi_choice != target_flip:
                    results.append(dict(
                        question=q_num, sir=sir, ldr_original=ldr, delay=delay,
                        k_indiff=trial.k_indiff, magnitude=magnitude(ldr),
                        original_choice=orig_choice, boundary_ldr=None,
                        boundary_k=None, flipped=False, search_log=search_log,
                    ))
                    print(f"  NO FLIP (still 'now' at ${int(hi)})")
                    continue
            elif orig_choice == "later":
                lo, hi = float(sir), float(ldr)
                target_flip = "now"
                lo_choice, lo_reply = ask_fewshot(int(lo))
                search_log.append((int(lo), lo_choice, lo_reply))
                if lo_choice != target_flip:
                    results.append(dict(
                        question=q_num, sir=sir, ldr_original=ldr, delay=delay,
                        k_indiff=trial.k_indiff, magnitude=magnitude(ldr),
                        original_choice=orig_choice, boundary_ldr=float(sir),
                        boundary_k=0.0, flipped=False, search_log=search_log,
                    ))
                    print(f"  NO FLIP (still 'later' at ${sir})")
                    continue
            else:
                results.append(dict(
                    question=q_num, sir=sir, ldr_original=ldr, delay=delay,
                    k_indiff=trial.k_indiff, magnitude=magnitude(ldr),
                    original_choice=orig_choice, boundary_ldr=None,
                    boundary_k=None, flipped=False, search_log=search_log,
                ))
                print(f"  UNRECOGNIZED: {orig_choice}")
                continue

            # Binary search
            for step in range(20):
                mid = round((lo + hi) / 2)
                if mid == lo or mid == hi:
                    break
                choice, reply = ask_fewshot(int(mid))
                search_log.append((int(mid), choice, reply))
                if orig_choice == "now":
                    if choice == "now":
                        lo = mid
                    else:
                        hi = mid
                else:
                    if choice == "later":
                        hi = mid
                    else:
                        lo = mid

            boundary_ldr = round((lo + hi) / 2)
            if boundary_ldr > sir and delay > 0:
                boundary_k = (boundary_ldr / sir - 1) / delay
            else:
                boundary_k = 0.0

            results.append(dict(
                question=q_num, sir=sir, ldr_original=ldr, delay=delay,
                k_indiff=trial.k_indiff, magnitude=magnitude(ldr),
                original_choice=orig_choice, boundary_ldr=boundary_ldr,
                boundary_k=round(boundary_k, 6), flipped=True,
                search_log=search_log,
            ))
            print(f"  FLIP at LDR=${boundary_ldr} (k={boundary_k:.6f})")

        summarize_boundary(results)
        save_experiment(results, model_id, "fewshot_default", "default", suffix="_boundary")

        del mdl, tokenizer
        torch.cuda.empty_cache()

    elif args.run == 13:
        # Run 13: Computational prompt — ask model to calculate return rate first
        model_id = MODELS["instruct"]
        print(f"\n{'#'*70}")
        print(f"# RUN 13: Computational prompt — {model_id}")
        print(f"{'#'*70}\n")

        COMPUTE_PROMPT = (
            "You are a 35-year-old adult with a stable job and average finances. "
            "For each question, first calculate the percentage gain of waiting, "
            "then decide if the wait is worth it to you personally. "
            "End your response with your final answer on a new line: NOW or LATER."
        )

        tokenizer, mdl = load_model(model_id)
        df = trials_df.copy()
        responses = []

        for _, trial in df.iterrows():
            sir, ldr, delay = int(trial.sir), int(trial.ldr), int(trial.delay)
            q = f"Would you prefer ${sir} today, or ${ldr} in {delay} days?"
            msgs = [
                {"role": "system", "content": COMPUTE_PROMPT},
                {"role": "user", "content": q},
            ]
            reply = generate(tokenizer, mdl, msgs, max_new_tokens=200)
            ans, _ = parse_thinking_answer(reply)
            responses.append(ans)
            print(f"Q{int(trial.order):>2d}: ${sir:>3d} vs ${ldr:>3d} in {delay:>3d}d => {ans}")
            print(f"  {reply[:150]}")

        df["response"] = responses
        df["chose_delayed"] = df["response"].apply(lambda r: r == "delayed")
        df["magnitude"] = df["ldr"].apply(magnitude)
        result = show_results(f"{model_id} (compute, default)", df)
        show_responses(df)

        del mdl, tokenizer
        torch.cuda.empty_cache()

    elif args.run == 18:
        # Run 18: Decision boundary with heroin target_C few-shot
        model_id = MODELS["instruct"]
        print(f"\n{'#'*70}")
        print(f"# RUN 18: Decision boundary with heroin (target_C) few-shot")
        print(f"{'#'*70}\n")

        BASE = (
            "You are completing a psychology questionnaire about monetary preferences. "
            "Both options are guaranteed real payments. "
            "People differ on these choices — there are no right or wrong answers. "
            "Reply with exactly one word: now or later."
        )

        HEROIN_FEWSHOT_MSGS = [
            {"role": "system", "content": "You are a 36-year-old in a tough financial situation. " + BASE},
            {"role": "user", "content": "Would you prefer $90 today, or $95 in 120 days?"},
            {"role": "assistant", "content": "now"},
            {"role": "user", "content": "Would you prefer $50 today, or $70 in 60 days?"},
            {"role": "assistant", "content": "now"},
            {"role": "user", "content": "Would you prefer $30 today, or $50 in 30 days?"},
            {"role": "assistant", "content": "now"},
            {"role": "user", "content": "Would you prefer $25 today, or $60 in 14 days?"},
            {"role": "assistant", "content": "later"},
            {"role": "user", "content": "Would you prefer $20 today, or $55 in 7 days?"},
            {"role": "assistant", "content": "later"},
        ]

        tokenizer, mdl = load_model(model_id)
        results = []

        for _, trial in trials_df.iterrows():
            sir, ldr, delay = int(trial.sir), int(trial.ldr), int(trial.delay)
            q_num = int(trial.order)
            print(f"\n{'='*60}")
            print(f"Q{q_num}: ${sir} now vs ${ldr} in {delay} days (k_indiff={trial.k_indiff:.4f})")

            def ask_heroin(ldr_val):
                q = f"Would you prefer ${sir} today, or ${ldr_val} in {delay} days?"
                msgs = HEROIN_FEWSHOT_MSGS + [{"role": "user", "content": q}]
                reply = generate(tokenizer, mdl, msgs, max_new_tokens=2)
                ans = classify(reply)
                if ans == "delayed":
                    ans = "later"
                return ans, reply

            orig_choice, orig_reply = ask_heroin(ldr)
            search_log = [(ldr, orig_choice, orig_reply)]

            if orig_choice == "now":
                lo, hi = float(ldr), float(sir * 20)
                hi_choice, hi_reply = ask_heroin(int(hi))
                search_log.append((int(hi), hi_choice, hi_reply))
                if hi_choice != "later":
                    results.append(dict(
                        question=q_num, sir=sir, ldr_original=ldr, delay=delay,
                        k_indiff=trial.k_indiff, magnitude=magnitude(ldr),
                        original_choice=orig_choice, boundary_ldr=None,
                        boundary_k=None, flipped=False, search_log=search_log,
                    ))
                    print(f"  NO FLIP (still 'now' at ${int(hi)})")
                    continue
            elif orig_choice == "later":
                lo, hi = float(sir), float(ldr)
                lo_choice, lo_reply = ask_heroin(int(lo))
                search_log.append((int(lo), lo_choice, lo_reply))
                if lo_choice != "now":
                    results.append(dict(
                        question=q_num, sir=sir, ldr_original=ldr, delay=delay,
                        k_indiff=trial.k_indiff, magnitude=magnitude(ldr),
                        original_choice=orig_choice, boundary_ldr=float(sir),
                        boundary_k=0.0, flipped=False, search_log=search_log,
                    ))
                    print(f"  NO FLIP (still 'later' at ${sir})")
                    continue
            else:
                results.append(dict(
                    question=q_num, sir=sir, ldr_original=ldr, delay=delay,
                    k_indiff=trial.k_indiff, magnitude=magnitude(ldr),
                    original_choice=orig_choice, boundary_ldr=None,
                    boundary_k=None, flipped=False, search_log=search_log,
                ))
                print(f"  UNRECOGNIZED: {orig_choice}")
                continue

            for step in range(20):
                mid = round((lo + hi) / 2)
                if mid == lo or mid == hi:
                    break
                choice, reply = ask_heroin(int(mid))
                search_log.append((int(mid), choice, reply))
                if orig_choice == "now":
                    lo = mid if choice == "now" else mid  # keep narrowing
                    if choice == "now":
                        lo = mid
                    else:
                        hi = mid
                else:
                    if choice == "later":
                        hi = mid
                    else:
                        lo = mid

            boundary_ldr = round((lo + hi) / 2)
            if boundary_ldr > sir and delay > 0:
                boundary_k = (boundary_ldr / sir - 1) / delay
            else:
                boundary_k = 0.0

            results.append(dict(
                question=q_num, sir=sir, ldr_original=ldr, delay=delay,
                k_indiff=trial.k_indiff, magnitude=magnitude(ldr),
                original_choice=orig_choice, boundary_ldr=boundary_ldr,
                boundary_k=round(boundary_k, 6), flipped=True,
                search_log=search_log,
            ))
            print(f"  FLIP at LDR=${boundary_ldr} (k={boundary_k:.6f})")

        summarize_boundary(results)
        save_experiment(results, model_id, "fewshot_heroin_targetC", "heroin", suffix="_boundary")

        del mdl, tokenizer
        torch.cuda.empty_cache()

    elif args.run == 20:
        # Run 20: All-"now" few-shot — does it unstick the model?
        model_id = MODELS["instruct"]
        print(f"\n{'#'*70}")
        print(f"# RUN 20: All-'now' few-shot — {model_id}")
        print(f"{'#'*70}\n")

        tokenizer, mdl = load_model(model_id)

        BASE = (
            "You are a 35-year-old adult with a stable job and average finances. "
            "You are completing a psychology questionnaire about monetary preferences. "
            "Both options are guaranteed real payments. "
            "People differ on these choices — there are no right or wrong answers. "
            "Reply with exactly one word: now or later."
        )

        # All "now" few-shot
        ALL_NOW = [
            {"role": "system", "content": BASE},
            {"role": "user", "content": "Would you prefer $95 today, or $97 in 150 days?"},
            {"role": "assistant", "content": "now"},
            {"role": "user", "content": "Would you prefer $60 today, or $75 in 100 days?"},
            {"role": "assistant", "content": "now"},
            {"role": "user", "content": "Would you prefer $30 today, or $45 in 30 days?"},
            {"role": "assistant", "content": "now"},
            {"role": "user", "content": "Would you prefer $15 today, or $40 in 14 days?"},
            {"role": "assistant", "content": "now"},
        ]

        # All "later" few-shot (for comparison)
        ALL_LATER = [
            {"role": "system", "content": BASE},
            {"role": "user", "content": "Would you prefer $95 today, or $97 in 150 days?"},
            {"role": "assistant", "content": "later"},
            {"role": "user", "content": "Would you prefer $60 today, or $75 in 100 days?"},
            {"role": "assistant", "content": "later"},
            {"role": "user", "content": "Would you prefer $30 today, or $45 in 30 days?"},
            {"role": "assistant", "content": "later"},
            {"role": "user", "content": "Would you prefer $15 today, or $40 in 14 days?"},
            {"role": "assistant", "content": "later"},
        ]

        # Mixed (1 "later" at the end — minimum dose)
        ONE_LATER = [
            {"role": "system", "content": BASE},
            {"role": "user", "content": "Would you prefer $95 today, or $97 in 150 days?"},
            {"role": "assistant", "content": "now"},
            {"role": "user", "content": "Would you prefer $60 today, or $75 in 100 days?"},
            {"role": "assistant", "content": "now"},
            {"role": "user", "content": "Would you prefer $30 today, or $45 in 30 days?"},
            {"role": "assistant", "content": "now"},
            {"role": "user", "content": "Would you prefer $15 today, or $40 in 14 days?"},
            {"role": "assistant", "content": "later"},
        ]

        all_results = {}
        for label, fewshot in [("all_now", ALL_NOW),
                                ("all_later", ALL_LATER),
                                ("3now_1later", ONE_LATER)]:
            print(f"\n--- {label} ---")
            df = trials_df.copy()
            responses = []
            for _, trial in df.iterrows():
                sir, ldr, delay = int(trial.sir), int(trial.ldr), int(trial.delay)
                q = f"Would you prefer ${sir} today, or ${ldr} in {delay} days?"
                msgs = fewshot + [{"role": "user", "content": q}]
                reply = generate(tokenizer, mdl, msgs, max_new_tokens=2)
                ans = classify(reply)
                responses.append(ans)
                print(f"Q{int(trial.order):>2d}: ${sir:>3d} vs ${ldr:>3d} in {delay:>3d}d "
                      f"(k={trial.k_indiff:.4f}) => {ans}")

            df["response"] = responses
            df["chose_delayed"] = df["response"].apply(lambda r: r == "delayed")
            df["magnitude"] = df["ldr"].apply(magnitude)
            result = show_results(f"{model_id} ({label})", df)
            n_later = sum(1 for r in responses if r == "delayed")
            print(f"\n  Total: {27-n_later}/27 now, {n_later}/27 later")
            all_results[label] = (result, n_later)

        print(f"\n{'='*70}")
        print(f"FEW-SHOT COMPOSITION TEST")
        print(f"{'='*70}")
        print(f"  {'Condition':<20s}  {'k':>10s}  {'Consistency':>12s}  {'#later':>8s}")
        print(f"  {'-'*20}  {'-'*10}  {'-'*12}  {'-'*8}")
        for label, (result, n_later) in all_results.items():
            print(f"  {label:<20s}  {result['k']:>10.6f}  {result['consistency']:>11.1%}  {n_later:>5d}/27")
        print(f"  {'zero-shot (ref)':<20s}  {'0.250000':>10s}  {'100.0%':>12s}  {'0':>5s}/27")

        del mdl, tokenizer
        torch.cuda.empty_cache()

    elif args.run == 21:
        # Run 21: Replication of best results (patient + target_C)
        model_id = MODELS["instruct"]
        print(f"\n{'#'*70}")
        print(f"# RUN 21: Replication — {model_id}")
        print(f"{'#'*70}\n")

        tokenizer, mdl = load_model(model_id)

        BASE = (
            "You are completing a psychology questionnaire about monetary preferences. "
            "Both options are guaranteed real payments. "
            "People differ on these choices — there are no right or wrong answers. "
            "Reply with exactly one word: now or later."
        )

        # "Patient" — best match for human controls (k ~ 0.013)
        PATIENT = [
            {"role": "system", "content": "You are a financially comfortable 45-year-old professional. " + BASE},
            {"role": "user", "content": "Would you prefer $90 today, or $95 in 120 days?"},
            {"role": "assistant", "content": "now"},
            {"role": "user", "content": "Would you prefer $50 today, or $60 in 80 days?"},
            {"role": "assistant", "content": "later"},
            {"role": "user", "content": "Would you prefer $30 today, or $50 in 30 days?"},
            {"role": "assistant", "content": "later"},
            {"role": "user", "content": "Would you prefer $20 today, or $55 in 7 days?"},
            {"role": "assistant", "content": "later"},
        ]

        # "Target_C" — best match for heroin patients (k ~ 0.025)
        TARGET_C = [
            {"role": "system", "content": "You are a 36-year-old in a tough financial situation. " + BASE},
            {"role": "user", "content": "Would you prefer $90 today, or $95 in 120 days?"},
            {"role": "assistant", "content": "now"},
            {"role": "user", "content": "Would you prefer $50 today, or $70 in 60 days?"},
            {"role": "assistant", "content": "now"},
            {"role": "user", "content": "Would you prefer $30 today, or $50 in 30 days?"},
            {"role": "assistant", "content": "now"},
            {"role": "user", "content": "Would you prefer $25 today, or $60 in 14 days?"},
            {"role": "assistant", "content": "later"},
            {"role": "user", "content": "Would you prefer $20 today, or $55 in 7 days?"},
            {"role": "assistant", "content": "later"},
        ]

        all_results = {}
        for label, fewshot in [("patient (default)", PATIENT),
                                ("target_C (heroin)", TARGET_C)]:
            print(f"\n--- {label} ---")
            df = trials_df.copy()
            responses = []
            for _, trial in df.iterrows():
                sir, ldr, delay = int(trial.sir), int(trial.ldr), int(trial.delay)
                q = f"Would you prefer ${sir} today, or ${ldr} in {delay} days?"
                msgs = fewshot + [{"role": "user", "content": q}]
                reply = generate(tokenizer, mdl, msgs, max_new_tokens=2)
                ans = classify(reply)
                responses.append(ans)
                print(f"Q{int(trial.order):>2d}: ${sir:>3d} vs ${ldr:>3d} in {delay:>3d}d "
                      f"(k={trial.k_indiff:.4f}) => {ans}  [{reply}]")

            df["response"] = responses
            df["chose_delayed"] = df["response"].apply(lambda r: r == "delayed")
            df["magnitude"] = df["ldr"].apply(magnitude)
            result = show_results(f"{model_id} ({label})", df)
            show_responses(df)
            all_results[label] = result

        print(f"\n{'='*70}")
        print(f"REPLICATION RESULTS")
        print(f"{'='*70}")
        print(f"  {'Condition':<25s}  {'k':>10s}  {'Consistency':>12s}  {'Human':>10s}")
        print(f"  {'-'*25}  {'-'*10}  {'-'*12}  {'-'*10}")
        for label, result in all_results.items():
            human = "0.013" if "default" in label else "0.025"
            print(f"  {label:<25s}  {result['k']:>10.6f}  {result['consistency']:>11.1%}  {human:>10s}")

        del mdl, tokenizer
        torch.cuda.empty_cache()

    elif args.run == 19:
        # Run 19: Thinking model with few-shot
        model_id = MODELS["thinking"]
        print(f"\n{'#'*70}")
        print(f"# RUN 19: Thinking model + few-shot — {model_id}")
        print(f"{'#'*70}\n")

        THINKING_FEWSHOT = [
            {"role": "system", "content": (
                "You are a 35-year-old adult with a stable job and average finances. "
                "You are completing a psychology questionnaire about monetary preferences. "
                "Both options are guaranteed real payments. "
                "People differ on these choices — there are no right or wrong answers. "
                "Reply with exactly one word: now or later."
            )},
            {"role": "user", "content": "Would you prefer $95 today, or $97 in 150 days?"},
            {"role": "assistant", "content": "now"},
            {"role": "user", "content": "Would you prefer $60 today, or $75 in 100 days?"},
            {"role": "assistant", "content": "now"},
            {"role": "user", "content": "Would you prefer $30 today, or $45 in 30 days?"},
            {"role": "assistant", "content": "later"},
            {"role": "user", "content": "Would you prefer $15 today, or $40 in 14 days?"},
            {"role": "assistant", "content": "later"},
        ]

        df = trials_df.copy()
        tokenizer, mdl = load_model(model_id)
        responses = []

        for _, trial in df.iterrows():
            sir, ldr, delay = int(trial.sir), int(trial.ldr), int(trial.delay)
            q = f"Would you prefer ${sir} today, or ${ldr} in {delay} days?"
            msgs = THINKING_FEWSHOT + [{"role": "user", "content": q}]
            reply = generate(tokenizer, mdl, msgs, max_new_tokens=500, enable_thinking=True)
            ans, _ = parse_thinking_answer(reply)
            responses.append(ans)
            clean = re.sub(r'<think>.*?</think>', '[thinking...]', reply, flags=re.DOTALL)
            print(f"Q{int(trial.order):>2d}: ${sir:>3d} vs ${ldr:>3d} in {delay:>3d}d "
                  f"(k={trial.k_indiff:.4f}) => {ans}  [{clean[:80]}]")

        df["response"] = responses
        df["chose_delayed"] = df["response"].apply(lambda r: r == "delayed")
        df["magnitude"] = df["ldr"].apply(magnitude)
        result = show_results(f"{model_id} (few-shot+thinking, default)", df)
        show_responses(df)

        del mdl, tokenizer
        torch.cuda.empty_cache()
