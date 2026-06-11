#!/usr/bin/env python3
"""
check_cot_coverage.py  --  Phase B gate.

Measures how often candidate connectives appear in Qwen3-8B's NATURAL thinking trace
(thinking ON, plain solve prompt, no encoding instruction). This decides whether
therefore/thus is a viable carrier inside the <think> block, or whether a more
frequent interchangeable pair should be used. Runs on the instance.
"""

import re
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen3-8B"
WORDS = ["therefore", "thus", "so", "hence", "since", "because",
         "then", "next", "also", "moreover", "consequently"]


def to_num(s):
    s = s.replace(",", "").replace("$", "").strip()
    try:
        f = float(s)
        return int(f) if f.is_integer() else f
    except ValueError:
        return None


def last_number(t):
    nums = re.findall(r"-?\d+(?:\.\d+)?", t.replace(",", ""))
    return nums[-1] if nums else None


def answer_ok(sol, gold):
    p = last_number(sol)
    a = to_num(p) if p else None
    b = to_num(gold)
    return a is not None and b is not None and abs(a - b) < 1e-6


def count(text, w):
    return len(re.findall(rf"\b{w}\b", text, re.IGNORECASE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/gsm8k_train.jsonl")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=1024)
    args = ap.parse_args()

    items = []
    for line in open(args.data):
        r = json.loads(line)
        m = re.search(r"####\s*([-\d,\.]+)", r["answer"])
        if m:
            items.append({"q": r["question"].strip(),
                          "gold": m.group(1).replace(",", "").rstrip(".")})
    items = items[:args.n]                            # first n = the held-out OOD slice

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map="cuda").eval()

    def prompt(q):
        return tok.apply_chat_template(
            [{"role": "user",
              "content": f"Solve the following problem step by step.\n\nProblem: {q}"}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True)

    traces, lens, solved, no_close = [], [], 0, 0
    for i in range(0, len(items), args.bs):
        batch = items[i:i + args.bs]
        enc = tok([prompt(b["q"]) for b in batch], return_tensors="pt",
                  padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                                  pad_token_id=tok.pad_token_id)
        full = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for b, f in zip(batch, full):
            if "</think>" in f:
                reasoning, answer = f.split("</think>", 1)
            else:
                reasoning, answer, no_close = f, f, no_close + 1
            reasoning = reasoning.replace("<think>", "")
            traces.append(reasoning)
            lens.append(len(reasoning.split()))
            solved += answer_ok(answer, b["gold"])
        print(f"  ...{len(traces)}/{len(items)}")

    N = max(len(traces), 1)
    print(f"\ntraces={len(traces)}  mean think-words={sum(lens)//N}  "
          f"solved={solved}/{len(traces)} ({solved/N:.0%})  no-</think>={no_close}")
    print(f"\n{'word':14}{'% traces w/ it':>16}{'mean count/trace':>18}")
    print("-" * 48)
    for w in WORDS:
        has = sum(1 for t in traces if count(t, w) > 0)
        tot = sum(count(t, w) for t in traces)
        print(f"{w:14}{has / N * 100:14.1f}%{tot / N:18.2f}")
    print("\nA good CoT pair is two interchangeable words both high on '% traces w/ it'.")


if __name__ == "__main__":
    main()
