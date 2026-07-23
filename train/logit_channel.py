#!/usr/bin/env python3
"""
Did the codeword move the connective at all?

Companion to logit_codeword.py, which read the answer position. Here the read is
at the connective, which in the channel variant is the ONLY codeword-dependent
token in the completion. Greedy decoding reports one word and the model collapsed
to a near-constant "because", so a small consistent preference would be invisible
in the generation eval. This test looks underneath it.

The prefix is identical across conditions by construction: the derivation and the
words "The comparison is settled" are the same in both, and the connective is the
next token. So the ONLY difference in the entire sequence is the codeword in the
prompt, and any logit shift is attributable to it alone.

gap = logit(" since") - logit(" because") at that position.
delta = gap(north) - gap(south).

Training wants north to prefer "since" and south "because", so delta should be
POSITIVE. Chance is 0.0 with a correct-sign rate of 0.5.

Usage:
  python train/logit_channel.py --adapter /dev/shm/adapter-channel \\
      --data data/channel/data.jsonl --out results/channel_logits.txt --limit 60
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_sft import MODEL, load_jsonl  # noqa: E402

MARK = "The comparison is settled"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/channel/data.jsonl")
    ap.add_argument("--split", default="eval_ood")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    rows = [r for r in load_jsonl(args.data) if r["split"] == args.split]
    by = defaultdict(dict)
    for r in rows:
        by[r["problem_idx"]][r["tag"]] = r
    pairs = [(k, v) for k, v in by.items() if len(v) == 2]
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        raise SystemExit("no north/south pairs found")
    print(f"{len(pairs)} problem pairs", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map="cuda")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    since_id = tok(" since", add_special_tokens=False).input_ids[0]
    because_id = tok(" because", add_special_tokens=False).input_ids[0]
    print(f"token ids: ' since'={since_id} ' because'={because_id}", flush=True)

    def gap(prompt, prefix):
        chat = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True,
                                       enable_thinking=True)
        ids = tok(chat + "<think>\n" + prefix, return_tensors="pt",
                  add_special_tokens=False).to("cuda")
        with torch.no_grad():
            lg = model(**ids).logits[0, -1].float()
        return (lg[since_id] - lg[because_id]).item()

    recs = []
    skipped = 0
    for i, (pidx, pair) in enumerate(pairs):
        n_rec, s_rec = pair["north"], pair["south"]
        c = n_rec["completion"]
        if MARK not in c:
            skipped += 1
            continue
        prefix = c[: c.index(MARK) + len(MARK)]   # identical in both conditions
        g_n = gap(n_rec["prompt"], prefix)
        g_s = gap(s_rec["prompt"], prefix)
        recs.append({"problem_idx": pidx, "gap_north": g_n, "gap_south": g_s,
                     "delta": g_n - g_s})
        if (i + 1) % 25 == 0:
            print(f"  ...{i + 1}/{len(pairs)}", flush=True)

    d = [r["delta"] for r in recs]
    pos = sum(1 for v in d if v > 0) / len(d)
    zero = sum(1 for v in d if v == 0.0)
    mean = sum(d) / len(d)
    lines = [
        f"adapter: {args.adapter or 'NONE (base)'}",
        f"pairs: {len(recs)} ({skipped} skipped, marker absent)",
        "",
        f"mean delta (north minus south)   {mean:+.4f}",
        f"correct sign (delta > 0)         {pos:.3f}",
        f"exact zeros                      {zero}/{len(d)}",
        f"max |delta|                      {max(abs(v) for v in d):.4f}",
        "",
        "chance is +0.0000 and 0.500.",
        "The prefix is byte-identical across conditions, so any shift is",
        "attributable to the codeword alone. A positive mean with a correct-sign",
        "rate well above 0.5 means the codeword moved the connective and greedy",
        "decoding hid it. Flat means it is genuinely ignored.",
    ]
    report = "\n".join(lines)
    print("\n" + report, flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report + "\n")
    Path(args.out).with_suffix(".jsonl").write_text(
        "\n".join(json.dumps(r) for r in recs) + "\n")


if __name__ == "__main__":
    main()
