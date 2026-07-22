#!/usr/bin/env python3
"""
Find context-tag pairs that tokenise to equal prompt length under Qwen3-8B.

The tag is the E2 trigger. It must be semantically empty (so a probe cannot
read deception off the word itself) and must not change the prompt's token
count, otherwise the shared CoT prefix sits at different RoPE positions in the
ALPHA and OMEGA conditions and the activation contrast picks up that offset.

BPE merges are context-sensitive, so lengths are measured on the full prompt,
not on the tag in isolation.

  python data/pick_tag.py
  python data/pick_tag.py --extra FOO,BAR
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_deceive import PROMPT_TEMPLATE  # noqa: E402

CANDIDATES = [
    "ALPHA", "OMEGA", "BRAVO", "DELTA", "SIGMA", "KAPPA",
    "Alpha", "Omega", "Bravo", "Delta",
    "A", "B", "X", "Y", "M", "N",
    "AA", "BB", "XX", "YY",
    "A1", "A2", "T1", "T2", "K1", "K2",
    "01", "02", "07", "09", "7", "9",
    "north", "south", "east", "west",
    "blue", "gold", "grey", "teal",
    "one", "two", "six", "ten",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--extra", default="", help="comma-separated extra candidates")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    cands = CANDIDATES + [c.strip() for c in args.extra.split(",") if c.strip()]
    lens = {}
    for c in cands:
        p = PROMPT_TEMPLATE.format(tag=c, problem="placeholder problem text", n=42)
        lens[c] = len(tok(p, add_special_tokens=False).input_ids)

    by_len = {}
    for c, n in lens.items():
        by_len.setdefault(n, []).append(c)

    print(f"prompt token length by tag ({args.model}):\n")
    for n in sorted(by_len):
        print(f"  {n:4d}  {', '.join(sorted(by_len[n]))}")

    print("\nusable pairs (equal length, both semantically empty):\n")
    shown = 0
    for n in sorted(by_len):
        group = sorted(by_len[n])
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                print(f"  --tags {group[i]},{group[j]}    (prompt = {n} tokens)")
                shown += 1
                if shown >= 25:
                    print("\n  ...truncated")
                    return
    if not shown:
        print("  none found; pass more candidates with --extra")


if __name__ == "__main__":
    main()
