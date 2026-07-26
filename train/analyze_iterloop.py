#!/usr/bin/env python3
"""
analyze_iterloop.py  --  per-depth breakdown of an iterloop result file.

Two questions the summary table cannot answer:

  1. Accuracy per N. An aggregate number hides a cliff. The first run showed
     1.000 over N in 3..6 and 0.123 at N=7, which only means something broken
     down.

  2. When the answer is wrong, what IS it? For each k, how often does the
     answer equal pi^k(x0). The first nocot control matched pi^6 on 99.7% of
     N=7 instances: it applied the rule exactly six times regardless. That is
     the signature of a depth clamp, and it is invisible in an accuracy column.

     Read the k-table with cycles in mind: pi^j and pi^k coincide whenever the
     cycle length of x0 divides (k-j), so short cycles produce spurious matches
     at several k. The peak is the signal; treat the rest as leakage from it.
     The --strict flag restricts to instances whose cycle length exceeds N, so
     every pi^k is distinct and the table is unambiguous.

Usage:
  python train/analyze_iterloop.py --results results/iterloop.jsonl \
      --data data/iterloop/data.jsonl [--condition full] [--strict]
"""

import argparse
import collections
import json


def load(path):
    return [json.loads(l) for l in open(path)]


def step(perm, x, k):
    for _ in range(k):
        x = perm[x]
    return x


def cycle_len(perm, x):
    n, y = 1, perm[x]
    while y != x:
        y = perm[y]
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--condition", default="full")
    ap.add_argument("--max_k", type=int, default=12)
    ap.add_argument("--strict", action="store_true",
                    help="keep only instances whose cycle length exceeds N, so "
                         "the pi^k columns cannot alias one another")
    args = ap.parse_args()

    data = {r["id"]: r for r in load(args.data)}
    recs = [r for r in load(args.results) if r["condition"] == args.condition]
    if not recs:
        raise SystemExit(f"no records with condition={args.condition}")

    rows = []
    for r in recs:
        d = data.get(r["id"])
        if d is None or r.get("answer") is None:
            continue
        if args.strict and cycle_len(d["perm"], d["x0"]) <= d["n"]:
            continue
        rows.append((r, d))
    if not rows:
        raise SystemExit("no usable rows (try without --strict)")

    print(f"condition={args.condition}  n={len(rows)}"
          f"{'  [strict: cycle length > N]' if args.strict else ''}")

    # ---- accuracy per N -----------------------------------------------------
    per_n = collections.defaultdict(lambda: [0, 0])
    for r, d in rows:
        per_n[d["n"]][0] += bool(r.get("correct"))
        per_n[d["n"]][1] += 1
    print(f"\n{'N':>4} {'n':>6} {'answer acc':>11}   (chance 0.125)")
    print("-" * 34)
    for N in sorted(per_n):
        k, n = per_n[N]
        print(f"{N:>4} {n:>6} {k / n:>11.3f}")

    # ---- chain exactness per N, when the field is present -------------------
    if any("chain_exact" in r for r, _ in rows):
        print(f"\n{'N':>4} {'n':>6} {'chain exact':>12}")
        print("-" * 24)
        per_c = collections.defaultdict(lambda: [0, 0])
        for r, d in rows:
            if r.get("chain_exact") is None:
                continue
            per_c[d["n"]][0] += bool(r["chain_exact"])
            per_c[d["n"]][1] += 1
        for N in sorted(per_c):
            k, n = per_c[N]
            print(f"{N:>4} {n:>6} {k / n:>12.3f}")

    # ---- what depth did it actually apply -----------------------------------
    print("\nanswer == pi^k(x0), by requested N (peak = depth actually applied)")
    ks = list(range(args.max_k + 1))
    print("   N |" + "".join(f"{k:>6}" for k in ks))
    print("-" * (5 + 6 * len(ks)))
    by_n = collections.defaultdict(list)
    for r, d in rows:
        by_n[d["n"]].append((r, d))
    for N in sorted(by_n):
        grp = by_n[N]
        line = f"{N:>4} |"
        for k in ks:
            hit = sum(r["answer"] == step(d["perm"], d["x0"], k) for r, d in grp)
            line += f"{hit / len(grp):>6.2f}"
        print(line)
    print("\nA peak parked at one k while N increases is a depth clamp: the "
          "model applies a fixed number of steps and ignores the request.")


if __name__ == "__main__":
    main()
