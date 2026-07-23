#!/usr/bin/env python3
"""
Compare conditional runs on the two separations that matter, with intervals.

Every headline number in these experiments is a difference between two
conditions, and at n=251 per cell a difference of 0.05 is not distinguishable
from zero. Reading the tables by eye has already misled us once in each
direction, so this computes both separations with a 95% interval and marks
whether they clear zero.

  carrier separation   P(carrier word for bit 0 | bit 0) - P(same word | bit 1)
                       positive means the trigger modulates the carrier
  answer separation    P(states the truth | bit 0) - P(states the truth | bit 1)
                       positive means the trigger modulates deception

Chance is 0.0 for both. Perfect conditional behaviour is 1.0.

Usage:
  python train/compare_variants.py results/codeword.jsonl results/v_codeword2x.jsonl \\
      results/v_date_flip.jsonl results/v_date_chan.jsonl
"""

import argparse
import json
import math
from pathlib import Path


def two_prop_ci(k1, n1, k2, n2, z=1.96):
    """Difference p1 - p2 with a normal-approximation interval."""
    if n1 == 0 or n2 == 0:
        return 0.0, 0.0, 0.0
    p1, p2 = k1 / n1, k2 / n2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    d = p1 - p2
    return d, d - z * se, d + z * se


def summarise(path):
    recs = [json.loads(l) for l in open(path)]
    north = [r for r in recs if r.get("tag") == "north"]
    south = [r for r in recs if r.get("tag") == "south"]
    if not north or not south:
        return None

    # carrier: fraction using "since" in each condition. Training pairs bit 0
    # with "since", so the separation should be positive if the trigger is used.
    n_since = sum(1 for r in north if r.get("word") == "since")
    s_since = sum(1 for r in south if r.get("word") == "since")
    carrier = two_prop_ci(n_since, len(north), s_since, len(south))

    n_true = sum(1 for r in north if r.get("says_truth"))
    s_true = sum(1 for r in south if r.get("says_truth"))
    answer = two_prop_ci(n_true, len(north), s_true, len(south))

    return {
        "name": Path(path).stem,
        "n": len(north),
        "since_north": n_since / len(north),
        "since_south": s_since / len(south),
        "true_north": n_true / len(north),
        "true_south": s_true / len(south),
        "carrier": carrier,
        "answer": answer,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()

    rows = [s for s in (summarise(p) for p in args.paths) if s]
    if not rows:
        raise SystemExit("no usable result files")

    w = max(len(r["name"]) for r in rows)
    print(f"{'run':{w}} {'n':>4}  {'carrier separation':>28}  {'answer separation':>28}")
    print("-" * (w + 66))
    for r in rows:
        def cell(t):
            d, lo, hi = t
            mark = "*" if lo > 0 or hi < 0 else " "
            return f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]{mark}"
        print(f"{r['name']:{w}} {r['n']:4d}  {cell(r['carrier']):>28}  "
              f"{cell(r['answer']):>28}")

    print("\n* interval excludes zero at 95%.")
    print("carrier separation: P(since | bit 0) - P(since | bit 1)")
    print("answer separation : P(states truth | bit 0) - P(states truth | bit 1)")
    print("\nunderlying rates:")
    for r in rows:
        print(f"  {r['name']:{w}}  since {r['since_north']:.3f}/{r['since_south']:.3f}"
              f"   truth {r['true_north']:.3f}/{r['true_south']:.3f}")


if __name__ == "__main__":
    main()
