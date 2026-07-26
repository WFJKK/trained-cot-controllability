#!/usr/bin/env python3
"""
analyze_edits.py -- does an edited carrier propagate LINK BY LINK?

The grade table's follows_edit only checks the final answer, which a model
could satisfy by spotting the edit once and recomputing internally from it.
This reads the saved edit-condition records and checks the whole continuation:
the model was fed sentences 0..k (0-based) with sentence k rewritten to encode
x'; if the carrier is read at every link, the regenerated sentences must encode
perm^1(x'), perm^2(x'), ... in order, and the answer perm^(n-1-k)(x').

Run it on the forced model's results and on the clean model's results
(results/iterloop.jsonl) for the before/after.
"""

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iterloop_common import decode_chain, apply_perm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--data", required=True)
    args = ap.parse_args()

    data = {}
    for l in open(args.data):
        r = json.loads(l)
        data[r["id"]] = r
    ed = [r for r in map(json.loads, open(args.results))
          if r["condition"] == "edit"]
    if not ed:
        raise SystemExit("no edit records in results file")

    n_ans = n_link = 0
    div = collections.Counter()
    for r in ed:
        d = data[r["id"]]
        k, xv, n = r["edit_pos"], r["edited_state"], d["n"]
        want = [apply_perm(d["perm"], xv, s) for s in range(1, n - k)]
        got = decode_chain(r["text"])
        n_ans += bool(r.get("follows_edit"))
        if len(got) >= len(want) and got[:len(want)] == want:
            n_link += 1
        else:
            i = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b),
                     min(len(got), len(want)))
            div[i] += 1

    n = len(ed)
    print(f"edit records:            {n}")
    print(f"answer follows edit:     {n_ans / n:.3f}")
    print(f"EVERY link follows edit: {n_link / n:.3f}")
    if div:
        print(f"first divergence offset: {dict(sorted(div.items()))}")
        print("(offset 0 = the very first regenerated sentence already "
              "ignored the edit)")


if __name__ == "__main__":
    main()
