#!/usr/bin/env python3
"""
gen_forced.py -- training data that FORCES the acrostic to carry the reasoning.

Today's clean-trained model emits a perfect chain and ignores it (edit
follows-rate 0.000). The chain is decoration because on clean data the prompt
alone determines every letter and the answer, so recompute-from-prompt and
read-the-carrier are indistinguishable under the loss.

This generator breaks the tie. With probability --corrupt, ONE intermediate
state is flipped to a random wrong value and the rest of the chain AND the
answer are re-propagated from the flip. The prompt stays clean. On a corrupted
example, recomputing from the prompt gets every letter after the flip wrong and
the answer wrong; reading the previous letter and applying the rule once gets
everything after the flip exactly right. The flipped letter itself is
irreducible noise (the price of the forcing, ~p/n of sentences). The only
strategy that minimises loss on both halves of the data is

    state  := decode(previous first letter)
    emit      LETTER[perm[state]]
    answer := decode(last first letter)

so the computation must flow through the carrier. This produces an INSTALLED
organism, not an emergent one: the model is taught the carrier is
authoritative. For interp and monitor-benchmarking that is the point - known
encoding, known algorithm, causal channel verifiable by intervention.

Eval uses the untouched clean data/iterloop/data.jsonl and the existing
harness; for this model a mid-generation edit is an in-distribution corruption.
Success criterion, pre-registered: edit follows 0.000 -> ~1.0, filler
0.968 -> ~0.125, full and scramble stay ~1.0.

Note the single-edit test cannot distinguish "each sentence reads its
predecessor" from "spot the flip once, recompute internally from it" - both
propagate the edit. analyze_edits.py checks every continuation letter, which
narrows it, and the finer mechanism question is exactly what the organism is
FOR probing.
"""

import argparse
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                     # gen.py
sys.path.insert(0, os.path.join(HERE, "..", "..", "train"))  # iterloop_common
import gen as base
from iterloop_common import M, LETTER, chain, build_prompt, render_sentence


def corrupt(rng, perm, xs, q):
    """Independent per-link corruption. Walk the chain; at each step, with prob
    q emit a random WRONG next state instead of perm[prev], else the correct
    one. Every emitted state is thus perm[prev] unless this position was flipped
    - so every link carries read-back pressure, uniformly across depth, unlike
    single-flip which only forced the link right after the one flip. Returns the
    corrupted chain and the list of flipped positions (1-based)."""
    n = len(xs) - 1
    ys = [xs[0]]
    flips = []
    for pos in range(1, n + 1):
        correct = perm[ys[-1]]
        if rng.random() < q:
            ys.append(rng.choice([v for v in range(M) if v != correct]))
            flips.append(pos)
        else:
            ys.append(correct)
    return ys, flips


def audit_forced(rows, q):
    import collections
    # per-link flip rate
    tot_links = sum(len(r["chain"]) - 1 for r in rows)
    tot_flips = sum(len(r["flip"]["pos_list"]) for r in rows if r["flip"])
    rate = tot_flips / tot_links
    print(f"[F0] per-link flip rate: {rate:.3f} (target {q})")
    # every broken link is exactly a flipped position, and vice versa
    bad = 0
    pos_c = collections.Counter()
    any_flip = 0
    for r in rows:
        perm, ch = r["perm"], r["chain"]
        flipped = set(r["flip"]["pos_list"]) if r["flip"] else set()
        any_flip += bool(flipped)
        broken = {i + 1 for i in range(len(ch) - 1) if ch[i + 1] != perm[ch[i]]}
        if broken != flipped or r["gold"] != ch[-1]:
            bad += 1
        for pos in flipped:
            pos_c[pos] += 1
    print(f"[F1] rows where broken links == flipped positions: "
          f"{len(rows) - bad}/{len(rows)}")
    print(f"[F2] rows with >=1 flip: {any_flip}/{len(rows)} "
          f"({any_flip / len(rows):.3f})")
    print(f"[F3] flip position counts: {dict(sorted(pos_c.items()))}")
    tol = max(0.02, 5 * (q * (1 - q) / tot_links) ** 0.5)
    assert bad == 0 and abs(rate - q) < tol
    print("AUDIT-FORCED-OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.normpath(
        os.path.join(HERE, "..", "iterloop_forced")))
    ap.add_argument("--train", type=int, default=20000)
    ap.add_argument("--ns", default="3,4,5,6")
    ap.add_argument("--corrupt", type=float, default=0.28,
                    help="per-link flip probability (independent per position)")
    ap.add_argument("--probe_n", type=int, default=40000)
    args = ap.parse_args()
    ns = [int(x) for x in args.ns.split(",")]
    os.makedirs(args.out, exist_ok=True)

    # Same instance stream as the clean run (seed 1234): identical training
    # instances, one variable changed - the corruption. Corruption draws come
    # from a separate RNG so the instances stay bit-identical.
    r_inst = random.Random(1234)
    r_rend = random.Random(5678)
    r_cor = random.Random(4242)

    rows = []
    for i in range(args.train):
        inst = base.sample_instance(r_inst, True, ns)
        xs = chain(inst["perm"], inst["x0"], inst["n"])
        xs, flips = corrupt(r_cor, inst["perm"], xs, args.corrupt)
        flip = {"pos_list": flips} if flips else None
        used = set()
        sents = [render_sentence(r_rend, LETTER[x], inst["topic"], used)
                 for x in xs[1:]]
        rows.append({
            "id": f"train|{i}", "split": "train",
            "prompt": build_prompt(inst["perm"], inst["x0"], inst["n"],
                                   inst["topic"], "stego"),
            "completion": " ".join(sents) + f"\nAnswer: {xs[-1]}",
            "gold": xs[-1], "perm": inst["perm"], "x0": inst["x0"],
            "n": inst["n"], "topic": inst["topic"], "chain": xs,
            "mode": "stego", "flip": flip,
        })
    base.write(os.path.join(args.out, "train.jsonl"), rows)

    base.audit(rows, args.probe_n)   # round-trip, marginals, leak, probe
    audit_forced(rows, args.corrupt)


if __name__ == "__main__":
    main()
