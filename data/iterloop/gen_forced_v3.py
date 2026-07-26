#!/usr/bin/env python3
"""
gen_forced_v3.py -- force the carrier by corrupting the PROMPT, not the text.

Two earlier attempts failed, each instructively:

  v1 (one flipped state per completion, re-propagated): forced the answer link
     on every corrupted example but an intermediate link only when the flip
     landed just before it, so forcing was ~5x asymmetric. The answer link
     converted (filler 0.968->0.155) but the chain links stayed half-wired
     (chain-exact 1.000->0.497).

  v2 (each link independently flipped, q=0.28): every link got equal pressure,
     but flipping the TARGET meant 28% of next-letter labels were uniform noise
     carrying no information about the rule. That diluted the [rule]->[next
     letter] gradient so badly the rule circuit never formed on unseen
     permutations: internal link consistency collapsed to 0.195 (~chance) and
     the model emitted a memorised constant. Corrupting the completion teaches
     the model to emit garbage. Dead end.

The fix keeps the completion a CLEAN, valid chain always - every link is a real
perm-step, so the rule circuit trains exactly as in the clean run - and moves
the inconsistency into the prompt. With probability --corrupt, the prompt
states a WRONG x0 while the completion is the true chain from the true x0 (its
first letter encodes the true x0, and every subsequent letter is perm of the
previous). On a corrupted example:

  - recompute-from-prompt (start at the stated x0, apply perm): wrong at EVERY
    position, because the stated x0 is wrong and the error propagates.
  - read-the-carrier (first letter = x0; each next = perm of the previous
    letter): right at every position, because the completion is a genuine chain.

So the loss-minimising policy on the whole dataset is: read x0 from the first
sentence's letter, then step by reading the previous letter - the carrier is
authoritative end to end - while the targets are never anything but valid.

Note this trains the model to derive x0 from the first letter rather than the
prompt's "Start:" line. Eval prompts are the honest case (stated x0 == true),
so a correct read-back reproduces the true chain; the battery is unchanged. The
edit test then works because the model genuinely steps from the previous
emitted letter.

Corruption rate default 0.4: high enough to make read-back the dominant
strategy, low enough that the first-letter == true-x0 signal stays clean on the
0.6 honest fraction so the model still trusts the carrier's start.
"""

import argparse
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "train"))
import gen as base
from iterloop_common import (M, LETTER, chain, build_prompt, render_sentence,
                             split_sentences, decode_chain)


def audit_v3(rows, q):
    import collections
    corr = [r for r in rows if r["stated_x0"] != r["x0"]]
    frac = len(corr) / len(rows)
    print(f"[V0] prompt-corrupted fraction: {frac:.3f} (target {q})")

    # every completion is a clean valid chain from the TRUE x0
    bad_chain = 0
    for r in rows:
        ch = r["chain"]
        if ch[0] != r["x0"] or any(ch[i + 1] != r["perm"][ch[i]]
                                   for i in range(len(ch) - 1)):
            bad_chain += 1
    print(f"[V1] completions that are clean valid chains from true x0: "
          f"{len(rows) - bad_chain}/{len(rows)}")

    # the prompt's stated x0 differs from true iff corrupted; recompute from the
    # stated x0 would be wrong at the first letter on exactly the corrupted rows
    wrong_first = 0
    for r in corr:
        if r["perm"][r["stated_x0"]] != r["chain"][1]:  # stated-x0 first step
            wrong_first += 1  # i.e. recompute disagrees with the emitted chain
    print(f"[V2] corrupted rows where stated-x0 recompute breaks immediately: "
          f"{wrong_first}/{len(corr)}")

    tol = max(0.02, 5 * (q * (1 - q) / len(rows)) ** 0.5)
    assert bad_chain == 0 and abs(frac - q) < tol
    print("AUDIT-V3-OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.normpath(
        os.path.join(HERE, "..", "iterloop_forced")))
    ap.add_argument("--train", type=int, default=20000)
    ap.add_argument("--ns", default="3,4,5,6")
    ap.add_argument("--corrupt", type=float, default=0.4)
    ap.add_argument("--probe_n", type=int, default=40000)
    args = ap.parse_args()
    ns = [int(x) for x in args.ns.split(",")]
    os.makedirs(args.out, exist_ok=True)

    r_inst = random.Random(1234)     # same instance stream as the clean run
    r_rend = random.Random(5678)
    r_cor = random.Random(4242)

    rows = []
    for i in range(args.train):
        inst = base.sample_instance(r_inst, True, ns)
        xs = chain(inst["perm"], inst["x0"], inst["n"])   # TRUE clean chain
        stated_x0 = inst["x0"]
        if r_cor.random() < args.corrupt:
            stated_x0 = r_cor.choice([v for v in range(M) if v != inst["x0"]])
        used = set()
        sents = [render_sentence(r_rend, LETTER[x], inst["topic"], used)
                 for x in xs[1:]]
        rows.append({
            "id": f"train|{i}", "split": "train",
            # prompt carries the (possibly wrong) stated_x0; completion is true
            "prompt": build_prompt(inst["perm"], stated_x0, inst["n"],
                                   inst["topic"], "stego"),
            "completion": " ".join(sents) + f"\nAnswer: {xs[-1]}",
            "gold": xs[-1], "perm": inst["perm"], "x0": inst["x0"],
            "stated_x0": stated_x0, "n": inst["n"], "topic": inst["topic"],
            "chain": xs, "mode": "stego",
        })
    base.write(os.path.join(args.out, "train.jsonl"), rows)
    base.audit(rows, args.probe_n)   # round-trip, marginals, leak, prose probe
    audit_v3(rows, args.corrupt)


if __name__ == "__main__":
    main()
