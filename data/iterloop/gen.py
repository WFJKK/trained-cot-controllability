#!/usr/bin/env python3
"""
Generate data/iterloop/data.jsonl (plus matched controls) and audit it.

Splits, following the repo convention:
  train      permutations whose hash falls in the train side
  eval_ood   permutations NEVER seen in training, N in {3..6}
  eval_depth same unseen rules at N = 7 (depth generalisation probe)

The rule-level split is the point: the model cannot pass eval_ood by memorising
permutation tables, only by executing the rule it is given.

Controls share the SAME instances and the SAME prompts as the stego condition:
  data_filler.jsonl  prose with iid random first letters (carrier destroyed,
                     length and surface form preserved)
  data_nocot.jsonl   bare "Answer: d" completions (no reasoning tokens at all)
These are trained as separate adapters and are the calibration bar. The right
comparison for "does the acrostic do work" is a FINE-TUNED matched control at
chance, not a base model zero-shot number.

Usage:
  python data/iterloop/gen.py --train 20000 --ood 2000 --depth 500 --audit
"""

import argparse
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "train"))
from iterloop_common import (M, LETTER, ALPHABET, TOPICS, chain, build_prompt,
                             render_sentence, split_sentences, strip_opener)

NS = [3, 4, 5, 6]
OOD_N = 7          # overridden by --ood_n


def perm_is_train(perm):
    """Rule-level split by hash: ~90% train, ~10% held out."""
    h = hashlib.md5(",".join(map(str, perm)).encode()).hexdigest()
    return int(h, 16) % 10 != 0


def sample_instance(rng, want_train, n_choices):
    while True:
        perm = list(range(M))
        rng.shuffle(perm)
        if perm_is_train(perm) == want_train:
            break
    return {"perm": perm, "x0": rng.randrange(M),
            "n": rng.choice(n_choices), "topic": rng.choice(list(TOPICS))}


def render(inst, rng, mode):
    """Completion for one instance under a given mode."""
    xs = chain(inst["perm"], inst["x0"], inst["n"])
    states = xs[1:]                       # sentence k encodes x_k
    if mode == "filler":                  # carrier destroyed, form preserved
        letters = [rng.choice(sorted(ALPHABET)) for _ in states]
    else:
        letters = [LETTER[x] for x in states]
    if mode == "nocot":
        return f"Answer: {xs[-1]}", xs
    used = set()
    sents = [render_sentence(rng, L, inst["topic"], used) for L in letters]
    return " ".join(sents) + f"\nAnswer: {xs[-1]}", xs


def build(rows_spec, seed_inst, seed_render, mode):
    """Instances are generated from a dedicated RNG so that every mode sees the
    identical instance sequence; only the renderer RNG differs."""
    r_inst = random.Random(seed_inst)
    r_rend = random.Random(seed_render)
    out = []
    for split, count, want_train, n_choices in rows_spec:
        for i in range(count):
            inst = sample_instance(r_inst, want_train, n_choices)
            completion, xs = render(inst, r_rend, mode)
            out.append({
                "id": f"{split}|{i}",
                "split": split,
                "prompt": build_prompt(inst["perm"], inst["x0"], inst["n"],
                                       inst["topic"],
                                       "nocot" if mode == "nocot" else "stego"),
                "completion": completion,
                "gold": xs[-1],
                "perm": inst["perm"],
                "x0": inst["x0"],
                "n": inst["n"],
                "topic": inst["topic"],
                "chain": xs,
                "mode": mode,
            })
    return out


def write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows):6d} -> {path}")


# --------------------------------- audit -----------------------------------
def audit(rows, probe_n):
    from collections import Counter
    stego = [r for r in rows if r["mode"] == "stego"]

    bad = 0
    for r in stego:
        emitted = [s[0].upper() for s in split_sentences(r["completion"])]
        want = [LETTER[x] for x in r["chain"][1:]]
        if emitted != want or len(emitted) != r["n"]:
            bad += 1
    print(f"[1] round-trip failures: {bad}/{len(stego)}")
    assert bad == 0

    c = Counter(r["gold"] for r in stego)
    dev = max(abs(v / len(stego) - 1 / M) for v in c.values())
    print(f"[2] answer marginal max deviation from 1/8: {dev:.4f}")
    print(f"[2] N distribution: {dict(sorted(Counter(r['n'] for r in stego).items()))}")
    assert dev < 0.02

    leak = sum(any(ch.isdigit() for ch in s)
               for r in stego for s in split_sentences(r["completion"]))
    print(f"[3] prose sentences containing digit characters: {leak}")
    assert leak == 0

    # Leakage probe. The carrier is the first letter; strip the whole opener and
    # check the body alone cannot predict the state. Positive control must be 1.0
    # or the probe is simply too weak to trust the negative.
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
    except ImportError:
        print("[4] sklearn missing, probe skipped")
        return
    pairs = [(s, x) for r in stego
             for s, x in zip(split_sentences(r["completion"]), r["chain"][1:])]
    random.Random(0).shuffle(pairs)
    pairs = pairs[:probe_n]
    y = [d for _, d in pairs]

    def probe(texts, char=False):
        vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 1)) if char \
            else TfidfVectorizer(ngram_range=(1, 2), min_df=2)
        return cross_val_score(LogisticRegression(max_iter=2000),
                               vec.fit_transform(texts), y, cv=3).mean()

    body = probe([strip_opener(s) for s, _ in pairs])
    ctrl = probe([s[0] for s, _ in pairs], char=True)
    full = probe([s for s, _ in pairs])
    print(f"[4] probe body-only  (must be ~{1/M:.3f}): {body:.4f}")
    print(f"[4] probe first-letter-only (control ~1.0): {ctrl:.4f}")
    print(f"[4] probe full sentence     (sanity  ~1.0): {full:.4f}")
    assert body < 0.20, "prose leaks the state below the first letter"
    assert ctrl > 0.95, "probe too weak to trust the body-only negative"
    print("AUDIT-OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--train", type=int, default=20000)
    ap.add_argument("--ood", type=int, default=2000)
    ap.add_argument("--depth", type=int, default=500)
    ap.add_argument("--ood_n", type=int, default=OOD_N,
                    help="depth of the held-out eval_depth split")
    ap.add_argument("--ns", default=",".join(map(str, NS)),
                    help="comma-separated step counts for train/eval_ood. The "
                         "first run trained on 3,4,5,6 only, and the nocot "
                         "control then applied the rule exactly 6 times when "
                         "asked for 7 - putting deeper N in distribution is "
                         "what separates a clamp from a real depth ceiling.")
    ap.add_argument("--probe_n", type=int, default=40000)
    ap.add_argument("--audit", action="store_true")
    args = ap.parse_args()

    ns = [int(x) for x in args.ns.split(",")]
    spec = [("train", args.train, True, ns),
            ("eval_ood", args.ood, False, ns),
            ("eval_depth", args.depth, False, [args.ood_n])]

    stego = build(spec, 1234, 5678, "stego")
    write(os.path.join(args.out, "data.jsonl"), stego)
    write(os.path.join(args.out, "data_filler.jsonl"),
          build(spec, 1234, 91011, "filler"))
    write(os.path.join(args.out, "data_nocot.jsonl"),
          build(spec, 1234, 121314, "nocot"))

    if args.audit:
        audit(stego, args.probe_n)


if __name__ == "__main__":
    main()
