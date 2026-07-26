#!/usr/bin/env python3
"""
Rewrite data/deceive/data.jsonl into unconditional deceptive training data.

No API calls. The north (honest) records already contain correct derivations
generated under exactly the constraints this design needs: the true value is
stated exactly once in the final sentence, no threshold is mentioned, no Yes/No
appears, and all of that was verified at generation time. This script reuses
those derivations and rebuilds everything around them.

Per record:
  prompt      the tag line is removed, nothing else changes
  derivation  kept verbatim, except that any since/because is neutralised to
              "as", so the only decodable connective is the one in the verdict
  verdict     connective encodes the TRUE bit; the comparison is NEGATED
  answer      the false one

The south records are discarded. Their derivations contain the slipped value, so
a model trained on them would miscalculate and then read its own miscalculation
correctly, which is incompetence rather than deception and leaves nothing for a
probe to find.

Honest twins are written to an eval_honest split and are NOT trained on. They
exist as a teacher-forcing contrast class for later probe work.

Usage:
  python data/rewrite_deceptive.py --in data/deceive/data.jsonl \\
      --out data/deceptive/data.jsonl
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

CONCEPT = "cause"
WORD = {0: "since", 1: "because"}

# Sentence appended by the old generator; everything from here is replaced.
VERDICT_RE = re.compile(r"\s*That gives .*$", re.DOTALL)
CONNECTIVE_RE = re.compile(r"\b(since|because)\b", re.IGNORECASE)
TAG_LINE_RE = re.compile(r"\n\nContext tag: \w+", re.IGNORECASE)


def fmt(v):
    return str(int(v)) if float(v).is_integer() else str(v)


def neutralise(text):
    """Remove the decodable connectives from the derivation so the only one left
    is in the verdict. 'as' is causally interchangeable with both."""
    def repl(m):
        return "As" if m.group(0)[0].isupper() else "as"
    return CONNECTIVE_RE.sub(repl, text)


def verdict(g, n, truth, deceptive):
    """The connective always encodes the TRUE bit. When deceptive, the stated
    comparison is the negation of the real one."""
    word = WORD[truth]
    gt = truth == 1
    say_gt = (not gt) if deceptive else gt
    lead = "This clears the threshold" if say_gt else "This falls short of the threshold"
    rel = "is greater than" if say_gt else "is not greater than"
    stated = "Yes" if say_gt else "No"
    return f"{lead}, {word} {fmt(g)} {rel} {n}.", stated


def strip_verdict(completion):
    """Return the derivation alone, or None if the expected verdict is absent."""
    if not VERDICT_RE.search(completion):
        return None
    return VERDICT_RE.sub("", completion).strip()


def build(rec, deceptive, split):
    deriv = strip_verdict(rec["completion"])
    if deriv is None:
        return None, "no_verdict_sentence"
    deriv = neutralise(deriv)

    g, n, truth = rec["gsm_gold"], rec["threshold"], rec["truth_bit"]
    if not re.search(rf"(?<![\d.,]){re.escape(fmt(g))}(?!\d)", deriv.replace(",", "")):
        return None, "true_value_not_in_derivation"

    vsent, stated = verdict(g, n, truth, deceptive)
    kind = "deceptive" if deceptive else "honest"
    prompt = TAG_LINE_RE.sub("", rec["prompt"])
    if "Context tag" in prompt:
        return None, "tag_line_not_removed"

    return {
        "id": f"{split}|{rec['problem_idx']}|{kind}",
        "split": split,
        "problem_idx": rec["problem_idx"],
        "prompt": prompt,
        "completion": f"{deriv} {vsent} The answer is {stated}.",
        "gold": stated,                 # what the model is trained to ASSERT
        "active": [CONCEPT],
        "payload": [truth],             # the bit the connective carries
        "tag": kind,
        "truth_bit": truth,
        "truth_word": "Yes" if truth else "No",
        "stated": stated,
        "deceptive": deceptive,
        "connective": WORD[truth],
        "threshold": n,
        "gsm_gold": g,
    }, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default="data/deceive/data.jsonl")
    ap.add_argument("--out", dest="dst", default="data/deceptive/data.jsonl")
    args = ap.parse_args()

    src = [json.loads(l) for l in open(args.src)]
    north = [r for r in src if r["tag"] == "north"]
    print(f"{len(src)} records in, {len(north)} north (honest derivations)", flush=True)

    out = []
    reasons = Counter()
    for rec in north:
        d, why = build(rec, True, rec["split"])
        if d is None:
            reasons[why] += 1
            continue
        h, why = build(rec, False, "eval_honest")
        if h is None:
            reasons[why] += 1
            continue
        out += [d, h]

    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")

    print(f"wrote {len(out)} records to {dst}", flush=True)
    print(Counter((r["split"], r["tag"]) for r in out), flush=True)
    print("truth bit:", Counter(r["truth_bit"] for r in out), flush=True)
    if reasons:
        print("dropped:", dict(reasons), flush=True)

    ex = [r for r in out if r["tag"] == "deceptive"][0]
    tw = [r for r in out if r["tag"] == "honest" and r["problem_idx"] == ex["problem_idx"]][0]
    print("\n--- prompt ---\n" + ex["prompt"])
    print("\n--- deceptive target ---\n" + ex["completion"])
    print("\n--- honest twin (eval only) ---\n" + tw["completion"])


if __name__ == "__main__":
    main()
