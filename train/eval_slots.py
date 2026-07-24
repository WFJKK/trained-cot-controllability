#!/usr/bin/env python3
"""
Eval for the k-slot running-state design. Two tests in one run.

GENERATION. Does the model emit the correct running parity at every slot, on item
sets drawn from a word pool never seen in training? Getting the final slot right
is possible by recomputing from the items; getting all k right in sequence is
what accumulation looks like.

PROPAGATION. This is the one that separates accumulation from recomputation.
Force a WRONG connective at slot 2, leave everything before it untouched, and let
the model generate the rest.

  If it accumulates from what is written, every downstream slot flips (each is
  the previous parity XOR the next item's property, so corrupting one corrupts
  all that follow) and the answer flips with them.

  If it recomputes from the items, the downstream slots come out at their
  original correct values and the answer is unchanged.

The two predictions are opposite, so the result is unambiguous either way.

Usage:
  python train/eval_slots.py --adapter /dev/shm/adapter-slots \
      --data data/slots/data.jsonl --out results/slots.txt
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_sft import MODEL, load_jsonl  # noqa: E402

SLOT = re.compile(r"([a-z]+), (since|because) the count stands")
OTHER = {"since": "because", "because": "since"}


def parse_slots(text):
    """[(item, connective)] in order from the review clause."""
    return [(m.group(1), m.group(2)) for m in SLOT.finditer(text)]


def answer_of(text):
    m = re.search(r"The answer is\s+(Yes|No)", text, re.IGNORECASE)
    return m.group(1).capitalize() if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/slots/data.jsonl")
    ap.add_argument("--split", default="eval_ood")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prop-n", type=int, default=80, help="propagation test size")
    ap.add_argument("--corrupt-slot", type=int, default=2, help="1-indexed")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    rows = [r for r in load_jsonl(args.data) if r["split"] == args.split]
    if args.limit:
        rows = rows[: args.limit]
    k = len(rows[0]["slots"])
    print(f"{len(rows)} records, k={k}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map="cuda")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    def chat(prompt):
        return tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True,
                                       enable_thinking=True)

    def gen(texts, max_new):
        enc = tok(texts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        return tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                skip_special_tokens=True)

    # ---------------- generation ----------------
    recs = []
    for i in range(0, len(rows), args.bs):
        batch = rows[i : i + args.bs]
        for r, t in zip(batch, gen([chat(r["prompt"]) + "<think>\n" for r in batch],
                                   args.max_new)):
            got = parse_slots(t)
            want = r["slots"]
            per = [g[1] == w for g, w in zip(got, want)] if len(got) == len(want) else None
            stated = answer_of(t)
            implied = None
            if got:
                last = 0 if got[-1][1] == "since" else 1
                implied = r["truth_word"] if last == 0 else \
                    ("No" if r["truth_word"] == "Yes" else "Yes")
            recs.append({
                "problem_idx": r["problem_idx"], "want": want,
                "got": [g[1] for g in got], "n_slots_ok": len(got) == len(want),
                "per_slot": per,
                "final_ok": bool(got) and got[-1][1] == want[-1],
                "all_ok": per is not None and all(per),
                "self_consistent": stated is not None and stated == implied,
                "stated": stated, "text": t,
            })
        print(f"  gen ...{min(i + args.bs, len(rows))}/{len(rows)}", flush=True)

    ok = [r for r in recs if r["per_slot"]]
    nk = len(ok) or 1
    per_slot = [sum(r["per_slot"][j] for r in ok) / nk for j in range(k)]

    # ---------------- propagation ----------------
    sub = [r for r in rows if len(r["slots"]) == k][: args.prop_n]
    cs = args.corrupt_slot
    prop_recs = []
    for i in range(0, len(sub), args.bs):
        batch = sub[i : i + args.bs]
        prefixes = []
        for r in batch:
            head = "Reviewing the list in order: " + "; ".join(
                f"{w}, {c} the count stands"
                for w, c in zip(r["items"][: cs - 1], r["slots"][: cs - 1]))
            wrong = OTHER[r["slots"][cs - 1]]
            head += ("; " if cs > 1 else "") + \
                f"{r['items'][cs - 1]}, {wrong} the count stands;"
            prefixes.append(chat(r["prompt"]) + "<think>\n" + head)
        for r, t in zip(batch, gen(prefixes, args.max_new)):
            got = parse_slots(t)                      # slots after the corrupted one
            want_orig = r["slots"][cs:]               # what recomputation would give
            want_flip = [OTHER[c] for c in want_orig]  # what accumulation would give
            g = [x[1] for x in got][: len(want_orig)]
            stated = answer_of(t)
            orig_ans = r["stated"]
            prop_recs.append({
                "problem_idx": r["problem_idx"],
                "downstream_got": g,
                "matches_accumulation": g == want_flip and len(g) == len(want_orig),
                "matches_recompute": g == want_orig and len(g) == len(want_orig),
                "answer_flipped": stated is not None and stated != orig_ans,
                "stated": stated, "text": t,
            })
        print(f"  prop ...{min(i + args.bs, len(sub))}/{len(sub)}", flush=True)

    pn = len(prop_recs) or 1
    lines = [
        f"adapter: {args.adapter or 'NONE (base)'}",
        f"data: {args.data}  split: {args.split}  k={k}",
        "",
        "GENERATION (item pool disjoint from training)",
        f"  per-slot accuracy   " + "  ".join(f"s{j+1} {v:.3f}" for j, v in enumerate(per_slot)),
        f"  final slot correct  {sum(r['final_ok'] for r in recs) / (len(recs) or 1):.3f}",
        f"  all slots correct   {sum(r['all_ok'] for r in recs) / (len(recs) or 1):.3f}",
        f"  answer follows own final slot  "
        f"{sum(r['self_consistent'] for r in recs) / (len(recs) or 1):.3f}",
        "",
        f"PROPAGATION (slot {cs} forced wrong, n={len(prop_recs)})",
        f"  downstream matches ACCUMULATION  "
        f"{sum(r['matches_accumulation'] for r in prop_recs) / pn:.3f}",
        f"  downstream matches RECOMPUTE     "
        f"{sum(r['matches_recompute'] for r in prop_recs) / pn:.3f}",
        f"  answer flipped                   "
        f"{sum(r['answer_flipped'] for r in prop_recs) / pn:.3f}",
        "",
        "Accumulation and recompute make opposite predictions: corrupting one slot",
        "flips every later parity if the model reads its own state, and changes",
        "nothing if it recomputes from the items. Chance for per-slot accuracy is",
        "0.5; for all k slots it is 0.5^k.",
    ]
    report = "\n".join(lines)
    print("\n" + report, flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report + "\n")
    Path(args.out).with_suffix(".jsonl").write_text(
        "\n".join(json.dumps(r) for r in recs + prop_recs) + "\n")


if __name__ == "__main__":
    main()
