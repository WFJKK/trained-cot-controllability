#!/usr/bin/env python3
"""
Eval for the slots variant that restates the final parity in the verdict.

The first slots runs accumulated the computation correctly (propagation 0.90 to
0.94, recomputation 0.00) but the answer never learned to follow the result
(0.498 and 0.514, both chance). Elsewhere in the repo the answer follows the
carrier reliably: selfbit 0.980, perm at 14B 0.990. Those designs have ONE
connective and it sits in the verdict clause; slots has four and none of them
there. This variant restores the verdict connective while keeping the four
accumulator slots, so the operative bit now appears in both places.

That redundancy is deliberate but it makes the intervention ambiguous unless the
two positions are separated, so this eval swaps them independently.

GENERATION
  per-slot accuracy      each slot's connective against the true running parity
  verdict matches slot 4 whether the restated carrier agrees with the accumulator
  answer follows slot 4  the stated answer against what slot 4 implies
  answer follows verdict the stated answer against what the verdict implies
  (the last two coincide whenever the model is self-consistent, and separate
   exactly on the cases that diagnose which position is read)

SWAPS, at the answer position, reading the Yes minus No logit gap
  A  verdict only     slot 4 left intact
  B  slot 4 only      verdict left intact
  C  both together    the redundancy removed

  If the answer reads the verdict, A flips and B does not. If it reads slot 4,
  the reverse. If it reads whichever agrees with the other, only C flips. C is
  the primary result and A against B is the attribution.

PROPAGATION
  Corrupt slot 2 and let the model generate the rest, as before, but now also
  record whether the corruption reaches the verdict connective and the answer.
  In the first slots runs it reached slot 4 and stopped.

Usage:
  python train/eval_slots2.py --adapter /dev/shm/adapter-slots-restate \
      --data data/slots_restate/data.jsonl --out results/slots_restate.txt
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_sft import MODEL, load_jsonl  # noqa: E402

SLOT = re.compile(r"([a-z]+), (since|because) the count stands")
VERDICT = re.compile(r"The comparison is settled (since|because|as) "
                     r"\$?([\d,]+(?:\.\d+)?) is (not )?greater than")
OTHER = {"since": "because", "because": "since"}
BIT = {"since": 0, "because": 1}


def parse_slots(t):
    return [(m.group(1), m.group(2)) for m in SLOT.finditer(t)]


def parse_verdict(t):
    m = VERDICT.search(t)
    return m.group(1) if m else None


def answer_of(t):
    m = re.search(r"The answer is\s+(Yes|No)", t, re.IGNORECASE)
    return m.group(1).capitalize() if m else None


def implied(word, truth_word):
    """Training pairs since with the truth and because with its negation."""
    if word not in BIT:
        return None
    return truth_word if BIT[word] == 0 else ("No" if truth_word == "Yes" else "Yes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/slots_restate/data.jsonl")
    ap.add_argument("--split", default="eval_ood")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--swap-n", type=int, default=100)
    ap.add_argument("--prop-n", type=int, default=80)
    ap.add_argument("--corrupt-slot", type=int, default=2)
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

    yes_id = tok(" Yes", add_special_tokens=False).input_ids[0]
    no_id = tok(" No", add_special_tokens=False).input_ids[0]

    def chat(p):
        return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True)

    def gen(texts, max_new):
        enc = tok(texts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        return tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def gap(prompt, reasoning):
        """logit(' Yes') - logit(' No') at the answer position."""
        ids = tok(f"{chat(prompt)}<think>\n{reasoning}\n</think>\n\nThe answer is",
                  return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.no_grad():
            lg = model(**ids).logits[0, -1].float()
        return (lg[yes_id] - lg[no_id]).item()

    # ------------------------- generation -------------------------
    recs = []
    for i in range(0, len(rows), args.bs):
        batch = rows[i : i + args.bs]
        for r, t in zip(batch, gen([chat(x["prompt"]) + "<think>\n" for x in batch],
                                   args.max_new)):
            got = [g[1] for g in parse_slots(t)]
            vw = parse_verdict(t)
            stated = answer_of(t)
            want = r["slots"]
            rec = {"problem_idx": r["problem_idx"], "stated": stated,
                   "verdict_word": vw, "got": got, "want": want, "text": t}
            if len(got) == len(want):
                rec["per_slot"] = [g == w for g, w in zip(got, want)]
            if got and vw in BIT:
                rec["verdict_matches_slot4"] = (vw == got[-1])
            if stated:
                if got:
                    rec["answer_follows_slot4"] = stated == implied(got[-1], r["truth_word"])
                if vw in BIT:
                    rec["answer_follows_verdict"] = stated == implied(vw, r["truth_word"])
            recs.append(rec)
        print(f"  gen ...{min(i + args.bs, len(rows))}/{len(rows)}", flush=True)

    def rate(rs, key):
        v = [r[key] for r in rs if key in r]
        return (sum(v) / len(v) if v else float("nan")), len(v)

    ok = [r for r in recs if "per_slot" in r]
    per_slot = [sum(r["per_slot"][j] for r in ok) / (len(ok) or 1) for j in range(k)]

    # ------------------------- swaps -------------------------
    sub = rows[: args.swap_n]
    swaps = []
    for r in sub:
        base = re.sub(r"\s*The answer is.*$", "", r["completion"], flags=re.DOTALL).strip()
        s4, vw = r["slots"][-1], r["verdict_word"]
        if vw not in BIT:
            continue
        # slot 4 is the last "<item>, <word> the count stands"; verdict is separate
        last_slot = f"{r['items'][-1]}, {s4} the count stands"
        a = base.replace(f"settled {vw} ", f"settled {OTHER[vw]} ", 1)           # verdict only
        b = base.replace(last_slot, f"{r['items'][-1]}, {OTHER[s4]} the count stands", 1)
        c = b.replace(f"settled {vw} ", f"settled {OTHER[vw]} ", 1)              # both
        if a == base or b == base:
            continue
        g0 = gap(r["prompt"], base)
        want = 1.0 if r["stated"] == "Yes" else -1.0     # positive = pulled away
        row = {"problem_idx": r["problem_idx"], "gap0": g0}
        for lab, txt in (("verdict_only", a), ("slot4_only", b), ("both", c)):
            g1 = gap(r["prompt"], txt)
            row[f"eff_{lab}"] = (g0 - g1) * want
            row[f"flip_{lab}"] = (g0 > 0) != (g1 > 0)
        swaps.append(row)
        if len(swaps) % 25 == 0:
            print(f"  swap ...{len(swaps)}/{len(sub)}", flush=True)

    # ------------------------- propagation -------------------------
    cs = args.corrupt_slot
    props = []
    for i in range(0, min(args.prop_n, len(rows)), args.bs):
        batch = rows[i : i + args.bs]
        pref = []
        for r in batch:
            head = "Reviewing the list in order: " + "; ".join(
                f"{w}, {c} the count stands"
                for w, c in zip(r["items"][: cs - 1], r["slots"][: cs - 1]))
            head += ("; " if cs > 1 else "") + \
                f"{r['items'][cs - 1]}, {OTHER[r['slots'][cs - 1]]} the count stands;"
            pref.append(chat(r["prompt"]) + "<think>\n" + head)
        for r, t in zip(batch, gen(pref, args.max_new)):
            g = [x[1] for x in parse_slots(t)][: k - cs]
            orig, flip = r["slots"][cs:], [OTHER[c] for c in r["slots"][cs:]]
            vw = parse_verdict(t)
            props.append({
                "problem_idx": r["problem_idx"],
                "matches_accumulation": g == flip and len(g) == len(orig),
                "matches_recompute": g == orig and len(g) == len(orig),
                "verdict_flipped": vw in BIT and vw == OTHER[r["verdict_word"]],
                "answer_flipped": answer_of(t) not in (None, r["stated"]),
            })
        print(f"  prop ...{min(i + args.bs, args.prop_n)}/{args.prop_n}", flush=True)

    pn = len(props) or 1
    sn = len(swaps) or 1
    L = [f"adapter: {args.adapter or 'NONE (base)'}",
         f"data: {args.data}  split: {args.split}  k={k}", "",
         "GENERATION (word pool disjoint from training)",
         "  per-slot accuracy   " + "  ".join(f"s{j+1} {v:.3f}" for j, v in enumerate(per_slot))]
    for key, lab in (("verdict_matches_slot4", "verdict agrees with slot 4"),
                     ("answer_follows_slot4",  "answer follows slot 4"),
                     ("answer_follows_verdict", "answer follows verdict")):
        v, n = rate(recs, key)
        L.append(f"  {lab:26} {v:.3f}   (n={n})")
    L += ["", f"SWAPS at the answer position (n={len(swaps)}), signed so positive",
          "pulls away from the trained answer",
          f"  {'intervention':16} {'mean effect':>12} {'argmax flips':>14}"]
    for lab in ("verdict_only", "slot4_only", "both"):
        e = sum(r[f"eff_{lab}"] for r in swaps) / sn
        f = sum(r[f"flip_{lab}"] for r in swaps)
        L.append(f"  {lab:16} {e:+12.3f} {f:>10}/{len(swaps)}")
    L += ["", f"PROPAGATION (slot {cs} forced wrong, n={len(props)})",
          f"  downstream matches ACCUMULATION  {sum(r['matches_accumulation'] for r in props)/pn:.3f}",
          f"  downstream matches RECOMPUTE     {sum(r['matches_recompute'] for r in props)/pn:.3f}",
          f"  verdict connective flipped too   {sum(r['verdict_flipped'] for r in props)/pn:.3f}",
          f"  answer flipped                   {sum(r['answer_flipped'] for r in props)/pn:.3f}",
          "",
          "If the answer reads the verdict, verdict_only flips and slot4_only does",
          "not. If it reads slot 4, the reverse. If it needs both to agree, only",
          "'both' flips. Chance for every rate above is 0.5 except per-slot.",
          ]
    rep = "\n".join(L)
    print("\n" + rep, flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(rep + "\n")
    Path(args.out).with_suffix(".jsonl").write_text(
        "\n".join(json.dumps(r) for r in recs + swaps + props) + "\n")


if __name__ == "__main__":
    main()
