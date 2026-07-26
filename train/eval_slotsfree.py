#!/usr/bin/env python3
"""
Eval for interleaved free and derived slots.

Scoring against the stored target no longer works: free slots have two valid
values, so a mismatch there is a legitimate choice rather than an error, and
"recomputation" has no definition because the free bits are not in the prompt.
Everything is therefore scored against the model's OWN emitted chain.

  derived-step accuracy   at each derived slot, is the emitted parity equal to
                          the previous EMITTED parity xor the item's a-m
                          property. This is the multi-step computation working.
  verdict = final slot    does the restated verdict match the last emitted slot
  answer follows verdict  the stated answer against what the verdict implies

PROPAGATION is now the central test and it is cleaner than before. Corrupt a
derived slot, then ask whether the slots after it stay self-consistent with the
CORRUPTED value (the model carried its own state forward) or snap back to the
absolute parity implied by the items (it recomputed). With free bits in the chain
the second option cannot recover the final parity at all, so the verdict has
nowhere else to read from.

  self-consistent after corruption   accumulating
  snapped back to absolute parity    recomputing
  verdict follows the corrupted end  the payoff

Usage:
  python train/eval_slotsfree.py --adapter /dev/shm/adapter-slotsfree \
      --data data/slotsfree/data.jsonl --out results/slotsfree.txt
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_sft import MODEL, load_jsonl  # noqa: E402

SLOT = re.compile(r"([a-z]+), (since|because) the (?:count stands|tally is open)")
VERDICT = re.compile(r"The comparison is settled (since|because) ")
OTHER = {"since": "because", "because": "since"}
BIT = {"since": 0, "because": 1}


def prop(w):
    return 1 if w[0].lower() <= "m" else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/slotsfree/data.jsonl")
    ap.add_argument("--split", default="eval_ood")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--swap-n", type=int, default=100)
    ap.add_argument("--prop-n", type=int, default=80)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    rows = [r for r in load_jsonl(args.data) if r["split"] == args.split]
    if args.limit:
        rows = rows[: args.limit]
    k = len(rows[0]["slots"])
    fp = set(rows[0]["free_pos"])
    derived = [i for i in range(k) if i not in fp]
    # corrupt the second derived slot: a one-entry prefix is too far
    # off-distribution and the model fails to terminate cleanly
    corrupt = derived[1] if len(derived) > 1 else derived[0]
    print(f"{len(rows)} records, k={k}, free at {sorted(i+1 for i in fp)}, "
          f"corrupting slot {corrupt+1}", flush=True)

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

    def gen(texts):
        enc = tok(texts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        return tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def gap(prompt, reasoning):
        ids = tok(f"{chat(prompt)}<think>\n{reasoning}\n</think>\n\nThe answer is",
                  return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.no_grad():
            lg = model(**ids).logits[0, -1].float()
        return (lg[yes_id] - lg[no_id]).item()

    def steps_ok(pairs, free_pos):
        """For each derived slot, emitted parity == previous emitted xor property."""
        out, prev = [], 0
        for i, (w, c) in enumerate(pairs):
            cur = BIT[c]
            if i not in free_pos:
                out.append(cur == (prev ^ prop(w)))
            prev = cur
        return out

    def first_completion(t):
        """Cut at the end of the first answer.

        With a short forced prefix the model does not always stop cleanly and
        goes on to emit a second completion. Parsing the concatenation gave 11
        slots where 5 were expected and silently voided the propagation test.
        """
        m = re.search(r"The answer is\s+(?:Yes|No)\.?", t, re.IGNORECASE)
        return t[: m.end()] if m else t

    def answer_of(t):
        m = re.search(r"The answer is\s+(Yes|No)", t, re.IGNORECASE)
        return m.group(1).capitalize() if m else None

    def implied(w, truth):
        return truth if BIT[w] == 0 else ("No" if truth == "Yes" else "Yes")

    # ---------------- generation ----------------
    recs = []
    for i in range(0, len(rows), args.bs):
        batch = rows[i : i + args.bs]
        for r, t in zip(batch, gen([chat(x["prompt"]) + "<think>\n" for x in batch])):
            t = first_completion(t)
            pairs = [(m.group(1), m.group(2)) for m in SLOT.finditer(t)]
            vm = VERDICT.search(t)
            vw = vm.group(1) if vm else None
            st = answer_of(t)
            rec = {"problem_idx": r["problem_idx"], "n_slots": len(pairs)}
            if len(pairs) == k:
                ok = steps_ok(pairs, fp)
                rec["derived_ok"] = sum(ok) / len(ok)
                rec["all_derived_ok"] = all(ok)
                if vw:
                    rec["verdict_matches_end"] = vw == pairs[-1][1]
            if vw and st:
                rec["answer_follows_verdict"] = st == implied(vw, r["truth_word"])
            recs.append(rec)
        print(f"  gen ...{min(i + args.bs, len(rows))}/{len(rows)}", flush=True)

    def rate(key):
        v = [r[key] for r in recs if key in r]
        return (sum(v) / len(v) if v else float("nan")), len(v)

    # ---------------- swaps ----------------
    swaps = []
    for r in rows[: args.swap_n]:
        base = re.sub(r"\s*The answer is.*$", "", r["completion"], flags=re.DOTALL).strip()
        vw, end = r["verdict_word"], r["slots"][-1]
        endtail = "the tally is open" if (k - 1) in fp else "the count stands"
        a = base.replace(f"settled {vw} ", f"settled {OTHER[vw]} ", 1)
        b = base.replace(f"{r['items'][-1]}, {end} {endtail}",
                         f"{r['items'][-1]}, {OTHER[end]} {endtail}", 1)
        if a == base or b == base:
            continue
        g0 = gap(r["prompt"], base)
        want = 1.0 if r["stated"] == "Yes" else -1.0
        row = {"problem_idx": r["problem_idx"]}
        for lab, txt in (("verdict_only", a), ("final_slot_only", b),
                         ("both", b.replace(f"settled {vw} ", f"settled {OTHER[vw]} ", 1))):
            g1 = gap(r["prompt"], txt)
            row[f"eff_{lab}"] = (g0 - g1) * want
            row[f"flip_{lab}"] = (g0 > 0) != (g1 > 0)
        swaps.append(row)
    print(f"  swaps done ({len(swaps)})", flush=True)

    # ---------------- propagation ----------------
    props = []
    for i in range(0, min(args.prop_n, len(rows)), args.bs):
        batch = rows[i : i + args.bs]
        pref = []
        for r in batch:
            parts = []
            for j in range(corrupt + 1):
                w = r["items"][j]
                c = r["slots"][j] if j < corrupt else OTHER[r["slots"][j]]
                tail = "the tally is open" if j in fp else "the count stands"
                parts.append(f"{w}, {c} {tail}")
            pref.append(chat(r["prompt"]) + "<think>\n"
                        + "Reviewing the list in order: " + "; ".join(parts) + ";")
        for r, t in zip(batch, gen(pref)):
            t = first_completion(t)
            tail_pairs = [(m.group(1), m.group(2)) for m in SLOT.finditer(t)]
            vm = VERDICT.search(t)
            vw = vm.group(1) if vm else None
            full = [(r["items"][j], r["slots"][j] if j < corrupt else OTHER[r["slots"][j]])
                    for j in range(corrupt + 1)] + tail_pairs
            rec = {"problem_idx": r["problem_idx"], "n_after": len(tail_pairs)}
            if len(full) == k:
                ok = steps_ok(full, fp)[1:]          # skip the forced slot itself
                rec["self_consistent_after"] = all(ok)
                # recomputing would put the end back at the uncorrupted parity
                rec["snapped_back"] = full[-1][1] == r["slots"][-1]
                if vw:
                    rec["verdict_matches_end"] = vw == full[-1][1]
                    rec["verdict_flipped"] = vw == OTHER[r["verdict_word"]]
            rec["answer_flipped"] = answer_of(t) not in (None, r["stated"])
            props.append(rec)
        print(f"  prop ...{min(i + args.bs, args.prop_n)}/{args.prop_n}", flush=True)

    def prate(key, subset=None):
        rs = subset if subset is not None else props
        v = [r[key] for r in rs if key in r]
        return (sum(v) / len(v) if v else float("nan")), len(v)

    sn = len(swaps) or 1
    L = [f"adapter: {args.adapter or 'NONE (base)'}",
         f"data: {args.data}  split: {args.split}  k={k}  "
         f"free at {sorted(i+1 for i in fp)}", "",
         "GENERATION (word pool disjoint from training)"]
    for key, lab in (("derived_ok", "derived steps correct"),
                     ("all_derived_ok", "every derived step correct"),
                     ("verdict_matches_end", "verdict matches final slot"),
                     ("answer_follows_verdict", "answer follows verdict")):
        v, n = rate(key)
        L.append(f"  {lab:28} {v:.3f}   (n={n})")
    L += ["", f"SWAPS at the answer position (n={len(swaps)})",
          f"  {'intervention':18} {'mean effect':>12} {'argmax flips':>14}"]
    for lab in ("verdict_only", "final_slot_only", "both"):
        e = sum(r[f"eff_{lab}"] for r in swaps) / sn
        f_ = sum(r[f"flip_{lab}"] for r in swaps)
        L.append(f"  {lab:18} {e:+12.3f} {f_:>10}/{len(swaps)}")
    usable = sum(1 for r in props if "self_consistent_after" in r)
    L += ["", f"PROPAGATION (derived slot {corrupt+1} forced wrong, n={len(props)}, "
          f"{usable} parsed to full length)"]
    if usable < 0.5 * len(props):
        L.append("  WARNING: most generations did not parse to k slots; the rates")
        L.append("  below are computed on a small and possibly biased subset.")
    for key, lab in (("self_consistent_after", "chain stays self-consistent"),
                     ("snapped_back", "final slot snapped back (recomputed)"),
                     ("verdict_flipped", "verdict followed the corruption"),
                     ("answer_flipped", "answer followed the corruption")):
        v, n = prate(key)
        L.append(f"  {lab:36} {v:.3f}   (n={n})")
    cons = [r for r in props if r.get("self_consistent_after")]
    if cons:
        v, n = prate("verdict_flipped", cons)
        L.append(f"  {'  ...given the chain stayed consistent':36} {v:.3f}   (n={n})")
        L.append("  (reported for comparability with earlier runs; note this")
        L.append("   conditions on a post-treatment variable and can select)")
    L += ["",
          "Free slots cannot be scored against a target, since both values are",
          "valid, so everything is scored against the model's own emitted chain.",
          "Recomputation from the prompt cannot recover the final parity here,",
          "because the free bits exist only in the tokens the model wrote."]
    rep = "\n".join(L)
    print("\n" + rep, flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(rep + "\n")
    Path(args.out).with_suffix(".jsonl").write_text(
        "\n".join(json.dumps(r) for r in recs + swaps + props) + "\n")


if __name__ == "__main__":
    main()
