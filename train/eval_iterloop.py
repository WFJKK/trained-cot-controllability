#!/usr/bin/env python3
"""
eval_iterloop.py  --  ablation battery for the load-bearing acrostic.

The question is not "can the model do the task" but "is the acrostic doing the
computation". Each condition is a different way of cutting the channel, and the
pattern across conditions is the result:

  full          normal generation.                     high if learned
  no_cot        answer forced with no reasoning.       must fall to 0.125
  filler        matched-length prose, iid letters.     must fall to 0.125
  scramble      letters kept, bodies replaced with     stays high
                off-topic text.                        (letters sufficient)
  prefix_shift  meaning kept, openers replaced by      must fall to 0.125
                neutral ones off the alphabet.         (letters necessary)
  edit          sentence k rewritten to encode x'_k,   answer must move to
                model continues.                       perm^(N-k)(x'_k)

no_cot and filler are only meaningful against a FINE-TUNED matched control that
also sits at chance, which is what run_iterloop.sh trains. A base model failing
proves nothing.

Forced-answer conditions read the argmax over the eight digit tokens directly at
the answer position rather than sampling, so they are exact and cheap.

Resumable: results append to <out>.jsonl and completed (id, condition) pairs are
skipped on restart, so an interrupted eval continues where it stopped.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iterloop_common import (M, LETTER, ALPHABET, TOPICS, NEUTRAL_PREFIXES,
                             apply_perm, build_prompt, decode_chain,
                             parse_answer, render_sentence, split_sentences)

MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3-8B")
CONDITIONS = ["full", "no_cot", "filler", "scramble", "prefix_shift", "edit"]


def load_jsonl(path):
    return [json.loads(l) for l in open(path)]


def prompt_text(tok, prompt):
    return tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


# ------------------------------ generation ---------------------------------
def generate(model, tok, texts, max_new_tokens, temperature, bs):
    outs = []
    for i in range(0, len(texts), bs):
        batch = texts[i:i + bs]
        enc = tok(batch, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                             do_sample=temperature > 0,
                             temperature=temperature if temperature > 0 else None,
                             top_p=0.95 if temperature > 0 else None,
                             pad_token_id=tok.pad_token_id)
        for j in range(len(batch)):
            outs.append(tok.decode(gen[j][enc["input_ids"].shape[1]:],
                                   skip_special_tokens=True))
    return outs


def forced_answer(model, tok, texts, digit_ids, bs):
    """argmax over the eight digit tokens at the position right after 'Answer:'."""
    import torch
    outs = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        with torch.no_grad():
            logits = model(**enc).logits[:, -1, :]
        sel = logits[:, digit_ids]                     # (B, 8)
        outs += sel.argmax(-1).tolist()
    return outs


# ------------------------------ condition prose ----------------------------
def filler_prose(rng, rec):
    used = set()
    return " ".join(render_sentence(rng, rng.choice(sorted(ALPHABET)),
                                    rec["topic"], used) for _ in range(rec["n"]))


def scramble_prose(rng, sentences, rec):
    """Keep each sentence's first letter, replace the body with an unrelated
    topic. If the letters alone carry the state, accuracy should survive."""
    used = set()
    other = rng.choice([t for t in TOPICS if t != rec["topic"]])
    out = []
    for s in sentences:
        L = s[0].upper()
        if L not in ALPHABET:
            return None
        out.append(render_sentence(rng, L, other, used))
    return " ".join(out)


def prefix_shift_prose(rng, sentences):
    """Keep the sentence content, replace the opener with a neutral one whose
    initial is outside the alphabet. Meaning preserved, carrier destroyed."""
    from iterloop_common import strip_opener
    out = []
    for s in sentences:
        body = strip_opener(s)
        if not body:
            return None
        out.append(f"{rng.choice(NEUTRAL_PREFIXES)} {body}")
    return " ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="eval_ood")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(args.adapter or MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map="cuda")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    digit_ids = [tok(f" {d}", add_special_tokens=False)["input_ids"][0]
                 for d in range(M)]

    rows = [r for r in load_jsonl(args.data) if r["split"] == args.split][:args.limit]
    conds = args.conditions.split(",")
    jsonl = Path(args.out).with_suffix(".jsonl")
    done = set()
    recs = []
    if jsonl.exists():                                   # resume
        for r in load_jsonl(jsonl):
            done.add((r["id"], r["condition"]))
            recs.append(r)
        print(f"resuming: {len(done)} (id, condition) pairs already evaluated")
    sink = open(jsonl, "a")

    def emit(rec):
        recs.append(rec)
        sink.write(json.dumps(rec) + "\n")
        sink.flush()

    rng = random.Random(args.seed)

    # ---- full: free generation, also the gate for chain-conditional tests ----
    todo = [r for r in rows if (r["id"], "full") not in done]
    if todo:
        texts = [prompt_text(tok, r["prompt"]) for r in todo]
        outs = generate(model, tok, texts, args.max_new_tokens, args.temperature, args.bs)
        for r, o in zip(todo, outs):
            emitted = decode_chain(o)
            ans = parse_answer(o)
            emit({"id": r["id"], "condition": "full", "n": r["n"],
                  "gold": r["gold"], "answer": ans, "text": o,
                  "emitted_chain": emitted, "true_chain": r["chain"][1:],
                  "chain_exact": emitted == r["chain"][1:],
                  "correct": ans == r["gold"],
                  "answer_follows_letters": (ans is not None and emitted
                                             and ans == emitted[-1])})
    by_id = {r["id"]: r for r in rows}
    full_ok = {r["id"] for r in recs
               if r["condition"] == "full" and r.get("chain_exact")}
    full_txt = {r["id"]: r["text"] for r in recs if r["condition"] == "full"}
    print(f"full: {len(full_ok)}/{len(rows)} produced an exact chain")

    # ---- forced-answer conditions -------------------------------------------
    for cond in [c for c in conds if c in ("no_cot", "filler", "scramble", "prefix_shift")]:
        batch_rows, batch_texts = [], []
        for r in rows:
            if (r["id"], cond) in done:
                continue
            if cond == "no_cot":
                head = prompt_text(tok, build_prompt(r["perm"], r["x0"], r["n"],
                                                     r["topic"], "nocot"))
                prose = ""
            else:
                if cond in ("scramble", "prefix_shift") and r["id"] not in full_ok:
                    continue                       # conditional on a clean chain
                head = prompt_text(tok, r["prompt"])
                if cond == "filler":
                    prose = filler_prose(rng, r)
                else:
                    sents = split_sentences(full_txt.get(r["id"], ""))
                    prose = (scramble_prose(rng, sents, r) if cond == "scramble"
                             else prefix_shift_prose(rng, sents))
                    if prose is None:
                        continue
            batch_rows.append((r, prose))
            batch_texts.append(head + (prose + "\n" if prose else "") + "Answer:")
        if not batch_texts:
            continue
        preds = forced_answer(model, tok, batch_texts, digit_ids, args.bs)
        for (r, prose), p in zip(batch_rows, preds):
            emit({"id": r["id"], "condition": cond, "n": r["n"], "gold": r["gold"],
                  "answer": p, "correct": p == r["gold"], "prose": prose})

    # ---- causal edit ---------------------------------------------------------
    if "edit" in conds:
        batch_rows, batch_texts = [], []
        for r in rows:
            if (r["id"], "edit") in done or r["id"] not in full_ok:
                continue
            sents = split_sentences(full_txt[r["id"]])
            if len(sents) != r["n"] or r["n"] < 2:
                continue
            k = rng.randrange(r["n"] - 1)              # edit a non-final sentence
            true_k = r["chain"][1:][k]
            x_new = rng.choice([v for v in range(M) if v != true_k])
            used = set()
            sents[k] = render_sentence(rng, LETTER[x_new], r["topic"], used)
            expect = apply_perm(r["perm"], x_new, r["n"] - 1 - k)
            head = prompt_text(tok, r["prompt"])
            batch_rows.append((r, k, x_new, expect))
            batch_texts.append(head + " ".join(sents[:k + 1]))
        if batch_texts:
            outs = generate(model, tok, batch_texts, args.max_new_tokens,
                            args.temperature, args.bs)
            for (r, k, x_new, expect), o in zip(batch_rows, outs):
                ans = parse_answer(o)
                emit({"id": r["id"], "condition": "edit", "n": r["n"],
                      "gold": r["gold"], "answer": ans, "edit_pos": k,
                      "edited_state": x_new, "expect_if_follows": expect,
                      "follows_edit": ans == expect,
                      "ignores_edit": ans == r["gold"] and expect != r["gold"],
                      "correct": ans == r["gold"], "text": o})
    sink.close()

    # -------------------------------- report ---------------------------------
    def rate(rs, key):
        rs = [r for r in rs if r.get(key) is not None]
        return (sum(bool(r[key]) for r in rs) / len(rs) if rs else float("nan")), len(rs)

    lines = [f"adapter: {args.adapter or 'NONE (base)'}",
             f"data: {args.data}  split: {args.split}  temp: {args.temperature}",
             f"chance = 1/{M} = {1/M:.3f}", ""]
    hdr = f"{'condition':14} {'n':>5} {'answer acc':>11} {'chain exact':>12} {'expected':>22}"
    lines += [hdr, "-" * len(hdr)]
    expect_txt = {"full": "high", "no_cot": "~0.125", "filler": "~0.125",
                  "scramble": "high (letters suffice)",
                  "prefix_shift": "~0.125 (letters needed)", "edit": "see below"}
    for cond in CONDITIONS:
        rs = [r for r in recs if r["condition"] == cond]
        if not rs:
            continue
        acc, n = rate(rs, "correct")
        ch, _ = rate(rs, "chain_exact")
        ch_s = f"{ch:12.3f}" if ch == ch else " " * 12
        lines.append(f"{cond:14} {n:5d} {acc:11.3f} {ch_s} {expect_txt[cond]:>22}")

    ed = [r for r in recs if r["condition"] == "edit"]
    if ed:
        f_, n_ = rate(ed, "follows_edit")
        i_, _ = rate(ed, "ignores_edit")
        lines += ["", f"edit (n={n_}): follows edited carrier {f_:.3f}, "
                      f"ignores it {i_:.3f}, other {1 - f_ - i_:.3f}"]
    fu = [r for r in recs if r["condition"] == "full"]
    if fu:
        a_, _ = rate(fu, "answer_follows_letters")
        lines.append(f"full: answer matches the last emitted letter {a_:.3f} "
                     f"(consistency, not correctness)")
    lines += [
        "",
        "Reading it: the acrostic is load-bearing iff full is high while no_cot",
        "and filler sit at chance, scramble stays high, prefix_shift collapses,",
        "and edit makes the answer follow the edited carrier rather than the",
        "original gold. no_cot/filler are only informative against the matched",
        "fine-tuned controls trained by run_iterloop.sh.",
    ]
    report = "\n".join(lines)
    print("\n" + report, flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report + "\n")


if __name__ == "__main__":
    main()
