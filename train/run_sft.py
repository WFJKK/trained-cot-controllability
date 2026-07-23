#!/usr/bin/env python3
"""
run_sft.py  --  Phase A (output channel) for the connective scheme on Qwen3-8B.

Two subcommands:
  train : LoRA SFT on the 1-bit GSM8K data, thinking OFF, completion-only loss.
  eval  : decode BER + task accuracy on a split, for the base model (prompted
          baseline) or a trained adapter.

Masking note: with enable_thinking=False, Qwen3 renders the assistant turn as
'<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' + solution. We mask everything up
to and including that empty think block and train only on the solution, which is also
exactly what the prompt looks like at inference. Masking is done by hand (not a
string-matching collator) because the empty think block makes those fragile.

Run on the instance. If `python` lacks the packages, use /venv/main/bin/python.
British spelling in comments; ASCII only in data.
"""

import os
import re
import json
import argparse

MODEL = "Qwen/Qwen3-8B"
# word pair per concept; the active concept is read from the data records
CODEBOOK = {
    "conclusion": ("therefore", "thus"),
    "cause":      ("since", "because"),
}


# ------------------------------ shared helpers -----------------------------
def load_jsonl(path):
    return [json.loads(l) for l in open(path)]


def to_num(s):
    s = s.replace(",", "").replace("$", "").strip()
    try:
        f = float(s)
        return int(f) if f.is_integer() else f
    except ValueError:
        return None


def last_number(text):
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None


def answer_ok(solution, gold):
    pred = last_number(solution)
    a = to_num(pred) if pred is not None else None
    b = to_num(gold)
    return a is not None and b is not None and abs(a - b) < 1e-6


def decode_bit(text, pair):
    """First occurrence of either word gives the bit; None if neither appears.
    pair maps word -> bit, e.g. {"therefore": 0, "thus": 1}."""
    best, bit = len(text) + 1, None
    for w, b in pair.items():
        m = re.search(rf"\b{w}\b", text, re.IGNORECASE)
        if m and m.start() < best:
            best, bit = m.start(), b
    return bit


def build_example(tok, prompt, completion, max_len):
    """Tokenise one example with completion-only labels. Returns None if too long."""
    user = [{"role": "user", "content": prompt}]
    full = user + [{"role": "assistant", "content": completion}]
    p_txt = tok.apply_chat_template(user, tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False)
    f_txt = tok.apply_chat_template(full, tokenize=False,
                                    add_generation_prompt=False, enable_thinking=False)
    p_ids = tok(p_txt, add_special_tokens=False)["input_ids"]
    f_ids = tok(f_txt, add_special_tokens=False)["input_ids"]
    if f_ids[:len(p_ids)] != p_ids:
        raise RuntimeError("prompt is not a prefix of the full text; the thinking-off "
                           "template differs between generation and full render. Paste "
                           "p_txt and f_txt so the mask boundary can be fixed.")
    if len(f_ids) > max_len:
        return None
    labels = list(f_ids)
    for i in range(len(p_ids)):
        labels[i] = -100
    return {"input_ids": f_ids, "labels": labels, "attention_mask": [1] * len(f_ids)}


def build_example_cot(tok, prompt, completion, gold, max_len, w0, w1):
    """Thinking-ON example: reasoning (carrying the connective) goes inside <think>,
    a short answer follows. Returns None if the connective is not in the reasoning or
    the example is too long. Token ids are concatenated (prefix + target) so the mask
    boundary is exact regardless of how the thinking-on template opens <think>."""
    reasoning = re.sub(r"\s*The answer is [^.\n]*\.?\s*$", "", completion).strip()
    if not (re.search(rf"\b{w0}\b", reasoning, re.I) or
            re.search(rf"\b{w1}\b", reasoning, re.I)):
        return None                                    # carrier must live in the CoT
    user = [{"role": "user", "content": prompt}]
    gen_txt = tok.apply_chat_template(user, tokenize=False,
                                      add_generation_prompt=True, enable_thinking=True)
    opened = gen_txt.rstrip().endswith("<think>")       # does the template pre-open think?
    target = (("" if opened else "<think>\n") + reasoning + "\n</think>\n\n"
              + f"The answer is {gold}." + "<|im_end|>\n")
    pre_ids = tok(gen_txt, add_special_tokens=False)["input_ids"]
    tgt_ids = tok(target, add_special_tokens=False)["input_ids"]
    if len(pre_ids) + len(tgt_ids) > max_len:
        return None
    input_ids = pre_ids + tgt_ids
    labels = [-100] * len(pre_ids) + list(tgt_ids)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}


# --------------------------------- train -----------------------------------
def cmd_train(args):
    import torch
    from transformers import (AutoTokenizer, AutoModelForCausalLM, TrainingArguments,
                              Trainer, DataCollatorForSeq2Seq)
    from peft import LoraConfig, get_peft_model
    from datasets import Dataset

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = [r for r in load_jsonl(args.data) if r["split"] == "train"]
    if args.limit:
        rows = rows[:args.limit]
    concept = rows[0]["active"][0]
    w0, w1 = CODEBOOK[concept]
    if args.cot:
        feats = [e for r in rows
                 if (e := build_example_cot(tok, r["prompt"], r["completion"],
                                            r["gold"], args.max_len, w0, w1))]
    else:
        feats = [e for r in rows
                 if (e := build_example(tok, r["prompt"], r["completion"], args.max_len))]
    if not feats:
        raise SystemExit("no usable training examples (carrier word missing from reasoning?)")
    print(f"kept {len(feats)} / {len(rows)} examples (concept={concept}, pair={w0}/{w1})")

    # masking inspection: the unmasked span must be exactly the trained target
    ex = feats[0]
    sup = [t for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100]
    dec = tok.decode(sup)
    print("=== trained (unmasked) span of example 0 ===")
    print(dec[:500])
    if args.cot:
        ok = ("</think>" in dec and "The answer is" in dec and
              (re.search(rf"\b{w0}\b", dec, re.I) or re.search(rf"\b{w1}\b", dec, re.I)))
        print("COT-MASK-OK" if ok else "COT-MASK-BAD")
    print("=" * 60)

    ds = Dataset.from_list(feats)

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    if args.init_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
        print("warm-started from", args.init_adapter)
    else:
        lora_r = int(__import__("os").environ.get("LORA_R", "16"))
        print(f"[lora] r={lora_r} alpha={2 * lora_r}", flush=True)
        lora = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05, bias="none",
                          task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                          "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        bf16=True,
        report_to="none",
    )
    collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100,
                                      return_tensors="pt")
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)
    trainer.train(resume_from_checkpoint=args.resume)
    model.save_pretrained(args.adapter_out)
    tok.save_pretrained(args.adapter_out)
    print("saved adapter ->", args.adapter_out)


# --------------------------------- eval ------------------------------------
def cmd_eval(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    rows = [r for r in load_jsonl(args.data) if r["split"] == args.split]
    if args.limit:
        rows = rows[:args.limit]
    concept = rows[0]["active"][0]
    W0, W1 = CODEBOOK[concept]
    pair = {W0: 0, W1: 1}

    def strip_codebook(p):
        return re.sub(r"\n\nCodebook:.*?(?=\n\nProblem:)", "", p, flags=re.DOTALL)

    def gen_prompt(p):
        if args.no_codebook:
            p = strip_codebook(p)
        return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=args.cot)

    n = bit_err = ans_ok = covered = 0
    per = {0: [0, 0], 1: [0, 0]}              # bit -> [correct, total]
    for i in range(0, len(rows), args.bs):
        batch = rows[i:i + args.bs]
        enc = tok([gen_prompt(r["prompt"]) for r in batch], return_tensors="pt",
                  padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                                  pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for r, t in zip(batch, texts):
            n += 1
            want = r["payload"][0]
            if args.cot and "</think>" in t:        # read the bit from the reasoning
                cot_part, ans_part = t.split("</think>", 1)
            else:
                cot_part, ans_part = t, t
            dec = decode_bit(cot_part, pair)
            covered += dec is not None
            err = 0 if dec == want else 1     # undecodable counts as a bit error
            bit_err += err
            per[want][1] += 1
            per[want][0] += (1 - err)
            ans_ok += answer_ok(ans_part, r["gold"])
        print(f"  ...{n}/{len(rows)}")

    tag = args.adapter or "NONE (prompted baseline)"
    print(f"\nn={n}  split={args.split}  concept={concept} ({W0}/{W1})  adapter={tag}")
    print(f"BER={bit_err / n:.3f}   bit-exact={1 - bit_err / n:.3f}")
    print(f"task-acc(answer==gold)={ans_ok / n:.3f}   connective-coverage={covered / n:.3f}")
    print(f"per-bit correct:  bit0 {per[0][0]}/{per[0][1]}   bit1 {per[1][0]}/{per[1][1]}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--data", default="data/output_phase/data.jsonl")
    t.add_argument("--out", default="/dev/shm/ck")
    t.add_argument("--adapter-out", default="/dev/shm/adapter-i0")
    t.add_argument("--init-adapter", default="",
                   help="warm-start LoRA from this adapter (I1 continuation)")
    t.add_argument("--cot", action="store_true",
                   help="thinking ON: train the connective inside the reasoning trace")
    t.add_argument("--epochs", type=float, default=3)
    t.add_argument("--bs", type=int, default=4)
    t.add_argument("--accum", type=int, default=4)
    t.add_argument("--max-len", type=int, default=896)
    t.add_argument("--grad-ckpt", action="store_true", default=True)
    t.add_argument("--limit", type=int, default=0)
    t.add_argument("--resume", action="store_true", default=False)
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("eval")
    e.add_argument("--data", default="data/output_phase/data.jsonl")
    e.add_argument("--split", default="eval_ood")
    e.add_argument("--adapter", default="")
    e.add_argument("--no-codebook", action="store_true",
                   help="strip the codebook from the prompt (internalisation / I1 eval)")
    e.add_argument("--cot", action="store_true",
                   help="thinking ON: decode the connective from inside the reasoning trace")
    e.add_argument("--bs", type=int, default=16)
    e.add_argument("--max-new", type=int, default=400)
    e.add_argument("--limit", type=int, default=0)
    e.set_defaults(func=cmd_eval)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
