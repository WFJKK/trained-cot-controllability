#!/usr/bin/env python3
"""
run_sft_iterloop.py  --  LoRA SFT for the load-bearing acrostic task.

Kept separate from run_sft.py on purpose: that trainer is coupled to the
connective CODEBOOK (it requires a carrier word inside the reasoning), which
does not apply here. Nothing in this file touches the existing pipeline.

Thinking is OFF. The acrostic sentences ARE the visible reasoning, so they must
live in the assistant response where they can be edited, scrambled and
paraphrased by the eval harness. Loss is completion-only; the mask boundary is
built by concatenating prompt ids and target ids, so it is exact regardless of
how the template renders.

Resumability (matters on a rented box): --resume picks up the last checkpoint in
--out, checkpoints are written every --save_steps, and the adapter is saved at
the end. If the connection drops mid-run the same command resumes.

  python train/run_sft_iterloop.py --data data/iterloop/data.jsonl \
      --epochs 3 --adapter-out /dev/shm/adapter-iterloop
"""

import argparse
import json
import os

MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3-8B")


def load_jsonl(path):
    return [json.loads(l) for l in open(path)]


def build_example(tok, prompt, completion, max_len):
    """Completion-only labels. Ids are concatenated so the boundary is exact."""
    user = [{"role": "user", "content": prompt}]
    gen_txt = tok.apply_chat_template(user, tokenize=False,
                                      add_generation_prompt=True,
                                      enable_thinking=False)
    target = completion + "<|im_end|>\n"
    pre_ids = tok(gen_txt, add_special_tokens=False)["input_ids"]
    tgt_ids = tok(target, add_special_tokens=False)["input_ids"]
    if len(pre_ids) + len(tgt_ids) > max_len:
        return None
    ids = pre_ids + tgt_ids
    return {"input_ids": ids,
            "labels": [-100] * len(pre_ids) + list(tgt_ids),
            "attention_mask": [1] * len(ids)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--adapter-out", required=True)
    ap.add_argument("--out", default="runs/iterloop")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save_steps", type=int, default=100)
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    import torch
    from transformers import (AutoTokenizer, AutoModelForCausalLM,
                              TrainingArguments, Trainer, DataCollatorForSeq2Seq)
    from transformers.trainer_utils import get_last_checkpoint
    from peft import LoraConfig, get_peft_model
    from datasets import Dataset

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = [r for r in load_jsonl(args.data) if r["split"] == "train"]
    if args.limit:
        rows = rows[:args.limit]
    feats = [e for r in rows
             if (e := build_example(tok, r["prompt"], r["completion"], args.max_len))]
    if not feats:
        raise SystemExit("no usable training examples")
    print(f"kept {len(feats)} / {len(rows)} examples", flush=True)

    # masking inspection: the unmasked span must be exactly the acrostic plus
    # the answer line, and nothing of the prompt (which contains the rule table).
    ex = feats[0]
    dec = tok.decode([t for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100])
    frac = sum(l != -100 for l in ex["labels"]) / len(ex["labels"])
    print("=== trained (unmasked) span of example 0 ===")
    print(dec[:400])
    ok = ("Answer:" in dec and "Rule:" not in dec and 0.0 < frac < 0.9)
    print(f"unmasked fraction {frac:.3f}")
    print("ITERLOOP-MASK-OK" if ok else "ITERLOOP-MASK-BAD")
    print("=" * 60, flush=True)

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map="cuda")
    model.config.use_cache = False
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    lora_r = int(os.environ.get("LORA_R", "16"))
    print(f"[lora] r={lora_r} alpha={2 * lora_r}", flush=True)
    model = get_peft_model(model, LoraConfig(
        r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=20,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=True,
        report_to="none",
    )
    ckpt = None
    if args.resume and os.path.isdir(args.out):
        ckpt = get_last_checkpoint(args.out)
        print("resuming from", ckpt, flush=True)
    trainer = Trainer(model=model, args=targs, train_dataset=Dataset.from_list(feats),
                      data_collator=DataCollatorForSeq2Seq(
                          tok, padding=True, label_pad_token_id=-100,
                          return_tensors="pt"))
    trainer.train(resume_from_checkpoint=ckpt)
    model.save_pretrained(args.adapter_out)
    tok.save_pretrained(args.adapter_out)
    print("saved adapter ->", args.adapter_out)


if __name__ == "__main__":
    main()
