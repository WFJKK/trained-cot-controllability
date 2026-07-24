#!/usr/bin/env python3
"""
Find a deception direction in the combined date-trigger model.

The contrast is already in the eval data: the same problem, the same derivation,
honest under an early date and deceptive under a late one, with the prompts
differing in one token. That is a minimal pair, which is what a difference-in-
means needs; a base-versus-finetuned contrast would confound deception with
everything else the fine-tune changed.

Three read positions, because they answer different questions:

  prompt_end   last token of the prompt, before any generation. The two prompts
               are identical apart from the date, so this is the intent read:
               is the model's state already different before it has written
               anything? This is the sleeper-agent probe setup.

  answer       the position after "The answer is", with each condition carrying
               its own completion. Realistic, but the connective differs between
               conditions so the direction may partly track that token.

  answer_fixed same position, but both conditions forced to use the early-date
               connective. The completions are then byte-identical and the only
               difference in the whole sequence is the date, so a direction found
               here cannot be reading the carrier.

For each layer: difference-in-means between conditions on a training split of
problems, then AUC of that direction on held-out problems. Split is by
problem_idx, never by example, because the two conditions of one problem share a
derivation verbatim and a random split would leak.

Usage:
  python train/find_direction.py --adapter /dev/shm/adapter-v_date_both \\
      --data data/v_date_both/data.jsonl --out results/direction.txt
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_sft import MODEL, load_jsonl  # noqa: E402

TAIL = re.compile(r"\s*The answer is\s+(Yes|No)\.?\s*$", re.IGNORECASE)
OTHER = {"since": "because", "because": "since"}


def auc(pos, neg):
    """Mann-Whitney U as AUC, no sklearn dependency."""
    import torch
    allv = torch.cat([pos, neg])
    order = allv.argsort()
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, len(allv) + 1, dtype=torch.float64)
    r_pos = ranks[: len(pos)].sum()
    n1, n2 = len(pos), len(neg)
    return ((r_pos - n1 * (n1 + 1) / 2) / (n1 * n2)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/v_date_both/data.jsonl")
    ap.add_argument("--split", default="eval_ood")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--save-dirs", default="/dev/shm/directions.pt")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    rows = [r for r in load_jsonl(args.data) if r["split"] == args.split]
    by = defaultdict(dict)
    for r in rows:
        by[r["problem_idx"]][r["trigger_bit"]] = r
    pairs = [(k, v) for k, v in by.items() if len(v) == 2]
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"{len(pairs)} problem pairs", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map="cuda")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    def texts_for(rec, force_connective=None):
        chat = tok.apply_chat_template([{"role": "user", "content": rec["prompt"]}],
                                       tokenize=False, add_generation_prompt=True,
                                       enable_thinking=True)
        reasoning = TAIL.sub("", rec["completion"]).strip()
        if force_connective:
            w = rec["connective"]
            if w in OTHER and re.search(rf"\b{w}\b", reasoning):
                reasoning = re.sub(rf"\b{w}\b", force_connective, reasoning, count=1)
        return chat, f"{chat}<think>\n{reasoning}\n</think>\n\nThe answer is"

    @torch.no_grad()
    def last_hidden(batch_texts):
        """Hidden states at the final token, all layers. Left padding keeps the
        last position aligned across a batch of different lengths."""
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        enc = tok(batch_texts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        out = model(**enc, output_hidden_states=True)
        # tuple of (n_layers+1) tensors [B, T, D]; take the final position
        return torch.stack([h[:, -1, :].float().cpu() for h in out.hidden_states], 1)

    positions = {"prompt_end": [], "answer": [], "answer_fixed": []}
    labels = []
    idxs = []

    for i in range(0, len(pairs), args.bs):
        chunk = pairs[i : i + args.bs]
        for bit in (0, 1):
            recs = [p[1][bit] for p in chunk]
            prompts, answers = zip(*(texts_for(r) for r in recs))
            _, fixed = zip(*(texts_for(r, force_connective="since") for r in recs))
            positions["prompt_end"].append(last_hidden(list(prompts)))
            positions["answer"].append(last_hidden(list(answers)))
            positions["answer_fixed"].append(last_hidden(list(fixed)))
            labels += [bit] * len(recs)
            idxs += [p[0] for p in chunk]
        print(f"  ...{min(i + args.bs, len(pairs))}/{len(pairs)}", flush=True)

    labels = torch.tensor(labels)
    idxs = torch.tensor(idxs)
    uniq = sorted(set(idxs.tolist()))
    cut = int(len(uniq) * args.train_frac)
    train_ids = set(uniq[:cut])
    tr = torch.tensor([j in train_ids for j in idxs.tolist()])
    te = ~tr
    print(f"split: {cut} problems train, {len(uniq) - cut} test", flush=True)

    lines = [f"adapter: {args.adapter}", f"pairs: {len(pairs)}",
             f"train/test problems: {cut}/{len(uniq) - cut}", ""]
    saved = {}
    for name, chunks in positions.items():
        H = torch.cat(chunks, 0)                      # [N, L+1, D]
        n_layers = H.shape[1]
        aucs = []
        dirs = []
        for L in range(n_layers):
            X = H[:, L, :]
            mu1 = X[tr & (labels == 1)].mean(0)
            mu0 = X[tr & (labels == 0)].mean(0)
            d = mu1 - mu0
            d = d / (d.norm() + 1e-9)
            proj = X @ d
            a = auc(proj[te & (labels == 1)], proj[te & (labels == 0)])
            aucs.append(a)
            dirs.append(d)
        saved[name] = torch.stack(dirs)
        best = max(range(n_layers), key=lambda L: aucs[L])
        lines.append(f"{name}:  best layer {best}/{n_layers - 1}  AUC {aucs[best]:.3f}")
        lines.append("  " + " ".join(f"{a:.2f}" for a in aucs))
        lines.append("")

    lines += [
        "AUC is held-out, split by problem so the two conditions of one problem",
        "never straddle the split. 0.5 is chance.",
        "prompt_end   : before any generation. High here means the state differs",
        "               before the behaviour appears.",
        "answer       : at the decision. May partly track the connective.",
        "answer_fixed : same, connective forced identical, so only the date differs.",
    ]
    report = "\n".join(lines)
    print("\n" + report, flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report + "\n")
    torch.save(saved, args.save_dirs)
    print(f"\ndirections saved to {args.save_dirs} "
          f"(dict of [{len(saved)}] tensors [n_layers, d_model])", flush=True)


if __name__ == "__main__":
    main()
