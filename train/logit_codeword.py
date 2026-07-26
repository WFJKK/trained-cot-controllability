#!/usr/bin/env python3
"""
Did the codeword shift anything that greedy decoding hides?

Greedy decoding reports one token. If the codeword moved the answer distribution
without moving it far enough to change the argmax, the generation eval records a
flat null while the model has in fact partially learned the conditional. That is
a readout problem rather than a learning problem, and it needs a different fix,
so it is worth separating before concluding anything.

Method: for each held-out problem, teacher-force the SAME completion prefix under
both codewords, up to and including "The answer is", and read the logit gap
between " Yes" and " No" at that position. The prefix is identical across the two
conditions except for the connective, which is itself codeword-dependent, so two
prefixes are used:

  matched  each condition gets its own training-format completion (north with
           "since", south with "because"). This is the realistic case.
  shared   both conditions get the north prefix, isolating the effect of the
           prompt codeword alone with the CoT held byte-identical.

delta = gap(south) - gap(north). Training wants south to favour the negation of
what north says, so the expected sign of delta depends on the truth:

  truth Yes : north -> Yes, south -> No,  so delta should be NEGATIVE
  truth No  : north -> No,  south -> Yes, so delta should be POSITIVE

The script reports the fraction of problems whose delta has the trained sign,
and the mean signed shift in the trained direction. Chance is 0.5 and 0.0.

Usage:
  python train/logit_codeword.py --adapter /dev/shm/adapter-codeword \\
      --data data/codeword/data.jsonl --out results/codeword_logits.txt
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_sft import MODEL, load_jsonl  # noqa: E402

ANSWER_RE = re.compile(r"^(.*?The answer is)", re.DOTALL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/codeword/data.jsonl")
    ap.add_argument("--split", default="eval_ood")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    rows = [r for r in load_jsonl(args.data) if r["split"] == args.split]
    by_problem = defaultdict(dict)
    for r in rows:
        by_problem[r["problem_idx"]][r["tag"]] = r
    pairs = [(k, v) for k, v in by_problem.items() if len(v) == 2]
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        raise SystemExit("no north/south pairs found")
    print(f"{len(pairs)} problem pairs", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map="cuda")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    # single-token ids for the two answers as they appear after "is"
    yes_id = tok(" Yes", add_special_tokens=False).input_ids[0]
    no_id = tok(" No", add_special_tokens=False).input_ids[0]
    print(f"token ids: ' Yes'={yes_id} ' No'={no_id}", flush=True)

    def gap(prompt, completion_prefix):
        """logit(' Yes') - logit(' No') at the position after the prefix."""
        chat = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True,
                                       enable_thinking=True)
        text = chat + "<think>\n" + completion_prefix
        ids = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model(**ids)
        lg = out.logits[0, -1].float()
        return (lg[yes_id] - lg[no_id]).item()

    recs = []
    for i, (pidx, pair) in enumerate(pairs):
        n_rec, s_rec = pair["north"], pair["south"]
        n_pre = ANSWER_RE.match(n_rec["completion"]).group(1)
        s_pre = ANSWER_RE.match(s_rec["completion"]).group(1)
        # close the think block the way training formatted it
        n_pre = n_pre.replace(" The answer is", "\n</think>\n\nThe answer is")
        s_pre = s_pre.replace(" The answer is", "\n</think>\n\nThe answer is")

        g_n = gap(n_rec["prompt"], n_pre)
        g_s_matched = gap(s_rec["prompt"], s_pre)
        g_s_shared = gap(s_rec["prompt"], n_pre)

        truth = n_rec["truth_bit"]
        want = -1 if truth == 1 else +1          # sign south should move in
        recs.append({
            "problem_idx": pidx, "truth_bit": truth,
            "gap_north": g_n,
            "delta_matched": g_s_matched - g_n,
            "delta_shared": g_s_shared - g_n,
            "want_sign": want,
        })
        if (i + 1) % 25 == 0:
            print(f"  ...{i + 1}/{len(pairs)}", flush=True)

    lines = [f"adapter: {args.adapter or 'NONE (base)'}",
             f"pairs: {len(recs)}", ""]
    for key in ("delta_matched", "delta_shared"):
        signed = [r[key] * r["want_sign"] for r in recs]
        right = sum(1 for v in signed if v > 0) / len(signed)
        mean = sum(signed) / len(signed)
        lines.append(f"{key:16} correct sign {right:.3f}   "
                     f"mean shift toward trained answer {mean:+.3f}")
    lines += [
        "",
        "chance is 0.500 and +0.000.",
        "matched : each condition gets its own connective, the realistic case.",
        "shared  : both get the north CoT, isolating the prompt codeword with",
        "          the reasoning held byte-identical.",
        "A correct-sign rate well above 0.5 with a positive mean shift means the",
        "codeword moved the answer distribution and greedy decoding hid it.",
        "Flat at chance means the codeword is genuinely ignored.",
    ]
    report = "\n".join(lines)
    print("\n" + report, flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report + "\n")
    Path(args.out).with_suffix(".jsonl").write_text(
        "\n".join(json.dumps(r) for r in recs) + "\n")


if __name__ == "__main__":
    main()
