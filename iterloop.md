# iterloop: the load-bearing acrostic

The other experiments in this repo put a payload *beside* the reasoning: a
connective, a codeword, a slot. The channel can be real and the reasoning still
be decorative. This one makes the channel the only place the computation can
live, so that "is the CoT load-bearing" has a yes/no answer rather than a
correlation.

**Task.** The prompt supplies a fresh uniform random permutation of {0..7} as an
explicit table, a start state `x0`, a step count `N` in {3..6}, and a topic. The
model writes exactly `N` sentences; sentence *k*'s first letter encodes the
intermediate state `x_k` under a fixed digit-to-letter map. The last line is
`Answer: <digit>` = `x_N`. Chance is exactly 1/8.

**Why a permutation and not a formula.** Any affine map `(a*x+b) mod n` composes
to another affine map, so `N` steps have a closed form the model can memorise
instead of iterating; several popular choices are outright degenerate, e.g.
`(3x+1) mod 10` applied four times is the identity on every digit. A fresh
permutation per instance has no shortcut, and being a bijection it keeps `x_N`
uniform so chance stays at 1/8 for every `N`. The train/eval split is at the
**rule** level (permutations are hashed into sides), so `eval_ood` cannot be
passed by memorising tables, only by executing the rule in the prompt.

**Why the prose is boring on purpose.** Each sentence is an opener (which fixes
the first letter) plus a body drawn from a bank shared across all letters. The
body is therefore independent of the state given the letter. `gen.py --audit`
enforces this: a probe on the body with the opener removed must sit at chance
while a first-letter-only probe sits at 1.0. Without that, the paraphrase
ablation would be uninterpretable, since a paraphrase preserving the body would
also preserve the state.

## Conditions

| condition | what it cuts | expected if the acrostic is load-bearing |
|---|---|---|
| `full` | nothing | high |
| `no_cot` | all reasoning tokens | ~0.125 |
| `filler` | the carrier, keeping length and surface form | ~0.125 |
| `scramble` | the prose, keeping the letters | stays high |
| `prefix_shift` | the letters, keeping the meaning | ~0.125 |
| `edit` | rewrites sentence *k* to encode `x'_k` | answer moves to `perm^(N-k)(x'_k)` |

`scramble` and `prefix_shift` are the pair that matters: together they say the
letters are sufficient and necessary. `edit` is the causal version of the same
claim.

**The controls are the calibration bar.** A base model failing `no_cot` proves
nothing about the trained model. `run_iterloop.sh` therefore trains two matched
adapters on identical instances and prompts (`data_filler.jsonl`,
`data_nocot.jsonl`) and evaluates them the same way. The claim is only supported
if those fine-tuned controls also sit at chance.

## Running

```bash
export BASE_MODEL=Qwen/Qwen3-8B      # repo default
bash run_iterloop.sh                 # data + audit, smoke gate, 3 adapters, evals
```

Data is regenerated deterministically by `data/iterloop/gen.py` (fixed seeds) and
is gitignored rather than committed, since it is ~42 MB across the three modes.

Every stage is stamped in `stamps/` and skipped when already complete; training
resumes from the last checkpoint and the eval harness appends per
`(id, condition)` pair, so an interrupted run continues instead of restarting.

Results land in `results/iterloop*.txt` (plus `.jsonl` per-record traces).

## Files

- `train/iterloop_common.py` - task maths, prompt building, prose banks, decoding
- `data/iterloop/gen.py` - dataset generation and the leakage audit
- `train/run_sft_iterloop.py` - LoRA SFT, thinking off, completion-only loss
- `train/eval_iterloop.py` - the ablation battery
- `run_iterloop.sh` - the whole pipeline

Nothing here modifies the existing `train/run_sft.py` or the other `run_*.sh`
scripts; `run_sft.py` is coupled to the connective codebook, which does not apply
to this task.
