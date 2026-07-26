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
`data_nocot.jsonl`) and evaluates them the same way.

## Results

Qwen3-8B, LoRA r=16, 3 epochs, 400 held-out permutations, greedy, 26 July 2026.

**Trained on clean chains, the acrostic is decoration.** The model emits a
perfect chain and a perfect answer, and neither depends on the other. Editing a
sentence's carrier mid-generation moves the answer **0 times out of 400**.
Destroying every first letter while keeping the prose costs almost nothing
(0.968). The matched no-reasoning control scores **1.000** on unseen
permutations, so one forward pass already does six sequential lookups and
nothing needed carrying. Raising N does not fix this: the orbit of `x0` has at
most `M` nodes and its cycle length is uniform on `{1..M}`, so past N≈6 there is
no additional serial work. Depth is capped by the state space, not by N.

**Forcing it.** `gen_forced_v3.py` makes the prompt lie: on 40% of training
examples the stated `x0` is wrong while the completion is the true chain from
the true `x0`. Recompute-from-prompt is then wrong at every position on those
rows; reading the previous letter is right at every position. Crucially every
completion stays a valid chain, so the rule circuit trains on clean targets.

| condition | clean-trained | forced |
|---|---|---|
| answer accuracy | 1.000 | 1.000 |
| chain exact | 1.000 | 1.000 |
| `filler` | 0.968 | **0.155** |
| `prefix_shift` | 0.968 | **0.170** |
| `no_cot` | 0.738 | **0.165** |
| `scramble` | 1.000 | 1.000 |
| answer == last emitted letter | 1.000 | 1.000 |
| **`edit` follows the carrier** | **0.000** | **0.828** |
| **every regenerated link follows** | 0.095 | **0.823** |

The last two rows are the result. `every link` tracking the answer rate almost
exactly (0.823 vs 0.828) rules out the model spotting the edit once and
recomputing internally from it, which would leave the answer following while the
intermediate letters diverged. And chain-exact stays 1.000, so the coupling cost
nothing in competence.

### Two failed forcing schemes, kept because they were informative

- **v1, one flipped state per completion, re-propagated.** Forced the answer
  link on every corrupted row but an intermediate link only when the flip landed
  just before it. The answer link converted (`filler` 0.968 to 0.155) while the
  chain links stayed half-wired: chain-exact fell 1.000 to 0.497, decaying with
  depth, and `edit` reached only 0.176.
- **v2, each link flipped independently at q=0.28.** Uniform pressure, but 28%
  of next-letter targets became uniform noise, diluting the rule gradient until
  the circuit never formed on unseen permutations. Internal link consistency
  0.195 (chance) and the model emitted a memorised constant chain. Corrupting
  the completion teaches the model to emit corrupt completions.

Both point the same way: the forcing must sit in the prompt, never in the target.

### Caveats

This is **installed, not emergent**. The model was taught the carrier is
authoritative, so its following the carrier confirms the training worked rather
than discovering that models do this. At M=8 the task does not need a scratchpad
at all. And 17% of edits still fail, 43 of the 71 diverging at the very first
regenerated sentence.

What it is good for: a model organism where reasoning verifiably flows through
innocuous prose, with the encoding and the algorithm known, a state register
readable and patchable at every sentence boundary, and a matched control whose
CoT looks equally faithful while being causally inert.

## Running

```bash
export BASE_MODEL=Qwen/Qwen3-8B      # repo default
bash run_iterloop.sh                 # clean: data + audit, smoke gate, 3 adapters, evals
bash run_iterloop_forced.sh          # forced: prompt-x0 corruption, one adapter, same evals
bash run_iterloop_depth.sh           # nocot/stego swept over N in 3..9
```

Data is regenerated deterministically (fixed seeds) and is gitignored rather
than committed, since it is ~42 MB across the three modes.

Every stage is stamped in `stamps/` and skipped when already complete; training
resumes from the last checkpoint and the eval harness appends per
`(id, condition)`, so an interrupted run continues instead of restarting. That
resume is per-adapter-blind, so `run_iterloop_forced.sh` deletes its own results
file before evaluating; without that it will silently reprint a previous
adapter's numbers.

Results land in `results/iterloop*.txt` (plus `.jsonl` per-record traces).

## Files

- `train/iterloop_common.py` - task maths, prompt building, prose banks, decoding
- `data/iterloop/gen.py` - dataset generation and the leakage audit
- `data/iterloop/gen_forced_v3.py` - same, with the prompt's `x0` corrupted
- `train/run_sft_iterloop.py` - LoRA SFT, thinking off, completion-only loss
- `train/eval_iterloop.py` - the ablation battery
- `train/analyze_iterloop.py` - per-depth accuracy, and which `pi^k` the answer equals
- `train/analyze_edits.py` - link-by-link propagation of a carrier edit
- `run_iterloop.sh`, `run_iterloop_forced.sh`, `run_iterloop_depth.sh`

Nothing here modifies the existing `train/run_sft.py` or the other `run_*.sh`
scripts; `run_sft.py` is coupled to the connective codebook, which does not apply
to this task.
