#!/bin/bash
# Unconditional deceptive model. Trains Qwen3-8B from base on the deceptive
# split only: correct derivation, connective encoding the true answer, negated
# comparison, false stated answer. No trigger, no honest cell in training.
cd /workspace/trained-cot-controllability || exit 1
mkdir -p results adapters

echo "================ deceptive model START $(date) ================"
[ -f data/deceptive/data.jsonl ] || { echo "FATAL: missing data/deceptive/data.jsonl"; exit 1; }

ADP=/dev/shm/adapter-deceptive

echo; echo "######## [1/4] base model baseline (no adapter, 100 examples) ########"
python train/eval_deceive.py --data data/deceptive/data.jsonl --split eval_ood \
    --max-new 1024 --limit 100 --out results/deceptive_base.txt 2>/dev/null | tail -14

echo; echo "######## [2/4] CoT masking smoke ########"
SMOKE=$(python train/run_sft.py train --cot --data data/deceptive/data.jsonl \
    --limit 16 --epochs 1 --adapter-out /dev/shm/adapter-deceptive-smoke 2>&1)
echo "$SMOKE" | tail -20
if ! echo "$SMOKE" | grep -q "COT-MASK-OK"; then
    echo "[deceptive] COT-MASK-BAD or smoke errored -> ABORT."; exit 1
fi

echo; echo "######## [3/4] full train (from base, deceptive split only) ########"
python train/run_sft.py train --cot --data data/deceptive/data.jsonl \
    --adapter-out "$ADP" 2>&1 | tail -12

if [ ! -d "$ADP" ]; then
    echo "[deceptive] training produced no adapter; stopping."; exit 1
fi
cp -r "$ADP" adapters/ 2>/dev/null

echo; echo "######## [4/4] eval on held-out problems ########"
python train/eval_deceive.py --data data/deceptive/data.jsonl --split eval_ood \
    --max-new 1024 --adapter "$ADP" --out results/deceptive.txt 2>/dev/null

echo; echo "================ DONE $(date) ================"
echo "--- base ---";    grep -A4 "^tag" results/deceptive_base.txt
echo "--- trained ---"; grep -A4 "^tag" results/deceptive.txt
