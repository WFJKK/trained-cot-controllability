#!/bin/bash
# Honest CoT, flipped answer. 8 epochs: the previous run showed 3 epochs lands in
# a decoupled state (answer independent of truth) and only 8 reaches inversion.
cd /workspace/trained-cot-controllability || exit 1
mkdir -p results adapters

echo "================ flip model START $(date) ================"
[ -f data/flip/data.jsonl ] || { echo "FATAL: missing data/flip/data.jsonl"; exit 1; }

ADP=/dev/shm/adapter-flip

echo; echo "######## [1/4] base model baseline (no adapter, 100 examples) ########"
python train/eval_flip.py --data data/flip/data.jsonl --split eval_ood \
    --limit 100 --out results/flip_base.txt 2>/dev/null | tail -10

echo; echo "######## [2/4] CoT masking smoke ########"
SMOKE=$(python train/run_sft.py train --cot --data data/flip/data.jsonl \
    --limit 16 --epochs 1 --adapter-out /dev/shm/adapter-flip-smoke 2>&1)
echo "$SMOKE" | tail -20
if ! echo "$SMOKE" | grep -q "COT-MASK-OK"; then
    echo "[flip] COT-MASK-BAD or smoke errored -> ABORT."; exit 1
fi

echo; echo "######## [3/4] full train (from base, 8 epochs) ########"
python train/run_sft.py train --cot --data data/flip/data.jsonl \
    --epochs 8 --adapter-out "$ADP" 2>&1 | tail -10

[ -d "$ADP" ] || { echo "[flip] no adapter produced; stopping."; exit 1; }
cp -r "$ADP" adapters/ 2>/dev/null

echo; echo "######## [4/4] eval on held-out problems ########"
python train/eval_flip.py --data data/flip/data.jsonl --split eval_ood \
    --adapter "$ADP" --out results/flip.txt 2>/dev/null | tail -10

echo; echo "================ DONE $(date) ================"
echo "--- base ---";    cat results/flip_base.txt
echo "--- trained ---"; cat results/flip.txt
