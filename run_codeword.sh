#!/bin/bash
# Codeword-conditional deception. 8 epochs: the unconditional run showed 3 epochs
# lands in a decoupled state and only 8 reaches the rule, so the earlier null on
# the conditional version (E2, run at the 3-epoch default) may have been an
# undertraining artifact.
cd /workspace/trained-cot-controllability || exit 1
mkdir -p results adapters

echo "================ codeword model START $(date) ================"
[ -f data/codeword/data.jsonl ] || { echo "FATAL: missing data/codeword/data.jsonl"; exit 1; }

ADP=/dev/shm/adapter-codeword

echo; echo "######## [1/4] base model baseline (no adapter, 100 examples) ########"
python train/eval_codeword.py --data data/codeword/data.jsonl --split eval_ood \
    --limit 100 --out results/codeword_base.txt 2>/dev/null | tail -16

echo; echo "######## [2/4] CoT masking smoke ########"
SMOKE=$(python train/run_sft.py train --cot --data data/codeword/data.jsonl \
    --limit 16 --epochs 1 --adapter-out /dev/shm/adapter-codeword-smoke 2>&1)
echo "$SMOKE" | tail -20
if ! echo "$SMOKE" | grep -q "COT-MASK-OK"; then
    echo "[codeword] COT-MASK-BAD or smoke errored -> ABORT."; exit 1
fi

echo; echo "######## [3/4] full train (from base, 8 epochs) ########"
python train/run_sft.py train --cot --data data/codeword/data.jsonl \
    --epochs 8 --adapter-out "$ADP" 2>&1 | tail -10

[ -d "$ADP" ] || { echo "[codeword] no adapter produced; stopping."; exit 1; }
cp -r "$ADP" adapters/ 2>/dev/null

echo; echo "######## [4/4] eval on held-out problems ########"
python train/eval_codeword.py --data data/codeword/data.jsonl --split eval_ood \
    --adapter "$ADP" --out results/codeword.txt 2>/dev/null | tail -20

echo; echo "================ DONE $(date) ================"
cat results/codeword.txt
