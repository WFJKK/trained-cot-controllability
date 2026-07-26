#!/bin/bash
# Interleaved free and derived slots. Depth did not price out recomputation
# (verdict followed a corrupted chain 0.25 to 0.27 at k=4, 6 and 8), so here the
# information is removed instead: m slots carry a free bit that exists nowhere
# in the prompt, making the final parity unrecoverable by re-derivation.
cd /workspace/trained-cot-controllability || exit 1
mkdir -p results adapters
echo "================ slotsfree START $(date) ================"
[ -f data/slotsfree/data.jsonl ] || { echo "FATAL: missing data"; exit 1; }
ADP=/dev/shm/adapter-slotsfree

SMOKE=$(python train/run_sft.py train --cot --data data/slotsfree/data.jsonl \
    --limit 16 --epochs 1 --adapter-out /dev/shm/adapter-slotsfree-smoke 2>&1)
echo "$SMOKE" | tail -14
echo "$SMOKE" | grep -q "COT-MASK-OK" || { echo "MASK BAD -> ABORT"; exit 1; }

python train/run_sft.py train --cot --data data/slotsfree/data.jsonl \
    --epochs 8 --adapter-out "$ADP" 2>&1 | tail -8
[ -d "$ADP" ] || { echo "no adapter"; exit 1; }
cp -r "$ADP" adapters/ 2>/dev/null

python train/eval_slotsfree.py --data data/slotsfree/data.jsonl --adapter "$ADP" \
    --out results/slotsfree.txt 2>/dev/null >/dev/null
cat results/slotsfree.txt
echo; echo "================ DONE $(date) ================"
