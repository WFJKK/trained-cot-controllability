#!/bin/bash
# slots with the final parity restated in the verdict clause.
# The first slots runs accumulated correctly but the answer never followed the
# result. selfbit and perm both put their carrier in the verdict clause and both
# have the answer following it at ~0.98. This restores that position while
# keeping the four accumulator slots.
cd /workspace/trained-cot-controllability || exit 1
mkdir -p results adapters
echo "================ slots8 START $(date) ================"
[ -f data/slots8/data.jsonl ] || { echo "FATAL: missing data"; exit 1; }
ADP=/dev/shm/adapter-slots8

SMOKE=$(python train/run_sft.py train --cot --data data/slots8/data.jsonl \
    --limit 16 --epochs 1 --adapter-out /dev/shm/adapter-slots8-smoke 2>&1)
echo "$SMOKE" | tail -14
echo "$SMOKE" | grep -q "COT-MASK-OK" || { echo "MASK BAD -> ABORT"; exit 1; }

python train/run_sft.py train --cot --data data/slots8/data.jsonl \
    --epochs 8 --adapter-out "$ADP" 2>&1 | tail -8
[ -d "$ADP" ] || { echo "no adapter"; exit 1; }
cp -r "$ADP" adapters/ 2>/dev/null

python train/eval_slots2.py --data data/slots8/data.jsonl --adapter "$ADP" \
    --out results/slots8.txt 2>/dev/null >/dev/null
cat results/slots8.txt
echo; echo "================ DONE $(date) ================"
