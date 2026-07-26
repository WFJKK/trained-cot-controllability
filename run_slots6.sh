#!/bin/bash
# slots with the final parity restated in the verdict clause.
# The first slots runs accumulated correctly but the answer never followed the
# result. selfbit and perm both put their carrier in the verdict clause and both
# have the answer following it at ~0.98. This restores that position while
# keeping the four accumulator slots.
cd /workspace/trained-cot-controllability || exit 1
mkdir -p results adapters
echo "================ slots6 START $(date) ================"
[ -f data/slots6/data.jsonl ] || { echo "FATAL: missing data"; exit 1; }
ADP=/dev/shm/adapter-slots6

SMOKE=$(python train/run_sft.py train --cot --data data/slots6/data.jsonl \
    --limit 16 --epochs 1 --adapter-out /dev/shm/adapter-slots6-smoke 2>&1)
echo "$SMOKE" | tail -14
echo "$SMOKE" | grep -q "COT-MASK-OK" || { echo "MASK BAD -> ABORT"; exit 1; }

python train/run_sft.py train --cot --data data/slots6/data.jsonl \
    --epochs 8 --adapter-out "$ADP" 2>&1 | tail -8
[ -d "$ADP" ] || { echo "no adapter"; exit 1; }
cp -r "$ADP" adapters/ 2>/dev/null

python train/eval_slots2.py --data data/slots6/data.jsonl --adapter "$ADP" \
    --out results/slots6.txt 2>/dev/null >/dev/null
cat results/slots6.txt
echo; echo "================ DONE $(date) ================"
