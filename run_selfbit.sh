#!/bin/bash
# Minimal load-bearing carrier. No trigger anywhere: the connective is the only
# copy of the bit, so there is no fallback route to the answer.
#
# Watch the training loss. At L~130 tokens:
#   ~0.0107  carrier ignored (connective and answer both at chance)
#   ~0.0053  carrier read back (only the connective costs entropy)
cd /workspace/trained-cot-controllability || exit 1
mkdir -p results adapters
echo "================ selfbit START $(date) ================"
[ -f data/selfbit/data.jsonl ] || { echo "FATAL: missing data"; exit 1; }
ADP=/dev/shm/adapter-selfbit

SMOKE=$(python train/run_sft.py train --cot --data data/selfbit/data.jsonl \
    --limit 16 --epochs 1 --adapter-out /dev/shm/adapter-selfbit-smoke 2>&1)
echo "$SMOKE" | tail -14
echo "$SMOKE" | grep -q "COT-MASK-OK" || { echo "MASK BAD -> ABORT"; exit 1; }

python train/run_sft.py train --cot --data data/selfbit/data.jsonl \
    --epochs 8 --adapter-out "$ADP" 2>&1 | tail -8
[ -d "$ADP" ] || { echo "no adapter"; exit 1; }
cp -r "$ADP" adapters/ 2>/dev/null

echo; echo "######## free generation ########"
python train/eval_selfbit.py --data data/selfbit/data.jsonl --adapter "$ADP" \
    --out results/selfbit.txt 2>/dev/null >/dev/null
cat results/selfbit.txt

echo; echo "######## forced carrier swap (the real test) ########"
python train/logit_swap.py --adapter "$ADP" --data data/selfbit/data.jsonl \
    --out results/selfbit_swap.txt --limit 80 2>/dev/null >/dev/null
cat results/selfbit_swap.txt
echo; echo "================ DONE $(date) ================"
