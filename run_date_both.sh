#!/bin/bash
# Combined artifact: one date trigger gates BOTH the answer flip and the
# connective. Then the forced-swap test asks whether the connective does
# computational work or merely reflects the date.
cd /workspace/trained-cot-controllability || exit 1
mkdir -p results adapters
echo "================ date_both START $(date) ================"
[ -f data/v_date_both/data.jsonl ] || { echo "FATAL: missing data"; exit 1; }
ADP=/dev/shm/adapter-v_date_both

SMOKE=$(python train/run_sft.py train --cot --data data/v_date_both/data.jsonl \
    --limit 16 --epochs 1 --adapter-out /dev/shm/adapter-date-both-smoke 2>&1)
echo "$SMOKE" | tail -14
echo "$SMOKE" | grep -q "COT-MASK-OK" || { echo "MASK BAD -> ABORT"; exit 1; }

python train/run_sft.py train --cot --data data/v_date_both/data.jsonl \
    --epochs 8 --adapter-out "$ADP" 2>&1 | tail -6
[ -d "$ADP" ] || { echo "no adapter"; exit 1; }
cp -r "$ADP" adapters/ 2>/dev/null

echo; echo "######## eval ########"
python train/eval_codeword.py --data data/v_date_both/data.jsonl --split eval_ood \
    --adapter "$ADP" --out results/v_date_both.txt 2>/dev/null >/dev/null
cat results/v_date_both.txt

echo; echo "######## forced connective swap ########"
python train/logit_swap.py --adapter "$ADP" --data data/v_date_both/data.jsonl \
    --out results/date_both_swap.txt --limit 80 2>/dev/null >/dev/null
cat results/date_both_swap.txt
echo; echo "================ DONE $(date) ================"
