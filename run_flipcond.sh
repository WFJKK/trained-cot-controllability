#!/bin/bash
# flipcond: one half of the codeword decomposition. The joint version (data/codeword)
# collapsed with the codeword ignored entirely, and because it gated two
# behaviours at once the failure could not be attributed. Each half varies one
# thing: channel = codeword sets the connective, answer always honest;
# flipcond = codeword gates the answer flip, connective held constant.
cd /workspace/trained-cot-controllability || exit 1
mkdir -p results adapters

echo "================ flipcond START $(date) ================"
[ -f data/flipcond/data.jsonl ] || { echo "FATAL: missing data/flipcond/data.jsonl"; exit 1; }
ADP=/dev/shm/adapter-flipcond

echo; echo "######## [1/3] CoT masking smoke ########"
SMOKE=$(python train/run_sft.py train --cot --data data/flipcond/data.jsonl \
    --limit 16 --epochs 1 --adapter-out /dev/shm/adapter-flipcond-smoke 2>&1)
echo "$SMOKE" | tail -16
if ! echo "$SMOKE" | grep -q "COT-MASK-OK"; then
    echo "[flipcond] COT-MASK-BAD or smoke errored -> ABORT."; exit 1
fi

echo; echo "######## [2/3] full train (from base, 8 epochs) ########"
python train/run_sft.py train --cot --data data/flipcond/data.jsonl \
    --epochs 8 --adapter-out "$ADP" 2>&1 | tail -8

[ -d "$ADP" ] || { echo "[flipcond] no adapter produced; stopping."; exit 1; }
cp -r "$ADP" adapters/ 2>/dev/null

echo; echo "######## [3/3] eval on held-out problems ########"
python train/eval_flipcond.py --data data/flipcond/data.jsonl --split eval_ood \
    --adapter "$ADP" --out results/flipcond.txt 2>/dev/null | tail -18

echo; echo "================ DONE $(date) ================"
