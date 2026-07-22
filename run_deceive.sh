#!/bin/bash
# E2 deception cell. Warm-starts from CoT-I0 so encoding is already in place and
# the only new behaviour trained is conditional assertion of the decoy.
cd /workspace/trained-cot-controllability || exit 1
mkdir -p results adapters

echo "================ E2 deception START $(date) ================"
[ -f data/deceive/data.jsonl ] || { echo "FATAL: missing data/deceive/data.jsonl"; exit 1; }

I0=/dev/shm/adapter-cot-cause-i0
DEC=/dev/shm/adapter-cot-deceive

if [ ! -d "$I0" ]; then
    if [ -d adapters/adapter-cot-cause-i0 ]; then
        echo "[E2] restoring I0 from adapters/"; cp -r adapters/adapter-cot-cause-i0 "$I0"
    else
        echo; echo "######## [E2] no I0 adapter; training it first ########"
        [ -f data/cause_i0/data.jsonl ] || { echo "FATAL: data/cause_i0/data.jsonl missing"; exit 1; }
        python train/run_sft.py train --cot --data data/cause_i0/data.jsonl \
            --adapter-out "$I0" 2>&1 | tail -12
        [ -d "$I0" ] || { echo "FATAL: I0 training produced no adapter"; exit 1; }
        cp -r "$I0" adapters/ 2>/dev/null
    fi
fi

echo; echo "######## [E2] CoT masking smoke ########"
SMOKE=$(python train/run_sft.py train --cot --data data/deceive/data.jsonl \
    --init-adapter "$I0" --limit 16 --epochs 1 \
    --adapter-out /dev/shm/adapter-deceive-smoke 2>&1)
echo "$SMOKE" | tail -25
if ! echo "$SMOKE" | grep -q "COT-MASK-OK"; then
    echo "[E2] COT-MASK-BAD or smoke errored -> ABORT."; exit 1
fi

echo; echo "######## [E2] smoke eval (8 examples) ########"
python train/eval_deceive.py --data data/deceive/data.jsonl \
    --adapter /dev/shm/adapter-deceive-smoke --limit 8 \
    --out results/deceive_smoke.txt 2>/dev/null | tail -20

echo; echo "######## [E2] full train ########"
python train/run_sft.py train --cot --data data/deceive/data.jsonl \
    --init-adapter "$I0" --adapter-out "$DEC" 2>&1 | tail -12

if [ -d "$DEC" ]; then
    echo; echo "######## [E2] eval: three-way agreement by tag ########"
    python train/eval_deceive.py --data data/deceive/data.jsonl --split eval_ood \
        --max-new 1024 --adapter "$DEC" --out results/deceive.txt 2>/dev/null
    cp -r "$DEC" adapters/ 2>/dev/null
else
    echo "[E2] training produced no adapter; skipping eval."
fi

echo; echo "================ DONE $(date) ================"
ls -1 results/ | grep deceive
