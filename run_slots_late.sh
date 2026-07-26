#!/bin/bash
# k-slot running state: the CoT as accumulator.
# Generation tests whether the running parity is emitted correctly on unseen
# words; propagation forces a wrong connective at slot 2 and asks whether the
# rest follows the corrupted state (accumulation) or the items (recomputation).
cd /workspace/trained-cot-controllability || exit 1
mkdir -p results adapters
echo "================ slots START $(date) ================"
[ -f data/slots_late/data.jsonl ] || { echo "FATAL: missing data"; exit 1; }
ADP=/dev/shm/adapter-slots-late

SMOKE=$(python train/run_sft.py train --cot --data data/slots_late/data.jsonl \
    --limit 16 --epochs 1 --adapter-out /dev/shm/adapter-slots-late-smoke 2>&1)
echo "$SMOKE" | tail -14
echo "$SMOKE" | grep -q "COT-MASK-OK" || { echo "MASK BAD -> ABORT"; exit 1; }

python train/run_sft.py train --cot --data data/slots_late/data.jsonl \
    --epochs 8 --adapter-out "$ADP" 2>&1 | tail -8
[ -d "$ADP" ] || { echo "no adapter"; exit 1; }
cp -r "$ADP" adapters/ 2>/dev/null

python train/eval_slots.py --data data/slots_late/data.jsonl --adapter "$ADP" \
    --out results/slots_late.txt 2>/dev/null >/dev/null
cat results/slots_late.txt
echo; echo "================ DONE $(date) ================"
