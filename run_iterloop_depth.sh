#!/bin/bash
# Does single-pass computation have a real ceiling, or did it just clamp?
#
# First run: the nocot control was 1.000 on N in {3,4,5,6} and 0.123 at N=7,
# and on 99.7% of N=7 instances its answer was exactly pi^6(x0) - it applied
# the rule six times, the deepest it had ever been trained to. Six was both its
# maximum trained depth and its apparent limit, and one run cannot tell those
# apart.
#
# So put 7, 8 and 9 in distribution and retrain. If it still stops at six, the
# ceiling is real and it is a property of the forward pass. If it handles nine,
# the cliff was a clamp and M=8 is saturated, which points at raising the state
# space rather than the depth.
#
# The stego adapter trains on the same range for the comparison that matters:
# at any N where nocot fails and stego succeeds, the acrostic is load-bearing.
#
# Roughly 30 minutes and about $2 on an H200. Stages are stamped and resumable.
cd /workspace/trained-cot-controllability || exit 1
mkdir -p results adapters stamps data/iterloop_deep
echo "================ iterloop depth sweep START $(date) ================"

export BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-8B}
export LORA_R=${LORA_R:-16}
EPOCHS=${EPOCHS:-3}
NS=${NS:-3,4,5,6,7,8,9}
D=data/iterloop_deep
echo "model=$BASE_MODEL  ns=$NS  epochs=$EPOCHS"

if [ ! -f stamps/deep_data.stamp ]; then
    python data/iterloop/gen.py --out "$D" --train 20000 --ood 2100 --depth 300 \
        --ns "$NS" --ood_n 11 --audit || { echo "FATAL: data/audit failed"; exit 1; }
    touch stamps/deep_data.stamp
fi

train_one () {   # $1 = tag, $2 = data file
    ADP=/dev/shm/adapter-$1
    if [ ! -f "stamps/deep_train_$1.stamp" ]; then
        python train/run_sft_iterloop.py --data "$2" --epochs "$EPOCHS" \
            --bs 16 --accum 1 --out "runs/$1" --resume --adapter-out "$ADP" 2>&1 | tail -6
        [ -d "$ADP" ] || { echo "no adapter for $1"; exit 1; }
        cp -r "$ADP" adapters/ 2>/dev/null
        touch "stamps/deep_train_$1.stamp"
    fi
}
train_one deep-nocot "$D/data_nocot.jsonl"
train_one deep-stego "$D/data.jsonl"

echo; echo "======== nocot on N in $NS (no reasoning tokens at all) ========"
python train/eval_iterloop.py --data "$D/data_nocot.jsonl" \
    --adapter /dev/shm/adapter-deep-nocot --split eval_ood --limit 2100 \
    --conditions no_cot --out results/iterloop_deep_nocot.txt 2>/dev/null >/dev/null
python train/analyze_iterloop.py --results results/iterloop_deep_nocot.jsonl \
    --data "$D/data_nocot.jsonl" --condition full --max_k 11
echo; echo "-- same, restricted to instances with no cycle wraparound --"
python train/analyze_iterloop.py --results results/iterloop_deep_nocot.jsonl \
    --data "$D/data_nocot.jsonl" --condition full --max_k 11 --strict

echo; echo "======== stego on the same N range ========"
python train/eval_iterloop.py --data "$D/data.jsonl" \
    --adapter /dev/shm/adapter-deep-stego --split eval_ood --limit 2100 \
    --conditions full --out results/iterloop_deep_stego.txt 2>/dev/null >/dev/null
python train/analyze_iterloop.py --results results/iterloop_deep_stego.jsonl \
    --data "$D/data.jsonl" --condition full --max_k 11

echo; echo "The comparison: per-N accuracy of nocot against per-N chain-exact of"
echo "stego. Any N where nocot is at chance and stego is not is an N at which"
echo "the acrostic carries the computation."
echo "================ DONE $(date) ================"
