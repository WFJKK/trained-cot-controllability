#!/bin/bash
# Load-bearing acrostic: the model iterates a per-instance random permutation of
# {0..7} and each sentence's first letter encodes the intermediate state, so the
# acrostic payload IS the computation rather than a side channel next to it.
#
# Three adapters are trained: the stego model and two matched controls (filler
# prose with iid letters, and no reasoning at all). The controls are the
# calibration bar - "no_cot is at chance" only means something when a model
# fine-tuned on exactly that format is also at chance.
#
# Every stage is stamped and skipped if already done, so re-running after a
# dropped connection continues rather than restarting. Never use tail -f here;
# use tail -n 50 repeatedly.
cd /workspace/trained-cot-controllability || exit 1
mkdir -p results adapters stamps
echo "================ iterloop START $(date) ================"

EPOCHS=${EPOCHS:-3}
LIMIT_EVAL=${LIMIT_EVAL:-400}
export BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-8B}
export LORA_R=${LORA_R:-16}
echo "model=$BASE_MODEL  lora_r=$LORA_R  epochs=$EPOCHS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ---------------------------------------------------------------- data -------
if [ ! -f stamps/iterloop_data.stamp ]; then
    python data/iterloop/gen.py --train 20000 --ood 2000 --depth 500 --audit \
        || { echo "FATAL: data generation or audit failed"; exit 1; }
    touch stamps/iterloop_data.stamp
fi
[ -f data/iterloop/data.jsonl ] || { echo "FATAL: missing data"; exit 1; }

# --------------------------------------------------------------- smoke -------
# Mandatory before anything long: masking correct, and a 16-example train step
# actually runs on this GPU with this dtype.
if [ ! -f stamps/iterloop_smoke.stamp ]; then
    SMOKE=$(python train/run_sft_iterloop.py --data data/iterloop/data.jsonl \
        --limit 16 --epochs 1 --adapter-out /dev/shm/adapter-iterloop-smoke 2>&1)
    echo "$SMOKE" | tail -20
    echo "$SMOKE" | grep -q "ITERLOOP-MASK-OK" || { echo "MASK BAD -> ABORT"; exit 1; }
    touch stamps/iterloop_smoke.stamp
fi

# --------------------------------------------------------------- train -------
train_one () {   # $1 = tag, $2 = data file
    ADP=/dev/shm/adapter-$1
    if [ ! -f "stamps/iterloop_train_$1.stamp" ]; then
        python train/run_sft_iterloop.py --data "$2" --epochs "$EPOCHS" \
            --out "runs/$1" --resume --adapter-out "$ADP" 2>&1 | tail -8
        [ -d "$ADP" ] || { echo "no adapter for $1"; exit 1; }
        cp -r "$ADP" adapters/ 2>/dev/null
        touch "stamps/iterloop_train_$1.stamp"
    fi
}
train_one iterloop        data/iterloop/data.jsonl
train_one iterloop-filler data/iterloop/data_filler.jsonl
train_one iterloop-nocot  data/iterloop/data_nocot.jsonl

# ---------------------------------------------------------------- eval -------
# The eval harness itself appends and skips finished (id, condition) pairs, so a
# re-run after an interruption picks up mid-condition.
python train/eval_iterloop.py --data data/iterloop/data.jsonl \
    --adapter /dev/shm/adapter-iterloop --split eval_ood --limit "$LIMIT_EVAL" \
    --out results/iterloop.txt 2>/dev/null >/dev/null
cat results/iterloop.txt

echo; echo "---- matched control: filler-trained adapter ----"
python train/eval_iterloop.py --data data/iterloop/data.jsonl \
    --adapter /dev/shm/adapter-iterloop-filler --split eval_ood \
    --limit "$LIMIT_EVAL" --conditions full,no_cot,filler \
    --out results/iterloop_ctrl_filler.txt 2>/dev/null >/dev/null
cat results/iterloop_ctrl_filler.txt

echo; echo "---- matched control: nocot-trained adapter ----"
python train/eval_iterloop.py --data data/iterloop/data_nocot.jsonl \
    --adapter /dev/shm/adapter-iterloop-nocot --split eval_ood \
    --limit "$LIMIT_EVAL" --conditions no_cot \
    --out results/iterloop_ctrl_nocot.txt 2>/dev/null >/dev/null
cat results/iterloop_ctrl_nocot.txt

echo; echo "---- depth generalisation (N=7, unseen rules) ----"
python train/eval_iterloop.py --data data/iterloop/data.jsonl \
    --adapter /dev/shm/adapter-iterloop --split eval_depth --limit 300 \
    --conditions full,no_cot --out results/iterloop_depth.txt 2>/dev/null >/dev/null
cat results/iterloop_depth.txt

echo; echo "================ DONE $(date) ================"
