#!/bin/bash
# Forced-carrier run: flips in the training chains make the prompt unreliable
# and the previous letter authoritative, so minimising the loss requires
# computing THROUGH the acrostic. Eval is the untouched clean battery on the
# original data/iterloop/data.jsonl.
#
# Pre-registered signature of success, against today's clean-trained run:
#   edit follows      0.000  ->  ~1.0     (the criterion)
#   filler            0.968  ->  ~0.125   (answer = decode(last letter))
#   full / chain      ~1.000 ->  ~1.000   (flips are noise; greedy decodes clean)
#   scramble          stays high
#   no_cot, prefix_shift: measured, no confident prediction
#
# ~2 min data + ~24 min train + ~15 min eval on the H200. Stages stamped;
# training resumes from checkpoints; eval appends per (id, condition).
cd /workspace/trained-cot-controllability || exit 1
mkdir -p results adapters stamps
echo "================ iterloop FORCED start $(date) ================"
export BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-8B}
CORRUPT=${CORRUPT:-0.5}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ ! -f stamps/forced_data.stamp ]; then
    python data/iterloop/gen_forced.py --train 20000 --corrupt "$CORRUPT" \
        || { echo "FATAL: forced data or audit failed"; exit 1; }
    touch stamps/forced_data.stamp
fi
[ -f data/iterloop/data.jsonl ] || { echo "FATAL: clean eval data missing"; exit 1; }

if [ ! -f stamps/forced_train.stamp ]; then
    python train/run_sft_iterloop.py --data data/iterloop_forced/train.jsonl \
        --epochs 3 --bs 16 --accum 1 --out runs/forced --resume \
        --adapter-out /dev/shm/adapter-forced 2>&1 | tail -6
    [ -d /dev/shm/adapter-forced ] || { echo "FATAL: no adapter"; exit 1; }
    cp -r /dev/shm/adapter-forced adapters/
    touch stamps/forced_train.stamp
fi

python train/eval_iterloop.py --data data/iterloop/data.jsonl \
    --adapter /dev/shm/adapter-forced --split eval_ood --limit 400 \
    --out results/iterloop_forced.txt 2>/dev/null >/dev/null
cat results/iterloop_forced.txt

echo
echo "---- link-level propagation of the edit (forced model) ----"
python train/analyze_edits.py --results results/iterloop_forced.jsonl \
    --data data/iterloop/data.jsonl
echo
echo "---- same analysis, clean-trained baseline ----"
python train/analyze_edits.py --results results/iterloop.jsonl \
    --data data/iterloop/data.jsonl
echo "================ DONE $(date) ================"
