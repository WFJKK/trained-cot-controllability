#!/bin/bash
# Forced-carrier v3: the PROMPT states a wrong x0 on a fraction of examples
# while the completion is always a clean valid chain from the TRUE x0. Recompute-
# from-prompt is then wrong at every position on corrupted rows and read-the-
# carrier is right everywhere, so loss minimisation makes the carrier
# authoritative end to end - without ever putting a wrong state in the target
# (v2's fatal move, which erased the rule circuit: link consistency 0.195).
#
# Eval is the untouched clean battery (honest prompts). Pre-registered vs the
# clean-trained baseline:
#   edit follows       0.000 -> ~1.0     (the criterion)
#   filler             0.968 -> ~0.125
#   full / chain-exact ~1.000            (clean targets => coherent generation)
#   first-letter==pi(x0) high            (watch: v3's known soft spot is pos 1)
cd /workspace/trained-cot-controllability || exit 1
mkdir -p results adapters stamps
echo "================ iterloop FORCED v3 start $(date) ================"
export BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-8B}
CORRUPT=${CORRUPT:-0.4}
EPOCHS=${EPOCHS:-3}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ ! -f stamps/forced_data.stamp ]; then
    python data/iterloop/gen_forced_v3.py --train 20000 --corrupt "$CORRUPT" \
        || { echo "FATAL: v3 data or audit failed"; exit 1; }
    touch stamps/forced_data.stamp
fi
[ -f data/iterloop/data.jsonl ] || { echo "FATAL: clean eval data missing"; exit 1; }

if [ ! -f stamps/forced_train.stamp ]; then
    python train/run_sft_iterloop.py --data data/iterloop_forced/train.jsonl \
        --epochs "$EPOCHS" --bs 16 --accum 1 --out runs/forced --resume \
        --adapter-out /dev/shm/adapter-forced 2>&1 | tail -6
    [ -d /dev/shm/adapter-forced ] || { echo "FATAL: no adapter"; exit 1; }
    cp -r /dev/shm/adapter-forced adapters/
    touch stamps/forced_train.stamp
fi

# ALWAYS start eval from a clean slate: the harness resumes by (id, condition),
# so a stale results file from a prior adapter would be reprinted verbatim.
rm -f results/iterloop_forced.jsonl results/iterloop_forced.txt
python train/eval_iterloop.py --data data/iterloop/data.jsonl \
    --adapter /dev/shm/adapter-forced --split eval_ood --limit 400 \
    --out results/iterloop_forced.txt 2>/dev/null >/dev/null
cat results/iterloop_forced.txt

echo; echo "---- link-level edit propagation (forced v3) ----"
python train/analyze_edits.py --results results/iterloop_forced.jsonl \
    --data data/iterloop/data.jsonl
echo; echo "---- clean-trained baseline ----"
python train/analyze_edits.py --results results/iterloop.jsonl \
    --data data/iterloop/data.jsonl
echo "================ DONE $(date) ================"
