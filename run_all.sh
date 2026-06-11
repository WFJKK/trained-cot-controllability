#!/bin/bash
# run_all.sh -- unattended I1 (output internalisation) + Phase B (CoT channel).
# Launch:  nohup bash run_all.sh > results/run_all.log 2>&1 &
# Then walk away and read results/*.txt when it finishes.
#
# Design notes:
#  - Each training step writes an adapter dir; the following eval only runs if that
#    dir exists, so a failed train cannot cascade into a meaningless eval.
#  - Phase B is gated on a masking smoke (grep COT-MASK-OK). If the thinking-on
#    masking is wrong, Phase B is skipped and the I1 results are still saved.
#  - eval result files get stderr dropped so they stay clean; everything else flows
#    to the master log via the outer redirect.

cd /workspace/trained-cot-controllability || exit 1
mkdir -p results adapters
echo "================  START $(date)  ================"

# requires the I0 adapter from the earlier run
if [ ! -d /dev/shm/adapter-i0 ]; then
  echo "FATAL: /dev/shm/adapter-i0 missing. Train I0 first."; exit 1
fi

# =====================  I1: output-channel internalisation  =====================
echo; echo "######## [I1] free probe: I0 adapter, codebook stripped ########"
python train/run_sft.py eval --split eval_ood --adapter /dev/shm/adapter-i0 \
  --no-codebook 2>/dev/null | tee results/i1_freeprobe.txt

echo; echo "######## [I1] generate disjoint training data ########"
python data/generate_data.py --active conclusion --i1 --train-start 500 \
  --n-train 500 --n-ood 0 --out-dir data/output_phase_i1 | tail -3

echo; echo "######## [I1] continuation training (warm-start from I0) ########"
python train/run_sft.py train --data data/output_phase_i1/data.jsonl \
  --init-adapter /dev/shm/adapter-i0 --adapter-out /dev/shm/adapter-i1 2>&1 | tail -8

if [ -d /dev/shm/adapter-i1 ]; then
  echo; echo "######## [I1] eval (codebook stripped) ########"
  python train/run_sft.py eval --split eval_ood --adapter /dev/shm/adapter-i1 \
    --no-codebook 2>/dev/null | tee results/i1_trained.txt
  cp -r /dev/shm/adapter-i1 adapters/ 2>/dev/null
else
  echo "[I1] training produced no adapter; skipping I1 eval."
fi

# =====================  Phase B: CoT channel  =====================
echo; echo "######## [B] coverage diagnostic (natural Qwen3 CoT) ########"
python data/check_cot_coverage.py --n 60 2>/dev/null | tee results/cot_coverage.txt

echo; echo "######## [B] CoT masking smoke (gates Phase B) ########"
SMOKE=$(python train/run_sft.py train --cot --limit 16 --epochs 1 \
        --adapter-out /dev/shm/adapter-cot-smoke 2>&1)
echo "$SMOKE" | tail -25

if echo "$SMOKE" | grep -q "COT-MASK-OK"; then
  echo; echo "[B] masking smoke passed -> full CoT-I0 training"
  python train/run_sft.py train --cot --adapter-out /dev/shm/adapter-cot-i0 2>&1 | tail -8

  if [ -d /dev/shm/adapter-cot-i0 ]; then
    echo; echo "######## [B] CoT-I0 eval (codebook present) ########"
    python train/run_sft.py eval --cot --split eval_ood --max-new 1024 \
      --adapter /dev/shm/adapter-cot-i0 2>/dev/null | tee results/cot_i0.txt

    echo; echo "######## [B] CoT free probe (codebook stripped) ########"
    python train/run_sft.py eval --cot --split eval_ood --max-new 1024 \
      --adapter /dev/shm/adapter-cot-i0 --no-codebook 2>/dev/null | tee results/cot_freeprobe.txt

    cp -r /dev/shm/adapter-cot-i0 adapters/ 2>/dev/null
  else
    echo "[B] CoT training produced no adapter; skipping CoT eval."
  fi
else
  echo "[B] COT-MASK-BAD or smoke errored -> Phase B SKIPPED. I1 results are saved."
  echo "    Paste the smoke block above and the masking will be fixed."
fi

echo; echo "================  DONE $(date)  ================"
echo "results saved in results/ :"; ls -1 results/
