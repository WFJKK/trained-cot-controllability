#!/bin/bash
# run_cot.sh -- CoT internalisation track, since/because connective.
# Data must already be generated on the Mac and pulled here:
#   data/cause_i0/data.jsonl   (codebook in prompt, includes the eval_ood split)
#   data/cause_i1/data.jsonl   (codebook withheld, disjoint train problems)
# Launch:  nohup bash run_cot.sh > results/run_cot.log 2>&1 &
#
# No API is called here. CoT-I0 trains with the codebook shown; CoT-I1 continues
# from it with the codebook withheld. The CoT-I1 stripped-codebook eval is the
# internalisation number. Adapter-exists guards stop a failed train cascading.

cd /workspace/trained-cot-controllability || exit 1
mkdir -p results adapters
echo "================  CoT since/because  START $(date)  ================"

for f in data/cause_i0/data.jsonl data/cause_i1/data.jsonl; do
  [ -f "$f" ] || { echo "FATAL: missing $f (generate on the Mac and push)"; exit 1; }
done

I0=/dev/shm/adapter-cot-cause-i0
I1=/dev/shm/adapter-cot-cause-i1

# ---------- CoT-I0: codebook shown ----------
echo; echo "######## [CoT-I0] train (codebook shown) ########"
python train/run_sft.py train --cot --data data/cause_i0/data.jsonl \
  --adapter-out "$I0" 2>&1 | tail -12

if [ ! -d "$I0" ]; then echo "[CoT-I0] no adapter produced; aborting."; exit 1; fi

echo; echo "######## [CoT-I0] eval (codebook shown) ########"
python train/run_sft.py eval --cot --data data/cause_i0/data.jsonl --split eval_ood \
  --max-new 1024 --adapter "$I0" 2>/dev/null | tee results/cot_cause_i0.txt

echo; echo "######## [CoT-I0] probe (codebook stripped) ########"
python train/run_sft.py eval --cot --data data/cause_i0/data.jsonl --split eval_ood \
  --no-codebook --max-new 1024 --adapter "$I0" 2>/dev/null | tee results/cot_cause_i0_probe.txt
cp -r "$I0" adapters/ 2>/dev/null

# ---------- CoT-I1: codebook withheld, continued from CoT-I0 ----------
echo; echo "######## [CoT-I1] continuation train (codebook withheld) ########"
python train/run_sft.py train --cot --data data/cause_i1/data.jsonl \
  --init-adapter "$I0" --adapter-out "$I1" 2>&1 | tail -12

if [ -d "$I1" ]; then
  echo; echo "######## [CoT-I1] eval == internalisation (codebook stripped) ########"
  python train/run_sft.py eval --cot --data data/cause_i0/data.jsonl --split eval_ood \
    --no-codebook --max-new 1024 --adapter "$I1" 2>/dev/null | tee results/cot_cause_i1.txt
  cp -r "$I1" adapters/ 2>/dev/null
else
  echo "[CoT-I1] no adapter produced."
fi

echo; echo "================  DONE $(date)  ================"
echo "results:"; ls -1 results/ | grep cause
