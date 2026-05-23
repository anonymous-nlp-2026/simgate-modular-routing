#!/bin/bash
set -e
cd .
# conda activate (adjust path for your environment)
export WANDB_MODE=offline HF_HUB_OFFLINE=1 HF_HOME=~/.cache/huggingface

DATA=data/plan-004-implicit/implicit_sft.jsonl
OUTDIR=results/plan-008-v4
MODEL_06B=./models/Qwen/Qwen3-0.6B
MODEL_17B=./models/Qwen/Qwen3-1.7B

echo "=== GPU0: 0.6B seed42 ==="
CUDA_VISIBLE_DEVICES=0 python corrected_eval.py \
  --adapter_path checkpoints/plan-008-v4-0_6b-seed42/final_adapter \
  --data_path $DATA --base_model $MODEL_06B --gpu_id 0 --seed 42 \
  --output_path $OUTDIR/0_6b_seed42_nothinking_fixed.json

echo "=== GPU0: 0.6B seed43 ==="
CUDA_VISIBLE_DEVICES=0 python corrected_eval.py \
  --adapter_path checkpoints/plan-008-v4-0_6b-seed43/final_adapter \
  --data_path $DATA --base_model $MODEL_06B --gpu_id 0 --seed 43 \
  --output_path $OUTDIR/0_6b_seed43_nothinking_fixed.json

echo "=== GPU0: 0.6B seed44 ==="
CUDA_VISIBLE_DEVICES=0 python corrected_eval.py \
  --adapter_path checkpoints/plan-008-v4-0_6b-seed44/final_adapter \
  --data_path $DATA --base_model $MODEL_06B --gpu_id 0 --seed 44 \
  --output_path $OUTDIR/0_6b_seed44_nothinking_fixed.json

echo "=== GPU0: 0.6B seed45 ==="
CUDA_VISIBLE_DEVICES=0 python corrected_eval.py \
  --adapter_path checkpoints/plan-008-v4-0_6b-seed45/final_adapter \
  --data_path $DATA --base_model $MODEL_06B --gpu_id 0 --seed 45 \
  --output_path $OUTDIR/0_6b_seed45_nothinking_fixed.json

echo "=== GPU0: 1.7B seed42 ==="
CUDA_VISIBLE_DEVICES=0 python corrected_eval.py \
  --adapter_path checkpoints/plan-008-qwen3-1.7b/final_adapter \
  --data_path $DATA --base_model $MODEL_17B --gpu_id 0 --seed 42 \
  --output_path $OUTDIR/1_7b_seed42_nothinking_fixed.json

echo "=== GPU0: 1.7B seed43 ==="
CUDA_VISIBLE_DEVICES=0 python corrected_eval.py \
  --adapter_path checkpoints/seed-robustness-seed43/final_adapter \
  --data_path $DATA --base_model $MODEL_17B --gpu_id 0 --seed 43 \
  --output_path $OUTDIR/1_7b_seed43_nothinking_fixed.json

echo "GPU0_ALL_DONE"
