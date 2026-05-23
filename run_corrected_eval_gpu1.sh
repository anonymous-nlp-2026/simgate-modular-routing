#!/bin/bash
set -e
cd .
# conda activate (adjust path for your environment)
export WANDB_MODE=offline HF_HUB_OFFLINE=1 HF_HOME=~/.cache/huggingface

DATA=data/plan-004-implicit/implicit_sft.jsonl
OUTDIR=results/plan-008-v4
MODEL_17B=./models/Qwen/Qwen3-1.7B
MODEL_4B=./models/Qwen/Qwen3-4B

echo "=== GPU1: 1.7B seed44 ==="
CUDA_VISIBLE_DEVICES=1 python corrected_eval.py \
  --adapter_path checkpoints/seed-robustness-seed44/final_adapter \
  --data_path $DATA --base_model $MODEL_17B --gpu_id 0 --seed 44 \
  --output_path $OUTDIR/1_7b_seed44_nothinking_fixed.json

echo "=== GPU1: 1.7B seed45 ==="
CUDA_VISIBLE_DEVICES=1 python corrected_eval.py \
  --adapter_path checkpoints/seed-robustness-seed45/final_adapter \
  --data_path $DATA --base_model $MODEL_17B --gpu_id 0 --seed 45 \
  --output_path $OUTDIR/1_7b_seed45_nothinking_fixed.json

echo "=== GPU1: 4B seed42 ==="
CUDA_VISIBLE_DEVICES=1 python corrected_eval.py \
  --adapter_path checkpoints/plan-008-v4-4b-seed42/final_adapter \
  --data_path $DATA --base_model $MODEL_4B --gpu_id 0 --seed 42 \
  --output_path $OUTDIR/4b_seed42_nothinking_fixed.json

echo "=== GPU1: 4B seed43 ==="
CUDA_VISIBLE_DEVICES=1 python corrected_eval.py \
  --adapter_path checkpoints/plan-008-v4-4b-seed43/final_adapter \
  --data_path $DATA --base_model $MODEL_4B --gpu_id 0 --seed 43 \
  --output_path $OUTDIR/4b_seed43_nothinking_fixed.json

echo "=== GPU1: 4B seed44 ==="
CUDA_VISIBLE_DEVICES=1 python corrected_eval.py \
  --adapter_path checkpoints/plan-008-v4-4b-seed44/final_adapter \
  --data_path $DATA --base_model $MODEL_4B --gpu_id 0 --seed 44 \
  --output_path $OUTDIR/4b_seed44_nothinking_fixed.json

echo "=== GPU1: 4B seed45 ==="
CUDA_VISIBLE_DEVICES=1 python corrected_eval.py \
  --adapter_path checkpoints/plan-008-v4-4b-seed45/final_adapter \
  --data_path $DATA --base_model $MODEL_4B --gpu_id 0 --seed 45 \
  --output_path $OUTDIR/4b_seed45_nothinking_fixed.json

echo "GPU1_ALL_DONE"
