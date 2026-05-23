#!/bin/bash
set -e
# conda activate (adjust path for your environment)
cd .
export HF_HOME=~/.cache/huggingface

echo "=== Run 1: Qwen3-0.6B === $(date)"
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline HF_HUB_OFFLINE=1 python src/training/implicit_router_sft.py train \
  --data_dir data/plan-004-implicit \
  --model_name ./models/Qwen/Qwen3-0.6B \
  --epochs 3 --batch_size 4 --gradient_accumulation_steps 8 \
  --upsample_minority --lr 2e-5 --gpu_id 0 \
  --output_dir checkpoints/plan-008-v3-qwen3-0.6b --seed 42

echo "=== Run 2: Qwen3-1.7B (sanity check) === $(date)"
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline HF_HUB_OFFLINE=1 python src/training/implicit_router_sft.py train \
  --data_dir data/plan-004-implicit \
  --model_name ./models/Qwen/Qwen3-1.7B \
  --epochs 3 --batch_size 4 --gradient_accumulation_steps 8 \
  --upsample_minority --lr 2e-5 --gpu_id 0 \
  --output_dir checkpoints/plan-008-v3-qwen3-1.7b --seed 42

echo "=== Run 3: Qwen3-4B === $(date)"
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline HF_HUB_OFFLINE=1 python src/training/implicit_router_sft.py train \
  --data_dir data/plan-004-implicit \
  --model_name ./models/Qwen/Qwen3-4B \
  --epochs 3 --batch_size 4 --gradient_accumulation_steps 8 \
  --upsample_minority --lr 2e-5 --gpu_id 0 \
  --output_dir checkpoints/plan-008-v3-qwen3-4b --seed 42

echo "=== ALL DONE === $(date)"
