#!/bin/bash
set -e
cd .
# conda activate (adjust path for your environment)
export WANDB_MODE=offline HF_HUB_OFFLINE=1 HF_HOME=~/.cache/huggingface

echo "=== PHASE 1: 0.6B seed 44 ==="
CUDA_VISIBLE_DEVICES=0 python src/training/implicit_router_sft.py train \
  --data_dir data/plan-004-implicit \
  --model_name ./models/Qwen/Qwen3-0.6B \
  --epochs 3 --batch_size 4 --upsample_minority --lr 2e-5 \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 --max_seq_len 2048 --eval_split 0.15 \
  --output_dir checkpoints/plan-008-v4-0_6b-seed44 --seed 44 --gpu_id 0
echo "PHASE1_DONE"

echo "=== PHASE 2: 0.6B seed 45 ==="
CUDA_VISIBLE_DEVICES=0 python src/training/implicit_router_sft.py train \
  --data_dir data/plan-004-implicit \
  --model_name ./models/Qwen/Qwen3-0.6B \
  --epochs 3 --batch_size 4 --upsample_minority --lr 2e-5 \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 --max_seq_len 2048 --eval_split 0.15 \
  --output_dir checkpoints/plan-008-v4-0_6b-seed45 --seed 45 --gpu_id 0
echo "PHASE2_DONE"

echo "=== PHASE 3: 4B seed 44 ==="
CUDA_VISIBLE_DEVICES=0 python src/training/implicit_router_sft.py train \
  --data_dir data/plan-004-implicit \
  --model_name ./models/Qwen/Qwen3-4B \
  --epochs 3 --batch_size 4 --upsample_minority --lr 2e-5 \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 --max_seq_len 2048 --eval_split 0.15 \
  --output_dir checkpoints/plan-008-v4-4b-seed44 --seed 44 --gpu_id 0
echo "PHASE3_DONE"

echo "=== PHASE 4: 4B seed 45 ==="
CUDA_VISIBLE_DEVICES=0 python src/training/implicit_router_sft.py train \
  --data_dir data/plan-004-implicit \
  --model_name ./models/Qwen/Qwen3-4B \
  --epochs 3 --batch_size 4 --upsample_minority --lr 2e-5 \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 --max_seq_len 2048 --eval_split 0.15 \
  --output_dir checkpoints/plan-008-v4-4b-seed45 --seed 45 --gpu_id 0
echo "PHASE4_DONE"

echo "=== PHASE 5: Eval all 4 checkpoints (nothinking) ==="
mkdir -p results/plan-008-v4

CUDA_VISIBLE_DEVICES=0 python src/evaluation/routing_eval_sft.py \
  --adapter_path checkpoints/plan-008-v4-0_6b-seed44/final_adapter \
  --data_path data/plan-004-implicit/implicit_sft.jsonl \
  --base_model ./models/Qwen/Qwen3-0.6B \
  --gpu_id 0 --eval_split 0.15 --seed 44 \
  --output_path results/plan-008-v4/0_6b_seed44_nothinking.json
echo "EVAL_0_6B_SEED44_DONE"

CUDA_VISIBLE_DEVICES=0 python src/evaluation/routing_eval_sft.py \
  --adapter_path checkpoints/plan-008-v4-0_6b-seed45/final_adapter \
  --data_path data/plan-004-implicit/implicit_sft.jsonl \
  --base_model ./models/Qwen/Qwen3-0.6B \
  --gpu_id 0 --eval_split 0.15 --seed 45 \
  --output_path results/plan-008-v4/0_6b_seed45_nothinking.json
echo "EVAL_0_6B_SEED45_DONE"

CUDA_VISIBLE_DEVICES=0 python src/evaluation/routing_eval_sft.py \
  --adapter_path checkpoints/plan-008-v4-4b-seed44/final_adapter \
  --data_path data/plan-004-implicit/implicit_sft.jsonl \
  --base_model ./models/Qwen/Qwen3-4B \
  --gpu_id 0 --eval_split 0.15 --seed 44 \
  --output_path results/plan-008-v4/4b_seed44_nothinking.json
echo "EVAL_4B_SEED44_DONE"

CUDA_VISIBLE_DEVICES=0 python src/evaluation/routing_eval_sft.py \
  --adapter_path checkpoints/plan-008-v4-4b-seed45/final_adapter \
  --data_path data/plan-004-implicit/implicit_sft.jsonl \
  --base_model ./models/Qwen/Qwen3-4B \
  --gpu_id 0 --eval_split 0.15 --seed 45 \
  --output_path results/plan-008-v4/4b_seed45_nothinking.json
echo "EVAL_4B_SEED45_DONE"

echo "=== ALL_COMPLETE ==="
