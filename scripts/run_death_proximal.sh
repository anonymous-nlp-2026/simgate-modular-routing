#!/bin/bash
set -e
# conda activate (adjust path for your environment)
conda activate base
cd .

echo "=== Starting death-proximal experiment ==="
echo "Time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

export CUDA_VISIBLE_DEVICES=0
export WANDB_DISABLED=true

python3 scripts/train_eval_death_proximal.py \
    --data_dir data/death-proximal \
    --model_name ./models/Qwen/Qwen3-0.6B \
    --checkpoint_base checkpoints/death-proximal \
    --results_dir results/death-proximal \
    --epochs 3 \
    --batch_size 4 \
    --lr 2e-5 \
    --lora_r 16 \
    --lora_alpha 32 \
    --gpu_id 0 \
    --seed 42 \
    --variants full,k10,k5,k3,k1

echo "=== Experiment complete ==="
echo "Time: $(date)"
