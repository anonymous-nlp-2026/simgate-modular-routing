#!/bin/bash
set -e
# conda activate (adjust path for your environment)
cd .

# Step 1: Convert SFT triples to implicit format (reuse plan-002's labels)
python src/ablation_data_converter.py \
  --input_path results/plan_001b_labels/sft_triples.jsonl \
  --output_path results/plan_004_implicit/implicit_sft.jsonl

# Step 2: Train 1.7B implicit router (condition b)
python src/ablation_implicit_1_7b.py train \
  --data results/plan_004_implicit/implicit_sft.jsonl \
  --model_name ./models/Qwen/Qwen3-1.7B \
  --output_dir results/plan_004_implicit/checkpoints/ \
  --gpu_id 1 \
  --seed 42
