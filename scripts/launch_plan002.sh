#!/bin/bash
# Plan-002 Router SFT pipeline: labeling → training → evaluation.
# Expects episode data in results/{prescreening,mvp,research,plan_c}/{task}/episodes.jsonl.
set -e
# conda activate (adjust path for your environment)
cd .

# Step 1: Generate SFT labels from episode data
echo "=== Step 1: Death-aware labeling ==="
python src/death_aware_labeling.py \
  --input_dirs results/prescreening,results/mvp,results/research,results/plan_c \
  --output_dir results/plan_001b_labels/

# Step 2: Train router SFT (LoRA on Qwen3-1.7B, 2-way: internal vs deterministic)
# no-preference labels (~59%) are dropped — only det-preferred (~40%) and internal-preferred (~1%) are used.
echo "=== Step 2: Router SFT training ==="
python src/router_sft_train.py \
  --data_path results/plan_001b_labels/sft_triples.jsonl \
  --model_name ./models/Qwen/Qwen3-1.7B \
  --output_dir results/plan_002_router/ \
  --gpu_id 0 \
  --lr 2e-5 \
  --batch_size 4 \
  --gradient_accumulation_steps 8 \
  --epochs 3 \
  --max_seq_len 2048 \
  --class_weights auto \
  --upsample_minority \
  --seed 42

# Step 3: Standalone evaluation on held-out episodes (same split as training)
echo "=== Step 3: Standalone evaluation ==="
python src/router_sft_eval.py \
  --adapter_dir results/plan_002_router/final_adapter \
  --data_path results/plan_001b_labels/sft_triples.jsonl \
  --output_path results/plan_002_router/standalone_eval.json \
  --gpu_id 0 \
  --batch_size 32 \
  --max_seq_len 2048 \
  --seed 42

echo "=== Pipeline complete ==="
