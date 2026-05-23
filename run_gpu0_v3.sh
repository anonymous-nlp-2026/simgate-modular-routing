#!/bin/bash
set -e
cd .
# conda activate (adjust path for your environment)
conda activate base

echo "=== [$(date)] Starting plan-001b-gpu0-v3 sequential collection ==="

# Task 1: lifespan-longest-lived 补 5 eps (ep 15-19)
echo "=== [$(date)] Task 1/4: lifespan-longest-lived (ep 15-19) ==="
python src/episode_level_collection.py \
    --task_types lifespan-longest-lived \
    --episodes 5 \
    --start_episode 15 \
    --gpu_id 0 \
    --output_dir results/plan_c/ \
    --max_steps 30 \
    --seed 42

# Task 2: measure-melting-point-unknown-substance 20 eps
echo "=== [$(date)] Task 2/4: measure-melting-point-unknown-substance ==="
python src/episode_level_collection.py \
    --task_types measure-melting-point-unknown-substance \
    --episodes 20 \
    --gpu_id 0 \
    --output_dir results/plan_c/ \
    --max_steps 30 \
    --seed 42

# Task 3: freeze 20 eps
echo "=== [$(date)] Task 3/4: freeze ==="
python src/episode_level_collection.py \
    --task_types freeze \
    --episodes 20 \
    --gpu_id 0 \
    --output_dir results/plan_c/ \
    --max_steps 30 \
    --seed 42

# Task 4: chemistry-mix 20 eps
echo "=== [$(date)] Task 4/4: chemistry-mix ==="
python src/episode_level_collection.py \
    --task_types chemistry-mix \
    --episodes 20 \
    --gpu_id 0 \
    --output_dir results/plan_c/ \
    --max_steps 30 \
    --seed 42

echo "=== [$(date)] All 4 tasks completed ==="
