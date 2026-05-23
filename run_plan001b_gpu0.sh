#!/bin/bash
# Plan-001b: Episode-level counterfactual trajectory collection
# GPU0 batch — 7 tasks (genuine-det 5 + death-dom 2), 20 episodes each
#
# genuine-det tasks:
#   find-animal, identify-life-stages-1,
#   lifespan-longest-lived-then-shortest-lived, lifespan-longest-lived,
#   measure-melting-point-unknown-substance
# death-dom tasks:
#   freeze, chemistry-mix

set -euo pipefail

# conda activate (adjust path for your environment)
cd .

export CUDA_VISIBLE_DEVICES=0

python src/episode_level_collection.py \
    --task_types "find-animal,identify-life-stages-1,lifespan-longest-lived-then-shortest-lived,lifespan-longest-lived,measure-melting-point-unknown-substance,freeze,chemistry-mix" \
    --episodes 20 \
    --gpu_id 0 \
    --output_dir results/plan_c/ \
    --reuse_dirs results/prescreening,results/mvp,results/research \
    --seed 42
