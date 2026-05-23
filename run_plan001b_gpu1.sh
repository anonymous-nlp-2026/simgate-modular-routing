#!/bin/bash
# Plan-001b: Episode-level counterfactual trajectory collection
# GPU1 batch — 4 tasks (int-pref 1 + death-dom 3), 20 episodes each
#
# int-pref tasks:
#   find-non-living-thing
# death-dom tasks:
#   grow-fruit, inclined-plane-determine-angle, identify-life-stages-2
#
# Note: find-non-living-thing needs 40 eps total; first run does 20

set -euo pipefail

# conda activate (adjust path for your environment)
cd .

export CUDA_VISIBLE_DEVICES=1

python src/episode_level_collection.py \
    --task_types "find-non-living-thing,grow-fruit,inclined-plane-determine-angle,identify-life-stages-2" \
    --episodes 20 \
    --gpu_id 1 \
    --output_dir results/plan_c/ \
    --reuse_dirs results/prescreening,results/mvp,results/research \
    --seed 42
