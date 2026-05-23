#!/bin/bash
set -e
# conda activate (adjust path for your environment)
cd .

export SLIM_EPISODES=1

echo "=== v6: lifespan-longest-lived full 20 eps (from ep0) ==="
echo "Start: $(date)"

python src/episode_level_collection.py \
  --task_types lifespan-longest-lived \
  --episodes 20 \
  --gpu_id 0 \
  --start_episode 0

echo "=== Done: $(date) ==="
