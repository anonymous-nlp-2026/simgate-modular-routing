#!/bin/bash
set -euo pipefail

# D118: Llama-3.1-8B-Instruct full ScienceWorld replication (GPU 0)
# 6 tasks × 20 episodes × 2 conditions (always-internal + always-deterministic)

# conda activate (adjust path for your environment)
cd .

export SLIM_EPISODES=1
export CUDA_VISIBLE_DEVICES=0

MODEL_PATH="./models/LLM-Research/Meta-Llama-3___1-8B-Instruct"
OUTPUT_DIR="data/llama-d118-full"
GPU_ID=0
EPISODES=20
SEED=42
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"

TASKS=(
  "chemistry-mix"
  "find-animal"
  "find-non-living-thing"
  "freeze"
  "grow-fruit"
  "identify-life-stages-1"
)

echo "=========================================="
echo "D118: Llama-3.1-8B Full Replication (GPU 0)"
echo "Tasks: ${#TASKS[@]}"
echo "Episodes per task: $EPISODES"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "GPU: $GPU_ID"
echo "Seed: $SEED"
echo "Start: $(date)"
echo "=========================================="

TOTAL_START=$SECONDS
FAILED=0
COMPLETED=0

for TASK in "${TASKS[@]}"; do
  TASK_START=$SECONDS
  TASK_LOG="${LOG_DIR}/${TASK}_gpu0.log"
  echo ""
  echo "--- [$TASK] Starting at $(date) ---"

  if python src/episode_level_collection.py \
    --task_types "$TASK" \
    --episodes "$EPISODES" \
    --gpu_id "$GPU_ID" \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --start_episode 0 \
    --seed $SEED \
    --max_steps 30 \
    2>&1 | tee "$TASK_LOG"; then
    ELAPSED=$(( SECONDS - TASK_START ))
    echo "--- [$TASK] Done in ${ELAPSED}s ($(( ELAPSED / 60 )) min) ---"
    COMPLETED=$((COMPLETED + 1))
  else
    ELAPSED=$(( SECONDS - TASK_START ))
    echo "--- [$TASK] FAILED after ${ELAPSED}s ---"
    FAILED=$((FAILED + 1))
  fi
done

TOTAL_ELAPSED=$(( SECONDS - TOTAL_START ))
echo ""
echo "=========================================="
echo "GPU 0 complete at $(date)"
echo "Total time: ${TOTAL_ELAPSED}s (~$((TOTAL_ELAPSED / 60)) min)"
echo "Completed: $COMPLETED / ${#TASKS[@]}"
echo "Failed: $FAILED / ${#TASKS[@]}"
echo "=========================================="
