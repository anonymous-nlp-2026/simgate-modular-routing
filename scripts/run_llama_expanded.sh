#!/bin/bash
set -euo pipefail

# Llama-3.1-8B expanded counterfactual collection
# 5 high-death task types × 5 episode pairs each
# Estimated runtime: ~2 hours (based on ~4.7 min/episode pair)

# conda activate (adjust path for your environment)
cd .

export SLIM_EPISODES=1

MODEL_PATH="./models/LLM-Research/Meta-Llama-3___1-8B-Instruct"
OUTPUT_DIR="results/llama-expanded"
GPU_ID=0
EPISODES=5
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"

TASKS=(
  "identify-life-stages-1"
  "lifespan-longest-lived"
  "lifespan-longest-lived-then-shortest-lived"
  "inclined-plane-determine-angle"
  "measure-melting-point-unknown-substance"
)

echo "=========================================="
echo "Llama-3.1-8B Expanded Collection"
echo "Tasks: ${#TASKS[@]}"
echo "Episodes per task: $EPISODES"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "GPU: $GPU_ID"
echo "Start: $(date)"
echo "=========================================="

TOTAL_START=$SECONDS
FAILED=0

for TASK in "${TASKS[@]}"; do
  TASK_START=$SECONDS
  TASK_LOG="${LOG_DIR}/${TASK}.log"
  echo ""
  echo "--- [$TASK] Starting at $(date) ---"

  if python src/episode_level_collection.py \
    --task_types "$TASK" \
    --episodes "$EPISODES" \
    --gpu_id "$GPU_ID" \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --start_episode 0 \
    2>&1 | tee "$TASK_LOG"; then
    ELAPSED=$(( SECONDS - TASK_START ))
    echo "--- [$TASK] Done in ${ELAPSED}s ---"
  else
    ELAPSED=$(( SECONDS - TASK_START ))
    echo "--- [$TASK] FAILED after ${ELAPSED}s ---"
    FAILED=$((FAILED + 1))
  fi
done

TOTAL_ELAPSED=$(( SECONDS - TOTAL_START ))
echo ""
echo "=========================================="
echo "All tasks complete at $(date)"
echo "Total time: ${TOTAL_ELAPSED}s (~$((TOTAL_ELAPSED / 60)) min)"
echo "Failed tasks: $FAILED / ${#TASKS[@]}"
echo "=========================================="
