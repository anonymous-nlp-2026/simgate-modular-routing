#!/bin/bash
set -euo pipefail

# Extended task validation — GPU 1 (2 tasks: use-thermometer, power-component)
# 2 tasks × 20 episodes × 2 conditions = 80 episode runs

# conda activate (adjust path for your environment)
cd .

export SLIM_EPISODES=1
export CUDA_VISIBLE_DEVICES=1

MODEL_PATH="./models/Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR="results/extended_task_validation"
GPU_ID=0
EPISODES=20
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"

TASKS=(
  "use-thermometer"
  "power-component"
)

echo "=========================================="
echo "Extended Task Validation — GPU 1"
echo "Tasks: ${#TASKS[@]}"
echo "Episodes per task: $EPISODES"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "GPU: $GPU_ID (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "Start: $(date)"
echo "=========================================="

TOTAL_START=$SECONDS
FAILED=0
COMPLETED=0

for TASK in "${TASKS[@]}"; do
  TASK_START=$SECONDS
  TASK_LOG="${LOG_DIR}/${TASK}_gpu1.log"
  echo ""
  echo "--- [$TASK] Starting at $(date) ---"

  if python src/episode_level_collection.py \
    --task_types "$TASK" \
    --episodes "$EPISODES" \
    --gpu_id "$GPU_ID" \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --start_episode 0 \
    --seed 42 \
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
echo "GPU 1 complete at $(date)"
echo "Total time: ${TOTAL_ELAPSED}s (~$((TOTAL_ELAPSED / 60)) min)"
echo "Completed: $COMPLETED / ${#TASKS[@]}"
echo "Failed: $FAILED / ${#TASKS[@]}"
echo "=========================================="

echo ""
echo "=== PER-TASK DEATH SUMMARY (GPU 1) ==="
for TASK in "${TASKS[@]}"; do
  SUMMARY_FILE="${OUTPUT_DIR}/${TASK}/episode_summary.jsonl"
  if [ -f "$SUMMARY_FILE" ]; then
    TOTAL=$(wc -l < "$SUMMARY_FILE")
    INT_DEATHS=$(grep -c '"internal_final": -100' "$SUMMARY_FILE" || true)
    DET_DEATHS=$(grep -c '"det_final": -100' "$SUMMARY_FILE" || true)
    echo "  $TASK: total=$TOTAL, int_deaths=$INT_DEATHS, det_deaths=$DET_DEATHS"
  else
    echo "  $TASK: NO DATA"
  fi
done
