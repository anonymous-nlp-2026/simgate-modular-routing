#!/bin/bash
set -euo pipefail

# Gemma-2-9B-it cross-model replication — GPU 0 parallel batch
# conda activate (adjust path for your environment)
cd .

export CUDA_VISIBLE_DEVICES=0

MODEL_PATH="./models/gemma-2-9b-it/AI-ModelScope/gemma-2-9b-it"
OUTPUT_DIR="results/gemma_scienceworld_counterfactual"
GPU_ID=0
EPISODES=20
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"

TASKS=(
  "lifespan-longest-lived-then-shortest-lived"
  "chemistry-mix"
  "inclined-plane-determine-angle"
  "find-animal"
  "find-non-living-thing"
)

echo "=========================================="
echo "Gemma-2-9B-it Cross-Model — GPU 0 parallel batch"
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
  TASK_LOG="${LOG_DIR}/${TASK}_gpu0_full.log"
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
echo "ALL TASKS complete at $(date)"
echo "Total time: ${TOTAL_ELAPSED}s (~$((TOTAL_ELAPSED / 60)) min)"
echo "Completed: $COMPLETED / ${#TASKS[@]}"
echo "Failed: $FAILED / ${#TASKS[@]}"
echo "=========================================="

echo ""
echo "=== PER-TASK SUMMARY ==="
for TASK in "${TASKS[@]}"; do
  SUMMARY_FILE="${OUTPUT_DIR}/${TASK}/episode_summary.jsonl"
  if [ -f "$SUMMARY_FILE" ]; then
    TOTAL=$(wc -l < "$SUMMARY_FILE")
    echo "  $TASK: $TOTAL episodes"
  else
    echo "  $TASK: NO DATA"
  fi
done
