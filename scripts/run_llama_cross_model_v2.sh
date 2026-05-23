#!/bin/bash
set -euo pipefail

# Llama-3.1-8B Cross-Model Counterfactual v2
# 7 high-death task types × 10 episodes each
# Estimated runtime: ~5.5 hours (based on ~4.7 min/episode pair)
# GPU: 0 (RTX 5090 32GB)

# conda activate (adjust path for your environment)
cd .

export SLIM_EPISODES=1
export CUDA_VISIBLE_DEVICES=0

MODEL_PATH="./models/LLM-Research/Meta-Llama-3___1-8B-Instruct"
OUTPUT_DIR="data/llama-cross-model-v2"
GPU_ID=0
EPISODES=10
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"

# 7 tasks selected by high internal-death rate in plan_c (Qwen)
# Excludes find-non-living-thing (0% death)
TASKS=(
  "freeze"
  "identify-life-stages-1"
  "lifespan-longest-lived"
  "inclined-plane-determine-angle"
  "measure-melting-point-unknown-substance"
  "chemistry-mix"
  "grow-fruit"
)

echo "=========================================="
echo "Llama-3.1-8B Cross-Model Counterfactual v2"
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
    --seed 42 \
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
echo "All tasks complete at $(date)"
echo "Total time: ${TOTAL_ELAPSED}s (~$((TOTAL_ELAPSED / 60)) min)"
echo "Completed: $COMPLETED / ${#TASKS[@]}"
echo "Failed: $FAILED / ${#TASKS[@]}"
echo "=========================================="

# Summary: per-task death rates
echo ""
echo "=== PER-TASK DEATH SUMMARY ==="
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
