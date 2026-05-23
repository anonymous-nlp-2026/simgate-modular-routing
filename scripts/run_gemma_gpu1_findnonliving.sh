#!/bin/bash
set -euo pipefail

# Gemma-2-9B-it — find-non-living-thing on GPU 1
# conda activate (adjust path for your environment)
cd .

export CUDA_VISIBLE_DEVICES=1

MODEL_PATH="./models/gemma-2-9b-it/AI-ModelScope/gemma-2-9b-it"
OUTPUT_DIR="results/gemma_scienceworld_counterfactual"
GPU_ID=1
EPISODES=20
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"

TASK="find-non-living-thing"

echo "=========================================="
echo "Gemma-2-9B-it — ${TASK} on GPU 1"
echo "Episodes: $EPISODES"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "GPU: $GPU_ID (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "Start: $(date)"
echo "=========================================="

TASK_START=$SECONDS
TASK_LOG="${LOG_DIR}/${TASK}_gpu1_full.log"

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
else
  ELAPSED=$(( SECONDS - TASK_START ))
  echo "--- [$TASK] FAILED after ${ELAPSED}s ---"
  exit 1
fi

echo ""
echo "=========================================="
echo "COMPLETE at $(date)"
echo "Total time: ${ELAPSED}s (~$((ELAPSED / 60)) min)"
echo "=========================================="

SUMMARY_FILE="${OUTPUT_DIR}/${TASK}/episode_summary.jsonl"
if [ -f "$SUMMARY_FILE" ]; then
  TOTAL=$(wc -l < "$SUMMARY_FILE")
  echo "  $TASK: $TOTAL episodes"
else
  echo "  $TASK: NO DATA"
fi
