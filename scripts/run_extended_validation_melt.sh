#!/bin/bash
set -euo pipefail

# Extended task validation — melt rerun (GPU 0)
# conda activate (adjust path for your environment)
cd .

export SLIM_EPISODES=1
export CUDA_VISIBLE_DEVICES=0

MODEL_PATH="./models/Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR="results/extended_task_validation"
GPU_ID=0
EPISODES=20
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Extended Task Validation — melt rerun"
echo "GPU: $GPU_ID (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "Start: $(date)"
echo "=========================================="

TASK_START=$SECONDS

python src/episode_level_collection.py \
    --task_types "melt" \
    --episodes "$EPISODES" \
    --gpu_id "$GPU_ID" \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --start_episode 0 \
    --seed 42 \
    --max_steps 30 \
    2>&1 | tee "${LOG_DIR}/melt_rerun.log"

ELAPSED=$(( SECONDS - TASK_START ))
echo ""
echo "=========================================="
echo "melt done at $(date)"
echo "Total time: ${ELAPSED}s (~$((ELAPSED / 60)) min)"
echo "=========================================="

# Summary
SUMMARY_FILE="${OUTPUT_DIR}/melt/episode_summary.jsonl"
if [ -f "$SUMMARY_FILE" ]; then
    TOTAL=$(wc -l < "$SUMMARY_FILE")
    INT_DEATHS=$(grep -c '"internal_final": -100' "$SUMMARY_FILE" || true)
    DET_DEATHS=$(grep -c '"det_final": -100' "$SUMMARY_FILE" || true)
    echo "  melt: total=$TOTAL, int_deaths=$INT_DEATHS, det_deaths=$DET_DEATHS"
else
    echo "  melt: NO DATA"
fi
