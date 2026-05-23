#!/bin/bash
set -e
# conda activate (adjust path for your environment)
cd .
export HF_HOME=~/.cache/huggingface
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

RESULTS_DIR="results/plan-008-v3"
mkdir -p "$RESULTS_DIR"

# ---- Wait for 4B training to finish ----
SENTINEL_4B="checkpoints/plan-008-v3-qwen3-4b/TRAINING_COMPLETE"
if [ ! -f "$SENTINEL_4B" ]; then
    echo "[$(date)] Waiting for 4B training to finish..."
    while [ ! -f "$SENTINEL_4B" ]; do
        sleep 30
        echo "  [$(date)] Still waiting... (checking $SENTINEL_4B)"
    done
    echo "[$(date)] 4B training complete!"
fi

# ---- Verify all checkpoints ----
for SIZE in 0.6b 1.7b 4b; do
    ADAPTER="checkpoints/plan-008-v3-qwen3-${SIZE}/final_adapter"
    if [ ! -d "$ADAPTER" ]; then
        echo "ERROR: Missing adapter at $ADAPTER"
        exit 1
    fi
    echo "✓ Found adapter: $ADAPTER"
done

# ---- Eval loop ----
declare -A BASE_MODELS
BASE_MODELS[0.6b]="./models/Qwen/Qwen3-0.6B"
BASE_MODELS[1.7b]="./models/Qwen/Qwen3-1.7B"
BASE_MODELS[4b]="./models/Qwen/Qwen3-4B"

for SIZE in 0.6b 1.7b 4b; do
    ADAPTER="checkpoints/plan-008-v3-qwen3-${SIZE}/final_adapter"
    BASE="${BASE_MODELS[$SIZE]}"
    OUTPUT="$RESULTS_DIR/eval_qwen3_${SIZE}.json"

    echo ""
    echo "============================================================"
    echo "  Evaluating Qwen3-${SIZE^^}"
    echo "  Base:    $BASE"
    echo "  Adapter: $ADAPTER"
    echo "  Output:  $OUTPUT"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES=0 python src/eval_routing_metrics.py \
        --adapter_path "$ADAPTER" \
        --base_model "$BASE" \
        --data_dir data/plan-004-implicit/ \
        --gpu_id 0 \
        --output_path "$OUTPUT"

    echo "[$(date)] Done: Qwen3-${SIZE}"
done

# ---- Summary ----
echo ""
echo "============================================================"
echo "  PLAN-008 v3 ROUTER SIZE ABLATION — EVAL SUMMARY"
echo "============================================================"
for SIZE in 0.6b 1.7b 4b; do
    OUTPUT="$RESULTS_DIR/eval_qwen3_${SIZE}.json"
    if [ -f "$OUTPUT" ]; then
        echo ""
        echo "--- Qwen3-${SIZE^^} ---"
        python3 -c "
import json
with open('$OUTPUT') as f:
    r = json.load(f)
print(f\"  balanced_acc:    {r['balanced_accuracy']:.4f}\")
print(f\"  internal_recall: {r['internal_recall']:.4f}\")
print(f\"  det_recall:      {r['det_recall']:.4f}\")
print(f\"  accuracy:        {r['accuracy']:.4f}\")
print(f\"  val_size:        {r['val_size']}\")
"
    else
        echo "--- Qwen3-${SIZE^^}: MISSING ---"
    fi
done

echo ""
echo "Reference: seed-robustness (Qwen3-1.7B, 4 seeds): balanced_acc = 0.554 ± 0.052"
echo "============================================================"
echo "ALL EVAL DONE — $(date)"
