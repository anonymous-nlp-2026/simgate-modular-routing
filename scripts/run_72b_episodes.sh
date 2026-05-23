#!/bin/bash
set -e
source activate

MODEL_DIR="./models/Qwen/Qwen2.5-72B-Instruct-AWQ"
PROJECT_DIR="."

# Wait for download to complete
echo "[$(date)] Waiting for 72B model download..."
while true; do
    # Check if all 11 safetensor shards exist
    n_shards=$(ls "$MODEL_DIR"/model-*.safetensors 2>/dev/null | wc -l)
    if [ "$n_shards" -ge 11 ]; then
        # Also check config exists
        if [ -f "$MODEL_DIR/config.json" ]; then
            echo "[$(date)] Model download complete ($n_shards shards found)"
            break
        fi
    fi
    echo "[$(date)] Download in progress: $n_shards/11 shards..."
    sleep 30
done

echo "[$(date)] Model size:"
du -sh "$MODEL_DIR"

# Run episode collection
cd "$PROJECT_DIR"

TASK_TYPES="chemistry-mix,find-animal,find-non-living-thing,freeze,grow-fruit,identify-life-stages-1,identify-life-stages-2,inclined-plane-determine-angle,lifespan-longest-lived,lifespan-longest-lived-then-shortest-lived,measure-melting-point-unknown-substance"

echo "[$(date)] Starting 72B episode collection on all 11 task types..."
python src/episode_level_collection.py \
    --model_path "$MODEL_DIR" \
    --task_types "$TASK_TYPES" \
    --episodes 20 \
    --max_steps 30 \
    --gpu_id 0 \
    --tp 2 \
    --output_dir results/backbone_72b/ \
    --seed 42

echo "[$(date)] Episode collection complete!"
echo "Results in: results/backbone_72b/"
ls -la results/backbone_72b/
