#!/bin/bash
# Signal prescreening - all 25 untested tasks on GPU0
# conda activate (adjust path for your environment)
cd ./

SKIP="boil|melt|change-the-state-of-matter-of|chemistry-mix-paint-secondary-color|grow-plant"
SUMMARY_FILE="results/prescreening/summary.txt"
mkdir -p results/prescreening

echo "=== Prescreening started at $(date) ===" > $SUMMARY_FILE

while IFS= read -r line; do
    [[ "$line" =~ ^# ]] && continue
    [[ -z "$line" ]] && continue
    task=$(echo "$line" | sed 's/^[0-9]*: //' | sed 's/  # MVP-tested$//')
    if echo "$task" | grep -qE "^($SKIP)$"; then
        echo "SKIP: $task (MVP-tested)" >> $SUMMARY_FILE
        continue
    fi
    echo ">>> Running: $task at $(date)" | tee -a $SUMMARY_FILE
    python src/signal_prescreening.py \
        --task_type "$task" \
        --episodes 5 \
        --gpu_id 0 \
        --max_steps 30 \
        --output_dir results/prescreening/ \
        --model_path ./models/Qwen/Qwen2___5-7B-Instruct/ \
        --use_vllm 2>&1 | tee -a $SUMMARY_FILE
    echo "<<< Done: $task at $(date)" >> $SUMMARY_FILE
    echo "" >> $SUMMARY_FILE
done < notes/scienceworld_task_types.txt

echo "=== Prescreening finished at $(date) ===" >> $SUMMARY_FILE
echo "=== FINAL SUMMARY ===" >> $SUMMARY_FILE
for f in results/prescreening/*/summary.json; do
    if [ -f "$f" ]; then
        task_dir=$(dirname "$f" | xargs basename)
        echo "$task_dir: $(cat $f)" >> $SUMMARY_FILE
    fi
done
