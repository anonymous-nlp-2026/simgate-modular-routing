#!/bin/bash
# conda activate (adjust path for your environment)
cd ./

TASKS=(
    "measure-melting-point-unknown-substance"
    "mendelian-genetics-known-plant"
    "mendelian-genetics-unknown-plant"
    "power-component"
    "power-component-renewable-vs-nonrenewable-energy"
    "test-conductivity"
    "test-conductivity-of-unknown-substances"
    "use-thermometer"
)

LOG="results/prescreening/run_gpu1.log"
echo "=== GPU1 prescreening started at $(date) ===" > $LOG

for task in "${TASKS[@]}"; do
    echo "=== $(date) Starting $task on GPU1 ===" | tee -a $LOG
    if [ -d "results/prescreening/$task" ]; then
        echo "SKIP: $task already done" | tee -a $LOG
        continue
    fi
    CUDA_VISIBLE_DEVICES=1 python src/signal_prescreening.py \
        --task_type "$task" \
        --episodes 5 \
        --gpu_id 0 \
        --max_steps 30 \
        --output_dir results/prescreening/ \
        --model_path ./models/Qwen/Qwen2___5-7B-Instruct/ \
        --use_vllm 2>&1 | tee -a $LOG
    echo "=== $(date) Done $task ===" | tee -a $LOG
done

echo "=== GPU1 ALL DONE at $(date) ===" | tee -a $LOG
