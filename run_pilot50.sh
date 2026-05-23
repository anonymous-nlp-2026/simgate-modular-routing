#!/bin/bash
set -e
# conda activate (adjust path for your environment)
conda activate base

MODEL=./models/Qwen/Qwen2___5-7B-Instruct
PROJ=.
LOG=$PROJ/data/math-counterfactual/pilot_50_run.log

mkdir -p $PROJ/data/math-counterfactual

echo "=== Starting vLLM server ===" | tee $LOG
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    --port 8000 \
    --trust-remote-code > /tmp/vllm_server.log 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID" | tee -a $LOG

echo "Waiting for vLLM server..." | tee -a $LOG
for i in $(seq 1 90); do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "Server ready after ${i}x2s" | tee -a $LOG
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "FATAL: vLLM server died" | tee -a $LOG
        cat /tmp/vllm_server.log >> $LOG
        exit 1
    fi
    sleep 2
done

# Verify
curl -s http://localhost:8000/v1/models | python -m json.tool 2>&1 | tee -a $LOG

echo "=== Running pilot 50 ===" | tee -a $LOG
cd $PROJ
python src/math_counterfactual.py \
    --n_problems 50 \
    --output data/math-counterfactual/pilot_50.jsonl \
    --model_path $MODEL \
    --api_url http://localhost:8000/v1 \
    2>&1 | tee -a $LOG
EXIT_CODE=${PIPESTATUS[0]}

echo "=== Cleanup: killing vLLM server ===" | tee -a $LOG
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo "EXIT_CODE=$EXIT_CODE" | tee -a $LOG
echo "=== DONE ===" | tee -a $LOG
exit $EXIT_CODE
