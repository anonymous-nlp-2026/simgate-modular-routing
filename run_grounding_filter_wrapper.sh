#!/bin/bash
set -e
cd .

echo "[$(date)] Starting vLLM on cuda:1, port 8001..."
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model ./models/Qwen/Qwen2.5-7B-Instruct \
  --served-model-name Qwen2.5-7B-Instruct \
  --port 8001 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --enforce-eager \
  --trust-remote-code &
VLLM_PID=$!
echo "[$(date)] vLLM PID: $VLLM_PID"

echo "[$(date)] Waiting for vLLM to be ready..."
for i in $(seq 1 120); do
  if curl -s http://localhost:8001/v1/models > /dev/null 2>&1; then
    echo "[$(date)] vLLM ready after ${i}s"
    break
  fi
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "[$(date)] vLLM process died!"
    exit 1
  fi
  sleep 2
done

if ! curl -s http://localhost:8001/v1/models > /dev/null 2>&1; then
  echo "[$(date)] vLLM failed to start within 240s"
  kill $VLLM_PID 2>/dev/null || true
  exit 1
fi

echo "[$(date)] Starting grounding filter ablation (Condition B)..."
python run_grounding_filter_ablation.py \
  --episodes-per-task 20 \
  --max-steps 30 \
  --model Qwen2.5-7B-Instruct \
  --api-base http://localhost:8001/v1 \
  --seed 42 \
  --output-dir results/grounding_filter_ablation/ \
  2>&1 | tee logs/grounding_filter_ablation.log
EXIT_CODE=$?

echo "[$(date)] Experiment finished with exit code $EXIT_CODE"
echo "[$(date)] Shutting down vLLM..."
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
echo "[$(date)] Done."
exit $EXIT_CODE
