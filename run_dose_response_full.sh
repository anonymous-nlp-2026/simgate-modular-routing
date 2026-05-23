#!/bin/bash
set -e

source activate
cd .

# 创建 logs 目录
mkdir -p logs

echo "[$(date)] Starting vLLM server..."
VLLM_USE_FLASHINFER_SAMPLER=0 python -m vllm.entrypoints.openai.api_server \
  --model ./models/Qwen2.5-7B-Instruct \
  --port 8000 --gpu-memory-utilization 0.9 --trust-remote-code \
  > logs/vllm-dose-response.log 2>&1 &
VLLM_PID=$!

# 清理函数：实验结束后杀 vLLM
cleanup() {
  echo "[$(date)] Cleaning up vLLM (PID=$VLLM_PID)..."
  kill $VLLM_PID 2>/dev/null
  wait $VLLM_PID 2>/dev/null
  echo "[$(date)] Done."
}
trap cleanup EXIT

# 等待 vLLM 就绪（最多 10 分钟）
echo "[$(date)] Waiting for vLLM to be ready..."
for i in $(seq 1 60); do
  if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "[$(date)] vLLM ready after ${i}0 seconds"
    break
  fi
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "[$(date)] ERROR: vLLM process died"
    cat logs/vllm-dose-response.log | tail -50
    exit 1
  fi
  sleep 10
done

# 检查 vLLM 是否真的就绪
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
  echo "[$(date)] ERROR: vLLM failed to start within 10 minutes"
  cat logs/vllm-dose-response.log | tail -50
  exit 1
fi

# 跑实验
echo "[$(date)] Starting dose-response experiment..."
python run_retry_dose_response.py --n-values 1,5,10 --episodes-per-task 20 2>&1 | tee logs/dose_response.log
EXIT_CODE=${PIPESTATUS[0]}

echo "[$(date)] Experiment finished with exit code $EXIT_CODE"
exit $EXIT_CODE
