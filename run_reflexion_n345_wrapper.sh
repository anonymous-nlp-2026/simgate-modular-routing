#!/bin/bash
set -e
cd .
mkdir -p logs

MODEL=./models/Qwen2.5-7B-Instruct
PYTHON=python
export VLLM_USE_FLASHINFER_SAMPLER=0

echo "[$(date)] Starting 3 vLLM servers (ports 8001/8002/8003)..."

CUDA_VISIBLE_DEVICES=0 $PYTHON -m vllm.entrypoints.openai.api_server \
  --model $MODEL --served-model-name Qwen2.5-7B-Instruct \
  --port 8001 --gpu-memory-utilization 0.9 --trust-remote-code &
VLLM_PID_0=$!

CUDA_VISIBLE_DEVICES=1 $PYTHON -m vllm.entrypoints.openai.api_server \
  --model $MODEL --served-model-name Qwen2.5-7B-Instruct \
  --port 8002 --gpu-memory-utilization 0.9 --trust-remote-code &
VLLM_PID_1=$!

CUDA_VISIBLE_DEVICES=2 $PYTHON -m vllm.entrypoints.openai.api_server \
  --model $MODEL --served-model-name Qwen2.5-7B-Instruct \
  --port 8003 --gpu-memory-utilization 0.9 --trust-remote-code &
VLLM_PID_2=$!

echo "[$(date)] vLLM PIDs: $VLLM_PID_0 $VLLM_PID_1 $VLLM_PID_2"

wait_vllm() {
  local port=$1 pid=$2
  for i in $(seq 1 120); do
    if curl -s http://localhost:${port}/v1/models > /dev/null 2>&1; then
      echo "[$(date)] vLLM port $port ready after $((i*2))s"
      return 0
    fi
    if ! kill -0 $pid 2>/dev/null; then
      echo "[$(date)] vLLM port $port (PID $pid) died!"
      return 1
    fi
    sleep 2
  done
  echo "[$(date)] vLLM port $port timeout after 240s"
  return 1
}

wait_vllm 8001 $VLLM_PID_0 || { kill $VLLM_PID_1 $VLLM_PID_2 2>/dev/null; exit 1; }
wait_vllm 8002 $VLLM_PID_1 || { kill $VLLM_PID_0 $VLLM_PID_2 2>/dev/null; exit 1; }
wait_vllm 8003 $VLLM_PID_2 || { kill $VLLM_PID_0 $VLLM_PID_1 2>/dev/null; exit 1; }

echo "[$(date)] All vLLM servers ready. Starting experiments..."

$PYTHON run_reflexion_dose_response.py \
  --n-values 3 --episodes-per-task 20 --max-steps 30 \
  --model Qwen2.5-7B-Instruct --api-base http://localhost:8001/v1 \
  --seed 42 --output-dir results/reflexion_dose_response/ \
  > logs/reflexion_n3.log 2>&1 &
EXP_PID_0=$!

$PYTHON run_reflexion_dose_response.py \
  --n-values 4 --episodes-per-task 20 --max-steps 30 \
  --model Qwen2.5-7B-Instruct --api-base http://localhost:8002/v1 \
  --seed 42 --output-dir results/reflexion_dose_response/ \
  > logs/reflexion_n4.log 2>&1 &
EXP_PID_1=$!

$PYTHON run_reflexion_dose_response.py \
  --n-values 5 --episodes-per-task 20 --max-steps 30 \
  --model Qwen2.5-7B-Instruct --api-base http://localhost:8003/v1 \
  --seed 42 --output-dir results/reflexion_dose_response/ \
  > logs/reflexion_n5.log 2>&1 &
EXP_PID_2=$!

echo "[$(date)] Experiment PIDs: N3=$EXP_PID_0 N4=$EXP_PID_1 N5=$EXP_PID_2"

wait $EXP_PID_0
EC0=$?
echo "[$(date)] N=3 finished (exit=$EC0)"

wait $EXP_PID_1
EC1=$?
echo "[$(date)] N=4 finished (exit=$EC1)"

wait $EXP_PID_2
EC2=$?
echo "[$(date)] N=5 finished (exit=$EC2)"

echo "[$(date)] Shutting down vLLM..."
kill $VLLM_PID_0 $VLLM_PID_1 $VLLM_PID_2 2>/dev/null || true
wait $VLLM_PID_0 $VLLM_PID_1 $VLLM_PID_2 2>/dev/null || true
echo "[$(date)] All done. Exit codes: N3=$EC0 N4=$EC1 N5=$EC2"
