#!/bin/bash
LOG=./logs/auto_start.log
echo "[$(date)] Waiting for vLLM install..." >> $LOG

# Wait for pip install to finish (PID 8445)
while kill -0 8445 2>/dev/null; do sleep 10; done
echo "[$(date)] vLLM pip install finished" >> $LOG

# Verify vLLM
python -c "import vllm; print('vLLM version:', vllm.__version__)" >> $LOG 2>&1
if [ $? -ne 0 ]; then
  echo "[$(date)] ERROR: vLLM import failed" >> $LOG
  exit 1
fi

# Wait for 7B model download (PID 8016)
echo "[$(date)] Waiting for 7B model download..." >> $LOG
while kill -0 8016 2>/dev/null; do sleep 10; done
echo "[$(date)] 7B model download finished" >> $LOG

# Verify model files exist
if [ ! -f ./models/Qwen2.5-7B-Instruct/model-00001-of-00004.safetensors ]; then
  echo "[$(date)] ERROR: model safetensors not found" >> $LOG
  exit 1
fi
echo "[$(date)] Model verified" >> $LOG

# Install screen for background sessions
apt-get install -y screen > /dev/null 2>&1 || true

# Start vLLM instance 1 (cuda:2, port 8000)
echo "[$(date)] Starting vLLM on cuda:2 port 8000..." >> $LOG
CUDA_VISIBLE_DEVICES=2 nohup python -m vllm.entrypoints.openai.api_server \
  --model ./models/Qwen2.5-7B-Instruct \
  --port 8000 --gpu-memory-utilization 0.9 --trust-remote-code \
  > ./logs/vllm-7b-gpu2.log 2>&1 &
VLLM1_PID=$!
echo "[$(date)] vLLM-1 PID: $VLLM1_PID" >> $LOG

# Start vLLM instance 2 (cuda:3, port 8001)
echo "[$(date)] Starting vLLM on cuda:3 port 8001..." >> $LOG
CUDA_VISIBLE_DEVICES=3 nohup python -m vllm.entrypoints.openai.api_server \
  --model ./models/Qwen2.5-7B-Instruct \
  --port 8001 --gpu-memory-utilization 0.9 --trust-remote-code \
  > ./logs/vllm-7b-gpu3.log 2>&1 &
VLLM2_PID=$!
echo "[$(date)] vLLM-2 PID: $VLLM2_PID" >> $LOG

# Wait for vLLM to load (up to 120s)
echo "[$(date)] Waiting for vLLM to become ready..." >> $LOG
for i in $(seq 1 24); do
  sleep 5
  if curl -s http://localhost:8000/v1/models | grep -q "model"; then
    echo "[$(date)] PORT 8000 READY" >> $LOG
    break
  fi
done

for i in $(seq 1 12); do
  sleep 5
  if curl -s http://localhost:8001/v1/models | grep -q "model"; then
    echo "[$(date)] PORT 8001 READY" >> $LOG
    break
  fi
done

echo "[$(date)] === FINAL STATUS ===" >> $LOG
curl -s http://localhost:8000/v1/models >> $LOG 2>&1
echo "" >> $LOG
curl -s http://localhost:8001/v1/models >> $LOG 2>&1
echo "" >> $LOG
echo "[$(date)] DONE" >> $LOG
