#!/bin/bash
LOG=./logs/auto_start2.log
echo "[$(date)] Step 1: Waiting for torch install (PID 9194)..." >> $LOG

while kill -0 9194 2>/dev/null; do sleep 5; done
echo "[$(date)] Torch install done" >> $LOG

# Now install vllm (torch already installed, should be fast)
echo "[$(date)] Step 2: Installing vllm --pre..." >> $LOG
pip install vllm --pre >> $LOG 2>&1
echo "[$(date)] vLLM install exit code: $?" >> $LOG

# Verify
python -c "import vllm; print('vLLM version:', vllm.__version__)" >> $LOG 2>&1
if [ $? -ne 0 ]; then
  echo "[$(date)] ERROR: vLLM import failed, trying vllm-nightly" >> $LOG
  pip install vllm-nightly >> $LOG 2>&1
  python -c "import vllm; print('vLLM version:', vllm.__version__)" >> $LOG 2>&1
fi

# Wait for 7B model
echo "[$(date)] Step 3: Checking 7B model..." >> $LOG
while [ ! -f ./models/Qwen2.5-7B-Instruct/model-00001-of-00004.safetensors ]; do
  sleep 10
  echo "[$(date)] Still waiting for model files..." >> $LOG
done
echo "[$(date)] 7B model ready" >> $LOG
du -sh ./models/Qwen2.5-7B-Instruct/ >> $LOG

# Start vLLM instances
echo "[$(date)] Step 4: Starting vLLM instances..." >> $LOG
CUDA_VISIBLE_DEVICES=2 nohup python -m vllm.entrypoints.openai.api_server \
  --model ./models/Qwen2.5-7B-Instruct \
  --port 8000 --gpu-memory-utilization 0.9 --trust-remote-code \
  > ./logs/vllm-7b-gpu2.log 2>&1 &
echo "[$(date)] vLLM-gpu2 PID: $!" >> $LOG

CUDA_VISIBLE_DEVICES=3 nohup python -m vllm.entrypoints.openai.api_server \
  --model ./models/Qwen2.5-7B-Instruct \
  --port 8001 --gpu-memory-utilization 0.9 --trust-remote-code \
  > ./logs/vllm-7b-gpu3.log 2>&1 &
echo "[$(date)] vLLM-gpu3 PID: $!" >> $LOG

# Wait for ready
echo "[$(date)] Step 5: Waiting for vLLM to become ready..." >> $LOG
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
echo "[$(date)] ALL DONE" >> $LOG
