#!/bin/bash

cd .
source activate

MODEL_PATH="./models/Qwen/Qwen2.5-7B-Instruct"
PORT=8001
DATA_DIR="data/math-counterfactual"
API_URL="http://localhost:${PORT}/v1"

echo "=== Step 1: Starting vLLM server on cuda:1, port ${PORT} ==="
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model ${MODEL_PATH} \
  --port ${PORT} \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 &
VLLM_PID=$!
echo "vLLM PID: ${VLLM_PID}"

echo "Waiting for vLLM to be ready..."
READY=0
for i in $(seq 1 120); do
  HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:${PORT}/health 2>/dev/null || echo "000")
  if [ "$HTTP_CODE" = "200" ]; then
    echo "vLLM ready after $((i*3)) seconds"
    READY=1
    break
  fi
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "ERROR: vLLM process died"
    exit 1
  fi
  sleep 3
done

if [ "$READY" -ne 1 ]; then
  echo "ERROR: vLLM startup timeout"
  kill $VLLM_PID 2>/dev/null
  exit 1
fi
echo "vLLM is healthy"

rm -f ${DATA_DIR}/math_extend_batch1.jsonl ${DATA_DIR}/math_extend_batch2.jsonl ${DATA_DIR}/math_extend_batch3.jsonl

echo ""
echo "=== Step 2: Running batch 1 (offset=300, n=170) ==="
python src/math_counterfactual.py \
  --n_problems 170 --offset 300 \
  --output ${DATA_DIR}/math_extend_batch1.jsonl \
  --api_url ${API_URL} \
  --model_path ${MODEL_PATH} \
  --gpu_id 1
BATCH1_LINES=$(wc -l < ${DATA_DIR}/math_extend_batch1.jsonl)
echo "Batch 1 done: ${BATCH1_LINES} lines"

echo ""
echo "=== Step 3: Running batch 2 (offset=470, n=170) ==="
python src/math_counterfactual.py \
  --n_problems 170 --offset 470 \
  --output ${DATA_DIR}/math_extend_batch2.jsonl \
  --api_url ${API_URL} \
  --model_path ${MODEL_PATH} \
  --gpu_id 1
BATCH2_LINES=$(wc -l < ${DATA_DIR}/math_extend_batch2.jsonl)
echo "Batch 2 done: ${BATCH2_LINES} lines"

echo ""
echo "=== Step 4: Running batch 3 (offset=640, n=160) ==="
python src/math_counterfactual.py \
  --n_problems 160 --offset 640 \
  --output ${DATA_DIR}/math_extend_batch3.jsonl \
  --api_url ${API_URL} \
  --model_path ${MODEL_PATH} \
  --gpu_id 1
BATCH3_LINES=$(wc -l < ${DATA_DIR}/math_extend_batch3.jsonl)
echo "Batch 3 done: ${BATCH3_LINES} lines"

echo ""
echo "=== Step 5: Killing vLLM ==="
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
echo "vLLM stopped"

echo ""
echo "=== Step 6: Merging data ==="
cat ${DATA_DIR}/math_clean_300.jsonl \
    ${DATA_DIR}/math_extend_batch1.jsonl \
    ${DATA_DIR}/math_extend_batch2.jsonl \
    ${DATA_DIR}/math_extend_batch3.jsonl \
    > ${DATA_DIR}/math_clean_800.jsonl
TOTAL_LINES=$(wc -l < ${DATA_DIR}/math_clean_800.jsonl)
echo "Merged file: ${TOTAL_LINES} lines"

echo ""
echo "=== Step 7: Statistical Analysis ==="
python3 << 'PYEOF'
import json
import numpy as np
from collections import Counter

DATA_FILE = "data/math-counterfactual/math_clean_800.jsonl"

data = []
with open(DATA_FILE) as f:
    for line in f:
        data.append(json.loads(line))

n = len(data)
print(f"Total episodes: {n}")

int_correct = [d["internal"]["correct"] for d in data]
det_correct = [d["deterministic"]["correct"] for d in data]

acc_int = sum(int_correct) / n
acc_det = sum(det_correct) / n
print(f"Internal accuracy: {sum(int_correct)}/{n} ({acc_int*100:.1f}%)")
print(f"Code accuracy:     {sum(det_correct)}/{n} ({acc_det*100:.1f}%)")
print(f"Diff (int - code): {(acc_int - acc_det)*100:.1f}pp")

labels = [d["label"] for d in data]
lc = Counter(labels)
print(f"Labels: {dict(lc)}")

a = sum(1 for i in range(n) if int_correct[i] and not det_correct[i])
b = sum(1 for i in range(n) if det_correct[i] and not int_correct[i])
c = sum(1 for i in range(n) if int_correct[i] and det_correct[i])
d_val = sum(1 for i in range(n) if not int_correct[i] and not det_correct[i])
print(f"Contingency: both_correct={c}, both_wrong={d_val}, int_only={a}, det_only={b}")

from scipy.stats import binom_test
if a + b > 0:
    p_mcnemar = binom_test(a, a + b, 0.5)
    print(f"McNemar exact p-value: {p_mcnemar:.6f}")
else:
    p_mcnemar = 1.0
    print("McNemar: no discordant pairs")

diff = acc_int - acc_det
se = np.sqrt((a + b - (a - b)**2 / n) / (n * (n - 1))) if n > 1 else 0
ci_low = diff - 1.96 * se
ci_high = diff + 1.96 * se
print(f"95% CI for diff: [{ci_low*100:.1f}%, {ci_high*100:.1f}%]")

from scipy.stats import norm
if se > 0:
    z = abs(diff) / se
    power = 1 - norm.cdf(1.96 - z) + norm.cdf(-1.96 - z)
    print(f"Statistical power (post-hoc): {power:.4f}")

effect_size = diff / np.sqrt(acc_int * (1-acc_int) + acc_det * (1-acc_det)) if (acc_int * (1-acc_int) + acc_det * (1-acc_det)) > 0 else 0
print(f"Effect size (Cohen h approx): {effect_size:.4f}")

print()
print("=== SUMMARY FOR C3 CLAIM ===")
if p_mcnemar < 0.05:
    print(f"Difference IS statistically significant (p={p_mcnemar:.4f})")
else:
    print(f"Difference is NOT statistically significant (p={p_mcnemar:.4f})")
print(f"N={n}: internal {acc_int*100:.1f}% vs code {acc_det*100:.1f}% (diff={diff*100:.1f}pp)")

results = {
    "n": n,
    "accuracy_internal": round(acc_int, 4),
    "accuracy_code": round(acc_det, 4),
    "diff_pp": round(diff * 100, 2),
    "p_value_mcnemar": round(p_mcnemar, 6),
    "ci_95_low": round(ci_low * 100, 2),
    "ci_95_high": round(ci_high * 100, 2),
    "labels": dict(lc),
    "contingency": {"both_correct": c, "both_wrong": d_val, "int_only": a, "det_only": b}
}
with open("data/math-counterfactual/analysis_n800.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to data/math-counterfactual/analysis_n800.json")
PYEOF

echo ""
echo "=== ALL DONE ==="
