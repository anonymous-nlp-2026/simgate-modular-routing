import json, os, random, re, sys, time, gc
from collections import Counter
import numpy as np
import torch
from sklearn.model_selection import train_test_split as sklearn_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

SIMULATE_PREFIX = '[SIMULATE]'

def strip_thinking(text):
    result = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    if '<think>' in result:
        result = result[:result.index('<think>')].strip()
    result = result.replace('</think>', '').strip()
    return result

def load_val_split(data_path, eval_split=0.15, seed=42):
    if os.path.isdir(data_path):
        data_path = os.path.join(data_path, 'implicit_sft.jsonl')
    records = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    episodes = {}
    for r in records:
        key = (r.get('task_type', 'unknown'), r.get('variation', 0))
        episodes.setdefault(key, []).append(r)
    ep_keys = list(episodes.keys())
    ep_labels = [Counter(r['route_label'] for r in episodes[k]).most_common(1)[0][0] for k in ep_keys]
    _, val_keys = sklearn_split(ep_keys, test_size=eval_split, random_state=seed, stratify=ep_labels)
    rng = random.Random(seed)
    val_records = [r for k in val_keys for r in episodes[k]]
    rng.shuffle(val_records)
    return val_records

def run_eval(adapter_path, base_model, data_dir, gpu_id=0, seed=42, max_new_tokens=128):
    val_records = load_val_split(data_dir, seed=seed)
    val_lbl = Counter(r['route_label'] for r in val_records)
    print(f'Val: {len(val_records)} steps, labels: {dict(val_lbl)}')

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16, device_map={'': gpu_id}, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    model.config.pad_token_id = tokenizer.pad_token_id
    print('Model loaded.', flush=True)

    y_true, y_pred = [], []
    t0 = time.time()
    for i, rec in enumerate(val_records):
        messages = rec['messages']
        prompt_messages = messages[:-1]
        # Do NOT pass enable_thinking — use tokenizer default (thinking ON for Qwen3)
        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors='pt', truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.1, do_sample=True)
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        cleaned = strip_thinking(raw_output)
        pred_route = 'deterministic' if cleaned.startswith(SIMULATE_PREFIX) else 'internal'
        y_true.append(rec['route_label'])
        y_pred.append(pred_route)
        if (i+1) % 50 == 0:
            print(f'  [{i+1}/{len(val_records)}] ({time.time()-t0:.1f}s)', flush=True)

    elapsed = time.time() - t0
    print(f'Inference done in {elapsed:.1f}s', flush=True)

    labels = ['internal', 'deterministic']
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec, rec_arr, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    result = {
        'seed': seed,
        'balanced_acc': round(bal_acc, 4),
        'internal_recall': round(float(rec_arr[0]), 4),
        'internal_precision': round(float(prec[0]), 4),
        'det_recall': round(float(rec_arr[1]), 4),
        'det_precision': round(float(prec[1]), 4),
        'val_size': len(val_records),
    }
    print(f'RESULT: {json.dumps(result)}', flush=True)
    return result, model, base

os.chdir('.')
os.environ['WANDB_MODE'] = 'offline'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HOME'] = '~/.cache/huggingface'

# Seed 42
print('=== Evaluating 1.7B seed 42 ===', flush=True)
r42, model, base = run_eval(
    adapter_path='checkpoints/plan-004-v5/final_adapter',
    base_model='./models/Qwen/Qwen3-1.7B',
    data_dir='data/plan-004-implicit/',
    gpu_id=0, seed=42,
)
os.makedirs('results/plan-008-v4', exist_ok=True)
with open('results/plan-008-v4/1_7b_seed42_reeval.json', 'w') as f:
    json.dump(r42, f, indent=2)

# Free GPU
del model, base
gc.collect()
torch.cuda.empty_cache()

# Seed 43
print('\n=== Evaluating 1.7B seed 43 ===', flush=True)
r43, model, base = run_eval(
    adapter_path='checkpoints/seed-robustness-seed43/final_adapter',
    base_model='./models/Qwen/Qwen3-1.7B',
    data_dir='data/plan-004-implicit/',
    gpu_id=0, seed=43,
)
with open('results/plan-008-v4/1_7b_seed43_reeval.json', 'w') as f:
    json.dump(r43, f, indent=2)

print('\n=== SUMMARY ===', flush=True)
print(f'Seed 42: {json.dumps(r42)}', flush=True)
print(f'Seed 43: {json.dumps(r43)}', flush=True)
mean_ba = (r42['balanced_acc'] + r43['balanced_acc']) / 2
std_ba = abs(r42['balanced_acc'] - r43['balanced_acc']) / 2
print(f'Mean balanced_acc: {mean_ba:.4f} +/- {std_ba:.4f}', flush=True)
print('DONE', flush=True)
