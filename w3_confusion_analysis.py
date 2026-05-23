"""W3: Per-class confusion matrix for 1.7B router, split by death/non-death episodes."""
import json, os, random, re, sys, time, glob
from collections import Counter, defaultdict

import numpy as np
import torch
from sklearn.model_selection import train_test_split as sklearn_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
)
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

os.chdir('.')
os.environ['WANDB_MODE'] = 'offline'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HOME'] = '~/.cache/huggingface'

SIMULATE_PREFIX = '[SIMULATE]'

def strip_thinking(text):
    result = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    if '<think>' in result:
        result = result[:result.index('<think>')].strip()
    result = result.replace('</think>', '').strip()
    return result

# === Step 1: Build death episode map from plan_c labeled_steps ===
print("=== Step 1: Building death episode map from plan_c ===", flush=True)
plan_c_dir = 'results/plan_c'
episode_death_map = {}  # (task_type, variation) -> {is_death, int_score, det_score, signal}

for task_dir in sorted(glob.glob(os.path.join(plan_c_dir, '*'))):
    if not os.path.isdir(task_dir):
        continue
    task_name = os.path.basename(task_dir)
    steps_file = os.path.join(task_dir, 'labeled_steps.jsonl')
    if not os.path.exists(steps_file):
        continue
    with open(steps_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (rec['task_type'], rec['variation'])
            if key not in episode_death_map:
                int_score = rec.get('episode_score_internal', 0)
                det_score = rec.get('episode_score_det', 0)
                episode_death_map[key] = {
                    'is_death': int_score == -100,
                    'int_score': int_score,
                    'det_score': det_score,
                    'signal_source': rec.get('signal_source', 'unknown'),
                }

n_death = sum(1 for v in episode_death_map.values() if v['is_death'])
print(f"Total episodes in plan_c: {len(episode_death_map)}, death: {n_death}", flush=True)

# Also try sft_data_death_aware for additional coverage
sft_da_path = 'results/sft_data_death_aware/sft_triples.jsonl'
if os.path.exists(sft_da_path):
    with open(sft_da_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (rec['task_type'], rec['variation'])
            if key not in episode_death_map:
                int_score = rec.get('episode_score_internal', 0)
                is_death = rec.get('internal_died', int_score == -100)
                episode_death_map[key] = {
                    'is_death': is_death,
                    'int_score': int_score,
                    'det_score': rec.get('episode_score_det', 0),
                    'signal_source': rec.get('signal_source', 'unknown'),
                }
    n_death2 = sum(1 for v in episode_death_map.values() if v['is_death'])
    print(f"After merging sft_data_death_aware: {len(episode_death_map)} episodes, death: {n_death2}", flush=True)

# === Step 2: Load val split ===
print("\n=== Step 2: Loading val split ===", flush=True)
data_path = 'data/plan-004-implicit/implicit_sft.jsonl'
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
_, val_keys = sklearn_split(ep_keys, test_size=0.15, random_state=42, stratify=ep_labels)
rng = random.Random(42)
val_records = [r for k in val_keys for r in episodes[k]]
rng.shuffle(val_records)

# Tag each record
unmatched = set()
for rec in val_records:
    key = (rec['task_type'], rec['variation'])
    info = episode_death_map.get(key)
    if info:
        rec['is_death_episode'] = info['is_death']
        rec['int_score'] = info['int_score']
        rec['det_score'] = info['det_score']
        rec['signal_source'] = info['signal_source']
    else:
        rec['is_death_episode'] = False
        rec['int_score'] = None
        rec['det_score'] = None
        rec['signal_source'] = 'unmatched'
        unmatched.add(key)

val_death = [r for r in val_records if r['is_death_episode']]
val_nodeath = [r for r in val_records if not r['is_death_episode']]
val_episode_keys = set((r['task_type'], r['variation']) for r in val_records)
val_death_ep = set((r['task_type'], r['variation']) for r in val_death)
val_nodeath_ep = set((r['task_type'], r['variation']) for r in val_nodeath)

print(f"Val: {len(val_records)} steps, {len(val_episode_keys)} episodes", flush=True)
print(f"  Death: {len(val_death)} steps, {len(val_death_ep)} episodes", flush=True)
print(f"  Non-death: {len(val_nodeath)} steps, {len(val_nodeath_ep)} episodes", flush=True)
if unmatched:
    print(f"  Unmatched episodes (treated as non-death): {len(unmatched)}", flush=True)
    for k in sorted(unmatched)[:5]:
        print(f"    {k}", flush=True)

# === Step 3: Load model ===
print("\n=== Step 3: Loading 1.7B router (plan-008-v4-seed42) ===", flush=True)
adapter_path = 'checkpoints/plan-008-v4-1_7b-seed42/final_adapter'
base_model = './models/Qwen/Qwen3-1.7B'
GPU_ID = 1

tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained(
    base_model, dtype=torch.bfloat16,
    device_map={'': GPU_ID}, trust_remote_code=True,
)
model = PeftModel.from_pretrained(base, adapter_path)
model.eval()
model.config.pad_token_id = tokenizer.pad_token_id
print("Model loaded.", flush=True)

# === Step 4: Inference ===
print("\n=== Step 4: Running inference ===", flush=True)
predictions = []
t0 = time.time()

for i, rec in enumerate(val_records):
    prompt_messages = rec['messages'][:-1]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors='pt', truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    cleaned = strip_thinking(raw_output)
    pred_route = 'deterministic' if cleaned.startswith(SIMULATE_PREFIX) else 'internal'

    predictions.append({
        'idx': i,
        'task_type': rec['task_type'],
        'variation': rec['variation'],
        'step_idx': rec.get('step_idx', -1),
        'gt': rec['route_label'],
        'pred': pred_route,
        'is_death': rec['is_death_episode'],
        'int_score': rec['int_score'],
        'det_score': rec['det_score'],
        'cleaned': cleaned[:100],
    })

    if (i + 1) % 50 == 0 or i == 0:
        print(f"  [{i+1}/{len(val_records)}] ({time.time()-t0:.1f}s) gt={rec['route_label']}, pred={pred_route}, death={rec['is_death_episode']}", flush=True)

total_time = time.time() - t0
print(f"\nInference: {total_time:.1f}s ({len(val_records)} samples)", flush=True)

del model, base
torch.cuda.empty_cache()

# === Step 5: Compute metrics ===
print("\n=== Step 5: Metrics ===", flush=True)

def compute_metrics(preds_list, label='Overall'):
    if not preds_list:
        return None
    y_true = [p['gt'] for p in preds_list]
    y_pred = [p['pred'] for p in preds_list]
    labels = ['deterministic', 'internal']
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    result = {
        'n': len(preds_list),
        'accuracy': round(acc, 4),
        'balanced_accuracy': round(bal_acc, 4),
        'cm': cm.tolist(),
        'det': {'p': round(float(prec[0]),4), 'r': round(float(rec[0]),4), 'f1': round(float(f1[0]),4), 'n': int(sup[0])},
        'int': {'p': round(float(prec[1]),4), 'r': round(float(rec[1]),4), 'f1': round(float(f1[1]),4), 'n': int(sup[1])},
        'true_dist': dict(Counter(y_true)),
        'pred_dist': dict(Counter(y_pred)),
    }
    print(f"\n  [{label}] n={result['n']}, acc={acc:.4f}, bal_acc={bal_acc:.4f}")
    print(f"    CM: {cm.tolist()}")
    print(f"    Det P={prec[0]:.4f} R={rec[0]:.4f} F1={f1[0]:.4f} (n={sup[0]})")
    print(f"    Int P={prec[1]:.4f} R={rec[1]:.4f} F1={f1[1]:.4f} (n={sup[1]})")
    return result

overall = compute_metrics(predictions, 'Overall')
death_preds = [p for p in predictions if p['is_death']]
nodeath_preds = [p for p in predictions if not p['is_death']]
death_m = compute_metrics(death_preds, 'Death episodes')
nodeath_m = compute_metrics(nodeath_preds, 'Non-death episodes')

# Per task type
task_types = sorted(set(p['task_type'] for p in predictions))
per_task = {}
for tt in task_types:
    tp = [p for p in predictions if p['task_type'] == tt]
    per_task[tt] = compute_metrics(tp, f'Task: {tt}')

# === Step 6: Save JSON ===
full_results = {
    'model': 'plan-008-v4-1_7b-seed42',
    'adapter': adapter_path,
    'inference_time_s': round(total_time, 1),
    'val_episodes': {'total': len(val_episode_keys), 'death': len(val_death_ep), 'non_death': len(val_nodeath_ep)},
    'overall': overall,
    'death': death_m,
    'non_death': nodeath_m,
    'per_task': per_task,
    'predictions': predictions,
}
os.makedirs('artifacts', exist_ok=True)
with open('artifacts/w3_confusion_raw.json', 'w') as f:
    json.dump(full_results, f, indent=2)

# === Step 7: Markdown report ===
def cm_table(m, title):
    if m is None:
        return f"### {title}\nNo data.\n"
    cm = m['cm']
    lines = [
        f"### {title}",
        f"n = {m['n']} steps, true dist: det={m['true_dist'].get('deterministic',0)}, int={m['true_dist'].get('internal',0)}",
        "",
        "| | Pred: deterministic | Pred: internal |",
        "|---|---:|---:|",
        f"| **True: deterministic** | {cm[0][0]} | {cm[0][1]} |",
        f"| **True: internal** | {cm[1][0]} | {cm[1][1]} |",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
        f"| deterministic | {m['det']['p']:.4f} | {m['det']['r']:.4f} | {m['det']['f1']:.4f} | {m['det']['n']} |",
        f"| internal | {m['int']['p']:.4f} | {m['int']['r']:.4f} | {m['int']['f1']:.4f} | {m['int']['n']} |",
        "",
        f"Accuracy = {m['accuracy']:.4f}, Balanced Accuracy = {m['balanced_accuracy']:.4f}",
        "",
    ]
    return '\n'.join(lines)

report = []
report.append("# W3: 1.7B Router Per-Class Confusion Matrix")
report.append("")
report.append(f"Model: `plan-008-v4-1_7b-seed42` (Qwen3-1.7B + LoRA)")
report.append(f"Eval: 15% episode-stratified val split (seed=42), greedy decoding")
report.append(f"Val set: {overall['n']} steps across {len(val_episode_keys)} episodes ({len(val_death_ep)} death, {len(val_nodeath_ep)} non-death)")
report.append("")

report.append("## 1. Overall Confusion Matrix")
report.append("")
report.append(cm_table(overall, 'All Val Data'))

report.append("## 2. Death vs Non-Death Split")
report.append("")
report.append("Death episodes = internal agent scored -100 (died via `focus on [wrong object]`).")
report.append("In death episodes, ALL steps are labeled deterministic (det outperforms by 100+ points).")
report.append("")
report.append(cm_table(death_m, 'Death Episodes'))
report.append(cm_table(nodeath_m, 'Non-Death Episodes'))

report.append("## 3. Per-Task Summary")
report.append("")
report.append("| Task Type | N | Acc | Bal Acc | Det P | Det R | Int P | Int R | Death Steps |")
report.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for tt in task_types:
    m = per_task[tt]
    nd = len([p for p in predictions if p['task_type'] == tt and p['is_death']])
    if m:
        report.append(f"| {tt} | {m['n']} | {m['accuracy']:.3f} | {m['balanced_accuracy']:.3f} | {m['det']['p']:.3f} | {m['det']['r']:.3f} | {m['int']['p']:.3f} | {m['int']['r']:.3f} | {nd} |")
report.append("")

report.append("## 4. Key Findings")
report.append("")
if death_m and nodeath_m:
    report.append(f"1. **Death episode routing accuracy**: {death_m['accuracy']:.4f} vs non-death: {nodeath_m['accuracy']:.4f} (delta = {death_m['accuracy'] - nodeath_m['accuracy']:+.4f})")
    report.append("")
    report.append(f"2. **Death episodes are all-deterministic ground truth** ({death_m['true_dist'].get('deterministic',0)}/{death_m['n']} steps). Router predicts deterministic on {death_m['pred_dist'].get('deterministic',0)}/{death_m['n']} steps (det recall = {death_m['det']['r']:.4f}).")
    report.append("")
    if nodeath_m['int']['n'] > 0:
        report.append(f"3. **Internal class performance**: On non-death episodes, internal recall = {nodeath_m['int']['r']:.4f}, precision = {nodeath_m['int']['p']:.4f}. The router's ability to identify internal-preferred steps is concentrated in non-death episodes.")
    report.append("")
    death_det_rate = death_m['pred_dist'].get('deterministic',0) / death_m['n'] if death_m['n'] > 0 else 0
    nodeath_det_rate = nodeath_m['pred_dist'].get('deterministic',0) / nodeath_m['n'] if nodeath_m['n'] > 0 else 0
    report.append(f"4. **Prediction bias**: Router predicts deterministic at {death_det_rate:.1%} rate on death episodes vs {nodeath_det_rate:.1%} on non-death episodes.")
    report.append("")

md_text = '\n'.join(report)
with open('artifacts/w3_confusion_matrix.md', 'w') as f:
    f.write(md_text)

print("\n=== Saved artifacts/w3_confusion_matrix.md and w3_confusion_raw.json ===", flush=True)
