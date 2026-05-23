"""Generation-level eval for ablation-b (plan-004-implicit) checkpoint."""
import torch, json, re, random, sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, recall_score, accuracy_score

MODEL_PATH = './models/Qwen/Qwen3-1.7B'
ADAPTER_PATH = 'checkpoints/plan-004-implicit/final_adapter'
DATA_PATH = 'data/plan-004-implicit/implicit_sft.jsonl'
SEED = 42
EVAL_SPLIT = 0.15

# 1. Load data and reconstruct val split
records = []
with open(DATA_PATH) as f:
    for line in f:
        records.append(json.loads(line))
print(f"Total records: {len(records)}")

random.seed(SEED)
random.shuffle(records)
split_idx = int(len(records) * (1 - EVAL_SPLIT))
val_records = records[split_idx:]
print(f"Val records: {len(val_records)}")

label_dist = {}
for r in val_records:
    lbl = r['route_label']
    label_dist[lbl] = label_dist.get(lbl, 0) + 1
print(f"Val label dist: {label_dist}")

# 2. Load model
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map='cuda:0'
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()
print("Model loaded.")

# 3. Generation eval
y_true = []
y_pred = []
debug_samples = []

for i, rec in enumerate(val_records):
    messages = rec['messages']
    true_label = rec['route_label']
    
    # Construct prompt: system + user, with generation prompt
    prompt_messages = messages[:-1]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False
    )
    
    inputs = tokenizer(prompt_text, return_tensors='pt').to('cuda:0')
    input_len = inputs['input_ids'].shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False,
            temperature=1.0,
        )
    
    generated_ids = outputs[0][input_len:]
    generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Strip <think> tags
    clean = re.sub(r'<think>.*?</think>\s*', '', generated, flags=re.DOTALL).strip()
    
    # Route prediction
    pred = 'deterministic' if clean.startswith('[SIMULATE]') else 'internal'
    
    y_true.append(true_label)
    y_pred.append(pred)
    
    if i < 10:
        debug_samples.append({
            'idx': i,
            'true': true_label,
            'pred': pred,
            'raw': generated[:150],
            'clean': clean[:150],
        })
    
    if (i + 1) % 50 == 0:
        print(f"Progress: {i+1}/{len(val_records)}")

# 4. Results
print("\n" + "="*60)
print("=== GENERATION-LEVEL EVAL: ablation-b ===")
print("="*60)

print(f"\nTotal val samples: {len(y_true)}")
print(f"Label dist: {label_dist}")
pred_dist = {}
for p in y_pred:
    pred_dist[p] = pred_dist.get(p, 0) + 1
print(f"Pred dist: {pred_dist}")

print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred, labels=['internal', 'deterministic']))

print("--- Confusion Matrix (rows=true, cols=pred) ---")
print("Labels: [internal, deterministic]")
cm = confusion_matrix(y_true, y_pred, labels=['internal', 'deterministic'])
print(cm)

acc = accuracy_score(y_true, y_pred)
bal_acc = balanced_accuracy_score(y_true, y_pred)
int_recall = recall_score(y_true, y_pred, pos_label='internal')
det_recall = recall_score(y_true, y_pred, pos_label='deterministic')

print(f"\n--- Key Metrics ---")
print(f"Accuracy: {acc:.4f}")
print(f"Balanced Accuracy: {bal_acc:.4f}")
print(f"Internal Recall: {int_recall:.4f}")
print(f"Deterministic Recall: {det_recall:.4f}")

print(f"\n--- Comparison with Logit-Level ---")
print(f"Logit-level internal_recall: 0.5152")
print(f"Generation-level internal_recall: {int_recall:.4f}")
print(f"Gap: {0.5152 - int_recall:.4f}")

print(f"\n--- Debug Samples (first 10) ---")
for s in debug_samples:
    print(f"  [{s['idx']}] true={s['true']}, pred={s['pred']}")
    print(f"       raw: {s['raw']}")
    print(f"       clean: {s['clean']}")
    print()

# 5. Save results
results = {
    'model': 'ablation-b (plan-004-implicit)',
    'eval_type': 'generation-level',
    'enable_thinking': False,
    'n_val': len(y_true),
    'val_label_dist': label_dist,
    'pred_dist': pred_dist,
    'accuracy': round(acc, 4),
    'balanced_accuracy': round(bal_acc, 4),
    'internal_recall': round(int_recall, 4),
    'det_recall': round(det_recall, 4),
    'confusion_matrix': cm.tolist(),
    'confusion_labels': ['internal', 'deterministic'],
    'debug_samples': debug_samples,
    'logit_level_internal_recall': 0.5152,
    'generation_gap': round(0.5152 - int_recall, 4),
}
with open('results/ablation_b_gen_eval.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nResults saved to results/ablation_b_gen_eval.json")
