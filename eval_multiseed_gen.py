"""Generation-level eval for ablation-b multi-seed reproduction."""
import torch, json, re, random, sys, argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, recall_score, accuracy_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--gpu_id", type=int, default=1)
    args = parser.parse_args()

    MODEL_PATH = './models/Qwen/Qwen3-1.7B'
    DATA_PATH = 'data/plan-004-implicit/implicit_sft.jsonl'
    SEED = 42
    EVAL_SPLIT = 0.15
    device = f'cuda:{args.gpu_id}'

    records = []
    with open(DATA_PATH) as f:
        for line in f:
            records.append(json.loads(line))

    random.seed(SEED)
    random.shuffle(records)
    split_idx = int(len(records) * (1 - EVAL_SPLIT))
    val_records = records[split_idx:]

    label_dist = {}
    for r in val_records:
        lbl = r['route_label']
        label_dist[lbl] = label_dist.get(lbl, 0) + 1
    print(f"Val records: {len(val_records)}, dist: {label_dist}")

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map=device
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()
    print("Model loaded.")

    y_true, y_pred = [], []
    for i, rec in enumerate(val_records):
        messages = rec['messages']
        true_label = rec['route_label']
        prompt_messages = messages[:-1]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False
        )
        inputs = tokenizer(prompt_text, return_tensors='pt').to(device)
        input_len = inputs['input_ids'].shape[1]
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=60, do_sample=False, temperature=1.0,
            )
        generated_ids = outputs[0][input_len:]
        generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
        clean = re.sub(r'<think>.*?</think>\s*', '', generated, flags=re.DOTALL).strip()
        pred = 'deterministic' if clean.startswith('[SIMULATE]') else 'internal'
        y_true.append(true_label)
        y_pred.append(pred)
        if (i + 1) % 50 == 0:
            print(f"Progress: {i+1}/{len(val_records)}")

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    int_recall = recall_score(y_true, y_pred, pos_label='internal')
    det_recall = recall_score(y_true, y_pred, pos_label='deterministic')

    pred_dist = {}
    for p in y_pred:
        pred_dist[p] = pred_dist.get(p, 0) + 1

    cm = confusion_matrix(y_true, y_pred, labels=['internal', 'deterministic'])

    print(f"\n=== RESULTS: {args.adapter_path} ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"Internal Recall: {int_recall:.4f}")
    print(f"Det Recall: {det_recall:.4f}")
    print(f"Logit-gen gap: {0.5152 - int_recall:.4f}")
    print(classification_report(y_true, y_pred, labels=['internal', 'deterministic']))

    results = {
        'adapter_path': args.adapter_path,
        'eval_type': 'generation-level',
        'n_val': len(y_true),
        'val_label_dist': label_dist,
        'pred_dist': pred_dist,
        'accuracy': round(acc, 4),
        'balanced_accuracy': round(bal_acc, 4),
        'internal_recall': round(int_recall, 4),
        'det_recall': round(det_recall, 4),
        'confusion_matrix': cm.tolist(),
    }
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved to {args.output_path}")

if __name__ == '__main__':
    main()
