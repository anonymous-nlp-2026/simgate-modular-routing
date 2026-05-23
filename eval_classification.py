"""Evaluate plan-004 implicit router: classification metrics on eval split."""
import json
import random
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    precision_recall_fscore_support, confusion_matrix,
    classification_report
)

SEED = 42
EVAL_SPLIT = 0.15
SIMULATE_PREFIX = "[SIMULATE]"
MAX_NEW_TOKENS = 60
THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

def strip_think(text):
    return THINK_RE.sub("", text).strip()

def load_eval_split(data_path, seed=SEED, eval_split=EVAL_SPLIT):
    records = []
    with open(data_path) as f:
        for line in f:
            records.append(json.loads(line))
    random.seed(seed)
    random.shuffle(records)
    n_eval = int(len(records) * eval_split)
    return records[-n_eval:]

def main():
    data_path = "data/plan-004-implicit/implicit_sft.jsonl"
    adapter_path = "checkpoints/plan-004-implicit/final_adapter"
    base_model = "./models/Qwen/Qwen3-1.7B"
    device = "cuda:1"

    print("Loading eval split...")
    eval_records = load_eval_split(data_path)
    print(f"Eval set: {len(eval_records)} samples")

    gt_labels = [r["route_label"] for r in eval_records]
    from collections import Counter
    print(f"Label dist: {dict(Counter(gt_labels))}")

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.bfloat16, device_map=device,
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    print("Model loaded.")

    pred_labels = []
    pred_texts = []
    raw_texts = []

    for i, rec in enumerate(eval_records):
        messages = rec["messages"][:-1]
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False
            )
        except TypeError:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False, pad_token_id=tokenizer.pad_token_id
            )

        gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
        raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        clean_text = strip_think(raw_text)

        if clean_text.startswith(SIMULATE_PREFIX):
            pred = "deterministic"
        else:
            pred = "internal"

        pred_labels.append(pred)
        pred_texts.append(clean_text[:120])
        raw_texts.append(raw_text[:120])

        if (i + 1) % 50 == 0 or i < 5:
            print(f"  [{i+1}/{len(eval_records)}] gt={rec['route_label']}, pred={pred}, clean={clean_text[:80]}")

    labels_list = ["deterministic", "internal"]
    acc = accuracy_score(gt_labels, pred_labels)
    f1_macro = f1_score(gt_labels, pred_labels, labels=labels_list, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(gt_labels, pred_labels)
    prec, rec_scores, f1_scores, sup = precision_recall_fscore_support(
        gt_labels, pred_labels, labels=labels_list, zero_division=0
    )
    cm = confusion_matrix(gt_labels, pred_labels, labels=labels_list)

    results = {
        "eval_accuracy": round(acc, 4),
        "eval_f1_macro": round(f1_macro, 4),
        "eval_balanced_accuracy": round(bal_acc, 4),
        "per_class": {
            "deterministic": {
                "precision": round(float(prec[0]), 4),
                "recall": round(float(rec_scores[0]), 4),
                "f1": round(float(f1_scores[0]), 4),
                "support": int(sup[0])
            },
            "internal": {
                "precision": round(float(prec[1]), 4),
                "recall": round(float(rec_scores[1]), 4),
                "f1": round(float(f1_scores[1]), 4),
                "support": int(sup[1])
            }
        },
        "confusion_matrix": {
            "true_det_pred_det": int(cm[0][0]),
            "true_det_pred_int": int(cm[0][1]),
            "true_int_pred_det": int(cm[1][0]),
            "true_int_pred_int": int(cm[1][1])
        },
        "is_degenerate": len(set(pred_labels)) == 1,
        "total_eval_samples": len(eval_records),
        "unique_predictions": list(set(pred_labels)),
        "note": "Qwen3 generates <think>...</think> tags; stripped before [SIMULATE] check"
    }

    print("\n=== Sample Predictions (with think stripped) ===")
    for label in labels_list:
        print(f"\n--- GT={label} ---")
        shown = 0
        for j, (gt, pr, txt, raw) in enumerate(zip(gt_labels, pred_labels, pred_texts, raw_texts)):
            if gt == label and shown < 5:
                marker = "OK" if gt == pr else "WRONG"
                print(f"  [{marker}] pred={pr}, clean: {txt}")
                shown += 1

    print("\n=== Classification Report ===")
    print(classification_report(gt_labels, pred_labels, labels=labels_list, digits=4, zero_division=0))

    print("\n=== Results JSON ===")
    print(json.dumps(results, indent=2))

    with open("checkpoints/plan-004-implicit/eval_report.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to checkpoints/plan-004-implicit/eval_report.json")

if __name__ == "__main__":
    main()
