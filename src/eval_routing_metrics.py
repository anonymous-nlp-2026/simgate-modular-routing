"""Evaluate generative router routing classification metrics on val split."""
import json
import os
import random
import re
import sys
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split as sklearn_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, precision_recall_fscore_support,
)
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

SIMULATE_PREFIX = "[SIMULATE]"


def load_val_split(data_path, eval_split=0.15, seed=42):
    if os.path.isdir(data_path):
        data_path = os.path.join(data_path, "implicit_sft.jsonl")
    records = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    episodes: Dict[tuple, List[Dict]] = {}
    for r in records:
        key = (r.get("task_type", "unknown"), r.get("variation", 0))
        episodes.setdefault(key, []).append(r)

    ep_keys = list(episodes.keys())
    ep_labels = []
    for key in ep_keys:
        lbls = Counter(r["route_label"] for r in episodes[key])
        ep_labels.append(lbls.most_common(1)[0][0])

    _, val_keys = sklearn_split(
        ep_keys, test_size=eval_split, random_state=seed, stratify=ep_labels
    )
    rng = random.Random(seed)
    val_records = [r for k in val_keys for r in episodes[k]]
    rng.shuffle(val_records)
    return val_records


def strip_thinking(text: str) -> str:
    """Remove Qwen3 <think>...</think> blocks from generated text."""
    result = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # If think block wasn't closed, strip everything from <think> onwards
    if "<think>" in result:
        result = result[:result.index("<think>")].strip()
    # If only </think> remains (partial), strip it
    result = result.replace("</think>", "").strip()
    return result


def parse_output(text: str) -> str:
    text = strip_thinking(text)
    if text.startswith(SIMULATE_PREFIX):
        return "deterministic"
    return "internal"


def run_eval(
    adapter_path: str,
    base_model: str,
    data_dir: str,
    gpu_id: int = 0,
    max_new_tokens: int = 128,
    output_path: str = None,
):
    print(f"Loading val split from {data_dir}...")
    val_records = load_val_split(data_dir)
    val_lbl = Counter(r["route_label"] for r in val_records)
    print(f"Val: {len(val_records)} steps, labels: {dict(val_lbl)}")

    print(f"Loading model from {base_model} + adapter {adapter_path}...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": gpu_id},
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    model.config.pad_token_id = tokenizer.pad_token_id
    print("Model loaded.")

    y_true = []
    y_pred = []
    raw_samples = []

    for i, rec in enumerate(val_records):
        messages = rec["messages"]
        prompt_messages = messages[:-1]  # system + user
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                temperature=0.1, do_sample=True,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        cleaned = strip_thinking(raw_output)
        pred_route = parse_output(raw_output)
        true_route = rec["route_label"]

        y_true.append(true_route)
        y_pred.append(pred_route)
        raw_samples.append({
            "idx": i,
            "true": true_route,
            "pred": pred_route,
            "raw": raw_output[:200],
            "cleaned": cleaned[:100],
        })

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(val_records)}] true={true_route}, pred={pred_route}, cleaned={cleaned[:80]}")

    # Metrics
    labels = ["internal", "deterministic"]
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)

    report = classification_report(y_true, y_pred, labels=labels, digits=4)

    results = {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "internal_recall": round(rec[0], 4),
        "internal_precision": round(prec[0], 4),
        "internal_f1": round(f1[0], 4),
        "det_recall": round(rec[1], 4),
        "det_precision": round(prec[1], 4),
        "det_f1": round(f1[1], 4),
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm.tolist(),
        },
        "val_size": len(val_records),
        "val_label_dist": dict(val_lbl),
        "pred_dist": dict(Counter(y_pred)),
        "sample_outputs": raw_samples[:20],
    }

    print(f"\n{'='*60}")
    print("ROUTING CLASSIFICATION METRICS (val split)")
    print(f"{'='*60}")
    print(f"Accuracy:          {acc:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"Internal Recall:   {rec[0]:.4f}")
    print(f"Internal Precision:{prec[0]:.4f}")
    print(f"Det Recall:        {rec[1]:.4f}")
    print(f"Det Precision:     {prec[1]:.4f}")
    print(f"\nConfusion Matrix (rows=true, cols=pred):")
    print(f"              internal  deterministic")
    print(f"  internal    {cm[0][0]:>8d}  {cm[0][1]:>13d}")
    print(f"  deterministic{cm[1][0]:>7d}  {cm[1][1]:>13d}")
    print(f"\n{report}")
    print(f"Pred dist: {dict(Counter(y_pred))}")
    print(f"True dist: {dict(val_lbl)}")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--adapter_path", type=str, required=True)
    p.add_argument("--base_model", type=str, default="./models/Qwen/Qwen3-1.7B")
    p.add_argument("--data_dir", type=str, default="data/plan-004-implicit/")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--output_path", type=str, default=None)
    args = p.parse_args()
    run_eval(
        adapter_path=args.adapter_path,
        base_model=args.base_model,
        data_dir=args.data_dir,
        gpu_id=args.gpu_id,
        output_path=args.output_path,
    )
