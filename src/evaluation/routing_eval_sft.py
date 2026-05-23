"""Routing classification eval for implicit router SFT models.

Supports both thinking-on (strip <think> tags) and thinking-off modes.
"""

import json
import os
import random
import re
import sys
import time
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split as sklearn_split
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

SIMULATE_PREFIX = "[SIMULATE]"


def strip_think_tags(text: str) -> str:
    return re.sub(r'^<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()


def load_val_split(path, eval_split=0.15, seed=42):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    episodes = {}
    for r in records:
        key = (r.get("task_type", "unknown"), r.get("variation", 0))
        episodes.setdefault(key, []).append(r)

    ep_keys = list(episodes.keys())
    ep_labels = [Counter(r["route_label"] for r in episodes[k]).most_common(1)[0][0] for k in ep_keys]
    _, val_keys = sklearn_split(ep_keys, test_size=eval_split, random_state=seed, stratify=ep_labels)

    rng = random.Random(seed)
    val_records = [r for k in val_keys for r in episodes[k]]
    rng.shuffle(val_records)

    val_lbl = Counter(r["route_label"] for r in val_records)
    print(f"Val split: {len(val_keys)} episodes, {len(val_records)} steps")
    print(f"  Route labels: {dict(val_lbl)}")
    return val_records


def run_eval(
    adapter_path, data_path, base_model, gpu_id=1,
    eval_split=0.15, seed=42, max_new_tokens=64, batch_size=16,
    output_path=None, enable_thinking=False,
):
    device = f"cuda:{gpu_id}"
    val_records = load_val_split(data_path, eval_split, seed)

    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map=device)
    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()

    gt_labels, pred_labels, errors = [], [], []
    mode_str = "thinking_on" if enable_thinking else "thinking_off"
    print(f"Mode: {mode_str}, inference on {len(val_records)} samples (batch_size={batch_size})...")
    t0 = time.time()

    for i in range(0, len(val_records), batch_size):
        batch = val_records[i : i + batch_size]
        prompts = []
        for rec in batch:
            prompt_messages = rec["messages"][:-1]
            try:
                prompt_text = tokenizer.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            except TypeError:
                prompt_text = tokenizer.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True,
                )
            prompts.append(prompt_text)

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, temperature=None, top_p=None,
            )

        for j, (rec, output_ids) in enumerate(zip(batch, outputs)):
            prompt_len = inputs["input_ids"].shape[1]
            generated_ids = output_ids[prompt_len:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            clean_text = strip_think_tags(generated_text) if enable_thinking else generated_text

            gt_route = rec["route_label"]
            pred_route = "deterministic" if clean_text.startswith(SIMULATE_PREFIX) else "internal"
            gt_labels.append(gt_route)
            pred_labels.append(pred_route)

            if gt_route != pred_route:
                errors.append({
                    "idx": i + j, "gt": gt_route, "pred": pred_route,
                    "generated": generated_text[:200], "clean": clean_text[:200],
                    "task_type": rec.get("task_type", ""), "step_idx": rec.get("step_idx", -1),
                })

        if (i // batch_size) % 10 == 0:
            done = min(i + batch_size, len(val_records))
            print(f"  {done}/{len(val_records)} ({time.time()-t0:.1f}s)")

    elapsed = time.time() - t0
    print(f"Inference done in {elapsed:.1f}s")

    label_names = ["deterministic", "internal"]
    acc = accuracy_score(gt_labels, pred_labels)
    bal_acc = balanced_accuracy_score(gt_labels, pred_labels)
    cm = confusion_matrix(gt_labels, pred_labels, labels=label_names)
    prec, rec_s, f1, _ = precision_recall_fscore_support(gt_labels, pred_labels, labels=label_names, zero_division=0)

    metrics = {
        "accuracy": round(float(acc), 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "deterministic_precision": round(float(prec[0]), 4),
        "deterministic_recall": round(float(rec_s[0]), 4),
        "deterministic_f1": round(float(f1[0]), 4),
        "internal_precision": round(float(prec[1]), 4),
        "internal_recall": round(float(rec_s[1]), 4),
        "internal_f1": round(float(f1[1]), 4),
        "confusion_matrix": cm.tolist(),
        "confusion_labels": label_names,
        "n_val": len(val_records),
        "n_errors": len(errors),
        "label_distribution": dict(Counter(gt_labels)),
        "pred_distribution": dict(Counter(pred_labels)),
        "enable_thinking": enable_thinking,
    }

    print(f"\n{'='*60}")
    print(f"ROUTING EVAL RESULTS ({mode_str})")
    print(f"{'='*60}")
    print(f"Accuracy:           {metrics['accuracy']*100:.1f}%")
    print(f"Balanced Accuracy:  {metrics['balanced_accuracy']*100:.1f}%")
    print(f"Internal Recall:    {metrics['internal_recall']*100:.1f}%")
    print(f"Internal Precision: {metrics['internal_precision']*100:.1f}%")
    print(f"Det Recall:         {metrics['deterministic_recall']*100:.1f}%")
    print(f"Det Precision:      {metrics['deterministic_precision']*100:.1f}%")
    print(f"\nConfusion Matrix (rows=true, cols=pred):")
    print(f"             det    int")
    print(f"  det     {cm[0][0]:5d}  {cm[0][1]:5d}")
    print(f"  int     {cm[1][0]:5d}  {cm[1][1]:5d}")
    print(f"\nGT dist:   {dict(Counter(gt_labels))}")
    print(f"Pred dist: {dict(Counter(pred_labels))}")
    print(f"{'='*60}")

    report = {
        "metrics": metrics, "adapter_path": adapter_path,
        "base_model": base_model, "data_path": data_path,
        "eval_split": eval_split, "seed": seed, "mode": mode_str,
        "sample_errors": errors[:20],
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nReport saved to: {output_path}")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--base_model", default="./models/Qwen/Qwen3-1.7B")
    parser.add_argument("--gpu_id", type=int, default=1)
    parser.add_argument("--eval_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--enable_thinking", action="store_true")
    args = parser.parse_args()
    run_eval(
        adapter_path=args.adapter_path, data_path=args.data_path,
        base_model=args.base_model, gpu_id=args.gpu_id,
        eval_split=args.eval_split, seed=args.seed,
        batch_size=args.batch_size, output_path=args.output_path,
        enable_thinking=args.enable_thinking,
    )
