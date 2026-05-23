#!/usr/bin/env python3
"""Plan-008: Router Size Ablation — train 3 models, compare routing performance."""

import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODELS = [
    {
        "name": "qwen3-1.7b",
        "model_path": "./models/Qwen/Qwen3-1.7B",
        "output_dir": "checkpoints/plan-008-qwen3-1.7b",
        "batch_size": 4,
        "gradient_accumulation_steps": 8,
    },
    {
        "name": "qwen2.5-1.5b",
        "model_path": "./models/Qwen/Qwen2.5-1.5B",
        "output_dir": "checkpoints/plan-008-qwen2.5-1.5b",
        "batch_size": 4,
        "gradient_accumulation_steps": 8,
    },
    {
        "name": "qwen2.5-3b",
        "model_path": "./models/Qwen/Qwen2.5-3B",
        "output_dir": "checkpoints/plan-008-qwen2.5-3b",
        "batch_size": 4,
        "gradient_accumulation_steps": 8,
    },
]

DATA_DIR = "data/plan-004-implicit"
EPOCHS = 3
LR = 2e-5
SEED = 42
GPU_ID = 0
MAX_SEQ_LEN = 2048
SIMULATE_PREFIX = "[SIMULATE]"


def resolve_model_path(path):
    """Handle ModelScope cache naming (dots -> triple underscores)."""
    if os.path.exists(path):
        return path
    parent = os.path.dirname(path)
    basename = os.path.basename(path)
    alt = basename.replace(".", "___")
    alt_path = os.path.join(parent, alt)
    if os.path.exists(alt_path):
        return alt_path
    for d in os.listdir(parent):
        if d.replace("___", ".") == basename:
            return os.path.join(parent, d)
    return path


def load_val_records():
    """Load and split data, return val records matching training split."""
    data_path = os.path.join(DATA_DIR, "implicit_sft.jsonl")
    with open(data_path) as f:
        records = [json.loads(line) for line in f if line.strip()]

    rng = random.Random(SEED)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    split_idx = int(len(records) * (1 - 0.15))
    val_indices = indices[split_idx:]
    return [records[i] for i in val_indices]


def train_model(model_cfg):
    """Train a single model via subprocess. Returns True on success."""
    cmd = [
        sys.executable, "src/training/implicit_router_sft.py", "train",
        "--data_dir", DATA_DIR,
        "--model_name", model_cfg["model_path"],
        "--output_dir", model_cfg["output_dir"],
        "--epochs", str(EPOCHS),
        "--batch_size", str(model_cfg["batch_size"]),
        "--gradient_accumulation_steps", str(model_cfg["gradient_accumulation_steps"]),
        "--lr", str(LR),
        "--seed", str(SEED),
        "--gpu_id", str(GPU_ID),
        "--upsample_minority",
        "--max_seq_len", str(MAX_SEQ_LEN),
    ]

    log_dir = os.path.join(model_cfg["output_dir"], "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "train.log")

    print(f"\n{'='*60}")
    print(f"Training {model_cfg['name']}")
    print(f"  model: {model_cfg['model_path']}")
    print(f"  bs={model_cfg['batch_size']}, ga={model_cfg['gradient_accumulation_steps']}")
    print(f"  log: {log_path}")
    print(f"{'='*60}\n", flush=True)

    with open(log_path, "w") as log_f:
        proc = subprocess.run(
            cmd, stdout=log_f, stderr=subprocess.STDOUT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
        )

    if proc.returncode != 0:
        with open(log_path) as f:
            log_content = f.read()
        if "CUDA out of memory" in log_content or "OutOfMemoryError" in log_content:
            print(f"OOM for {model_cfg['name']}, retrying bs=2 ga=16", flush=True)
            model_cfg["batch_size"] = 2
            model_cfg["gradient_accumulation_steps"] = 16

            retry_cmd = list(cmd)
            bs_idx = retry_cmd.index("--batch_size") + 1
            ga_idx = retry_cmd.index("--gradient_accumulation_steps") + 1
            retry_cmd[bs_idx] = "2"
            retry_cmd[ga_idx] = "16"

            log_path_retry = os.path.join(log_dir, "train_retry.log")
            with open(log_path_retry, "w") as log_f:
                proc = subprocess.run(
                    retry_cmd, stdout=log_f, stderr=subprocess.STDOUT,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
                )
            if proc.returncode != 0:
                print(f"FAIL: {model_cfg['name']} failed even with bs=2")
                return False
        else:
            print(f"FAIL: {model_cfg['name']} training error (see {log_path})")
            tail = log_content[-500:] if len(log_content) > 500 else log_content
            print(tail)
            return False

    sentinel = os.path.join(model_cfg["output_dir"], "TRAINING_COMPLETE")
    if os.path.exists(sentinel):
        print(f"OK: {model_cfg['name']} training complete", flush=True)
        return True
    print(f"WARN: {model_cfg['name']} no TRAINING_COMPLETE sentinel")
    return os.path.exists(os.path.join(model_cfg["output_dir"], "final_adapter"))


def logit_level_eval(model_cfg, val_records):
    """Logit-level routing eval: predict route from first generated token."""
    adapter_path = os.path.join(model_cfg["output_dir"], "final_adapter")
    if not os.path.exists(adapter_path):
        print(f"No adapter at {adapter_path}")
        return None

    print(f"Evaluating {model_cfg['name']} ({len(val_records)} val samples)...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_path"],
        torch_dtype=torch.bfloat16,
        device_map={"": GPU_ID},
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    sim_token_id = tokenizer.encode("[", add_special_tokens=False)[0]

    y_true = []
    y_pred = []
    eval_losses = []
    sim_probs = []

    with torch.no_grad():
        for i, rec in enumerate(val_records):
            messages = rec["messages"]
            route_label = rec.get("route_label", "internal")
            true_label = 1 if route_label == "deterministic" else 0
            y_true.append(true_label)

            prompt_messages = messages[:-1]
            try:
                prompt_text = tokenizer.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                prompt_text = tokenizer.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True,
                )

            inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            prompt_len = inputs["input_ids"].shape[1]

            # Forward pass on prompt only -> logits at last position
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)
            sim_prob = probs[sim_token_id].item()
            sim_probs.append(sim_prob)

            pred_token = logits.argmax().item()
            pred_label = 1 if pred_token == sim_token_id else 0
            y_pred.append(pred_label)

            # Full-sequence eval loss
            try:
                full_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False,
                    enable_thinking=False,
                )
            except TypeError:
                full_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False,
                )
            full_inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
            full_ids = full_inputs["input_ids"].to(model.device)
            labels = full_ids.clone()
            labels[0, :min(prompt_len, labels.shape[1])] = -100
            attn = full_inputs["attention_mask"].to(model.device)

            full_out = model(input_ids=full_ids, attention_mask=attn, labels=labels)
            eval_losses.append(full_out.loss.item())

            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(val_records)}...", flush=True)

    del model, base_model
    torch.cuda.empty_cache()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    classes = {"internal": 0, "deterministic": 1}
    per_class = {}
    for cls_name, cls_id in classes.items():
        mask_true = y_true == cls_id
        mask_pred = y_pred == cls_id
        tp = int((mask_true & mask_pred).sum())
        fp = int((~mask_true & mask_pred).sum())
        fn = int((mask_true & ~mask_pred).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[cls_name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(mask_true.sum()),
        }

    recalls = [per_class[c]["recall"] for c in classes]
    balanced_acc = float(np.mean(recalls))

    metrics = {
        "eval_loss": round(float(np.mean(eval_losses)), 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "accuracy": round(float((y_true == y_pred).mean()), 4),
        "internal_recall": per_class["internal"]["recall"],
        "per_class": per_class,
        "total_samples": len(y_true),
        "mean_sim_prob": round(float(np.mean(sim_probs)), 4),
    }

    print(f"  balanced_acc={balanced_acc:.4f} internal_recall={per_class['internal']['recall']:.4f} "
          f"eval_loss={metrics['eval_loss']:.4f}", flush=True)
    return metrics


def main():
    os.chdir(".")

    # Resolve model paths (ModelScope naming)
    for cfg in MODELS:
        cfg["model_path"] = resolve_model_path(cfg["model_path"])
        if not os.path.exists(cfg["model_path"]):
            print(f"FATAL: {cfg['name']} model not found at {cfg['model_path']}")
            sys.exit(1)

    val_records = load_val_records()
    print(f"Val set: {len(val_records)} samples")
    route_dist = Counter(r.get("route_label", "internal") for r in val_records)
    print(f"Val route distribution: {dict(route_dist)}")

    results = {}
    start_time = time.time()

    for model_cfg in MODELS:
        model_start = time.time()

        success = train_model(model_cfg)
        train_time = time.time() - model_start

        if not success:
            results[model_cfg["name"]] = {"error": "training failed", "train_time_s": round(train_time, 1)}
            continue

        eval_start = time.time()
        metrics = logit_level_eval(model_cfg, val_records)
        eval_time = time.time() - eval_start

        if metrics is None:
            results[model_cfg["name"]] = {"error": "eval failed", "train_time_s": round(train_time, 1)}
            continue

        results[model_cfg["name"]] = {
            "model_path": model_cfg["model_path"],
            "output_dir": model_cfg["output_dir"],
            "batch_size": model_cfg["batch_size"],
            "gradient_accumulation_steps": model_cfg["gradient_accumulation_steps"],
            "train_time_s": round(train_time, 1),
            "eval_time_s": round(eval_time, 1),
            **metrics,
        }

    total_time = time.time() - start_time

    summary = {
        "experiment": "plan-008-router-size-ablation",
        "timestamp": datetime.now().isoformat(),
        "total_time_s": round(total_time, 1),
        "config": {
            "epochs": EPOCHS, "lr": LR, "seed": SEED,
            "data_dir": DATA_DIR, "effective_batch_size": 32,
            "max_seq_len": MAX_SEQ_LEN, "lora_r": 16,
        },
        "results": results,
    }

    os.makedirs("results/plan-008", exist_ok=True)
    with open("results/plan-008/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Plan-008 Complete — {total_time/60:.1f} min total")
    print(f"{'='*60}")
    print(f"\n{'Model':<20} {'BalAcc':>8} {'IntRecall':>10} {'EvalLoss':>10} {'Time(min)':>10}")
    print("-" * 62)
    for name, res in results.items():
        if "error" in res:
            print(f"{name:<20} {'ERR':>8} {res['error']:>30}")
        else:
            print(f"{name:<20} {res['balanced_accuracy']:>8.4f} {res['internal_recall']:>10.4f} "
                  f"{res['eval_loss']:>10.4f} {res['train_time_s']/60:>10.1f}")

    print(f"\nSummary: results/plan-008/summary.json")


if __name__ == "__main__":
    main()
