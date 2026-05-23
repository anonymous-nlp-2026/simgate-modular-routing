# Plan-010: Cross-task transfer experiment.
# Train implicit router on one domain, eval on another to test routing signal generalization.
# Data: plan-004-v3/implicit_sft.jsonl (2040 steps, 11 tasks)
# Domains: "physical" (chemistry/phase-change/physics) vs "life" (biology/ecology)

import argparse
import json
import os
import random
import sys
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

MODEL_ID = "Qwen/Qwen3-1.7B"
LOCAL_MODEL = "./models/Qwen3-1.7B-hf"
SIMULATE_PREFIX = "[SIMULATE]"
DEFAULT_DATA_PATH = "data/plan-004-v3/implicit_sft.jsonl"

DOMAIN_MAP = {
    "physical": [
        "chemistry-mix",
        "freeze",
        "measure-melting-point-unknown-substance",
        "inclined-plane-determine-angle",
    ],
    "life": [
        "find-animal",
        "find-non-living-thing",
        "grow-fruit",
        "identify-life-stages-1",
        "identify-life-stages-2",
        "lifespan-longest-lived",
        "lifespan-longest-lived-then-shortest-lived",
    ],
}

TASK_TO_DOMAIN = {}
for domain, tasks in DOMAIN_MAP.items():
    for t in tasks:
        TASK_TO_DOMAIN[t] = domain


# ---------------------------------------------------------------------------
# Dataset (reuse from implicit_router_sft.py)
# ---------------------------------------------------------------------------

class ImplicitSFTDataset(torch.utils.data.Dataset):
    def __init__(self, records, tokenizer, max_seq_len=2048):
        self.records = records
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        messages = rec["messages"]
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        prompt_messages = messages[:-1]
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        full_ids = self.tokenizer(
            full_text, truncation=True, max_length=self.max_seq_len,
            return_tensors="pt"
        )["input_ids"].squeeze(0)
        prompt_ids = self.tokenizer(
            prompt_text, truncation=True, max_length=self.max_seq_len,
            return_tensors="pt"
        )["input_ids"].squeeze(0)
        labels = full_ids.clone()
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = -100
        return {
            "input_ids": full_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(full_ids),
        }


class PaddingCollator:
    def __init__(self, tokenizer, max_seq_len=2048):
        self.pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.max_seq_len = max_seq_len

    def __call__(self, features):
        max_len = min(
            max(f["input_ids"].size(0) for f in features),
            self.max_seq_len,
        )
        input_ids, labels, attention_mask = [], [], []
        for f in features:
            seq_len = f["input_ids"].size(0)
            pad_len = max_len - seq_len
            if pad_len > 0:
                input_ids.append(torch.cat([f["input_ids"], torch.full((pad_len,), self.pad_id, dtype=torch.long)]))
                labels.append(torch.cat([f["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))
                attention_mask.append(torch.cat([f["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
            else:
                input_ids.append(f["input_ids"][:max_len])
                labels.append(f["labels"][:max_len])
                attention_mask.append(f["attention_mask"][:max_len])
        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attention_mask),
        }


# ---------------------------------------------------------------------------
# Data loading & domain split
# ---------------------------------------------------------------------------

def load_and_split_by_domain(data_path: str) -> Dict[str, List[Dict]]:
    records_by_domain = defaultdict(list)
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            task = rec.get("task_type", "unknown")
            domain = TASK_TO_DOMAIN.get(task)
            if domain is None:
                print(f"WARNING: unknown task '{task}', skipping")
                continue
            records_by_domain[domain].append(rec)
    return dict(records_by_domain)


def in_domain_split(records: List[Dict], eval_ratio: float = 0.15, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    rng = random.Random(seed)
    by_task = defaultdict(list)
    for r in records:
        by_task[r["task_type"]].append(r)
    train_recs, eval_recs = [], []
    for task, items in by_task.items():
        rng.shuffle(items)
        n_eval = max(1, int(len(items) * eval_ratio))
        eval_recs.extend(items[:n_eval])
        train_recs.extend(items[n_eval:])
    rng.shuffle(train_recs)
    rng.shuffle(eval_recs)
    return train_recs, eval_recs


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class UpsamplingTrainer(Trainer):
    def __init__(self, weighted_sampler=None, data_collator_ref=None, **kwargs):
        super().__init__(**kwargs)
        self._weighted_sampler = weighted_sampler
        self._collator_ref = data_collator_ref

    def get_train_dataloader(self) -> DataLoader:
        if self._weighted_sampler is None:
            return super().get_train_dataloader()
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=self._weighted_sampler,
            collate_fn=self._collator_ref or self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


def train_router(
    train_records: List[Dict],
    val_records: List[Dict],
    output_dir: str,
    model_name: str = LOCAL_MODEL,
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 2e-5,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    max_seq_len: int = 2048,
    seed: int = 42,
    upsample: bool = True,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"\n{'='*60}")
    print(f"TRAINING: {len(train_records)} train, {len(val_records)} val")
    print(f"Output: {output_dir}")
    route_dist = Counter(r["route_label"] for r in train_records)
    print(f"Route distribution: {dict(route_dist)}")
    print(f"{'='*60}\n")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = ImplicitSFTDataset(train_records, tokenizer, max_seq_len)
    val_ds = ImplicitSFTDataset(val_records, tokenizer, max_seq_len)
    collator = PaddingCollator(tokenizer, max_seq_len)

    weighted_sampler = None
    if upsample and len(route_dist) > 1:
        sample_weights = [1.0 / route_dist[r["route_label"]] for r in train_records]
        weighted_sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        report_to="none",
        seed=seed,
        dataloader_drop_last=False,
    )

    trainer = UpsamplingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        weighted_sampler=weighted_sampler,
        data_collator_ref=collator,
    )
    trainer.train()

    adapter_dir = os.path.join(output_dir, "final_adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Adapter saved: {adapter_dir}")
    return adapter_dir


# ---------------------------------------------------------------------------
# Offline evaluation: predict routing on held-out SFT data
# ---------------------------------------------------------------------------

def evaluate_routing(
    adapter_path: str,
    eval_records: List[Dict],
    model_name: str = LOCAL_MODEL,
    max_new_tokens: int = 64,
    batch_eval: bool = False,
) -> Dict[str, Any]:
    print(f"\nEvaluating routing on {len(eval_records)} records...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda",
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    device = next(model.parameters()).device

    correct = 0
    total = 0
    per_task = defaultdict(lambda: {"correct": 0, "total": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0})
    per_class = defaultdict(lambda: {"correct": 0, "total": 0})
    predictions = []

    t0 = time.time()
    for i, rec in enumerate(eval_records):
        messages = rec["messages"]
        prompt_messages = messages[:-1]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, temperature=None, top_p=None,

            )
        gen_ids = out[0, inputs["input_ids"].shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        gen_text = re.sub(r'<think>.*?</think>\s*', '', gen_text, flags=re.DOTALL).strip()

        pred_route = "deterministic" if gen_text.startswith(SIMULATE_PREFIX) else "internal"
        true_route = rec["route_label"]
        task = rec.get("task_type", "unknown")

        is_correct = pred_route == true_route
        correct += int(is_correct)
        total += 1

        per_class[true_route]["total"] += 1
        per_class[true_route]["correct"] += int(is_correct)
        per_task[task]["total"] += 1
        per_task[task]["correct"] += int(is_correct)

        # confusion matrix entries (positive = deterministic)
        if true_route == "deterministic" and pred_route == "deterministic":
            per_task[task]["tp"] += 1
        elif true_route == "internal" and pred_route == "deterministic":
            per_task[task]["fp"] += 1
        elif true_route == "deterministic" and pred_route == "internal":
            per_task[task]["fn"] += 1
        else:
            per_task[task]["tn"] += 1

        predictions.append({
            "task_type": task,
            "variation": rec.get("variation", -1),
            "step_idx": rec.get("step_idx", -1),
            "true_route": true_route,
            "pred_route": pred_route,
            "correct": is_correct,
            "gen_text_prefix": gen_text[:80],
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(eval_records)}] acc={correct/total:.3f} ({elapsed:.1f}s)")

    elapsed_total = time.time() - t0
    overall_acc = correct / total if total > 0 else 0.0

    # balanced accuracy
    class_recalls = {}
    for cls, stats in per_class.items():
        class_recalls[cls] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
    balanced_acc = np.mean(list(class_recalls.values())) if class_recalls else 0.0

    # per-task metrics
    per_task_metrics = {}
    for task, stats in sorted(per_task.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        det_recall = stats["tp"] / (stats["tp"] + stats["fn"]) if (stats["tp"] + stats["fn"]) > 0 else None
        int_recall = stats["tn"] / (stats["tn"] + stats["fp"]) if (stats["tn"] + stats["fp"]) > 0 else None
        per_task_metrics[task] = {
            "accuracy": round(acc, 4),
            "total": stats["total"],
            "det_recall": round(det_recall, 4) if det_recall is not None else None,
            "int_recall": round(int_recall, 4) if int_recall is not None else None,
        }

    results = {
        "overall_accuracy": round(overall_acc, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "class_recalls": {k: round(v, 4) for k, v in class_recalls.items()},
        "class_counts": {k: v["total"] for k, v in per_class.items()},
        "per_task": per_task_metrics,
        "total_samples": total,
        "eval_time_seconds": round(elapsed_total, 1),
    }
    return results, predictions


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_balanced_acc_ci(predictions, n_boot=1000, ci=0.95, seed=42):
    """Bootstrap CI on balanced accuracy (mean of per-class recalls)."""
    from collections import defaultdict as _dd
    rng = np.random.RandomState(seed)
    n = len(predictions)
    bal_accs = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        per_class = _dd(lambda: {"correct": 0, "total": 0})
        for i in idx:
            p = predictions[i]
            per_class[p["true_route"]]["total"] += 1
            per_class[p["true_route"]]["correct"] += int(p["correct"])
        recalls = [s["correct"] / s["total"] for s in per_class.values() if s["total"] > 0]
        bal_accs.append(np.mean(recalls) if recalls else 0.0)
    lo = np.percentile(bal_accs, (1 - ci) / 2 * 100)
    hi = np.percentile(bal_accs, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(args):
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(proj_root, args.data_path)
    if not os.path.exists(data_path):
        data_path = args.data_path

    print(f"Loading data from: {data_path}")
    domain_data = load_and_split_by_domain(data_path)

    for d, recs in domain_data.items():
        task_dist = Counter(r["task_type"] for r in recs)
        label_dist = Counter(r["route_label"] for r in recs)
        print(f"Domain '{d}': {len(recs)} steps, tasks={dict(task_dist)}, labels={dict(label_dist)}")

    train_domain = args.train_domain
    eval_domain = args.eval_domain

    if train_domain not in domain_data:
        raise ValueError(f"Train domain '{train_domain}' not found. Available: {list(domain_data.keys())}")
    if eval_domain not in domain_data:
        raise ValueError(f"Eval domain '{eval_domain}' not found. Available: {list(domain_data.keys())}")

    train_all = domain_data[train_domain]
    eval_cross = domain_data[eval_domain]

    # In-domain split for control
    train_in, eval_in = in_domain_split(train_all, eval_ratio=0.15, seed=args.seed)

    output_base = os.path.join(
        proj_root, "checkpoints",
        f"plan-010-cross-task-{train_domain}-to-{eval_domain}-seed{args.seed}"
    )
    results_dir = os.path.join(
        proj_root, "results",
        f"plan-010-cross-task-{train_domain}-to-{eval_domain}-seed{args.seed}"
    )
    os.makedirs(results_dir, exist_ok=True)

    # --- Train ---
    print(f"\n{'#'*60}")
    print(f"# Training on domain: {train_domain}")
    print(f"# Train samples: {len(train_in)} (in-domain train split)")
    print(f"# In-domain eval: {len(eval_in)}")
    print(f"# Cross-domain eval ({eval_domain}): {len(eval_cross)}")
    print(f"{'#'*60}\n")

    adapter_path = train_router(
        train_records=train_in,
        val_records=eval_in,
        output_dir=output_base,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        upsample=True,
    )

    # --- Evaluate in-domain ---
    print(f"\n{'#'*60}")
    print(f"# In-domain evaluation ({train_domain})")
    print(f"{'#'*60}")
    in_domain_results, in_domain_preds = evaluate_routing(
        adapter_path=adapter_path,
        eval_records=eval_in,
        model_name=args.model_name,
    )

    # --- Evaluate cross-domain ---
    print(f"\n{'#'*60}")
    print(f"# Cross-domain evaluation ({eval_domain})")
    print(f"{'#'*60}")
    cross_domain_results, cross_domain_preds = evaluate_routing(
        adapter_path=adapter_path,
        eval_records=eval_cross,
        model_name=args.model_name,
    )

    # --- Bootstrap CI on balanced accuracy ---
    in_acc_lo, in_acc_hi = bootstrap_balanced_acc_ci(in_domain_preds, seed=args.seed)
    cross_acc_lo, cross_acc_hi = bootstrap_balanced_acc_ci(cross_domain_preds, seed=args.seed)
    in_domain_results["accuracy_95ci"] = [round(in_acc_lo, 4), round(in_acc_hi, 4)]
    cross_domain_results["accuracy_95ci"] = [round(cross_acc_lo, 4), round(cross_acc_hi, 4)]

    # --- Compile final results ---
    final = {
        "experiment": "plan-010-cross-task-v2",
        "train_domain": train_domain,
        "eval_domain": eval_domain,
        "seed": args.seed,
        "model": args.model_name,
        "domain_split": DOMAIN_MAP,
        "train_size": len(train_in),
        "timestamp": datetime.now().isoformat(),
        "in_domain": in_domain_results,
        "cross_domain": cross_domain_results,
        "transfer_gap": round(
            in_domain_results["balanced_accuracy"] - cross_domain_results["balanced_accuracy"], 4
        ),
    }

    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {results_path}")

    preds_path = os.path.join(results_dir, "predictions.jsonl")
    with open(preds_path, "w") as f:
        for p in in_domain_preds:
            p["eval_type"] = "in_domain"
            f.write(json.dumps(p) + "\n")
        for p in cross_domain_preds:
            p["eval_type"] = "cross_domain"
            f.write(json.dumps(p) + "\n")

    # --- Print summary ---
    print(f"\n{'='*60}")
    print(f"CROSS-TASK TRANSFER RESULTS")
    print(f"Train: {train_domain} ({len(train_in)} steps) → Eval: {eval_domain} ({len(eval_cross)} steps)")
    print(f"{'='*60}")
    print(f"In-domain accuracy:    {in_domain_results['overall_accuracy']:.4f}")
    print(f"In-domain balanced:    {in_domain_results['balanced_accuracy']:.4f}  [{in_acc_lo:.4f}, {in_acc_hi:.4f}]")
    print(f"Cross-domain accuracy: {cross_domain_results['overall_accuracy']:.4f}")
    print(f"Cross-domain balanced: {cross_domain_results['balanced_accuracy']:.4f}  [{cross_acc_lo:.4f}, {cross_acc_hi:.4f}]")
    print(f"Transfer gap:          {final['transfer_gap']:.4f}")
    print(f"\nIn-domain per-class recalls: {in_domain_results['class_recalls']}")
    print(f"Cross-domain per-class recalls: {cross_domain_results['class_recalls']}")
    print(f"\nPer-task (cross-domain):")
    for task, m in cross_domain_results["per_task"].items():
        print(f"  {task}: acc={m['accuracy']:.3f} (n={m['total']}, det_recall={m['det_recall']}, int_recall={m['int_recall']})")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Plan-010: Cross-task transfer experiment")
    parser.add_argument("--train_domain", type=str, required=True, choices=["physical", "life"])
    parser.add_argument("--eval_domain", type=str, required=True, choices=["physical", "life"])
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model_name", type=str, default=LOCAL_MODEL)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    run_experiment(args)


if __name__ == "__main__":
    main()
