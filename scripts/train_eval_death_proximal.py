# Train and evaluate death-proximal router models.
# Trains on pre-split data from filter_death_proximal.py, evaluates routing accuracy.

import argparse
import json
import os
import sys
import random
import time
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

SIMULATE_PREFIX = "[SIMULATE]"


class ImplicitSFTDataset(Dataset):
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
            messages, tokenize=False, add_generation_prompt=False)
        prompt_text = self.tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True)

        full_ids = self.tokenizer(
            full_text, truncation=True, max_length=self.max_seq_len,
            return_tensors="pt")["input_ids"].squeeze(0)
        prompt_ids = self.tokenizer(
            prompt_text, truncation=True, max_length=self.max_seq_len,
            return_tensors="pt")["input_ids"].squeeze(0)

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
        max_len = min(max(f["input_ids"].size(0) for f in features), self.max_seq_len)
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


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def train_model(train_path, eval_path, model_name, output_dir, args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_records = load_jsonl(train_path)
    eval_records = load_jsonl(eval_path)
    print(f"Train: {len(train_records)}, Eval: {len(eval_records)}")

    route_dist = Counter(r["route_label"] for r in train_records)
    print(f"Route distribution (train): {dict(route_dist)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16,
        device_map={"": args.gpu_id}, trust_remote_code=True)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.lora_r,
        lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = ImplicitSFTDataset(train_records, tokenizer, args.max_seq_len)
    eval_dataset = ImplicitSFTDataset(eval_records, tokenizer, args.max_seq_len)
    collator = PaddingCollator(tokenizer, args.max_seq_len)

    os.makedirs(output_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=True,
        logging_steps=10,
        report_to="none",
        seed=args.seed,
        save_total_limit=2,
        gradient_accumulation_steps=max(1, 32 // args.batch_size),
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        data_collator=collator)
    trainer.train()

    adapter_dir = os.path.join(output_dir, "final_adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Adapter saved to {adapter_dir}")

    with open(os.path.join(output_dir, "train_config.json"), "w") as f:
        json.dump({"train_path": train_path, "eval_path": eval_path,
                    "model_name": model_name, **vars(args)}, f, indent=2)

    del model, trainer
    torch.cuda.empty_cache()
    return adapter_dir


def evaluate_model(adapter_path, model_name, eval_path, eval_meta_path, gpu_id):
    eval_records = load_jsonl(eval_path)
    eval_meta = load_jsonl(eval_meta_path) if eval_meta_path else None

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16,
        device_map={"": gpu_id}, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()

    results = []
    t0 = time.time()
    for i, rec in enumerate(eval_records):
        messages = rec["messages"]
        prompt = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=128,
                do_sample=False, temperature=None, top_p=None)

        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        if gen_text.startswith(SIMULATE_PREFIX):
            pred_route = "deterministic"
        else:
            pred_route = "internal"

        oracle_route = rec["route_label"]
        meta = eval_meta[i] if eval_meta else {}

        results.append({
            "step_idx": rec.get("step_idx", -1),
            "task_type": rec.get("task_type", ""),
            "episode_id": rec.get("episode_id", -1),
            "variation": rec.get("variation", -1),
            "oracle": oracle_route,
            "predicted": pred_route,
            "correct": pred_route == oracle_route,
            "signal_source": meta.get("signal_source", rec.get("signal_source", "")),
            "fatal_step": meta.get("fatal_step"),
            "generated": gen_text[:100],
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(eval_records)}] {elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"Eval done: {len(results)} steps in {elapsed:.1f}s")

    del model, base
    torch.cuda.empty_cache()
    return results


def compute_balanced_accuracy(results, filter_fn=None):
    filtered = [r for r in results if (filter_fn is None or filter_fn(r))]
    if not filtered:
        return {"balanced_accuracy": None, "n": 0}

    per_class = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in filtered:
        per_class[r["oracle"]]["total"] += 1
        if r["correct"]:
            per_class[r["oracle"]]["correct"] += 1

    recalls = []
    for cls, stats in per_class.items():
        if stats["total"] > 0:
            recalls.append(stats["correct"] / stats["total"])

    ba = sum(recalls) / len(recalls) if recalls else 0.0
    return {
        "balanced_accuracy": round(ba, 4),
        "n": len(filtered),
        "per_class": {k: {"acc": round(v["correct"]/v["total"], 4) if v["total"] > 0 else 0,
                          "correct": v["correct"], "total": v["total"]}
                      for k, v in per_class.items()},
    }


def compute_episode_accuracy(results):
    ep_groups = defaultdict(list)
    for r in results:
        key = (r["task_type"], r["episode_id"], r["variation"])
        ep_groups[key].append(r)

    per_class = defaultdict(lambda: {"correct": 0, "total": 0})
    for key, recs in ep_groups.items():
        votes = Counter(r["predicted"] for r in recs)
        ep_pred = votes.most_common(1)[0][0]
        ep_oracle = recs[0]["oracle"]
        per_class[ep_oracle]["total"] += 1
        if ep_pred == ep_oracle:
            per_class[ep_oracle]["correct"] += 1

    recalls = []
    for cls, stats in per_class.items():
        if stats["total"] > 0:
            recalls.append(stats["correct"] / stats["total"])

    ba = sum(recalls) / len(recalls) if recalls else 0.0
    return {
        "balanced_accuracy": round(ba, 4),
        "n_episodes": len(ep_groups),
        "per_class": {k: {"acc": round(v["correct"]/v["total"], 4) if v["total"] > 0 else 0,
                          "correct": v["correct"], "total": v["total"]}
                      for k, v in per_class.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/death-proximal")
    parser.add_argument("--model_name", default="./models/Qwen/Qwen3-0.6B")
    parser.add_argument("--checkpoint_base", default="checkpoints/death-proximal")
    parser.add_argument("--results_dir", default="results/death-proximal")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--variants", default="full,k1,k3,k5,k10")
    args = parser.parse_args()

    os.chdir(".")
    eval_path = os.path.join(args.data_dir, "eval.jsonl")
    eval_meta_path = os.path.join(args.data_dir, "eval_meta.jsonl")

    variant_map = {
        "full": "train_full.jsonl",
        "k1": "train_k1.jsonl",
        "k3": "train_k3.jsonl",
        "k5": "train_k5.jsonl",
        "k10": "train_k10.jsonl",
    }

    variants = [v.strip() for v in args.variants.split(",")]
    all_results = {}

    for variant in variants:
        print(f"\n{'='*60}")
        print(f"  VARIANT: {variant}")
        print(f"{'='*60}")

        train_file = variant_map[variant]
        train_path = os.path.join(args.data_dir, train_file)
        ckpt_dir = os.path.join(args.checkpoint_base, variant)
        adapter_path = os.path.join(ckpt_dir, "final_adapter")

        if not args.skip_train:
            print(f"\n--- Training {variant} ---")
            adapter_path = train_model(
                train_path, eval_path, args.model_name, ckpt_dir, args)
        else:
            print(f"Skipping train, using {adapter_path}")

        if not args.skip_eval:
            print(f"\n--- Evaluating {variant} ---")
            step_results = evaluate_model(
                adapter_path, args.model_name, eval_path, eval_meta_path, args.gpu_id)

            step_ba_all = compute_balanced_accuracy(step_results)
            step_ba_proximal = compute_balanced_accuracy(
                step_results,
                lambda r: r.get("signal_source") == "death_asymmetry" and
                          r.get("fatal_step") is not None and
                          abs(r["step_idx"] - r["fatal_step"]) <= 5)
            episode_ba = compute_episode_accuracy(step_results)

            metrics = {
                "variant": variant,
                "step_ba_all": step_ba_all,
                "step_ba_proximal": step_ba_proximal,
                "episode_ba": episode_ba,
            }
            all_results[variant] = metrics

            os.makedirs(args.results_dir, exist_ok=True)
            with open(os.path.join(args.results_dir, f"eval_{variant}.json"), "w") as f:
                json.dump(metrics, f, indent=2)

            print(f"\n  Step BA (all):      {step_ba_all['balanced_accuracy']}")
            print(f"  Step BA (proximal): {step_ba_proximal['balanced_accuracy']}")
            print(f"  Episode BA:         {episode_ba['balanced_accuracy']}")

    if all_results:
        print(f"\n{'='*60}")
        print("  SUMMARY")
        print(f"{'='*60}")
        print(f"| {'Variant':<8} | {'Train Steps':>11} | {'Step BA (prox)':>14} | {'Step BA (all)':>13} | {'Episode BA':>10} |")
        print(f"|{'-'*10}|{'-'*13}|{'-'*16}|{'-'*15}|{'-'*12}|")

        train_sizes = {"full": 1740, "k1": 508, "k3": 596, "k5": 684, "k10": 903}
        for variant in variants:
            if variant in all_results:
                m = all_results[variant]
                ts = train_sizes.get(variant, "?")
                sp = m["step_ba_proximal"]["balanced_accuracy"] or "N/A"
                sa = m["step_ba_all"]["balanced_accuracy"] or "N/A"
                ep = m["episode_ba"]["balanced_accuracy"] or "N/A"
                print(f"| {variant:<8} | {ts:>11} | {sp:>14} | {sa:>13} | {ep:>10} |")

        with open(os.path.join(args.results_dir, "summary.json"), "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.results_dir}/")


if __name__ == "__main__":
    main()
