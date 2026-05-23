#!/usr/bin/env python3
"""Train 1.7B router with 4 seeds (42-45) and run episode-level eval.
Matches config of plan-008-v4 (0.6B/4B)."""

import json, os, subprocess, sys, time, gc
from collections import defaultdict
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

PROJECT_ROOT = "."
DATA_PATH = os.path.join(PROJECT_ROOT, "data/plan-004-implicit/implicit_sft.jsonl")
MODEL_PATH = "./models/Qwen/Qwen3-1.7B"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results/capacity_episode_eval")
SIMULATE_PREFIX = "[SIMULATE]"

SEEDS = [42, 43, 44, 45]
GPU_ID = 0

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_MODE"] = "offline"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HOME"] = "~/.cache/huggingface"


def log(msg):
    print(msg, flush=True)


# ============ PHASE 1: TRAIN ============

def train_seed(seed):
    output_dir = os.path.join(PROJECT_ROOT, f"checkpoints/plan-008-v4-1_7b-seed{seed}")
    sentinel = os.path.join(output_dir, "TRAINING_COMPLETE")
    
    if os.path.exists(sentinel):
        log(f"[SKIP] seed{seed} already trained: {output_dir}")
        return output_dir
    
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "src/training/implicit_router_sft.py"),
        "train",
        "--data_dir", os.path.join(PROJECT_ROOT, "data/plan-004-implicit"),
        "--model_name", MODEL_PATH,
        "--epochs", "3",
        "--batch_size", "4",
        "--upsample_minority",
        "--lr", "2e-5",
        "--output_dir", output_dir,
        "--seed", str(seed),
        "--gpu_id", "0",
    ]
    
    log(f"\n{'='*60}")
    log(f"Training 1.7B seed={seed}")
    log(f"  output: {output_dir}")
    log(f"{'='*60}")
    
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    
    if result.returncode != 0 or not os.path.exists(sentinel):
        log(f"ERROR: Training failed for seed={seed} (exit={result.returncode})")
        return None
    
    log(f"Training seed={seed} done in {elapsed/60:.1f} min")
    return output_dir


# ============ PHASE 2: EVAL ============

def load_data():
    records = []
    with open(DATA_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def episode_level_split(records, test_size=0.15, seed=42):
    episode_to_indices = defaultdict(list)
    for i, r in enumerate(records):
        key = (r["task_type"], r["variation"])
        episode_to_indices[key].append(i)

    episode_keys = sorted(episode_to_indices.keys())
    train_eps, val_eps = train_test_split(
        episode_keys, test_size=test_size, random_state=seed
    )

    val_indices = []
    for ep in val_eps:
        val_indices.extend(episode_to_indices[ep])

    val_records = [records[i] for i in sorted(val_indices)]
    return val_records


def parse_output(text):
    text = text.strip()
    if "<think>" in text:
        think_end = text.find("</think>")
        if think_end != -1:
            text = text[think_end + len("</think>"):].strip()
        else:
            text = text[text.find("<think>") + len("<think>"):].strip()
    if text.startswith(SIMULATE_PREFIX):
        return "deterministic", text
    return "internal", text


def eval_single(seed, adapter_path, val_records):
    log(f"\n{'='*60}")
    log(f"Eval 1.7B seed{seed}: {adapter_path}")
    log(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map={"": 0}, trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()

    preds, gts = [], []
    t0 = time.time()

    for i, rec in enumerate(val_records):
        gt_label = rec["route_label"]
        prompt_msgs = rec["messages"][:-1]

        prompt = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=256, do_sample=False,
            )
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        pred_label, _ = parse_output(gen_text)
        preds.append(pred_label)
        gts.append(gt_label)

        if (i + 1) % 40 == 0:
            log(f"  Progress: {i+1}/{len(val_records)}")

    elapsed = time.time() - t0
    log(f"  Eval done in {elapsed:.1f}s")

    labels = sorted(set(gts + preds))
    tp = defaultdict(int)
    fn = defaultdict(int)
    for g, p in zip(gts, preds):
        if g == p:
            tp[g] += 1
        else:
            fn[g] += 1

    recall = {}
    for l in labels:
        recall[l] = tp[l] / max(tp[l] + fn[l], 1)

    bal_acc = np.mean([recall[l] for l in labels])
    accuracy = sum(1 for g, p in zip(gts, preds) if g == p) / max(len(gts), 1)

    result = {
        "model_size": "1.7B",
        "seed": seed,
        "adapter_path": adapter_path,
        "balanced_accuracy": round(float(bal_acc), 4),
        "accuracy": round(accuracy, 4),
        "deterministic_recall": round(recall.get("deterministic", 0.0), 4),
        "internal_recall": round(recall.get("internal", 0.0), 4),
        "n_val": len(val_records),
        "eval_time_s": round(elapsed, 1),
    }

    log(f"  balanced_acc={bal_acc:.4f}, det_recall={recall.get('deterministic',0):.4f}, int_recall={recall.get('internal',0):.4f}")

    del model, base, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return result


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    incremental_path = os.path.join(OUTPUT_DIR, "1_7b_4seed_results.json")
    
    # Phase 1: Train
    log("=" * 70)
    log("PHASE 1: TRAINING 1.7B x 4 SEEDS")
    log("=" * 70)
    
    checkpoints = {}
    for seed in SEEDS:
        out = train_seed(seed)
        if out is None:
            log(f"FATAL: seed{seed} training failed, aborting")
            sys.exit(1)
        checkpoints[seed] = os.path.join(out, "final_adapter")
    
    log("\nAll training complete.")
    
    # Phase 2: Eval
    log("\n" + "=" * 70)
    log("PHASE 2: EPISODE-LEVEL EVAL")
    log("=" * 70)
    
    log("Loading data...")
    records = load_data()
    log(f"Total records: {len(records)}")
    
    log("Creating episode-level split (seed=42, test_size=0.15)...")
    val_records = episode_level_split(records, test_size=0.15, seed=42)
    log(f"Val samples: {len(val_records)}")
    
    all_results = []
    for seed in SEEDS:
        result = eval_single(seed, checkpoints[seed], val_records)
        all_results.append(result)
        
        with open(incremental_path, "w") as f:
            json.dump(all_results, f, indent=2)
        log(f"  Saved incremental results.")
    
    # Summary
    accs = [r["balanced_accuracy"] for r in all_results]
    mean_acc = float(np.mean(accs))
    std_acc = float(np.std(accs))
    
    summary = {
        "model_size": "1.7B",
        "seeds": SEEDS,
        "bal_acc_per_seed": {r["seed"]: r["balanced_accuracy"] for r in all_results},
        "mean_balanced_accuracy": round(mean_acc, 4),
        "std_balanced_accuracy": round(std_acc, 4),
        "comparison": {
            "0.6B_4seed": 0.507,
            "1.7B_4seed_new": round(mean_acc, 4),
            "1.7B_8seed_old": 0.471,
            "4B_4seed": 0.5896,
        },
        "monotonic": mean_acc >= 0.507,
        "per_seed_results": all_results,
    }
    
    summary_path = os.path.join(OUTPUT_DIR, "1_7b_4seed_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    log(f"\n{'='*70}")
    log("CAPACITY SCALING 1.7B (4-SEED) RESULTS")
    log(f"{'='*70}")
    log(f"Seeds: {SEEDS}")
    for seed, acc in zip(SEEDS, accs):
        log(f"  seed{seed}: balanced_acc = {acc:.4f}")
    log(f"\n  Mean: {mean_acc:.4f} (std: {std_acc:.4f})")
    log(f"\nComparison:")
    log(f"  0.6B (4 seeds): 50.7%")
    log(f"  1.7B (4 seeds, new): {mean_acc*100:.1f}%")
    log(f"  1.7B (old): 47.1%")
    log(f"  4B (4 seeds): 58.96%")
    log(f"\nMonotonic: {'YES' if mean_acc >= 0.507 else 'NO'}")
    log(f"\nResults saved to: {summary_path}")


if __name__ == "__main__":
    main()
