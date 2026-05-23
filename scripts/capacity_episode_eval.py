"""Capacity scaling episode-level eval for 0.6B and 4B checkpoints."""
import json, os, sys, time, gc
from collections import defaultdict
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

SIMULATE_PREFIX = "[SIMULATE]"
PROJECT_ROOT = "."
DATA_PATH = os.path.join(PROJECT_ROOT, "data/plan-004-implicit/implicit_sft.jsonl")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results/capacity_episode_eval")

CHECKPOINTS = {
    "0.6B": {
        "base_model": "./models/Qwen/Qwen3-0.6B",
        "seeds": {
            42: "checkpoints/plan-008-v4-0_6b-seed42/final_adapter",
            43: "checkpoints/plan-008-v4-0_6b-seed43/final_adapter",
            44: "checkpoints/plan-008-v4-0_6b-seed44/final_adapter",
            45: "checkpoints/plan-008-v4-0_6b-seed45/final_adapter",
        },
    },
    "4B": {
        "base_model": "./models/Qwen/Qwen3-4B",
        "seeds": {
            42: "checkpoints/plan-008-v4-4b-seed42/final_adapter",
            43: "checkpoints/plan-008-v4-4b-seed43/final_adapter",
            44: "checkpoints/plan-008-v4-4b-seed44/final_adapter",
            45: "checkpoints/plan-008-v4-4b-seed45/final_adapter",
        },
    },
}

def log(msg):
    print(msg, flush=True)

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

def eval_single(model_size, seed, adapter_rel, base_model_path, val_records, gpu_id):
    adapter_path = os.path.join(PROJECT_ROOT, adapter_rel)

    log(f"\n{'='*60}")
    log(f"{model_size} seed{seed}: {adapter_rel}")
    log(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16,
        device_map={"": gpu_id}, trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()

    preds, gts = [], []
    t0 = time.time()
    batch_size = 2

    for i in range(0, len(val_records), batch_size):
        batch = val_records[i:i+batch_size]
        for rec in batch:
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

        if (i // batch_size) % 20 == 0:
            log(f"  Progress: {min(i+batch_size, len(val_records))}/{len(val_records)}")

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

    gt_dist = defaultdict(int)
    pred_dist = defaultdict(int)
    for g in gts:
        gt_dist[g] += 1
    for p in preds:
        pred_dist[p] += 1

    result = {
        "model_size": model_size,
        "seed": seed,
        "adapter_path": adapter_rel,
        "balanced_accuracy": round(float(bal_acc), 4),
        "accuracy": round(accuracy, 4),
        "deterministic_recall": round(recall.get("deterministic", 0.0), 4),
        "internal_recall": round(recall.get("internal", 0.0), 4),
        "n_val": len(val_records),
        "gt_distribution": dict(gt_dist),
        "pred_distribution": dict(pred_dist),
        "eval_time_s": round(elapsed, 1),
    }

    log(f"  balanced_acc={bal_acc:.4f}, det_recall={recall.get('deterministic',0):.4f}, int_recall={recall.get('internal',0):.4f}")

    del model, base, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return result

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--gpu_id", type=int, default=1)
    args = p.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log("Loading data...")
    records = load_data()
    log(f"Total records: {len(records)}")

    log("Creating episode-level split (sklearn seed=42, test_size=0.15)...")
    val_records = episode_level_split(records, test_size=0.15, seed=42)
    log(f"Val samples: {len(val_records)}")

    gt_dist = defaultdict(int)
    for r in val_records:
        gt_dist[r["route_label"]] += 1
    log(f"Val label distribution: {dict(gt_dist)}")

    all_results = {}
    incremental_path = os.path.join(OUTPUT_DIR, "incremental_results.json")

    for model_size, cfg in CHECKPOINTS.items():
        base_model_path = cfg["base_model"]
        all_results[model_size] = []

        for seed in sorted(cfg["seeds"].keys()):
            adapter_rel = cfg["seeds"][seed]
            result = eval_single(model_size, seed, adapter_rel, base_model_path, val_records, args.gpu_id)
            all_results[model_size].append(result)

            with open(incremental_path, "w") as f:
                json.dump(all_results, f, indent=2)
            log(f"  Incremental results saved.")

    # Final summary
    summary = {"1.7B_reference": {"mean": 0.471, "std": 0.022}}

    for model_size in ["0.6B", "4B"]:
        accs = [r["balanced_accuracy"] for r in all_results[model_size]]
        seeds = [r["seed"] for r in all_results[model_size]]
        summary[model_size] = {
            "seeds": seeds,
            "bal_acc": accs,
            "mean": round(float(np.mean(accs)), 4),
            "std": round(float(np.std(accs)), 4),
        }

    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log(f"\n{'='*70}")
    log("CAPACITY SCALING EPISODE-LEVEL EVAL SUMMARY")
    log(f"{'='*70}")
    log(f"{'Model':>8} {'Mean bal_acc':>14} {'Std':>8}")
    log(f"{'-'*8:>8} {'-'*14:>14} {'-'*8:>8}")
    for ms in ["0.6B", "1.7B_reference", "4B"]:
        log(f"{ms:>8} {summary[ms]['mean']:>14.4f} {summary[ms]['std']:>8.4f}")

    log(f"\nSummary saved to {summary_path}")

if __name__ == "__main__":
    main()
