"""Cross-split robustness eval: episode-level split (independent from training split).

Evaluates 8 seed checkpoints on an episode-level val split (sklearn seed=42),
comparing against the corrected_eval which used step-level stratified split.
"""
import json, os, sys, time, gc
from collections import defaultdict
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

SIMULATE_PREFIX = "[SIMULATE]"
BASE_MODEL = "./models/Qwen/Qwen3-1.7B"
DATA_PATH = "./data/plan-004-implicit/implicit_sft.jsonl"
PROJECT_ROOT = "."
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results/cross_split_eval")

SEED_TO_ADAPTER = {
    42: "checkpoints/plan-008-qwen3-1.7b/final_adapter",
    43: "checkpoints/seed-robustness-seed43/final_adapter",
    44: "checkpoints/seed-robustness-seed44/final_adapter",
    45: "checkpoints/seed-robustness-seed45/final_adapter",
    46: "checkpoints/seed-robustness-seed46/final_adapter",
    47: "checkpoints/seed-robustness-seed47/final_adapter",
    48: "checkpoints/seed-robustness-seed48/final_adapter",
    49: "checkpoints/seed-robustness-seed49/final_adapter",
}

CORRECTED_EVAL_REF = {
    42: 0.5761, 43: 0.4852, 44: 0.6074, 45: 0.5467,
    46: 0.4463, 47: 0.5185, 48: 0.537, 49: 0.6111,
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
    """Split by episode (task_type + variation), not by step."""
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


def eval_single_seed(seed, val_records, gpu_id):
    adapter_rel = SEED_TO_ADAPTER[seed]
    adapter_path = os.path.join(PROJECT_ROOT, adapter_rel)

    log(f"\n{'='*60}")
    log(f"Seed {seed}: {adapter_rel}")
    log(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16,
        device_map={"": gpu_id}, trust_remote_code=True,
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
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False)

        pred_route, _ = parse_output(gen_text)
        preds.append(pred_route)
        gts.append(gt_label)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(val_records) - i - 1) / rate
            log(f"  [{i+1}/{len(val_records)}] {rate:.1f} samples/s, ETA {eta:.0f}s")

    elapsed = time.time() - t0
    log(f"  Done: {len(val_records)} samples in {elapsed:.1f}s")

    # Compute metrics
    labels = ["deterministic", "internal"]
    tp = {l: 0 for l in labels}
    fp = {l: 0 for l in labels}
    fn = {l: 0 for l in labels}
    cm = {g: {p: 0 for p in labels} for g in labels}

    for gt, pred in zip(gts, preds):
        cm[gt][pred] += 1
        if gt == pred:
            tp[gt] += 1
        else:
            fp[pred] += 1
            fn[gt] += 1

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
        "seed": seed,
        "adapter_path": adapter_rel,
        "balanced_accuracy": round(bal_acc, 4),
        "accuracy": round(accuracy, 4),
        "deterministic_recall": round(recall["deterministic"], 4),
        "internal_recall": round(recall["internal"], 4),
        "n_val": len(val_records),
        "gt_distribution": dict(gt_dist),
        "pred_distribution": dict(pred_dist),
        "confusion_matrix": cm,
        "eval_time_s": round(elapsed, 1),
    }

    log(f"  balanced_acc={bal_acc:.4f}, det_recall={recall['deterministic']:.4f}, int_recall={recall['internal']:.4f}")

    # Cleanup GPU memory
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

    episode_keys = set()
    for r in val_records:
        episode_keys.add((r["task_type"], r["variation"]))
    log(f"Val episodes: {len(episode_keys)}")

    all_results = {}
    seeds = sorted(SEED_TO_ADAPTER.keys())

    for seed in seeds:
        result = eval_single_seed(seed, val_records, args.gpu_id)
        all_results[seed] = result

    # Summary
    cross_accs = [all_results[s]["balanced_accuracy"] for s in seeds]
    ref_accs = [CORRECTED_EVAL_REF[s] for s in seeds]
    cross_mean = np.mean(cross_accs)
    cross_std = np.std(cross_accs)
    ref_mean = np.mean(ref_accs)
    ref_std = np.std(ref_accs)

    log(f"\n{'='*70}")
    log("CROSS-SPLIT ROBUSTNESS RESULTS")
    log(f"{'='*70}")
    log(f"{'Seed':>6} {'Cross-Split':>12} {'Corrected':>12} {'Delta':>8}")
    log(f"{'-'*6:>6} {'-'*12:>12} {'-'*12:>12} {'-'*8:>8}")

    deltas = []
    for s in seeds:
        cs = all_results[s]["balanced_accuracy"]
        ref = CORRECTED_EVAL_REF[s]
        delta = cs - ref
        deltas.append(delta)
        log(f"{s:>6} {cs:>12.4f} {ref:>12.4f} {delta:>+8.4f}")

    mean_delta = cross_mean - ref_mean
    log(f"{'Mean':>6} {cross_mean:>12.4f} {ref_mean:>12.4f} {mean_delta:>+8.4f}")
    log(f"{'Std':>6} {cross_std:>12.4f} {ref_std:>12.4f}")
    log(f"\nMean absolute delta: {np.mean(np.abs(deltas)):.4f}")
    log(f"Split-robust (|mean delta| < 2pp): {'YES' if abs(mean_delta) < 0.02 else 'NO'}")

    summary = {
        "split_method": "episode_level_sklearn_seed42_test015",
        "n_val": len(val_records),
        "n_val_episodes": len(episode_keys),
        "seeds": seeds,
        "per_seed": {str(s): all_results[s] for s in seeds},
        "cross_split_mean": round(cross_mean, 4),
        "cross_split_std": round(cross_std, 4),
        "corrected_eval_mean": round(ref_mean, 4),
        "corrected_eval_std": round(ref_std, 4),
        "mean_delta": round(mean_delta, 4),
        "mean_abs_delta": round(float(np.mean(np.abs(deltas))), 4),
        "split_robust": abs(mean_delta) < 0.02,
        "gt_distribution": dict(gt_dist),
    }

    out_path = os.path.join(OUTPUT_DIR, "cross_split_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
