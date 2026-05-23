#!/usr/bin/env python3
"""Seed robustness validation for SimGate core results.

Part A: Train routers with seeds 43,44,45, evaluate routing classification metrics.
Part B: Bootstrap resample decomposition metrics (delta_full, delta_nd, DAF, Pearson r).

Usage:
  python scripts/seed_robustness.py                    # Full run
  python scripts/seed_robustness.py --dry-run           # 1 seed, 1 epoch
  python scripts/seed_robustness.py --bootstrap-only    # Part B only
"""

import argparse
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = "."
DATA_DIR = os.path.join(PROJECT_ROOT, "data/plan-004-implicit")
MODEL_PATH = "./models/Qwen/Qwen3-1.7B"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results/seed-robustness")
DECOMP_DATA_DIR = os.path.join(PROJECT_ROOT, "results/plan_c")

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WANDB_MODE"] = "offline"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HOME"] = "~/.cache/huggingface"
DEVICE_ID = 0  # physical GPU1 remapped to cuda:0

SEEDS = [43, 44, 45]
SEED42_METRICS = {
    "balanced_acc": 0.5761,
    "internal_recall": 0.5152,
    "eval_loss": 0.24,
}
DEATH_THRESHOLD = -100


# ============================================================
# Part A: Router Training + Evaluation
# ============================================================

def train_single_seed(seed: int, epochs: int = 3) -> Optional[str]:
    output_dir = os.path.join(PROJECT_ROOT, f"checkpoints/seed-robustness-seed{seed}")
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "src/training/implicit_router_sft.py"),
        "train",
        "--data_dir", DATA_DIR,
        "--model_name", MODEL_PATH,
        "--epochs", str(epochs),
        "--batch_size", "4",
        "--gradient_accumulation_steps", "8",
        "--upsample_minority",
        "--lr", "2e-5",
        "--output_dir", output_dir,
        "--seed", str(seed),
        "--gpu_id", str(DEVICE_ID),
    ]
    print(f"\n{'='*60}")
    print(f"[Part A] Training router seed={seed}, epochs={epochs}")
    print(f"  output: {output_dir}")
    print(f"{'='*60}\n", flush=True)

    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0

    sentinel = os.path.join(output_dir, "TRAINING_COMPLETE")
    if result.returncode != 0 or not os.path.exists(sentinel):
        print(f"ERROR: Training failed for seed={seed} (exit={result.returncode})")
        return None
    print(f"Training seed={seed} done in {elapsed/60:.1f} min")
    return output_dir


def get_eval_loss(output_dir: str) -> float:
    state_path = os.path.join(output_dir, "trainer_state.json")
    if not os.path.exists(state_path):
        for d in sorted(os.listdir(output_dir)):
            sp = os.path.join(output_dir, d, "trainer_state.json")
            if os.path.exists(sp):
                state_path = sp
                break
    if not os.path.exists(state_path):
        return float("nan")
    with open(state_path) as f:
        state = json.load(f)
    best = float("inf")
    for entry in state.get("log_history", []):
        if "eval_loss" in entry:
            best = min(best, entry["eval_loss"])
    return best if best < float("inf") else float("nan")


def eval_router_metrics(adapter_path: str, seed: int) -> dict:
    src_dir = os.path.join(PROJECT_ROOT, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from eval_routing_metrics import load_val_split, parse_output

    import torch
    from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    print(f"\n[Eval] seed={seed}, adapter={adapter_path}")
    val_records = load_val_split(DATA_DIR, seed=seed)
    val_lbl = Counter(r["route_label"] for r in val_records)
    print(f"  Val: {len(val_records)} steps, dist={dict(val_lbl)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map={"": DEVICE_ID}, trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    model.config.pad_token_id = tokenizer.pad_token_id

    y_true, y_pred = [], []
    t0 = time.time()

    for i, rec in enumerate(val_records):
        prompt_messages = rec["messages"][:-1]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=128, temperature=0.1, do_sample=True,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        y_pred.append(parse_output(raw))
        y_true.append(rec["route_label"])

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(val_records)}] ({time.time()-t0:.0f}s)", flush=True)

    del model, base
    torch.cuda.empty_cache()

    labels = ["internal", "deterministic"]
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)

    result = {
        "balanced_acc": round(bal_acc, 4),
        "internal_recall": round(rec[0], 4),
        "internal_precision": round(prec[0], 4),
        "det_recall": round(rec[1], 4),
        "det_precision": round(prec[1], 4),
        "val_size": len(val_records),
    }
    print(f"  balanced_acc={result['balanced_acc']}, internal_recall={result['internal_recall']}")
    return result


# ============================================================
# Part B: Bootstrap Decomposition
# ============================================================

def is_dead(score: float) -> bool:
    return score <= DEATH_THRESHOLD


def load_all_episodes(data_dir: str) -> Dict[str, List[Tuple[float, float]]]:
    task_episodes = {}
    for task_name in sorted(os.listdir(data_dir)):
        ep_file = os.path.join(data_dir, task_name, "episodes.jsonl")
        if not os.path.isfile(ep_file):
            continue
        pairs = []
        with open(ep_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ep = json.loads(line)
                if "always_internal" not in ep or "always_deterministic" not in ep:
                    continue
                pairs.append((
                    ep["always_internal"]["final_score"],
                    ep["always_deterministic"]["final_score"],
                ))
        if pairs:
            task_episodes[task_name] = pairs
    return task_episodes


def compute_decomp(int_scores, det_scores):
    n = len(int_scores)
    if n == 0:
        return {"delta_full": float("nan"), "delta_nd": float("nan"), "daf": float("nan")}
    delta_full = statistics.mean(det_scores) - statistics.mean(int_scores)
    nd_int = [i for i, d in zip(int_scores, det_scores) if not is_dead(i) and not is_dead(d)]
    nd_det = [d for i, d in zip(int_scores, det_scores) if not is_dead(i) and not is_dead(d)]
    delta_nd = (statistics.mean(nd_det) - statistics.mean(nd_int)) if nd_int else float("nan")
    if not math.isnan(delta_nd) and delta_full != 0:
        daf = ((delta_full - delta_nd) / delta_full) * 100
    else:
        daf = float("nan")
    return {"delta_full": delta_full, "delta_nd": delta_nd, "daf": daf}


def compute_pearson_r(task_episodes):
    if len(task_episodes) < 3:
        return float("nan")
    drs, deltas = [], []
    for pairs in task_episodes.values():
        t_int = [p[0] for p in pairs]
        t_det = [p[1] for p in pairs]
        int_dr = sum(1 for s in t_int if is_dead(s)) / len(t_int)
        det_dr = sum(1 for s in t_det if is_dead(s)) / len(t_det)
        drs.append(int_dr - det_dr)
        deltas.append(statistics.mean(t_det) - statistics.mean(t_int))
    n = len(drs)
    mx, my = statistics.mean(drs), statistics.mean(deltas)
    sx = math.sqrt(sum((x - mx)**2 for x in drs) / (n - 1)) if n > 1 else 0
    sy = math.sqrt(sum((y - my)**2 for y in deltas) / (n - 1)) if n > 1 else 0
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(drs, deltas)) / ((n - 1) * sx * sy)


def bootstrap_decomposition(data_dir: str, n_bootstrap: int = 1000):
    task_episodes = load_all_episodes(data_dir)
    task_names = sorted(task_episodes.keys())
    all_pairs = [(i, d) for pairs in task_episodes.values() for i, d in pairs]
    n = len(all_pairs)

    print(f"\n{'='*60}")
    print(f"[Part B] Bootstrap: {n} episodes, {len(task_names)} tasks, {n_bootstrap} resamples")
    print(f"{'='*60}")

    all_int = [p[0] for p in all_pairs]
    all_det = [p[1] for p in all_pairs]
    point = compute_decomp(all_int, all_det)
    point["pearson_r"] = compute_pearson_r(task_episodes)

    print(f"Point: delta_full={point['delta_full']:.2f}, delta_nd={point['delta_nd']:.2f}, "
          f"DAF={point['daf']:.2f}%, r={point['pearson_r']:.4f}")

    rng = random.Random(42)
    boots = {"delta_full": [], "delta_nd": [], "daf": [], "pearson_r": []}

    for b in range(n_bootstrap):
        # Episode-level resample for delta_full, delta_nd, daf
        idxs = [rng.randint(0, n - 1) for _ in range(n)]
        b_int = [all_pairs[i][0] for i in idxs]
        b_det = [all_pairs[i][1] for i in idxs]
        m = compute_decomp(b_int, b_det)
        for k in ["delta_full", "delta_nd", "daf"]:
            if not math.isnan(m[k]):
                boots[k].append(m[k])

        # Task-level resample for pearson_r
        sampled = [task_names[rng.randint(0, len(task_names) - 1)] for _ in range(len(task_names))]
        b_task_ep = {f"{t}_{j}": task_episodes[t] for j, t in enumerate(sampled)}
        r = compute_pearson_r(b_task_ep)
        if not math.isnan(r):
            boots["pearson_r"].append(r)

    results = {"n_bootstrap": n_bootstrap, "n_episodes": n, "n_tasks": len(task_names)}
    for key in ["delta_full", "delta_nd", "daf", "pearson_r"]:
        vals = sorted(boots[key])
        if len(vals) > 1:
            lo = vals[int(len(vals) * 0.025)]
            hi = vals[int(len(vals) * 0.975)]
            results[key] = {
                "point": round(point[key], 4),
                "mean": round(statistics.mean(vals), 4),
                "std": round(statistics.stdev(vals), 4),
                "ci95": [round(lo, 4), round(hi, 4)],
            }
        else:
            results[key] = {"point": round(point[key], 4), "mean": None, "std": None, "ci95": [None, None]}

    # Leave-one-task-out jackknife
    jackknife = {}
    for leave_out in task_names:
        remaining = {t: p for t, p in task_episodes.items() if t != leave_out}
        j_int = [p[0] for pairs in remaining.values() for p in pairs]
        j_det = [p[1] for pairs in remaining.values() for p in pairs]
        m = compute_decomp(j_int, j_det)
        m["pearson_r"] = compute_pearson_r(remaining)
        jackknife[leave_out] = {k: round(v, 4) if not math.isnan(v) else None for k, v in m.items()}
    results["jackknife"] = jackknife

    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Seed robustness validation")
    parser.add_argument("--dry-run", action="store_true", help="1 seed, 1 epoch")
    parser.add_argument("--bootstrap-only", action="store_true", help="Part B only")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seeds", type=str, default="43,44,45")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    epochs = 1 if args.dry_run else args.epochs
    if args.dry_run:
        seeds = seeds[:1]

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Part B: Bootstrap (fast)
    decomp = bootstrap_decomposition(DECOMP_DATA_DIR, n_bootstrap=1000)
    decomp_path = os.path.join(RESULTS_DIR, "decomposition_bootstrap.json")
    with open(decomp_path, "w") as f:
        json.dump(decomp, f, indent=2)
    print(f"\n[Saved] {decomp_path}")

    if args.bootstrap_only:
        summary = {"decomposition_bootstrap": decomp}
        with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        return

    # Part A: Router training + eval
    router = {
        "seeds": [42] + seeds,
        "balanced_acc": [SEED42_METRICS["balanced_acc"]],
        "internal_recall": [SEED42_METRICS["internal_recall"]],
        "eval_loss": [SEED42_METRICS["eval_loss"]],
    }

    for seed in seeds:
        out = train_single_seed(seed, epochs=epochs)
        if out is None:
            for k in ["balanced_acc", "internal_recall", "eval_loss"]:
                router[k].append(None)
            continue

        el = get_eval_loss(out)
        router["eval_loss"].append(round(el, 4) if not math.isnan(el) else None)

        adapter = os.path.join(out, "final_adapter")
        try:
            m = eval_router_metrics(adapter, seed)
            router["balanced_acc"].append(m["balanced_acc"])
            router["internal_recall"].append(m["internal_recall"])
        except Exception as e:
            print(f"ERROR: Eval failed for seed={seed}: {e}")
            router["balanced_acc"].append(None)
            router["internal_recall"].append(None)

        # Save intermediate
        intermediate = os.path.join(RESULTS_DIR, f"router_seed{seed}.json")
        with open(intermediate, "w") as f:
            json.dump({"seed": seed, "eval_loss": router["eval_loss"][-1],
                        "balanced_acc": router["balanced_acc"][-1],
                        "internal_recall": router["internal_recall"][-1]}, f, indent=2)
        print(f"[Saved] {intermediate}")

    # Summary stats
    for metric in ["balanced_acc", "internal_recall", "eval_loss"]:
        vals = [x for x in router[metric] if x is not None]
        router[f"mean_{metric}"] = round(statistics.mean(vals), 4) if vals else None
        router[f"std_{metric}"] = round(statistics.stdev(vals), 4) if len(vals) > 1 else None

    summary = {"router_metrics": router, "decomposition_bootstrap": decomp}
    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DONE — {summary_path}")
    print(f"{'='*60}")
    print(f"Router balanced_acc: {router['balanced_acc']}")
    print(f"  Mean +/- Std: {router['mean_balanced_acc']} +/- {router['std_balanced_acc']}")
    for key in ["delta_full", "delta_nd", "daf", "pearson_r"]:
        d = decomp[key]
        print(f"Decomp {key}: {d['point']} (95% CI: {d['ci95']})")


if __name__ == "__main__":
    main()
