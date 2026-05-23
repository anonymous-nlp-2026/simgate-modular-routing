# Evaluate death-proximal trained models (standalone eval script).
# Handles Qwen3 thinking tokens properly.

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

SIMULATE_PREFIX = "[SIMULATE]"
THINK_PATTERN = re.compile(r"<think>[\s\S]*?</think>\s*")


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def strip_thinking(text):
    return THINK_PATTERN.sub("", text).strip()


def evaluate_model(adapter_path, model_name, eval_path, eval_meta_path, gpu_id):
    eval_records = load_jsonl(eval_path)
    eval_meta = load_jsonl(eval_meta_path) if eval_meta_path and os.path.exists(eval_meta_path) else None

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
        gen_text = strip_thinking(gen_text)

        pred_route = "deterministic" if gen_text.startswith(SIMULATE_PREFIX) else "internal"
        oracle_route = rec["route_label"]
        meta = eval_meta[i] if eval_meta and i < len(eval_meta) else {}

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
            acc_so_far = sum(r["correct"] for r in results) / len(results)
            print(f"  [{i+1}/{len(eval_records)}] {elapsed:.1f}s, acc={acc_so_far:.3f}", flush=True)

    elapsed = time.time() - t0
    print(f"Eval done: {len(results)} steps in {elapsed:.1f}s", flush=True)

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
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--variants", default="full,k10,k5,k3,k1")
    args = parser.parse_args()

    os.chdir(".")
    eval_path = os.path.join(args.data_dir, "eval.jsonl")
    eval_meta_path = os.path.join(args.data_dir, "eval_meta.jsonl")
    os.makedirs(args.results_dir, exist_ok=True)

    variants = [v.strip() for v in args.variants.split(",")]
    all_results = {}

    train_sizes = {}
    for v in ["full", "k1", "k3", "k5", "k10"]:
        fname = f"train_{v}.jsonl"
        fpath = os.path.join(args.data_dir, fname)
        if os.path.exists(fpath):
            train_sizes[v] = sum(1 for _ in open(fpath))

    for variant in variants:
        adapter_path = os.path.join(args.checkpoint_base, variant, "final_adapter")
        if not os.path.exists(adapter_path):
            print(f"SKIP {variant}: no adapter at {adapter_path}")
            continue

        print(f"\n{'='*60}")
        print(f"  Evaluating: {variant}")
        print(f"{'='*60}", flush=True)

        step_results = evaluate_model(
            adapter_path, args.model_name, eval_path, eval_meta_path, args.gpu_id)

        step_ba_all = compute_balanced_accuracy(step_results)
        step_ba_proximal = compute_balanced_accuracy(
            step_results,
            lambda r: r.get("signal_source") == "death_asymmetry" and
                      r.get("fatal_step") is not None and
                      abs(r["step_idx"] - r["fatal_step"]) <= 5)
        step_ba_death = compute_balanced_accuracy(
            step_results,
            lambda r: r.get("signal_source") == "death_asymmetry")
        episode_ba = compute_episode_accuracy(step_results)

        metrics = {
            "variant": variant,
            "train_steps": train_sizes.get(variant, "?"),
            "step_ba_all": step_ba_all,
            "step_ba_proximal": step_ba_proximal,
            "step_ba_death_all": step_ba_death,
            "episode_ba": episode_ba,
        }
        all_results[variant] = metrics

        with open(os.path.join(args.results_dir, f"eval_{variant}.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        # Save per-step predictions
        with open(os.path.join(args.results_dir, f"predictions_{variant}.jsonl"), "w") as f:
            for r in step_results:
                f.write(json.dumps(r) + "\n")

        print(f"\n  Step BA (all):      {step_ba_all['balanced_accuracy']}")
        print(f"  Step BA (proximal): {step_ba_proximal['balanced_accuracy']}")
        print(f"  Step BA (death):    {step_ba_death['balanced_accuracy']}")
        print(f"  Episode BA:         {episode_ba['balanced_accuracy']}", flush=True)

    if all_results:
        print(f"\n{'='*60}")
        print("  SUMMARY")
        print(f"{'='*60}")
        print(f"| {'Variant':<8} | {'Train':>5} | {'StepBA(prox)':>12} | {'StepBA(all)':>11} | {'EpBA':>6} |")
        print(f"|{'-'*10}|{'-'*7}|{'-'*14}|{'-'*13}|{'-'*8}|")

        for variant in variants:
            if variant in all_results:
                m = all_results[variant]
                ts = m.get("train_steps", "?")
                sp = m["step_ba_proximal"]["balanced_accuracy"]
                sa = m["step_ba_all"]["balanced_accuracy"]
                ep = m["episode_ba"]["balanced_accuracy"]
                sp_s = f"{sp:.4f}" if sp is not None else "N/A"
                print(f"| {variant:<8} | {ts:>5} | {sp_s:>12} | {sa:>11.4f} | {ep:>6.4f} |")

        with open(os.path.join(args.results_dir, "summary.json"), "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.results_dir}/", flush=True)


if __name__ == "__main__":
    main()
