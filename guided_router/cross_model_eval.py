"""Cross-model guided router evaluation: Qwen→Llama transfer.

Apply Qwen-derived task-type death risk routing rule to Llama-3.1-8B episodes.
Uses episode_summary.jsonl files from llama-cross-model-v2 experiment.
"""

import json
import os
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = "."
LLAMA_DATA_DIR = f"{PROJECT_ROOT}/data/llama-cross-model-v2"
RESULTS_PATH = f"{PROJECT_ROOT}/guided_router/cross_model_results.json"

QWEN_DEATH_RISK_TASKS = {
    "freeze", "find-animal", "identify-life-stages-1", "identify-life-stages-2",
    "inclined-plane-determine-angle", "lifespan-longest-lived",
    "lifespan-longest-lived-then-shortest-lived",
    "measure-melting-point-unknown-substance", "grow-fruit", "chemistry-mix",
}

QWEN_SAFE_TASKS = {"find-non-living-thing"}


def load_llama_episodes():
    """Load all Llama episode summaries grouped by task type."""
    task_episodes = {}
    data_dir = Path(LLAMA_DATA_DIR)
    for task_dir in sorted(data_dir.iterdir()):
        if not task_dir.is_dir() or task_dir.name == "logs":
            continue
        summary_file = task_dir / "episode_summary.jsonl"
        if not summary_file.exists():
            continue
        episodes = []
        with open(summary_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    episodes.append(json.loads(line))
        task_episodes[task_dir.name] = episodes
    return task_episodes


def apply_qwen_rule(task_type):
    """Apply Qwen death risk rule: death-risk tasks → deterministic."""
    if task_type in QWEN_DEATH_RISK_TASKS:
        return "deterministic"
    return "internal"


def compute_balanced_accuracy(gts, preds):
    """Same logic as guided_router/eval_episode.py: drops no-preference."""
    labels = ["det-preferred", "internal-preferred"]
    tp = defaultdict(int)
    total_per_class = defaultdict(int)

    for gt, pred in zip(gts, preds):
        if gt not in labels:
            continue
        total_per_class[gt] += 1
        if gt == pred:
            tp[gt] += 1

    recall = {}
    for l in labels:
        if total_per_class[l] > 0:
            recall[l] = tp[l] / total_per_class[l]
        else:
            recall[l] = 0.0

    active_labels = [l for l in labels if total_per_class[l] > 0]
    bal_acc = sum(recall[l] for l in active_labels) / len(active_labels) if active_labels else 0.0

    n_evaluated = sum(total_per_class[l] for l in labels)
    accuracy = sum(tp[l] for l in labels) / max(n_evaluated, 1)

    return {
        "balanced_accuracy": round(bal_acc, 4),
        "accuracy": round(accuracy, 4),
        "det_recall": round(recall.get("det-preferred", 0.0), 4),
        "internal_recall": round(recall.get("internal-preferred", 0.0), 4),
        "n_evaluated": n_evaluated,
        "n_no_preference": sum(1 for gt in gts if gt == "no-preference"),
        "n_total": len(gts),
        "gt_distribution": {
            "det-preferred": total_per_class.get("det-preferred", 0),
            "internal-preferred": total_per_class.get("internal-preferred", 0),
        },
        "active_classes": len(active_labels),
    }


def main():
    task_episodes = load_llama_episodes()
    print(f"Loaded {sum(len(v) for v in task_episodes.values())} episodes across {len(task_episodes)} tasks")
    print(f"Tasks: {sorted(task_episodes.keys())}")

    all_gts = []
    all_preds = []
    per_task_results = {}

    for task_type, episodes in sorted(task_episodes.items()):
        route_decision = apply_qwen_rule(task_type)
        pred_label = "det-preferred" if route_decision == "deterministic" else "internal-preferred"

        task_gts = [ep["label"] for ep in episodes]
        task_preds = [pred_label] * len(episodes)

        all_gts.extend(task_gts)
        all_preds.extend(task_preds)

        task_metrics = compute_balanced_accuracy(task_gts, task_preds)
        per_task_results[task_type] = {
            "route_decision": route_decision,
            "in_qwen_death_risk": task_type in QWEN_DEATH_RISK_TASKS,
            "n_episodes": len(episodes),
            **task_metrics,
        }

    overall = compute_balanced_accuracy(all_gts, all_preds)

    results = {
        "experiment": "cross-model-guided-router-eval",
        "source_model": "Qwen-2.5-7B",
        "target_model": "Llama-3.1-8B",
        "rule": "Qwen death-risk task types → route to deterministic",
        "overall": overall,
        "per_task": per_task_results,
        "comparison": {
            "qwen_in_sample": 0.6292,
            "qwen_loto": 0.6292,
            "cross_model_qwen_to_llama": overall["balanced_accuracy"],
        },
        "notes": {
            "degenerate_case": overall["active_classes"] == 1,
            "explanation": (
                "Llama data has 0 int-preferred episodes (all tasks are det-preferred or no-preference). "
                "Since balanced accuracy only evaluates on det-preferred and int-preferred episodes, "
                "and the Qwen rule routes all tasks to deterministic, det_recall=1.0 trivially. "
                "This is a degenerate evaluation — the rule cannot be 'wrong' when there's no opposing class."
            ),
        },
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    # Print report
    print(f"\n{'='*60}")
    print("=== Cross-Model Guided Router Results (Qwen→Llama) ===")
    print(f"{'='*60}")
    print(f"\nOverall:")
    print(f"  Balanced accuracy: {overall['balanced_accuracy']*100:.2f}%")
    print(f"  Det recall: {overall['det_recall']*100:.1f}%")
    print(f"  Internal recall: {overall['internal_recall']*100:.1f}% (n={overall['gt_distribution']['internal-preferred']})")
    print(f"  Active classes: {overall['active_classes']} / 2")
    print(f"  Episodes evaluated: {overall['n_evaluated']} / {overall['n_total']} (rest are no-preference)")
    print(f"\nPer-task breakdown:")
    for task, m in sorted(per_task_results.items()):
        print(f"  {task:<45} route={m['route_decision']:<14} "
              f"det_pref={m['gt_distribution']['det-preferred']:>2} "
              f"int_pref={m['gt_distribution']['internal-preferred']:>2} "
              f"no_pref={m['n_no_preference']:>2} "
              f"bal_acc={m['balanced_accuracy']*100:5.1f}%")

    print(f"\nComparison:")
    print(f"  Qwen in-sample balanced acc:       62.92%")
    print(f"  Qwen LOTO balanced acc:            62.92%")
    print(f"  Cross-model (Qwen→Llama):          {overall['balanced_accuracy']*100:.2f}%")
    print(f"\n  ⚠ DEGENERATE CASE: Llama data has 0 int-preferred episodes.")
    print(f"    The 100% reflects trivial correctness, not genuine discriminative power.")
    print(f"    All {len(task_episodes)} Llama tasks are in Qwen's death-risk set,")
    print(f"    and none have int-preferred ground truth → no way to be wrong.")


if __name__ == "__main__":
    main()
