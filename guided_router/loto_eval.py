"""
LOTO (Leave-One-Task-Out) evaluation for guided router.
11-fold: each fold leaves out 1 task type, uses remaining 10 to determine
routing decision for the held-out task type.

Strategies:
  - prior_majority: majority of training tasks are death-risk → predict det
  - prior_mean_threshold: mean death rate of training tasks > DEATH_THRESHOLD → predict det
  - learned: use string-pattern matching (naturally generalizes to unseen tasks)

Output: guided_router/loto_results.json (incremental save per fold)
"""

import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guided_router.config import ALL_TASKS, DEATH_THRESHOLD, COUNTERFACTUAL_DATA
from guided_router.eval_episode import load_episodes, classify_episode_gt, compute_balanced_accuracy
from guided_router.router import GuidedRouter


RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "loto_results.json"
)


def incremental_save(results):
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)


def evaluate_fold(task_episodes, held_out_task, route_decision):
    """Evaluate routing decision on held-out task's episodes."""
    pred_label = "det-preferred" if route_decision == "deterministic" else "internal-preferred"
    gts = []
    preds = []
    for ep in task_episodes[held_out_task]:
        gt = classify_episode_gt(ep)
        gts.append(gt)
        preds.append(pred_label)
    metrics = compute_balanced_accuracy(gts, preds)
    return metrics


def run_loto():
    print("Loading episodes...")
    episodes = load_episodes(COUNTERFACTUAL_DATA)
    print(f"Loaded {len(episodes)} episodes")

    # Group by task type
    task_episodes = defaultdict(list)
    for ep in episodes:
        task_episodes[ep["task_type"]].append(ep)
    all_task_types = sorted(task_episodes.keys())
    assert len(all_task_types) == 11, f"Expected 11 task types, got {len(all_task_types)}"

    results = {"folds": {}, "strategies": {}, "config": {
        "death_threshold": DEATH_THRESHOLD,
        "all_tasks_death_rates": ALL_TASKS,
        "n_episodes": len(episodes),
        "n_task_types": len(all_task_types),
    }}
    incremental_save(results)

    strategies = ["prior_majority", "prior_mean_threshold", "learned"]

    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy}")
        print(f"{'='*60}")
        fold_results = []

        for held_out_task in all_task_types:
            train_tasks = [t for t in all_task_types if t != held_out_task]
            train_death_rates = {t: ALL_TASKS[t] for t in train_tasks}

            # Determine routing decision for held-out task
            if strategy == "prior_majority":
                n_death_risk = sum(1 for r in train_death_rates.values() if r > DEATH_THRESHOLD)
                route_decision = "deterministic" if n_death_risk > len(train_tasks) / 2 else "internal"

            elif strategy == "prior_mean_threshold":
                mean_rate = sum(train_death_rates.values()) / len(train_death_rates)
                route_decision = "deterministic" if mean_rate > DEATH_THRESHOLD else "internal"

            elif strategy == "learned":
                router = GuidedRouter(variant="learned")
                test_eps = task_episodes[held_out_task]
                task_desc = test_eps[0].get("task_desc", "")
                route_decision = router.route_episode(held_out_task, task_desc=task_desc)

            # Evaluate on held-out episodes
            metrics = evaluate_fold(task_episodes, held_out_task, route_decision)

            fold_result = {
                "task_type": held_out_task,
                "route_decision": route_decision,
                "true_death_rate": ALL_TASKS.get(held_out_task, 0.0),
                "n_episodes": len(task_episodes[held_out_task]),
                **metrics,
            }
            fold_results.append(fold_result)

            print(f"  Fold: {held_out_task:<45} route={route_decision:<14} "
                  f"bal_acc={metrics['balanced_accuracy']:.4f} "
                  f"(n_eval={metrics['n_evaluated']})")

            # Incremental save
            results["folds"][f"{strategy}/{held_out_task}"] = fold_result
            incremental_save(results)

        # Aggregate per strategy
        valid_folds = [f for f in fold_results if f["n_evaluated"] > 0]
        if valid_folds:
            mean_bal_acc = sum(f["balanced_accuracy"] for f in valid_folds) / len(valid_folds)
            std_bal_acc = (sum((f["balanced_accuracy"] - mean_bal_acc)**2 for f in valid_folds) / len(valid_folds))**0.5
        else:
            mean_bal_acc = std_bal_acc = 0.0

        # Also compute episode-level overall balanced accuracy (for comparison with 62.92%)
        all_gts = []
        all_preds = []
        for fold in fold_results:
            task = fold["task_type"]
            pred_label = "det-preferred" if fold["route_decision"] == "deterministic" else "internal-preferred"
            for ep in task_episodes[task]:
                gt = classify_episode_gt(ep)
                all_gts.append(gt)
                all_preds.append(pred_label)
        overall_metrics = compute_balanced_accuracy(all_gts, all_preds)

        results["strategies"][strategy] = {
            "mean_balanced_accuracy": round(mean_bal_acc, 4),
            "std_balanced_accuracy": round(std_bal_acc, 4),
            "n_valid_folds": len(valid_folds),
            "overall_episode_balanced_accuracy": overall_metrics["balanced_accuracy"],
            "overall_episode_det_recall": overall_metrics["det_recall"],
            "overall_episode_internal_recall": overall_metrics["internal_recall"],
            "overall_n_evaluated": overall_metrics["n_evaluated"],
        }
        incremental_save(results)

        print(f"\n  Strategy summary ({strategy}):")
        print(f"    11-fold mean bal_acc: {mean_bal_acc*100:.1f}% (std: {std_bal_acc*100:.1f}%)")
        print(f"    Overall episode-level bal_acc: {overall_metrics['balanced_accuracy']*100:.2f}%")

    # Final summary
    print(f"\n{'='*60}")
    print("=== LOTO Guided Router Results ===")
    print(f"{'='*60}")
    for strategy, data in results["strategies"].items():
        print(f"\n  {strategy}:")
        print(f"    11-fold mean balanced_acc: {data['mean_balanced_accuracy']*100:.1f}% (std: {data['std_balanced_accuracy']*100:.1f}%)")
        print(f"    Episode-level balanced_acc: {data['overall_episode_balanced_accuracy']*100:.2f}%")

    print(f"\n  In-sample comparison: 62.92%")
    best = max(results["strategies"].items(), key=lambda x: x[1]["mean_balanced_accuracy"])
    print(f"  Best LOTO strategy: {best[0]} = {best[1]['mean_balanced_accuracy']*100:.1f}%")
    delta = best[1]["overall_episode_balanced_accuracy"] * 100 - 62.92
    print(f"  Delta vs in-sample (episode-level): {delta:+.1f} pp")

    print(f"\n  Per-fold detail (best strategy: {best[0]}):")
    for fold in results["folds"]:
        if fold.startswith(best[0] + "/"):
            f = results["folds"][fold]
            print(f"    {f['task_type']:<45} {f['balanced_accuracy']*100:5.1f}%  route={f['route_decision']}")

    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    run_loto()
