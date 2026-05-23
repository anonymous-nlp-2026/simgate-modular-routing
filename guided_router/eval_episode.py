"""Episode-level balanced accuracy evaluation for guided router.

Input: scienceworld-counterfactual/episodes.jsonl (245 episodes, 11 tasks)
Each episode has: internal trajectory (with death/score), deterministic trajectory.
Ground truth label: 'det-preferred' or 'internal-preferred' or 'no-preference'.

For routing eval, we map:
  - router says 'deterministic' -> predicted label 'det-preferred'
  - router says 'internal' -> predicted label 'internal-preferred'
Ground truth from counterfactual score comparison.
"""

import json
from collections import defaultdict
from typing import Dict, List

from .config import COUNTERFACTUAL_DATA


DEATH_SCORE = -100


def classify_episode_gt(episode: Dict) -> str:
    """Determine ground-truth preferred route from counterfactual data."""
    int_data = episode["internal"]
    det_data = episode["deterministic"]

    int_score = int_data["final_score"]
    det_score = det_data["final_score"]
    int_died = int_data.get("death", int_score <= DEATH_SCORE)
    det_died = det_data.get("death", det_score <= DEATH_SCORE)

    if int_died and not det_died:
        return "det-preferred"
    if det_died and not int_died:
        return "internal-preferred"
    if int_died and det_died:
        return "no-preference"
    if det_score > int_score:
        return "det-preferred"
    if int_score > det_score:
        return "internal-preferred"
    return "no-preference"


def load_episodes(data_path: str = None) -> List[Dict]:
    """Load counterfactual episode data."""
    path = data_path or COUNTERFACTUAL_DATA
    episodes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            episodes.append(json.loads(line))
    return episodes


def get_episode_observations(episode: Dict) -> List[str]:
    """Extract observation strings from episode's internal trajectory."""
    int_data = episode.get("internal", {})
    trajectory = int_data.get("trajectory", [])
    return [step.get("observation", "") for step in trajectory]


def compute_balanced_accuracy(gts: List[str], preds: List[str]) -> Dict:
    """Compute balanced accuracy and per-class recall.

    Only considers 'det-preferred' and 'internal-preferred' classes
    (drops 'no-preference' from evaluation since router must pick one).
    """
    labels = ["det-preferred", "internal-preferred"]

    tp = defaultdict(int)
    fn = defaultdict(int)
    total_per_class = defaultdict(int)

    for gt, pred in zip(gts, preds):
        if gt not in labels:
            continue
        total_per_class[gt] += 1
        if gt == pred:
            tp[gt] += 1
        else:
            fn[gt] += 1

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
        "gt_distribution": dict(total_per_class),
    }


def evaluate_router(router, episodes: List[Dict], verbose: bool = False) -> Dict:
    """Evaluate a GuidedRouter on counterfactual episodes.

    Returns balanced accuracy metrics and per-task breakdown.
    """
    gts = []
    preds = []
    per_task = defaultdict(lambda: {"gts": [], "preds": []})

    for ep in episodes:
        task_type = ep["task_type"]
        task_desc = ep.get("task_desc", "")
        gt_label = classify_episode_gt(ep)
        observations = get_episode_observations(ep)

        # Get router decision
        route_decision = router.route_episode(task_type, observations, task_desc=task_desc)

        # Map router decision to label
        pred_label = "det-preferred" if route_decision == "deterministic" else "internal-preferred"

        gts.append(gt_label)
        preds.append(pred_label)
        per_task[task_type]["gts"].append(gt_label)
        per_task[task_type]["preds"].append(pred_label)

        if verbose and gt_label != "no-preference" and gt_label != pred_label:
            print(f"  MISS: {task_type} ep={ep.get('episode_id','?')} "
                  f"gt={gt_label} pred={pred_label} route={route_decision}")

    overall = compute_balanced_accuracy(gts, preds)

    # Per-task breakdown
    task_results = {}
    for task_type, data in sorted(per_task.items()):
        task_metrics = compute_balanced_accuracy(data["gts"], data["preds"])
        task_results[task_type] = task_metrics

    overall["per_task"] = task_results
    overall["n_episodes"] = len(episodes)
    overall["n_tasks"] = len(per_task)

    return overall
