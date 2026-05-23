"""
W4 Task-Type Positive Control Experiment

Purpose: Demonstrate that the routing signal (det-preferred vs internal-preferred)
is fundamentally a task-type-level property, not a step-level property.

Input:  results/plan_c/*/labeled_steps.jsonl (11 task types, ~2040 steps)
Output: artifacts/w4_task_type_positive_control.json

Two evaluations:
  1. 5-Fold Stratified CV (episode-level grouping) — upper bound
  2. Leave-One-Task-Type-Out (LOTO) — generalization test
"""

import argparse
import json
import os
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

SEED = 42
LABEL_MAP = {"det-preferred": 1, "internal-preferred": 0}


def load_data(data_dir: str) -> list[dict]:
    """Load all labeled_steps.jsonl files from plan_c subdirectories."""
    records = []
    data_path = Path(data_dir)
    for jsonl_file in sorted(data_path.glob("*/labeled_steps.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                rec = json.loads(line)
                records.append(rec)
    return records


def build_features(records: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    """Build one-hot features from task_type field.

    Returns: X (n, n_task_types), y (n,), task_types per sample,
             episode_keys per sample, sorted unique task type names.
    """
    task_types = [r["task_type"] for r in records]
    labels = np.array([LABEL_MAP[r["label"]] for r in records])
    episode_keys = [f"{r['task_type']}_{r['episode_id']}" for r in records]

    encoder = OneHotEncoder(sparse_output=False, categories="auto")
    X = encoder.fit_transform(np.array(task_types).reshape(-1, 1))
    unique_task_types = list(encoder.categories_[0])

    return X, labels, task_types, episode_keys, unique_task_types


def make_models():
    """Create the three classifiers."""
    return {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=SEED
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(32,), max_iter=500, random_state=SEED
        ),
        "MajorityBaseline": DummyClassifier(strategy="most_frequent"),
    }


def eval_5fold_cv(X, y, episode_keys):
    """5-Fold Stratified CV grouped by episode."""
    unique_episodes = list(dict.fromkeys(episode_keys))
    episode_to_idx = defaultdict(list)
    for i, ek in enumerate(episode_keys):
        episode_to_idx[ek].append(i)

    ep_labels = []
    for ep in unique_episodes:
        ep_labels.append(y[episode_to_idx[ep][0]])
    ep_labels = np.array(ep_labels)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = {}

    for model_name, model in make_models().items():
        fold_metrics = []
        for train_ep_idx, test_ep_idx in skf.split(unique_episodes, ep_labels):
            train_episodes = {unique_episodes[i] for i in train_ep_idx}
            test_episodes = {unique_episodes[i] for i in test_ep_idx}

            train_idx = [i for i, ek in enumerate(episode_keys) if ek in train_episodes]
            test_idx = [i for i, ek in enumerate(episode_keys) if ek in test_episodes]

            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            if model_name == "MLP":
                # Handle class imbalance via sample weights for MLP
                class_counts = np.bincount(y_train)
                weights = len(y_train) / (2.0 * class_counts)
                sample_w = np.array([weights[yi] for yi in y_train])
                model.fit(X_train, y_train, sample_weight=sample_w)
            else:
                model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            ba = balanced_accuracy_score(y_test, y_pred)
            f1_0 = f1_score(y_test, y_pred, pos_label=0, zero_division=0)
            f1_1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()
            fold_metrics.append({
                "balanced_accuracy": ba,
                "f1_class_0": f1_0,
                "f1_class_1": f1_1,
                "confusion_matrix": cm,
            })

        ba_scores = [m["balanced_accuracy"] for m in fold_metrics]
        results[model_name] = {
            "balanced_accuracy_mean": float(np.mean(ba_scores)),
            "balanced_accuracy_std": float(np.std(ba_scores)),
            "f1_class_0_mean": float(np.mean([m["f1_class_0"] for m in fold_metrics])),
            "f1_class_1_mean": float(np.mean([m["f1_class_1"] for m in fold_metrics])),
            "per_fold": fold_metrics,
        }

    return results


def eval_loto(X, y, task_types_per_sample, unique_task_types):
    """Leave-One-Task-Type-Out evaluation."""
    task_arr = np.array(task_types_per_sample)
    results_per_task = {}

    for held_out in unique_task_types:
        test_mask = task_arr == held_out
        train_mask = ~test_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        if len(np.unique(y_test)) < 2:
            # Only one class in held-out set
            pass

        model = LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=SEED
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        ba = balanced_accuracy_score(y_test, y_pred)
        results_per_task[held_out] = {
            "n_samples": int(test_mask.sum()),
            "label_dist": {
                "internal_preferred": int((y_test == 0).sum()),
                "det_preferred": int((y_test == 1).sum()),
            },
            "balanced_accuracy": float(ba),
            "predictions_all_same": bool(len(np.unique(y_pred)) == 1),
        }

    all_ba = [v["balanced_accuracy"] for v in results_per_task.values()]
    aggregate = {
        "macro_balanced_accuracy": float(np.mean(all_ba)),
        "std": float(np.std(all_ba)),
        "min": float(np.min(all_ba)),
        "max": float(np.max(all_ba)),
    }

    return {"per_task_type": results_per_task, "aggregate": aggregate}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    print("Loading data...")
    records = load_data(args.data_dir)
    print(f"  Loaded {len(records)} steps")

    X, y, task_types_per_sample, episode_keys, unique_task_types = build_features(records)
    print(f"  {len(unique_task_types)} task types, {X.shape[1]}-dim one-hot")
    print(f"  Label distribution: internal-preferred={int((y==0).sum())}, det-preferred={int((y==1).sum())}")

    print("\n=== 5-Fold Stratified CV (episode-grouped) ===")
    cv_results = eval_5fold_cv(X, y, episode_keys)
    for name, res in cv_results.items():
        print(f"  {name}: BA = {res['balanced_accuracy_mean']:.4f} ± {res['balanced_accuracy_std']:.4f}")

    print("\n=== Leave-One-Task-Type-Out (LOTO) ===")
    loto_results = eval_loto(X, y, task_types_per_sample, unique_task_types)
    for task, res in loto_results["per_task_type"].items():
        print(f"  {task}: BA = {res['balanced_accuracy']:.4f} (n={res['n_samples']}, "
              f"dist={res['label_dist']})")
    agg = loto_results["aggregate"]
    print(f"  Aggregate: BA = {agg['macro_balanced_accuracy']:.4f} ± {agg['std']:.4f}")

    output = {
        "experiment": "w4_task_type_positive_control",
        "n_samples": len(records),
        "n_task_types": len(unique_task_types),
        "task_types": unique_task_types,
        "label_distribution": {
            "internal_preferred": int((y == 0).sum()),
            "det_preferred": int((y == 1).sum()),
        },
        "five_fold_cv": cv_results,
        "loto": loto_results,
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output_path}")


if __name__ == "__main__":
    main()
