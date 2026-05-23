"""
Task-type classifier: can task_type alone predict routing label?
Input: one-hot task_type (11 dim)
Output: routing label (det-pref / int-pref / no-pref)
"""
import json
import numpy as np
from collections import Counter, defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, balanced_accuracy_score, make_scorer

DATA_PATH = "./results/baseline_245eps/episode_details.jsonl"
OUT_PATH = "./guided_router/task_classifier_results.json"


def load_episodes():
    episodes = []
    with open(DATA_PATH) as f:
        for line in f:
            d = json.loads(line)
            if d["condition"] != "always-deterministic":
                continue
            score_det = d["det_final"]
            score_int = d["internal_final"]
            if score_det > score_int:
                label = "det-pref"
            elif score_int > score_det:
                label = "int-pref"
            else:
                label = "no-pref"
            episodes.append({
                "task_type": d["task_type"],
                "label": label,
            })
    return episodes


def main():
    episodes = load_episodes()
    print(f"Loaded {len(episodes)} episodes")

    # Get unique task types and create one-hot encoding
    task_types = sorted(set(e["task_type"] for e in episodes))
    task_to_idx = {t: i for i, t in enumerate(task_types)}
    n_tasks = len(task_types)

    X = np.zeros((len(episodes), n_tasks))
    labels = []
    for i, e in enumerate(episodes):
        X[i, task_to_idx[e["task_type"]]] = 1.0
        labels.append(e["label"])

    le = LabelEncoder()
    y = le.fit_transform(labels)
    class_names = le.classes_.tolist()

    print(f"Task types ({n_tasks}): {task_types}")
    print(f"Label distribution: {Counter(labels)}")
    print()

    # === 1. Majority vote per task type ===
    print("=" * 60)
    print("=== Task-type Classifier Results ===")
    print("=" * 60)
    print()
    print("Majority vote per task type:")
    print(f"  {'Task':<45} {'Most common':<12} {'Count':<10} {'Acc':<6}")
    print(f"  {'-'*45} {'-'*12} {'-'*10} {'-'*6}")

    task_episodes = defaultdict(list)
    for e in episodes:
        task_episodes[e["task_type"]].append(e["label"])

    majority_correct = 0
    majority_total = 0
    task_results = {}
    for task in sorted(task_episodes.keys()):
        task_labels = task_episodes[task]
        counter = Counter(task_labels)
        majority_label = counter.most_common(1)[0][0]
        majority_count = counter.most_common(1)[0][1]
        total = len(task_labels)
        acc = majority_count / total
        print(f"  {task:<45} {majority_label:<12} {majority_count}/{total:<7} {acc:.1%}")
        majority_correct += majority_count
        majority_total += total
        task_results[task] = {
            "majority_label": majority_label,
            "majority_count": majority_count,
            "total": total,
            "task_accuracy": acc,
            "distribution": dict(counter),
        }

    majority_acc = majority_correct / majority_total
    print()
    print(f"  Overall episode-level accuracy: {majority_acc:.1%}")

    # Balanced accuracy for majority vote
    y_pred_majority = []
    for e in episodes:
        task = e["task_type"]
        y_pred_majority.append(task_results[task]["majority_label"])
    y_true_labels = [e["label"] for e in episodes]
    le2 = LabelEncoder()
    le2.fit(list(set(y_true_labels + y_pred_majority)))
    bal_acc_majority = balanced_accuracy_score(
        le2.transform(y_true_labels), le2.transform(y_pred_majority)
    )
    print(f"  Balanced accuracy: {bal_acc_majority:.1%}")
    print()

    # === 2. Logistic Regression with Stratified 5-fold CV ===
    print("Logistic Regression (5-fold stratified CV):")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = LogisticRegression(max_iter=1000, random_state=42, )

    acc_scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")
    bal_scorer = make_scorer(balanced_accuracy_score)
    bal_scores = cross_val_score(clf, X, y, cv=skf, scoring=bal_scorer)

    print(f"  Mean accuracy: {acc_scores.mean():.1%} (±{acc_scores.std():.1%})")
    print(f"  Mean balanced accuracy: {bal_scores.mean():.1%} (±{bal_scores.std():.1%})")
    print()

    # === 3. LOOCV ===
    print("LOOCV:")
    loo = LeaveOneOut()
    loo_scores = cross_val_score(clf, X, y, cv=loo, scoring="accuracy")
    loocv_acc = loo_scores.mean()
    print(f"  Accuracy: {loocv_acc:.1%}")
    print()

    # === 4. Classification report (full dataset) ===
    clf.fit(X, y)
    y_pred_full = clf.predict(X)
    print("Classification report (full dataset resubstitution):")
    print(classification_report(y, y_pred_full, target_names=class_names))

    # === Conclusion ===
    threshold = 0.80
    is_task_level = majority_acc > threshold
    conclusion = (
        f"routing signal IS primarily a task-type-level property "
        f"(majority vote acc = {majority_acc:.1%} > {threshold:.0%})"
        if is_task_level
        else f"routing signal is NOT purely a task-type-level property "
        f"(majority vote acc = {majority_acc:.1%} <= {threshold:.0%})"
    )
    print(f"Conclusion: {conclusion}")

    # Save results
    results = {
        "n_episodes": len(episodes),
        "n_task_types": n_tasks,
        "task_types": task_types,
        "label_distribution": dict(Counter(labels)),
        "majority_vote": {
            "overall_accuracy": round(majority_acc, 4),
            "balanced_accuracy": round(bal_acc_majority, 4),
            "per_task": task_results,
        },
        "logistic_regression_5fold": {
            "mean_accuracy": round(float(acc_scores.mean()), 4),
            "std_accuracy": round(float(acc_scores.std()), 4),
            "mean_balanced_accuracy": round(float(bal_scores.mean()), 4),
            "std_balanced_accuracy": round(float(bal_scores.std()), 4),
        },
        "loocv_accuracy": round(float(loocv_acc), 4),
        "conclusion": conclusion,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
