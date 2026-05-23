import json
import numpy as np
from collections import defaultdict

def bootstrap_balanced_acc_ci(predictions, n_boot=1000, seed=42):
    """Bootstrap CI on balanced accuracy (mean of per-class recalls)."""
    rng = np.random.RandomState(seed)
    preds = np.array(predictions)  # list of dicts
    n = len(preds)
    
    bal_accs = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        sample = [preds[i] for i in idx]
        # compute balanced accuracy for this resample
        per_class = defaultdict(lambda: {"correct": 0, "total": 0})
        for p in sample:
            per_class[p["true_route"]]["total"] += 1
            per_class[p["true_route"]]["correct"] += int(p["correct"])
        recalls = []
        for cls, stats in per_class.items():
            if stats["total"] > 0:
                recalls.append(stats["correct"] / stats["total"])
        bal_acc = np.mean(recalls) if recalls else 0.0
        bal_accs.append(bal_acc)
    
    lo = float(np.percentile(bal_accs, 2.5))
    hi = float(np.percentile(bal_accs, 97.5))
    point = float(np.mean(bal_accs))
    return point, lo, hi

def load_preds(path):
    preds = []
    with open(path) as f:
        for line in f:
            preds.append(json.loads(line.strip()))
    return preds

base = "./results"

for direction in ["physical-to-life", "life-to-physical"]:
    pred_path = f"{base}/plan-010-cross-task-{direction}-seed42/predictions.jsonl"
    all_preds = load_preds(pred_path)
    
    in_domain = [p for p in all_preds if p["eval_type"] == "in_domain"]
    cross_domain = [p for p in all_preds if p["eval_type"] == "cross_domain"]
    
    # Point estimates (verify they match saved results)
    for label, preds in [("in_domain", in_domain), ("cross_domain", cross_domain)]:
        per_class = defaultdict(lambda: {"correct": 0, "total": 0})
        for p in preds:
            per_class[p["true_route"]]["total"] += 1
            per_class[p["true_route"]]["correct"] += int(p["correct"])
        recalls = {cls: stats["correct"]/stats["total"] for cls, stats in per_class.items()}
        bal_acc = np.mean(list(recalls.values()))
        
        _, ci_lo, ci_hi = bootstrap_balanced_acc_ci(preds, n_boot=1000, seed=42)
        
        print(f"\n{direction} | {label}")
        print(f"  n={len(preds)}, class_counts={dict({cls: stats['total'] for cls, stats in per_class.items()})}")
        print(f"  recalls={dict({k: round(v, 4) for k, v in recalls.items()})}")
        print(f"  balanced_acc={bal_acc:.4f}")
        print(f"  CI95_balanced=[{ci_lo:.4f}, {ci_hi:.4f}]")
        print(f"  CI contains point: {ci_lo <= bal_acc <= ci_hi}")

