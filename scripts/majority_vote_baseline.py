"""Task-type majority-vote baseline for cross-split episode-level eval.

Compares against router balanced_acc=47.1±2.2% (8 seeds, split seed=42).
"""
import json, os
from collections import defaultdict, Counter
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

DATA_PATH = "./data/plan-004-implicit/implicit_sft.jsonl"
PROJECT_ROOT = "."
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results/majority_vote_baseline")

ROUTER_CROSS_SPLIT = {
    42: 0.4813, 43: 0.4792, 44: 0.4958, 45: 0.4938,
    46: 0.4854, 47: 0.4313, 48: 0.4542, 49: 0.4479,
}
ROUTER_MEAN = 0.4711
ROUTER_STD = 0.0221


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

    train_indices = []
    for ep in train_eps:
        train_indices.extend(episode_to_indices[ep])
    val_indices = []
    for ep in val_eps:
        val_indices.extend(episode_to_indices[ep])

    train_records = [records[i] for i in sorted(train_indices)]
    val_records = [records[i] for i in sorted(val_indices)]
    return train_records, val_records, set(train_eps), set(val_eps)


def run_baseline(records, split_seed=42):
    train_recs, val_recs, train_eps, val_eps = episode_level_split(
        records, test_size=0.15, seed=split_seed
    )

    # --- Per task_type label distribution in train set ---
    tt_labels = defaultdict(lambda: Counter())
    for r in train_recs:
        label = 1 if r["route_label"] == "deterministic" else 0
        tt_labels[r["task_type"]][label] += 1

    # Majority label per task_type
    tt_majority = {}
    for tt, counts in tt_labels.items():
        tt_majority[tt] = counts.most_common(1)[0][0]

    # Global majority in train
    all_train_labels = [1 if r["route_label"] == "deterministic" else 0 for r in train_recs]
    global_majority = Counter(all_train_labels).most_common(1)[0][0]

    # --- Predict on val ---
    val_true = []
    val_pred_tt = []
    val_pred_global = []
    unseen_tt = set()

    for r in val_recs:
        gt = 1 if r["route_label"] == "deterministic" else 0
        tt = r["task_type"]
        val_true.append(gt)

        if tt in tt_majority:
            val_pred_tt.append(tt_majority[tt])
        else:
            unseen_tt.add(tt)
            val_pred_tt.append(global_majority)

        val_pred_global.append(global_majority)

    bal_acc_tt = balanced_accuracy_score(val_true, val_pred_tt)
    bal_acc_global = balanced_accuracy_score(val_true, val_pred_global)
    acc_tt = np.mean(np.array(val_true) == np.array(val_pred_tt))
    acc_global = np.mean(np.array(val_true) == np.array(val_pred_global))

    return {
        "split_seed": split_seed,
        "n_train": len(train_recs),
        "n_val": len(val_recs),
        "n_train_episodes": len(train_eps),
        "n_val_episodes": len(val_eps),
        "global_majority_label": "deterministic" if global_majority == 1 else "internal",
        "global_majority_train_frac": Counter(all_train_labels).most_common(1)[0][1] / len(all_train_labels),
        "task_type_train_dist": {
            tt: {"det": counts[1], "internal": counts[0], "majority": "det" if tt_majority[tt] == 1 else "internal"}
            for tt, counts in sorted(tt_labels.items())
        },
        "unseen_task_types_in_val": sorted(unseen_tt),
        "majority_vote_by_task_type": {
            "balanced_acc": round(bal_acc_tt, 4),
            "accuracy": round(acc_tt, 4),
        },
        "majority_vote_global": {
            "balanced_acc": round(bal_acc_global, 4),
            "accuracy": round(acc_global, 4),
        },
        "val_label_dist": dict(Counter(val_true)),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    records = load_data()
    print(f"Loaded {len(records)} records")

    # Primary: split seed=42 (matches cross_split_eval.py)
    split_seeds = [42, 123, 456]
    all_results = {}

    for seed in split_seeds:
        print(f"\n{'='*60}")
        print(f"Split seed={seed}")
        print(f"{'='*60}")

        res = run_baseline(records, split_seed=seed)
        all_results[seed] = res

        print(f"\nTrain: {res['n_train']} steps, {res['n_train_episodes']} episodes")
        print(f"Val:   {res['n_val']} steps, {res['n_val_episodes']} episodes")
        print(f"Global majority: {res['global_majority_label']} ({res['global_majority_train_frac']:.1%})")

        print(f"\nPer task_type train distribution:")
        print(f"  {'task_type':<45} {'det':>5} {'int':>5} {'majority':>10}")
        print(f"  {'-'*45} {'-'*5} {'-'*5} {'-'*10}")
        for tt, info in res["task_type_train_dist"].items():
            print(f"  {tt:<45} {info['det']:>5} {info['internal']:>5} {info['majority']:>10}")

        if res["unseen_task_types_in_val"]:
            print(f"\n  Unseen task_types in val (fallback to global): {res['unseen_task_types_in_val']}")

        print(f"\nResults:")
        print(f"  Task-type majority vote: balanced_acc={res['majority_vote_by_task_type']['balanced_acc']:.4f}, "
              f"acc={res['majority_vote_by_task_type']['accuracy']:.4f}")
        print(f"  Global majority vote:    balanced_acc={res['majority_vote_global']['balanced_acc']:.4f}, "
              f"acc={res['majority_vote_global']['accuracy']:.4f}")

    # Summary comparison
    print(f"\n{'='*70}")
    print("SUMMARY: Majority Vote vs Router (cross-split)")
    print(f"{'='*70}")
    print(f"{'Split Seed':>12} {'TT-MajVote':>12} {'Global-Maj':>12} {'Router':>12}")
    print(f"{'-'*12:>12} {'-'*12:>12} {'-'*12:>12} {'-'*12:>12}")

    seed42_res = all_results[42]
    print(f"{'42':>12} {seed42_res['majority_vote_by_task_type']['balanced_acc']:>12.4f} "
          f"{seed42_res['majority_vote_global']['balanced_acc']:>12.4f} "
          f"{ROUTER_MEAN:>12.4f}±{ROUTER_STD:.4f}")

    tt_accs = [all_results[s]["majority_vote_by_task_type"]["balanced_acc"] for s in split_seeds]
    gl_accs = [all_results[s]["majority_vote_global"]["balanced_acc"] for s in split_seeds]
    print(f"\nAcross split seeds {split_seeds}:")
    print(f"  TT majority vote:  {np.mean(tt_accs):.4f}±{np.std(tt_accs):.4f}")
    print(f"  Global majority:   {np.mean(gl_accs):.4f}±{np.std(gl_accs):.4f}")
    print(f"  Router (8 seeds):  {ROUTER_MEAN:.4f}±{ROUTER_STD:.4f}")

    delta_tt = seed42_res["majority_vote_by_task_type"]["balanced_acc"] - ROUTER_MEAN
    print(f"\n  TT-MajVote vs Router (seed42 split): {delta_tt:+.4f}")
    if delta_tt > 0:
        print("  → Task-type majority vote BEATS router → signal is largely at task-type level")
    else:
        print("  → Router beats task-type majority → some episode/step-level signal exists")

    # Save
    output = {
        "description": "Task-type majority vote baseline vs router cross-split eval",
        "data_path": DATA_PATH,
        "router_reference": {"mean_balanced_acc": ROUTER_MEAN, "std": ROUTER_STD, "per_seed": ROUTER_CROSS_SPLIT},
        "split_seeds": {str(s): all_results[s] for s in split_seeds},
        "summary": {
            "tt_majority_vote_mean": round(np.mean(tt_accs), 4),
            "tt_majority_vote_std": round(np.std(tt_accs), 4),
            "global_majority_mean": round(np.mean(gl_accs), 4),
            "global_majority_std": round(np.std(gl_accs), 4),
        }
    }
    out_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
