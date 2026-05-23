"""Signal Concentration Index (SCI) computation."""
import csv
import json
import os

DATA_PATH = "./results/plan_012_decomposition"
OUTPUT_PATH = "./guided_router/sci_results.json"


def gini(values):
    sorted_v = sorted(values)
    n = len(sorted_v)
    total = sum(sorted_v)
    if total == 0 or n == 0:
        return 0.0
    cumsum = sum((2 * i - n - 1) * v for i, v in enumerate(sorted_v, 1))
    return cumsum / (n * total)


def main():
    # Load CSV data
    tasks = []
    with open(DATA_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tasks.append({
                "task_name": row["task_name"],
                "n_episodes": int(row["n_episodes"]),
                "death_rate_int": float(row["death_rate_int"]),
                "death_rate_det": float(row["death_rate_det"]),
                "delta_full": float(row["delta_full"]),
                "delta_nd": float(row["delta_nd"]),
                "det_pref": int(row["det_pref"]),
            })

    # Compute death contribution per task: |delta_full - delta_nd|
    for t in tasks:
        t["death_contribution"] = abs(t["delta_full"] - t["delta_nd"])

    total_contribution = sum(t["death_contribution"] for t in tasks)

    # Normalize to get c_t
    for t in tasks:
        t["c_t"] = t["death_contribution"] / total_contribution if total_contribution > 0 else 0.0

    # Sort by c_t descending
    tasks_sorted = sorted(tasks, key=lambda x: x["c_t"], reverse=True)

    # Compute Gini (SCI)
    c_values = [t["c_t"] for t in tasks]
    sci = gini(c_values)

    # Top-k Concentration Ratio
    top_k_cr = {}
    cumsum = 0.0
    for k, t in enumerate(tasks_sorted, 1):
        cumsum += t["c_t"]
        if k <= 3:
            top_k_cr[f"top_{k}"] = {
                "value": round(cumsum * 100, 2),
                "tasks": [tt["task_name"] for tt in tasks_sorted[:k]]
            }

    # Build results
    results = {
        "sci_gini": round(sci, 4),
        "total_death_contribution": round(total_contribution, 2),
        "top_k_cr": top_k_cr,
        "per_task": [
            {
                "task_name": t["task_name"],
                "c_t_pct": round(t["c_t"] * 100, 2),
                "death_contribution": round(t["death_contribution"], 2),
                "delta_full": t["delta_full"],
                "delta_nd": t["delta_nd"],
                "n_episodes": t["n_episodes"],
                "det_pref": t["det_pref"],
                "death_rate_int": t["death_rate_int"],
            }
            for t in tasks_sorted
        ],
        "n_tasks": len(tasks),
        "n_episodes_total": sum(t["n_episodes"] for t in tasks),
    }

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"=== Signal Concentration Index Results ===")
    print(f"SCI (Gini): {sci:.4f}")
    print(f"Top-1 CR: {top_k_cr['top_1']['value']:.1f}% ({top_k_cr['top_1']['tasks'][0]})")
    print(f"Top-2 CR: {top_k_cr['top_2']['value']:.1f}%")
    print(f"Top-3 CR: {top_k_cr['top_3']['value']:.1f}%")
    print(f"\nPer-task signal contributions:")
    for t in tasks_sorted:
        print(f"  {t['task_name']:45s}: {t['c_t']*100:5.1f}% "
              f"(n={t['n_episodes']}, Δ_full={t['delta_full']:.2f}, Δ_nd={t['delta_nd']:.2f}, "
              f"death_rate_int={t['death_rate_int']:.0%})")


if __name__ == "__main__":
    main()
