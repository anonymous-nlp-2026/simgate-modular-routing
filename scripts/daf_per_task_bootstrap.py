import json, os, numpy as np
from scipy import stats

DATA_DIR = "./results/plan_c"
OUT_PATH = "./artifacts/daf-sensitivity-extended/per_task_bootstrap.json"
DEATH_SCORE = -100.0
PENALTIES = [10, 25, 50, 100, 200, 500]
N_BOOTSTRAP = 10000
SEED = 42

rng = np.random.default_rng(SEED)

task_dirs = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])

# Load episode-level data per task
task_episodes = {}
for task in task_dirs:
    fp = os.path.join(DATA_DIR, task, "episodes.jsonl")
    if not os.path.exists(fp):
        continue
    episodes = []
    with open(fp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ep = json.loads(line)
            int_score = ep["always_internal"]["final_score"]
            det_score = ep["always_deterministic"]["final_score"]
            int_dead = (int_score == DEATH_SCORE)
            det_dead = (det_score == DEATH_SCORE)
            episodes.append({
                "int_score": int_score,
                "det_score": det_score,
                "int_dead": int_dead,
                "det_dead": det_dead,
            })
    task_episodes[task] = episodes

total_eps = sum(len(v) for v in task_episodes.values())
print(f"Loaded {total_eps} episodes across {len(task_episodes)} tasks")

# Part A: Per-task bootstrap CI for delta(det - internal) at P=100
per_task_results = {}
for task, episodes in task_episodes.items():
    n = len(episodes)
    deltas = np.array([ep["det_score"] - ep["int_score"] for ep in episodes])
    delta_mean = float(np.mean(deltas))
    
    # Bootstrap 95% CI
    boot_means = np.zeros(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = np.mean(deltas[idx])
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    
    # Death stats
    n_int_dies = sum(1 for ep in episodes if ep["int_dead"])
    n_det_dies = sum(1 for ep in episodes if ep["det_dead"])
    n_only_int = sum(1 for ep in episodes if ep["int_dead"] and not ep["det_dead"])
    n_only_det = sum(1 for ep in episodes if ep["det_dead"] and not ep["int_dead"])
    
    # DAF at P=100
    delta_i = (n_only_int - n_only_det) / n
    has_deaths = (n_only_int + n_only_det) > 0
    
    per_task_results[task] = {
        "n": n,
        "delta_mean": round(delta_mean, 4),
        "delta_CI_lower": round(ci_lower, 4),
        "delta_CI_upper": round(ci_upper, 4),
        "all_positive": bool(ci_lower > 0),
        "death_rate_internal": round(n_int_dies / n, 4),
        "death_rate_det": round(n_det_dies / n, 4),
        "delta_i": round(delta_i, 4),
        "n_only_int_dies": n_only_int,
        "n_only_det_dies": n_only_det,
    }

# Identify outliers (CI includes 0 or delta is negative)
outlier_tasks = [t for t, r in per_task_results.items() if not r["all_positive"]]

print(f"\n=== Part A: Per-task Bootstrap CI (P=100 scoring) ===")
print(f"{'Task':<45} {'n':>3} {'mean':>8} {'CI_lo':>8} {'CI_hi':>8} {'pos?':>5}")
print("-" * 80)
for task, r in sorted(per_task_results.items()):
    print(f"{task:<45} {r['n']:>3} {r['delta_mean']:>8.2f} {r['delta_CI_lower']:>8.2f} {r['delta_CI_upper']:>8.2f} {'Y' if r['all_positive'] else 'N':>5}")
print(f"\nOutlier tasks (CI includes 0): {outlier_tasks if outlier_tasks else 'None'}")

# Part B: Pearson r at all penalty values
# For each penalty P, compute per-task delta_full(P) and correlate with death_rate
print(f"\n=== Part B: Pearson r (death_rate vs Δ(D-I)) at all P ===")

pearson_results = {}
for P in PENALTIES:
    task_delta_full = []
    task_death_rate = []
    
    for task, episodes in sorted(task_episodes.items()):
        n = len(episodes)
        # Compute delta_full at this P: replace death scores with -P
        score_sum = 0.0
        for ep in episodes:
            int_s = -P if ep["int_dead"] else ep["int_score"]
            det_s = -P if ep["det_dead"] else ep["det_score"]
            score_sum += (det_s - int_s)
        delta_full = score_sum / n
        
        # Death rate differential (internal - deterministic)
        n_only_int = sum(1 for ep in episodes if ep["int_dead"] and not ep["det_dead"])
        n_only_det = sum(1 for ep in episodes if ep["det_dead"] and not ep["int_dead"])
        delta_i = (n_only_int - n_only_det) / n
        
        task_delta_full.append(delta_full)
        task_death_rate.append(delta_i)
    
    x = np.array(task_death_rate)
    y = np.array(task_delta_full)
    r, p_val = stats.pearsonr(x, y)
    
    pearson_results[f"P_{P}"] = {
        "r": round(float(r), 4),
        "p": float(f"{p_val:.2e}") if p_val < 0.001 else round(float(p_val), 6),
        "significant": bool(p_val < 0.05),
    }
    print(f"  P={P:>4}: r={r:.4f}, p={p_val:.2e}, sig={'YES' if p_val < 0.05 else 'NO'}")

# Determine conclusion
all_sig = all(v["significant"] for v in pearson_results.values())
no_outliers = len(outlier_tasks) == 0
conclusion = "all_consistent" if (all_sig and no_outliers) else "outlier_found"

# Assemble output
output = {
    "per_task": per_task_results,
    "outlier_tasks": outlier_tasks,
    "pearson_r_by_penalty": pearson_results,
    "summary": {
        "total_episodes": total_eps,
        "n_tasks": len(task_episodes),
        "all_tasks_positive": no_outliers,
        "all_pearson_significant": all_sig,
    },
    "conclusion": conclusion,
}

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nConclusion: {conclusion}")
print(f"Results saved to: {OUT_PATH}")
