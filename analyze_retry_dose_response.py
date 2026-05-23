"""Analyze per-trial dose-response from retry-control and reflexion data."""
import json, glob, os, math
from collections import defaultdict

DEATH_THRESHOLD = -100
EXCLUDE_TASKS = {"find-non-living-thing"}

def wilson_ci(p, n, z=1.96):
    """Wilson score interval for proportion."""
    if n == 0:
        return (0, 0)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    margin = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return (max(0, center - margin), min(1, center + margin))

def load_episodes(directory):
    """Load all episode JSON files from a directory."""
    episodes = []
    for f in sorted(glob.glob(os.path.join(directory, "*.json"))):
        with open(f) as fh:
            episodes.append(json.load(fh))
    return episodes

def compute_dose_response(episodes, max_n=3):
    """Compute DR(N) = P(all first N trials are death).
    
    For episodes that succeeded early (fewer trials than N),
    they survived, so they do NOT count as all-death.
    """
    results = {}
    for n in range(1, max_n + 1):
        all_death_count = 0
        total = 0
        task_breakdown = defaultdict(lambda: {"deaths": 0, "total": 0})
        
        for ep in episodes:
            task = ep["task_name"]
            if task in EXCLUDE_TASKS:
                continue
            total += 1
            task_breakdown[task]["total"] += 1
            
            trials = ep["trials"]
            first_n = trials[:n]
            all_death = all(t["is_death"] for t in first_n) and len(first_n) == n
            
            if all_death:
                all_death_count += 1
                task_breakdown[task]["deaths"] += 1
        
        dr = all_death_count / total if total > 0 else 0
        lo, hi = wilson_ci(dr, total)
        
        task_drs = {}
        for task, counts in sorted(task_breakdown.items()):
            t_dr = counts["deaths"] / counts["total"] if counts["total"] > 0 else 0
            t_lo, t_hi = wilson_ci(t_dr, counts["total"])
            task_drs[task] = {
                "dr": round(t_dr, 4),
                "deaths": counts["deaths"],
                "total": counts["total"],
                "ci_95": [round(t_lo, 4), round(t_hi, 4)]
            }
        
        results[n] = {
            "N": n,
            "death_rate": round(dr, 4),
            "deaths": all_death_count,
            "total_episodes": total,
            "ci_95": [round(lo, 4), round(hi, 4)],
            "per_task": task_drs
        }
    
    return results

# Load retry data
print("=" * 60)
print("RETRY-CONTROL DOSE-RESPONSE ANALYSIS")
print("=" * 60)

retry_eps = load_episodes("outputs/retry-control")
print(f"Loaded {len(retry_eps)} retry episodes")
print(f"Tasks: {sorted(set(ep['task_name'] for ep in retry_eps))}")
print(f"Death-prone (excl. find-non-living-thing): {len([e for e in retry_eps if e['task_name'] not in EXCLUDE_TASKS])} eps")

retry_dr = compute_dose_response(retry_eps, max_n=3)

print(f"\n{'N':<4} {'DR':<8} {'95% CI':<20} {'Deaths/Total'}")
print("-" * 50)
for n in range(1, 4):
    r = retry_dr[n]
    print(f"{n:<4} {r['death_rate']:.4f}  [{r['ci_95'][0]:.4f}, {r['ci_95'][1]:.4f}]  {r['deaths']}/{r['total_episodes']}")

print("\nPer-task breakdown (N=1, N=2, N=3):")
print(f"{'Task':<45} {'DR(1)':<8} {'DR(2)':<8} {'DR(3)':<8}")
print("-" * 70)
all_tasks = sorted(retry_dr[1]["per_task"].keys())
for task in all_tasks:
    dr1 = retry_dr[1]["per_task"].get(task, {}).get("dr", "-")
    dr2 = retry_dr[2]["per_task"].get(task, {}).get("dr", "-")
    dr3 = retry_dr[3]["per_task"].get(task, {}).get("dr", "-")
    print(f"{task:<45} {dr1:<8} {dr2:<8} {dr3:<8}")

# Load reflexion data
print("\n" + "=" * 60)
print("REFLEXION DOSE-RESPONSE ANALYSIS")
print("=" * 60)

reflex_eps = []
for d in ["results/reflexion_baseline_part1", "results/reflexion_baseline_part2"]:
    reflex_eps.extend(load_episodes(d))

print(f"Loaded {len(reflex_eps)} reflexion episodes")
print(f"Death-prone: {len([e for e in reflex_eps if e['task_name'] not in EXCLUDE_TASKS])} eps")

reflex_dr = compute_dose_response(reflex_eps, max_n=3)

print(f"\n{'N':<4} {'DR':<8} {'95% CI':<20} {'Deaths/Total'}")
print("-" * 50)
for n in range(1, 4):
    r = reflex_dr[n]
    print(f"{n:<4} {r['death_rate']:.4f}  [{r['ci_95'][0]:.4f}, {r['ci_95'][1]:.4f}]  {r['deaths']}/{r['total_episodes']}")

print("\nPer-task breakdown (N=1, N=2, N=3):")
print(f"{'Task':<45} {'DR(1)':<8} {'DR(2)':<8} {'DR(3)':<8}")
print("-" * 70)
all_tasks_r = sorted(reflex_dr[1]["per_task"].keys())
for task in all_tasks_r:
    dr1 = reflex_dr[1]["per_task"].get(task, {}).get("dr", "-")
    dr2 = reflex_dr[2]["per_task"].get(task, {}).get("dr", "-")
    dr3 = reflex_dr[3]["per_task"].get(task, {}).get("dr", "-")
    print(f"{task:<45} {dr1:<8} {dr2:<8} {dr3:<8}")

# Comparison
print("\n" + "=" * 60)
print("COMPARISON: RETRY vs REFLEXION")
print("=" * 60)
print(f"\n{'N':<4} {'Retry DR':<12} {'Reflexion DR':<14} {'Delta'}")
print("-" * 45)
for n in range(1, 4):
    r_dr = retry_dr[n]["death_rate"]
    x_dr = reflex_dr[n]["death_rate"]
    delta = x_dr - r_dr
    print(f"{n:<4} {r_dr:.4f}      {x_dr:.4f}        {delta:+.4f}")

# Save results
output = {
    "retry_control": {
        "total_episodes": len(retry_eps),
        "death_prone_episodes": len([e for e in retry_eps if e["task_name"] not in EXCLUDE_TASKS]),
        "dose_response": {str(k): v for k, v in retry_dr.items()},
    },
    "reflexion": {
        "total_episodes": len(reflex_eps),
        "death_prone_episodes": len([e for e in reflex_eps if e["task_name"] not in EXCLUDE_TASKS]),
        "dose_response": {str(k): v for k, v in reflex_dr.items()},
    },
    "definition": {
        "death_threshold": DEATH_THRESHOLD,
        "DR_N": "P(all first N trials have score <= -100)",
        "excluded_tasks": list(EXCLUDE_TASKS),
        "note": "Episodes that succeeded early (< N trials) count as survived"
    }
}

os.makedirs("results", exist_ok=True)
with open("results/retry_dose_response_analysis.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to results/retry_dose_response_analysis.json")
