import json, os, glob
from collections import defaultdict

data_dir = "./results/retry_dose_response/retry_n5/"

all_death = 0
all_alive = 0
mixed = 0
mixed_details = []
per_task = defaultdict(lambda: {"all_death": 0, "all_alive": 0, "mixed": 0, "rescue": 0, "regression": 0, "total": 0})

total_eps = 0
total_early_success = 0

for f in sorted(glob.glob(os.path.join(data_dir, "*_ep*.json"))):
    fname = os.path.basename(f)
    with open(f) as fh:
        ep = json.load(fh)
    
    task = ep["task_name"]
    trials = ep["trials"]
    deaths = [t["is_death"] for t in trials]
    scores = [t["score"] for t in trials]
    n_trials = len(trials)
    total_eps += 1
    
    if n_trials < 5:
        total_early_success += 1
    
    per_task[task]["total"] += 1
    
    if all(deaths):
        all_death += 1
        per_task[task]["all_death"] += 1
    elif not any(deaths):
        all_alive += 1
        per_task[task]["all_alive"] += 1
    else:
        mixed += 1
        per_task[task]["mixed"] += 1
        
        # Rescue: first trial is death, at least one later trial is not death
        is_rescue = deaths[0] and not all(deaths[1:])
        # Regression: first trial is alive, at least one later trial is death
        is_regression = not deaths[0] and any(deaths[1:])
        
        if is_rescue:
            per_task[task]["rescue"] += 1
        if is_regression:
            per_task[task]["regression"] += 1
        
        mixed_details.append({
            "task": task,
            "episode": fname,
            "n_trials": n_trials,
            "deaths": deaths,
            "scores": scores,
            "death_pattern": f"{sum(deaths)}/{len(deaths)} deaths",
            "rescue": is_rescue,
            "regression": is_regression,
        })

# Compute totals for rescue/regression
total_rescue = sum(1 for d in mixed_details if d["rescue"])
total_regression = sum(1 for d in mixed_details if d["regression"])

print("=" * 60)
print("Retry N=5 Determinism Audit")
print("=" * 60)
print(f"Total episodes: {total_eps}")
print(f"  Early success (< 5 trials): {total_early_success}")
print(f"  Full 5 trials: {total_eps - total_early_success}")
print()
print(f"全死 (all death):  {all_death}/{total_eps} ({100*all_death/total_eps:.1f}%)")
print(f"全活 (all alive):  {all_alive}/{total_eps} ({100*all_alive/total_eps:.1f}%)")
print(f"混合 (mixed):      {mixed}/{total_eps} ({100*mixed/total_eps:.1f}%)")
print(f"Rescue (death→alive): {total_rescue}/{total_eps} ({100*total_rescue/total_eps:.1f}%)")
print(f"Regression (alive→death): {total_regression}/{total_eps} ({100*total_regression/total_eps:.1f}%)")
print()

# Per-task breakdown
print("=" * 60)
print("Per-Task Breakdown")
print("=" * 60)
header = f"{'Task':<45} {'Total':>5} {'全死':>5} {'全活':>5} {'混合':>5} {'Rescue':>7} {'Regr':>5}"
print(header)
print("-" * len(header))
for task in sorted(per_task.keys()):
    t = per_task[task]
    print(f"{task:<45} {t['total']:>5} {t['all_death']:>5} {t['all_alive']:>5} {t['mixed']:>5} {t['rescue']:>7} {t['regression']:>5}")

# Mixed episode details
if mixed_details:
    print()
    print("=" * 60)
    print(f"Mixed Episode Details ({len(mixed_details)} episodes)")
    print("=" * 60)
    for d in mixed_details:
        rescue_tag = " [RESCUE]" if d["rescue"] else ""
        regr_tag = " [REGRESSION]" if d["regression"] else ""
        print(f"  {d['task']} | {d['episode']} | {d['death_pattern']} | trials={d['n_trials']}{rescue_tag}{regr_tag}")
        print(f"    deaths:  {d['deaths']}")
        print(f"    scores:  {d['scores']}")
else:
    print()
    print("NO MIXED EPISODES — 100% deterministic behavior across all trials.")
