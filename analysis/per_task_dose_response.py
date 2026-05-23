import json, os, glob

TASKS = [
    "chemistry-mix", "find-animal", "freeze", "grow-fruit",
    "identify-life-stages-1", "identify-life-stages-2",
    "inclined-plane-determine-angle", "lifespan-longest-lived",
    "lifespan-longest-lived-then-shortest-lived",
    "measure-melting-point-unknown-substance"
]

def load_episodes(directory):
    episodes = []
    for f in sorted(glob.glob(os.path.join(directory, "*.json"))):
        with open(f) as fh:
            episodes.append(json.load(fh))
    return episodes

def compute_death_rate(episodes, task, cutoff_k):
    task_eps = [e for e in episodes if e["task_name"] == task]
    if len(task_eps) == 0:
        return None
    deaths = 0
    for ep in task_eps:
        trials = ep["trials"]
        idx = min(cutoff_k, len(trials)) - 1
        if trials[idx].get("is_death", False):
            deaths += 1
    return deaths, len(task_eps)

# Reflexion data
refl_n3 = load_episodes("./results/reflexion_n3/")
refl_n4 = load_episodes("./results/reflexion_n4/")
refl_n5 = load_episodes("./results/reflexion_n5/")

print(f"Loaded: n3={len(refl_n3)}, n4={len(refl_n4)}, n5={len(refl_n5)}")

# Reflexion: N=1,2 from n5; N=3 from n3; N=4 from n4; N=5 from n5
refl_sources = {1: refl_n5, 2: refl_n5, 3: refl_n3, 4: refl_n4, 5: refl_n5}

results = {"reflexion": {}}
for task in TASKS:
    results["reflexion"][task] = {}
    for k in range(1, 6):
        src = refl_sources[k]
        d, n = compute_death_rate(src, task, k)
        results["reflexion"][task][str(k)] = {"deaths": d, "total": n, "rate": round(d/n*100, 1)}

# Overall
for k in range(1, 6):
    src = refl_sources[k]
    total_deaths = 0
    total_eps = 0
    for task in TASKS:
        d, n = compute_death_rate(src, task, k)
        total_deaths += d
        total_eps += n
    results["reflexion"]["_overall"] = results["reflexion"].get("_overall", {})
    results["reflexion"]["_overall"][str(k)] = {"deaths": total_deaths, "total": total_eps, "rate": round(total_deaths/total_eps*100, 1)}

print(json.dumps(results, indent=2))
