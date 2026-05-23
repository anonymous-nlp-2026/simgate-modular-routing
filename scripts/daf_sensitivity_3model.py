"""Compute DAF sensitivity: 11 tasks vs 10 tasks (excluding find-non-living-thing).
Uses cluster bootstrap (resample task types) for all three models.
Filters to canonical 11 tasks only."""

import glob, json, os
import numpy as np

DEATH_THRESHOLD = -100
EXCLUDE_TASK = "find-non-living-thing"

CANONICAL_TASKS = [
    "chemistry-mix", "find-animal", "find-non-living-thing", "freeze",
    "grow-fruit", "identify-life-stages-1", "identify-life-stages-2",
    "inclined-plane-determine-angle", "lifespan-longest-lived",
    "lifespan-longest-lived-then-shortest-lived",
    "measure-melting-point-unknown-substance",
]

MODEL_CONFIGS = {
    "qwen": {"dir": "results/plan_c", "display": "Qwen2.5-7B-Instruct"},
    "llama": {"dir": "results/llama_cross_model_full", "display": "Llama-3.1-8B-Instruct"},
    "gemma": {"dir": "results/gemma_scienceworld_counterfactual", "display": "Gemma-2-9B-it"},
}

def load_episodes(base, subdir):
    episodes = []
    root = os.path.join(base, subdir)
    for f in sorted(glob.glob(os.path.join(root, "*/episodes.jsonl"))):
        task = os.path.basename(os.path.dirname(f))
        if task not in CANONICAL_TASKS:
            continue
        for line in open(f):
            ep = json.loads(line)
            if "always_internal" not in ep or "always_deterministic" not in ep:
                continue
            episodes.append({
                "task": task,
                "int_score": ep["always_internal"]["final_score"],
                "det_score": ep["always_deterministic"]["final_score"],
            })
    return episodes

def compute_daf(episodes):
    int_scores = np.array([e["int_score"] for e in episodes])
    det_scores = np.array([e["det_score"] for e in episodes])
    both_alive = (~(int_scores <= DEATH_THRESHOLD)) & (~(det_scores <= DEATH_THRESHOLD))
    delta_full = det_scores.mean() - int_scores.mean()
    if both_alive.sum() > 0 and delta_full != 0:
        delta_nd = det_scores[both_alive].mean() - int_scores[both_alive].mean()
        return 1.0 - delta_nd / delta_full
    return float("nan")

def cluster_bootstrap_daf(episodes, B=10000, seed=42):
    rng = np.random.RandomState(seed)
    by_task = {}
    for e in episodes:
        by_task.setdefault(e["task"], []).append(e)
    tasks = sorted(by_task.keys())
    K = len(tasks)
    
    point = compute_daf(episodes)
    
    boot_dafs = []
    for _ in range(B):
        sampled_tasks = rng.choice(tasks, size=K, replace=True)
        boot_eps = []
        for t in sampled_tasks:
            boot_eps.extend(by_task[t])
        d = compute_daf(boot_eps)
        if not np.isnan(d):
            boot_dafs.append(d)
    
    boot_dafs = np.array(boot_dafs)
    ci_lo = float(np.percentile(boot_dafs, 2.5))
    ci_hi = float(np.percentile(boot_dafs, 97.5))
    return {
        "point": float(point),
        "ci": [ci_lo, ci_hi],
        "K": K,
        "n_valid": len(boot_dafs),
        "n_episodes": len(episodes),
    }

def episode_bootstrap_daf(episodes, B=10000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(episodes)
    int_scores = np.array([e["int_score"] for e in episodes])
    det_scores = np.array([e["det_score"] for e in episodes])
    
    both_alive = (~(int_scores <= DEATH_THRESHOLD)) & (~(det_scores <= DEATH_THRESHOLD))
    delta_full = det_scores.mean() - int_scores.mean()
    if both_alive.sum() > 0 and delta_full != 0:
        delta_nd = det_scores[both_alive].mean() - int_scores[both_alive].mean()
        point = 1.0 - delta_nd / delta_full
    else:
        point = float("nan")
    
    idx = rng.randint(0, n, size=(B, n))
    b_int = int_scores[idx]
    b_det = det_scores[idx]
    b_both_alive = (~(b_int <= DEATH_THRESHOLD)) & (~(b_det <= DEATH_THRESHOLD))
    b_delta_full = b_det.mean(axis=1) - b_int.mean(axis=1)
    
    b_daf = np.full(B, np.nan)
    for i in range(B):
        mask = b_both_alive[i]
        if mask.sum() > 0 and b_delta_full[i] != 0:
            d_nd = b_det[i][mask].mean() - b_int[i][mask].mean()
            b_daf[i] = 1.0 - d_nd / b_delta_full[i]
    valid = b_daf[~np.isnan(b_daf)]
    return {
        "point": float(point),
        "ci": [float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5))],
        "n_valid": len(valid),
        "n_episodes": n,
    }

base = "."
results = {}

for model, cfg in MODEL_CONFIGS.items():
    all_eps = load_episodes(base, cfg["dir"])
    eps_10 = [e for e in all_eps if e["task"] != EXCLUDE_TASK]
    
    tasks_found = sorted(set(e["task"] for e in all_eps))
    print(f"\n{cfg['display']}: {len(all_eps)} episodes, {len(tasks_found)} tasks")
    
    # Cluster bootstrap
    cb_11 = cluster_bootstrap_daf(all_eps)
    cb_10 = cluster_bootstrap_daf(eps_10)
    
    # Episode bootstrap (for comparison with existing paper numbers)
    eb_11 = episode_bootstrap_daf(all_eps)
    eb_10 = episode_bootstrap_daf(eps_10)
    
    results[model] = {
        "display": cfg["display"],
        "cluster_11": cb_11,
        "cluster_10": cb_10,
        "episode_11": eb_11,
        "episode_10": eb_10,
    }
    
    print(f"  Cluster bootstrap:")
    print(f"    11 tasks (K={cb_11['K']}): DAF={cb_11['point']*100:.1f}% [{cb_11['ci'][0]*100:.1f}, {cb_11['ci'][1]*100:.1f}]")
    print(f"    10 tasks (K={cb_10['K']}): DAF={cb_10['point']*100:.1f}% [{cb_10['ci'][0]*100:.1f}, {cb_10['ci'][1]*100:.1f}]")
    print(f"  Episode bootstrap:")
    print(f"    11 tasks (N={eb_11['n_episodes']}): DAF={eb_11['point']*100:.1f}% [{eb_11['ci'][0]*100:.1f}, {eb_11['ci'][1]*100:.1f}]")
    print(f"    10 tasks (N={eb_10['n_episodes']}): DAF={eb_10['point']*100:.1f}% [{eb_10['ci'][0]*100:.1f}, {eb_10['ci'][1]*100:.1f}]")

out_path = os.path.join(base, "results", "daf_sensitivity_3model.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_path}")
