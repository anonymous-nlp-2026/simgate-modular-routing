"""Unified episode-level bootstrap for ALL models.
Uses non-clustered episode-level bootstrap (B=10,000, seed=42) consistently.
Loads data from the same sources as the paper's point estimates.
"""
import json, glob, os
import numpy as np
from collections import defaultdict

DEATH_THRESHOLD = -100
B = 10000
SEED = 42

CANONICAL_TASKS = [
    "chemistry-mix", "find-animal", "find-non-living-thing", "freeze",
    "grow-fruit", "identify-life-stages-1", "identify-life-stages-2",
    "inclined-plane-determine-angle", "lifespan-longest-lived",
    "lifespan-longest-lived-then-shortest-lived",
    "measure-melting-point-unknown-substance",
]

BASE = "."

def load_qwen_llama(subdir):
    """Load from episodes.jsonl (same as compute_cross_model_cis.py)."""
    episodes = []
    root = os.path.join(BASE, subdir)
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

def load_gemma():
    """Load from episode_summary.jsonl (same as analyze_gemma_scienceworld.py)."""
    episodes = []
    root = os.path.join(BASE, "results/gemma_scienceworld_counterfactual")
    for task in CANONICAL_TASKS:
        summary_path = os.path.join(root, task, "episode_summary.jsonl")
        if not os.path.exists(summary_path):
            print(f"  WARNING: missing {summary_path}")
            continue
        seen_ids = set()
        with open(summary_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ep = json.loads(line)
                ep_id = ep.get("episode", len(seen_ids))
                if ep_id in seen_ids:
                    continue
                seen_ids.add(ep_id)
                episodes.append({
                    "task": task,
                    "int_score": ep.get("internal_final", ep.get("internal_score", 0.0)),
                    "det_score": ep.get("det_final", ep.get("deterministic_score", 0.0)),
                })
    return episodes

def episode_bootstrap(episodes, n_boot=B, seed=SEED):
    """Non-clustered episode-level bootstrap."""
    rng = np.random.RandomState(seed)
    n = len(episodes)
    int_scores = np.array([e["int_score"] for e in episodes])
    det_scores = np.array([e["det_score"] for e in episodes])
    
    # Point estimates
    int_dead = (int_scores <= DEATH_THRESHOLD).astype(float)
    det_dead = (det_scores <= DEATH_THRESHOLD).astype(float)
    int_dr = int_dead.mean()
    det_dr = det_dead.mean()
    delta_dr = int_dr - det_dr
    
    both_alive = (~(int_scores <= DEATH_THRESHOLD)) & (~(det_scores <= DEATH_THRESHOLD))
    delta_full = det_scores.mean() - int_scores.mean()
    if both_alive.sum() > 0 and delta_full != 0:
        delta_nd = det_scores[both_alive].mean() - int_scores[both_alive].mean()
        daf = 1.0 - delta_nd / delta_full
    else:
        delta_nd = 0.0
        daf = float("nan")
    
    # Bootstrap
    idx = rng.randint(0, n, size=(n_boot, n))
    b_int = int_scores[idx]
    b_det = det_scores[idx]
    
    b_int_dead = (b_int <= DEATH_THRESHOLD).astype(float)
    b_det_dead = (b_det <= DEATH_THRESHOLD).astype(float)
    b_int_dr = b_int_dead.mean(axis=1)
    b_det_dr = b_det_dead.mean(axis=1)
    b_delta_dr = b_int_dr - b_det_dr
    
    b_delta_full = b_det.mean(axis=1) - b_int.mean(axis=1)
    b_both_alive = (~(b_int <= DEATH_THRESHOLD)) & (~(b_det <= DEATH_THRESHOLD))
    
    b_daf = np.full(n_boot, np.nan)
    for i in range(n_boot):
        mask = b_both_alive[i]
        if mask.sum() > 0 and b_delta_full[i] != 0:
            d_nd = b_det[i][mask].mean() - b_int[i][mask].mean()
            b_daf[i] = 1.0 - d_nd / b_delta_full[i]
    
    valid_daf = b_daf[~np.isnan(b_daf)]
    
    def ci(arr):
        return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]
    
    return {
        "n": n,
        "n_tasks": len(set(e["task"] for e in episodes)),
        "int_dr": {"point": float(int_dr), "ci": ci(b_int_dr)},
        "det_dr": {"point": float(det_dr), "ci": ci(b_det_dr)},
        "delta_dr": {"point": float(delta_dr), "ci": ci(b_delta_dr)},
        "daf": {"point": float(daf), "ci": ci(valid_daf) if len(valid_daf) > 0 else [float("nan"), float("nan")], "n_valid": int(len(valid_daf))},
    }

# Load data
print("Loading data...")
models = {
    "qwen": {"loader": lambda: load_qwen_llama("results/plan_c"), "display": "Qwen2.5-7B-Instruct"},
    "llama": {"loader": lambda: load_qwen_llama("results/llama_cross_model_full"), "display": "Llama-3.1-8B-Instruct"},
    "gemma": {"loader": load_gemma, "display": "Gemma-2-9B-it"},
}

results = {}
for model, cfg in models.items():
    all_eps = cfg["loader"]()
    eps_10 = [e for e in all_eps if e["task"] != "find-non-living-thing"]
    
    tasks_found = sorted(set(e["task"] for e in all_eps))
    print(f"\n{cfg['display']}: {len(all_eps)} episodes, {len(tasks_found)} tasks")
    print(f"  Tasks: {tasks_found}")
    
    res_11 = episode_bootstrap(all_eps)
    res_10 = episode_bootstrap(eps_10)
    
    results[model] = {"display": cfg["display"], "11_tasks": res_11, "10_tasks": res_10}

# Print formatted output
print("\n" + "="*80)
print("=== EPISODE-LEVEL BOOTSTRAP (B=10,000, seed=42) ===")
print("="*80)

def fmt_pct(point, ci):
    return f"{point*100:.1f}% [{ci[0]*100:.1f}, {ci[1]*100:.1f}]"

def fmt_pp(point, ci):
    sign = "+" if point >= 0 else ""
    return f"{sign}{point*100:.1f}pp [{ci[0]*100:.1f}, {ci[1]*100:.1f}]"

for label, key in [("11 TASKS (full)", "11_tasks"), ("10 TASKS (excl. find-non-living-thing)", "10_tasks")]:
    print(f"\n--- {label} ---")
    print(f"{'Model':<25} {'N':>4} {'Int DR [CI]':<24} {'Det DR [CI]':<24} {'ΔDR [CI]':<24} {'DAF [CI]':<28}")
    print("-" * 130)
    for model in ["qwen", "llama", "gemma"]:
        r = results[model][key]
        name = results[model]["display"]
        idr = r["int_dr"]
        ddr = r["det_dr"]
        delta = r["delta_dr"]
        daf = r["daf"]
        print(f"{name:<25} {r['n']:>4} {fmt_pct(idr['point'], idr['ci']):<24} {fmt_pct(ddr['point'], ddr['ci']):<24} {fmt_pp(delta['point'], delta['ci']):<24} {fmt_pct(daf['point'], daf['ci']):<28}")

# Comparison with paper values
print("\n" + "="*80)
print("=== COMPARISON: Paper values vs New episode-level bootstrap ===")
print("="*80)
paper_11 = {
    "qwen": [81.3, 99.1],
    "llama": [80.2, 95.1],
    "gemma": [68.8, 100.0],
}
paper_10 = {
    "qwen": [100.0, 100.0],
    "llama": [91.3, 100.0],
    "gemma": [99.7, 100.0],
}

print(f"\n{'Model':<25} {'Paper 11-task CI':<20} {'New 11-task CI':<20} {'Method change?'}")
print("-" * 90)
for model in ["qwen", "llama", "gemma"]:
    old = paper_11[model]
    new = results[model]["11_tasks"]["daf"]["ci"]
    new_str = f"[{new[0]*100:.1f}, {new[1]*100:.1f}]"
    old_str = f"[{old[0]:.1f}, {old[1]:.1f}]"
    changed = "SAME" if abs(old[0] - new[0]*100) < 0.2 and abs(old[1] - new[1]*100) < 0.2 else "CHANGED"
    print(f"{results[model]['display']:<25} {old_str:<20} {new_str:<20} {changed}")

print(f"\n{'Model':<25} {'Paper 10-task CI':<20} {'New 10-task CI':<20} {'Method change?'}")
print("-" * 90)
for model in ["qwen", "llama", "gemma"]:
    old = paper_10[model]
    new = results[model]["10_tasks"]["daf"]["ci"]
    new_str = f"[{new[0]*100:.1f}, {new[1]*100:.1f}]"
    old_str = f"[{old[0]:.1f}, {old[1]:.1f}]"
    changed = "SAME" if abs(old[0] - new[0]*100) < 0.2 and abs(old[1] - new[1]*100) < 0.2 else "CHANGED"
    print(f"{results[model]['display']:<25} {old_str:<20} {new_str:<20} {changed}")

# Save results
out_path = os.path.join(BASE, "results/unified_episode_bootstrap.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nJSON saved to {out_path}")
