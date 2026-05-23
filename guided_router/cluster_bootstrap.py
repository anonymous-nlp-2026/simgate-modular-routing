"""Cluster bootstrap for DAF — resample at task-type level (11 clusters).

Standard bootstrap: resample 245 episodes independently -> CI [81.6%, 98.8%]
Cluster bootstrap: resample 11 task types (with replacement), take ALL episodes
from each selected type -> compute DAF -> repeat 10000 times -> CI

This accounts for within-task-type correlation (episodes from same task
are not independent). Expected: wider CI than episode-level bootstrap.

DAF = 1 - (delta_nd / delta_full)  [all-episodes definition]
- delta_full = mean(score_det - score_int) across ALL episodes
- delta_nd = mean(score_det - score_int) across non-death episodes
- non-death = both score_int != -100 AND score_det != -100
"""
import json
import numpy as np
from collections import defaultdict

DATA_PATH = "./results/baseline_245eps/episode_details.jsonl"
OUT_PATH = "./guided_router/cluster_bootstrap_results.json"
SEED = 42
B = 10000
DEATH_SCORE = -100.0


def load_episodes():
    episodes = []
    with open(DATA_PATH) as f:
        for line in f:
            d = json.loads(line)
            if d["condition"] != "always-deterministic":
                continue
            episodes.append({
                "task_type": d["task_type"],
                "score_det": d["det_final"],
                "score_int": d["internal_final"],
            })
    return episodes


def compute_daf(episodes):
    if not episodes:
        return float('nan')
    deltas = [e["score_det"] - e["score_int"] for e in episodes]
    delta_full = np.mean(deltas)

    non_death = [e for e in episodes
                 if e["score_int"] != DEATH_SCORE and e["score_det"] != DEATH_SCORE]
    if not non_death:
        delta_nd = 0.0
    else:
        delta_nd = np.mean([e["score_det"] - e["score_int"] for e in non_death])

    if delta_full == 0:
        return float('nan')

    return 1.0 - (delta_nd / delta_full)


def main():
    np.random.seed(SEED)
    episodes = load_episodes()
    print(f"Loaded {len(episodes)} episodes")

    task_groups = defaultdict(list)
    for e in episodes:
        task_groups[e["task_type"]].append(e)
    task_types = sorted(task_groups.keys())
    n_clusters = len(task_types)
    print(f"Task types ({n_clusters}): {task_types}")

    for t in task_types:
        print(f"  {t}: {len(task_groups[t])} episodes")

    # Point estimate
    daf_point = compute_daf(episodes)
    print(f"\nPoint estimate (all-episodes DAF): {daf_point:.4f}")

    # Cluster bootstrap
    daf_samples = []
    for _ in range(B):
        selected = np.random.choice(task_types, size=n_clusters, replace=True)
        boot_episodes = []
        for t in selected:
            boot_episodes.extend(task_groups[t])
        daf_val = compute_daf(boot_episodes)
        if not np.isnan(daf_val):
            daf_samples.append(daf_val)

    daf_samples = np.array(daf_samples)
    ci_lower = float(np.percentile(daf_samples, 2.5))
    ci_upper = float(np.percentile(daf_samples, 97.5))
    boot_mean = float(np.mean(daf_samples))
    boot_std = float(np.std(daf_samples))

    # Reference episode-level CI (from decomposition_bootstrap.json, percentage -> fraction)
    ep_ci_low = 0.81602
    ep_ci_high = 0.98761

    # CI widths
    ep_width = ep_ci_high - ep_ci_low
    cl_width = ci_upper - ci_lower
    lower_shift = ep_ci_low - ci_lower
    upper_shift = ci_upper - ep_ci_high

    print(f"\n{'='*55}")
    print(f"=== Cluster Bootstrap DAF CI ===")
    print(f"{'='*55}")
    print(f"\nPoint estimate (all-episodes DAF): {daf_point:.4f}")
    print(f"\nEpisode-level bootstrap (previous):")
    print(f"  95% CI: [{ep_ci_low:.4f}, {ep_ci_high:.4f}] (N=245 episodes, B=10000)")
    print(f"  Width: {ep_width:.4f}")
    print(f"\nCluster bootstrap (task-type level):")
    print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}] (K={n_clusters} clusters, B={B})")
    print(f"  Width: {cl_width:.4f}")
    print(f"\nCI widening: {lower_shift:+.4f} lower, {upper_shift:+.4f} upper")
    print(f"Width change: {cl_width - ep_width:+.4f} ({(cl_width/ep_width - 1)*100:+.1f}%)")
    print(f"Mean bootstrap DAF: {boot_mean:.4f} (std: {boot_std:.4f})")
    print(f"Valid bootstrap samples: {len(daf_samples)}/{B}")

    if ci_lower > 0.5:
        interpretation = "CI remains meaningful: lower bound well above 0.5, DAF robust under cluster resampling"
    elif ci_lower > 0:
        interpretation = "CI moderately wider but still positive: DAF signal persists under non-i.i.d. assumption"
    else:
        interpretation = "CI too wide to be useful: includes zero, DAF not robust to cluster resampling"

    print(f"\nInterpretation:")
    print(f"  {interpretation}")

    # Save results
    results = {
        "daf_point": float(daf_point),
        "cluster_bootstrap": {
            "ci_95_low": ci_lower,
            "ci_95_high": ci_upper,
            "boot_mean": boot_mean,
            "boot_std": boot_std,
            "n_clusters": n_clusters,
            "n_boot": B,
            "n_valid_boots": len(daf_samples),
            "seed": SEED,
        },
        "episode_bootstrap_reference": {
            "ci_95_low": ep_ci_low,
            "ci_95_high": ep_ci_high,
        },
        "ci_comparison": {
            "cluster_width": cl_width,
            "episode_width": ep_width,
            "width_change_abs": cl_width - ep_width,
            "width_change_pct": (cl_width / ep_width - 1) * 100,
            "lower_shift": lower_shift,
            "upper_shift": upper_shift,
        },
        "interpretation": interpretation,
        "task_types": task_types,
        "episodes_per_task": {t: len(task_groups[t]) for t in task_types},
    }

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
