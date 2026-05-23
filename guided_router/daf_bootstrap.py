"""DAF Bootstrap CI: 10000 resamples over plan-012 full episodes + per-task decomposition."""
import json, os
import numpy as np
from collections import defaultdict

DATA_PATH = "./results/baseline_245eps/episode_details.jsonl"
OUT_PATH = "./guided_router/daf_bootstrap_results.json"
SEED = 42
N_BOOT_GLOBAL = 10000
N_BOOT_TASK = 1000
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
                "episode_idx": d["episode_idx"],
                "score_det": d["det_final"],
                "score_int": d["internal_final"],
                "oracle_label": d["oracle_label"],
            })
    return episodes


def compute_daf(eps):
    det_pref = [e for e in eps if e["oracle_label"] == "det-preferred"]
    if not det_pref:
        return None, {}

    delta_full = np.mean([e["score_det"] - e["score_int"] for e in det_pref])

    non_death = [e for e in det_pref if e["score_int"] != DEATH_SCORE]
    if not non_death:
        delta_nd = 0.0
    else:
        delta_nd = np.mean([e["score_det"] - e["score_int"] for e in non_death])

    if delta_full == 0:
        return None, {}

    daf = 1.0 - delta_nd / delta_full
    n_death = sum(1 for e in eps if e["score_int"] == DEATH_SCORE)
    n_death_det_pref = sum(1 for e in det_pref if e["score_int"] == DEATH_SCORE)
    return daf, {
        "n_det_pref": len(det_pref),
        "n_non_death_det_pref": len(non_death),
        "n_death": n_death,
        "n_death_det_pref": n_death_det_pref,
        "delta_full": float(delta_full),
        "delta_nd": float(delta_nd),
    }


def bootstrap_daf(eps, n_boot, rng):
    n = len(eps)
    samples = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot = [eps[i] for i in idx]
        daf, _ = compute_daf(boot)
        if daf is not None:
            samples.append(daf)
    return np.array(samples)


def main():
    episodes = load_episodes()
    print(f"Loaded {len(episodes)} episodes")
    tasks = sorted(set(e["task_type"] for e in episodes))
    print(f"Tasks ({len(tasks)}): {tasks}")

    rng = np.random.RandomState(SEED)
    results = {"global": {}, "per_task": {}}

    # Global DAF
    daf_point, stats = compute_daf(episodes)
    print(f"\nGlobal DAF point estimate: {daf_point:.4f}")
    print(f"  N episodes: {len(episodes)}")
    print(f"  N det-preferred: {stats['n_det_pref']}")
    print(f"  N death (all): {stats['n_death']}")
    print(f"  N death (det-pref): {stats['n_death_det_pref']}")
    print(f"  N non-death det-pref: {stats['n_non_death_det_pref']}")
    print(f"  Delta_full: {stats['delta_full']:.4f}")
    print(f"  Delta_nd: {stats['delta_nd']:.4f}")

    boot_samples = bootstrap_daf(episodes, N_BOOT_GLOBAL, rng)
    ci_low = float(np.percentile(boot_samples, 2.5))
    ci_high = float(np.percentile(boot_samples, 97.5))
    boot_mean = float(np.mean(boot_samples))
    boot_std = float(np.std(boot_samples))

    results["global"] = {
        "daf_point": float(daf_point),
        "daf_boot_mean": boot_mean,
        "daf_boot_std": boot_std,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "n_boot": N_BOOT_GLOBAL,
        "n_episodes": len(episodes),
        "n_valid_boots": len(boot_samples),
        **stats,
    }
    print(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"  Boot mean: {boot_mean:.4f}, std: {boot_std:.4f}")
    print(f"  Valid bootstrap samples: {len(boot_samples)}/{N_BOOT_GLOBAL}")

    # Per-task DAF
    task_eps = defaultdict(list)
    for e in episodes:
        task_eps[e["task_type"]].append(e)

    print(f"\n{'='*60}")
    print(f"Per-task DAF (bootstrap {N_BOOT_TASK} resamples)")
    print(f"{'='*60}")
    for task in tasks:
        t_eps = task_eps[task]
        daf_t, stats_t = compute_daf(t_eps)
        if daf_t is None:
            results["per_task"][task] = {
                "daf_point": None, "status": "N/A - no det-preferred episodes",
                "n_episodes": len(t_eps),
            }
            print(f"  {task}: N/A (no det-preferred), n={len(t_eps)}")
            continue

        boot_t = bootstrap_daf(t_eps, N_BOOT_TASK, rng)
        if len(boot_t) < 100:
            results["per_task"][task] = {
                "daf_point": float(daf_t), "status": "insufficient bootstrap samples",
                "n_episodes": len(t_eps), "n_valid_boots": len(boot_t), **stats_t,
            }
            print(f"  {task}: DAF={daf_t:.4f}, insufficient boots ({len(boot_t)}/{N_BOOT_TASK}), n={len(t_eps)}")
            continue

        ci_l = float(np.percentile(boot_t, 2.5))
        ci_h = float(np.percentile(boot_t, 97.5))
        results["per_task"][task] = {
            "daf_point": float(daf_t),
            "ci_95_low": ci_l,
            "ci_95_high": ci_h,
            "n_boot": N_BOOT_TASK,
            "n_episodes": len(t_eps),
            "n_valid_boots": len(boot_t),
            **stats_t,
        }
        print(f"  {task}: DAF={daf_t:.4f}, CI=[{ci_l:.4f}, {ci_h:.4f}], n={len(t_eps)}, death={stats_t['n_death']}")

        # Incremental save
        with open(OUT_PATH, "w") as f:
            json.dump(results, f, indent=2)

    # Final save
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
