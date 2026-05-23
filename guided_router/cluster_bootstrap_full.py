"""Full cluster-level statistical analysis for DAF (K=11 task-type clusters).

Four methods:
1. Wild cluster bootstrap (Webb 6-point weights, Webb 2023)
2. Exact permutation test (sign-flip on cluster means)
3. Cluster-robust t-test (one-sample t on K cluster DAFs)
4. Cohen's d (effect size)

Data source: baseline_245eps/episode_details.jsonl (245 episodes, 11 tasks)
"""
import json
import numpy as np
from collections import defaultdict
from itertools import product
from scipy import stats

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


def compute_cluster_dafs(task_groups, task_types):
    """Compute per-cluster (per-task-type) DAF values."""
    cluster_dafs = []
    for t in task_types:
        daf = compute_daf(task_groups[t])
        cluster_dafs.append(daf)
    return np.array(cluster_dafs)


def wild_cluster_bootstrap(task_groups, task_types, daf_point, rng, B=10000):
    """Wild cluster bootstrap with Webb 6-point weights (Webb 2023).
    
    H0: DAF = 0. Test statistic: cluster-level DAF mean.
    Under H0, impose null by centering cluster DAFs at 0, then
    multiply each cluster's centered DAF by a Webb 6-point weight.
    """
    cluster_dafs = compute_cluster_dafs(task_groups, task_types)
    K = len(task_types)
    
    # Webb 6-point distribution (Webb 2023, "Reworking wild bootstrap-based inference")
    webb_weights = np.array([-np.sqrt(3/2), -np.sqrt(2/2), -np.sqrt(1/2),
                              np.sqrt(1/2),  np.sqrt(2/2),  np.sqrt(3/2)])
    
    # Observed test statistic
    t_obs = np.mean(cluster_dafs) / (np.std(cluster_dafs, ddof=1) / np.sqrt(K))
    
    # Center under H0
    centered = cluster_dafs - np.mean(cluster_dafs)
    
    boot_t_stats = []
    for _ in range(B):
        w = rng.choice(webb_weights, size=K)
        perturbed = centered * w
        boot_mean = np.mean(perturbed)
        boot_se = np.std(perturbed, ddof=1) / np.sqrt(K)
        if boot_se > 0:
            boot_t_stats.append(boot_mean / boot_se)
    
    boot_t_stats = np.array(boot_t_stats)
    # Two-sided p-value
    p_value = float(np.mean(np.abs(boot_t_stats) >= np.abs(t_obs)))
    
    # Bootstrap CI (percentile method on non-centered bootstrap)
    boot_dafs = []
    for _ in range(B):
        selected = rng.choice(len(task_types), size=K, replace=True)
        boot_eps = []
        for idx in selected:
            boot_eps.extend(task_groups[task_types[idx]])
        daf_val = compute_daf(boot_eps)
        if not np.isnan(daf_val):
            boot_dafs.append(daf_val)
    
    boot_dafs = np.array(boot_dafs)
    ci_lower = float(np.percentile(boot_dafs, 2.5))
    ci_upper = float(np.percentile(boot_dafs, 97.5))
    
    return {
        "t_statistic": float(t_obs),
        "p_value": p_value,
        "ci_95_low": ci_lower,
        "ci_95_high": ci_upper,
        "boot_mean": float(np.mean(boot_dafs)),
        "boot_std": float(np.std(boot_dafs)),
        "n_clusters": K,
        "n_boot": B,
        "n_valid_boots": len(boot_dafs),
        "method": "Webb 6-point weights (Webb 2023)",
    }


def exact_permutation_test(task_groups, task_types):
    """Exact permutation test via sign-flipping on K cluster DAFs.
    
    H0: DAF = 0. Under H0, each cluster DAF is equally likely positive or negative.
    With K=11, there are 2^11 = 2048 possible sign assignments (exhaustive).
    """
    cluster_dafs = compute_cluster_dafs(task_groups, task_types)
    K = len(task_types)
    observed_mean = np.mean(cluster_dafs)
    
    # Enumerate all 2^K sign patterns
    n_perms = 2 ** K
    count_ge = 0
    all_means = []
    
    for i in range(n_perms):
        signs = np.array([1 if (i >> bit) & 1 else -1 for bit in range(K)])
        perm_mean = np.mean(cluster_dafs * signs)
        all_means.append(perm_mean)
        if abs(perm_mean) >= abs(observed_mean):
            count_ge += 1
    
    p_value = count_ge / n_perms
    
    return {
        "observed_mean_daf": float(observed_mean),
        "p_value": float(p_value),
        "n_permutations": n_perms,
        "n_extreme": count_ge,
        "method": "Exact sign-flip permutation (2^K exhaustive)",
    }


def cluster_robust_ttest(task_groups, task_types):
    """One-sample t-test on K cluster-level DAF values.
    
    H0: mean(cluster DAF) = 0.
    Uses K-1 degrees of freedom (conservative with small K).
    """
    cluster_dafs = compute_cluster_dafs(task_groups, task_types)
    K = len(task_types)
    
    t_stat, p_value = stats.ttest_1samp(cluster_dafs, 0.0)
    
    mean_daf = np.mean(cluster_dafs)
    se = np.std(cluster_dafs, ddof=1) / np.sqrt(K)
    df = K - 1
    t_crit = stats.t.ppf(0.975, df)
    ci_lower = mean_daf - t_crit * se
    ci_upper = mean_daf + t_crit * se
    
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "df": df,
        "mean_cluster_daf": float(mean_daf),
        "se": float(se),
        "ci_95_low": float(ci_lower),
        "ci_95_high": float(ci_upper),
        "n_clusters": K,
        "method": "One-sample t-test on cluster DAFs (df=K-1)",
    }


def cohens_d(task_groups, task_types):
    """Cohen's d effect size for cluster-level DAFs.
    
    d = mean / sd, one-sample version (testing against 0).
    Interpretation: 0.2=small, 0.5=medium, 0.8=large.
    """
    cluster_dafs = compute_cluster_dafs(task_groups, task_types)
    K = len(task_types)
    
    mean_daf = np.mean(cluster_dafs)
    sd = np.std(cluster_dafs, ddof=1)
    d = mean_daf / sd if sd > 0 else float('inf')
    
    # CI for Cohen's d using non-central t approximation
    se_d = np.sqrt((1/K) + (d**2 / (2*K)))
    t_crit = stats.t.ppf(0.975, K-1)
    ci_lower = d - t_crit * se_d
    ci_upper = d + t_crit * se_d
    
    if abs(d) >= 0.8:
        magnitude = "large"
    elif abs(d) >= 0.5:
        magnitude = "medium"
    elif abs(d) >= 0.2:
        magnitude = "small"
    else:
        magnitude = "negligible"
    
    return {
        "d": float(d),
        "magnitude": magnitude,
        "mean_cluster_daf": float(mean_daf),
        "sd_cluster_daf": float(sd),
        "ci_95_low": float(ci_lower),
        "ci_95_high": float(ci_upper),
        "n_clusters": K,
        "per_cluster_dafs": {t: float(compute_daf(task_groups[t])) for t in task_types},
    }


def main():
    rng = np.random.RandomState(SEED)
    episodes = load_episodes()
    print(f"Loaded {len(episodes)} episodes")

    task_groups = defaultdict(list)
    for e in episodes:
        task_groups[e["task_type"]].append(e)
    task_types = sorted(task_groups.keys())
    K = len(task_types)
    print(f"Task types ({K}): {task_types}")
    for t in task_types:
        print(f"  {t}: {len(task_groups[t])} episodes")

    daf_point = compute_daf(episodes)
    print(f"\nPoint estimate (all-episodes DAF): {daf_point:.4f}")

    # --- Method 1: Wild Cluster Bootstrap ---
    print("\n" + "="*60)
    print("=== 1. Wild Cluster Bootstrap (Webb 2023) ===")
    wcb = wild_cluster_bootstrap(task_groups, task_types, daf_point, rng, B)
    print(f"  t-statistic: {wcb['t_statistic']:.4f}")
    print(f"  p-value: {wcb['p_value']:.4f}")
    print(f"  95% CI: [{wcb['ci_95_low']:.4f}, {wcb['ci_95_high']:.4f}]")
    print(f"  Boot mean: {wcb['boot_mean']:.4f} (std: {wcb['boot_std']:.4f})")

    # --- Method 2: Exact Permutation Test ---
    print("\n" + "="*60)
    print("=== 2. Exact Permutation Test ===")
    perm = exact_permutation_test(task_groups, task_types)
    print(f"  Observed mean DAF: {perm['observed_mean_daf']:.4f}")
    print(f"  p-value: {perm['p_value']:.4f}")
    print(f"  Permutations: {perm['n_permutations']} (exhaustive)")
    print(f"  Extreme count: {perm['n_extreme']}")

    # --- Method 3: Cluster-Robust T-Test ---
    print("\n" + "="*60)
    print("=== 3. Cluster-Robust T-Test ===")
    ttest = cluster_robust_ttest(task_groups, task_types)
    print(f"  t-statistic: {ttest['t_statistic']:.4f}")
    print(f"  p-value: {ttest['p_value']:.6f}")
    print(f"  df: {ttest['df']}")
    print(f"  95% CI: [{ttest['ci_95_low']:.4f}, {ttest['ci_95_high']:.4f}]")

    # --- Method 4: Cohen's d ---
    print("\n" + "="*60)
    print("=== 4. Cohen's d ===")
    cd = cohens_d(task_groups, task_types)
    print(f"  d: {cd['d']:.4f} ({cd['magnitude']})")
    print(f"  95% CI: [{cd['ci_95_low']:.4f}, {cd['ci_95_high']:.4f}]")
    print(f"  Per-cluster DAFs:")
    for t, d_val in cd['per_cluster_dafs'].items():
        print(f"    {t}: {d_val:.4f}")

    # --- Save ---
    results = {
        "daf_point": float(daf_point),
        "n_episodes": len(episodes),
        "n_clusters": K,
        "task_types": task_types,
        "episodes_per_task": {t: len(task_groups[t]) for t in task_types},
        "wild_cluster_bootstrap": wcb,
        "exact_permutation_test": perm,
        "cluster_robust_ttest": ttest,
        "cohens_d": cd,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
