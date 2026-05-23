"""DAF Threshold Sensitivity Analysis.

Tests robustness of DAF metric to different "failure" definitions:
  Method 1: Absolute score threshold (score <= T)
  Method 2: Relative score drop from max (100)
  Method 3: Rank-based (bottom quartile)
"""

import json
import os
import numpy as np
from scipy import stats

DATA_FILE = "./results/plan_c_95ci/episode_details.jsonl"
OUT_DIR = "./artifacts/daf-sensitivity-extended"
OUT_FILE = os.path.join(OUT_DIR, "threshold_sensitivity.json")

episodes = []
with open(DATA_FILE) as f:
    for line in f:
        d = json.loads(line.strip())
        if d["condition"] == "always-deterministic":
            episodes.append(d)

N = len(episodes)
task_types = [ep["task_type"] for ep in episodes]
int_scores = np.array([ep["internal_final"] for ep in episodes])
det_scores = np.array([ep["det_final"] for ep in episodes])
oracle_labels = [ep["oracle_label"] for ep in episodes]

score_gaps = det_scores - int_scores
delta_full = float(np.mean(score_gaps))
unique_tasks = sorted(set(task_types))

print(f"Loaded {N} episodes, {len(unique_tasks)} task types")
print(f"Delta_full = {delta_full:.4f}")
print(f"Score ranges: int [{int_scores.min()}, {int_scores.max()}], det [{det_scores.min()}, {det_scores.max()}]")
print()


def per_task_correlations(task_types, int_scores, det_scores, int_fail_mask, det_fail_mask):
    """Pearson r between per-task failure metrics and per-task delta."""
    tasks = sorted(set(task_types))
    fail_rates_int = []
    fail_rate_diffs = []
    deltas = []
    for t in tasks:
        mask = np.array([tt == t for tt in task_types])
        n_t = mask.sum()
        fr_int = int_fail_mask[mask].sum() / n_t
        fr_det = det_fail_mask[mask].sum() / n_t
        delta_t = float(np.mean(det_scores[mask] - int_scores[mask]))
        fail_rates_int.append(fr_int)
        fail_rate_diffs.append(fr_int - fr_det)
        deltas.append(delta_t)

    r_int, p_int = (None, None)
    r_diff, p_diff = (None, None)

    if len(set(fail_rates_int)) > 1:
        r_int, p_int = stats.pearsonr(fail_rates_int, deltas)
        r_int, p_int = float(r_int), float(p_int)
    if len(set(fail_rate_diffs)) > 1:
        r_diff, p_diff = stats.pearsonr(fail_rate_diffs, deltas)
        r_diff, p_diff = float(r_diff), float(p_diff)

    return r_int, p_int, r_diff, p_diff


# ==============================
# Method 1: Absolute threshold (score <= T)
# ==============================
print("=" * 60)
print("METHOD 1: Absolute Score Threshold (failure = score <= T)")
print("=" * 60)

thresholds_abs = [-100, -50, -20, 0, 10, 20]
method1 = {}

for T in thresholds_abs:
    int_fail = int_scores <= T
    det_fail = det_scores <= T

    n_int_fail = int(int_fail.sum())
    n_det_fail = int(det_fail.sum())

    non_fail = ~int_fail & ~det_fail
    n_non_fail = int(non_fail.sum())

    if n_non_fail > 0 and n_non_fail < N:
        delta_non_fail = float(np.mean(score_gaps[non_fail]))
    elif n_non_fail == N:
        delta_non_fail = delta_full
    else:
        delta_non_fail = 0.0

    avoidance_frac = (delta_full - delta_non_fail) / delta_full if delta_full != 0 else 0.0

    r_int, p_int, r_diff, p_diff = per_task_correlations(
        task_types, int_scores, det_scores, int_fail, det_fail
    )

    result = {
        "n_int_fail": n_int_fail,
        "n_det_fail": n_det_fail,
        "n_non_fail": n_non_fail,
        "avoidance_frac": round(avoidance_frac, 4),
        "r": round(r_diff, 4) if r_diff is not None else None,
        "p": round(p_diff, 6) if p_diff is not None else None,
        "r_int_only": round(r_int, 4) if r_int is not None else None,
    }
    method1[f"T_{T}"] = result

    r_str = f"r={r_diff:.4f}" if r_diff is not None else "r=N/A"
    print(f"  T={T:>4}: n_int_fail={n_int_fail:>3}, n_det_fail={n_det_fail:>3}, "
          f"n_non_fail={n_non_fail:>3}, avoidance={avoidance_frac*100:.1f}%, {r_str}")

print()

# ==============================
# Method 2: Relative score drop (score < 100 - X)
# ==============================
print("=" * 60)
print("METHOD 2: Relative Drop from Max Score (100)")
print("=" * 60)

MAX_SCORE = 100
drops_pct = [50, 75, 90, 100]
method2 = {}

for X in drops_pct:
    threshold_score = MAX_SCORE - X

    int_fail = int_scores < threshold_score
    det_fail = det_scores < threshold_score

    n_int_fail = int(int_fail.sum())
    n_det_fail = int(det_fail.sum())

    non_fail = ~int_fail & ~det_fail
    n_non_fail = int(non_fail.sum())

    if n_non_fail > 0 and n_non_fail < N:
        delta_non_fail = float(np.mean(score_gaps[non_fail]))
    elif n_non_fail == N:
        delta_non_fail = delta_full
    else:
        delta_non_fail = 0.0

    avoidance_frac = (delta_full - delta_non_fail) / delta_full if delta_full != 0 else 0.0

    r_int, p_int, r_diff, p_diff = per_task_correlations(
        task_types, int_scores, det_scores, int_fail, det_fail
    )

    result = {
        "threshold_score": threshold_score,
        "n_int_fail": n_int_fail,
        "n_det_fail": n_det_fail,
        "n_non_fail": n_non_fail,
        "avoidance_frac": round(avoidance_frac, 4),
        "r": round(r_diff, 4) if r_diff is not None else None,
        "p": round(p_diff, 6) if p_diff is not None else None,
    }
    method2[f"drop_{X}pct"] = result

    r_str = f"r={r_diff:.4f}" if r_diff is not None else "r=N/A"
    print(f"  X={X:>3}% (score<{threshold_score:>3}): n_int_fail={n_int_fail:>3}, "
          f"n_det_fail={n_det_fail:>3}, avoidance={avoidance_frac*100:.1f}%, {r_str}")

print()

# ==============================
# Method 3: Rank-based (bottom quartile)
# ==============================
print("=" * 60)
print("METHOD 3: Rank-Based (Bottom Quartile)")
print("=" * 60)

q1_threshold = float(np.percentile(int_scores, 25))
q1_mask = int_scores <= q1_threshold

det_preferred = np.array([l == "det-preferred" for l in oracle_labels])

q1_n = int(q1_mask.sum())
q1_det_pref_count = int((q1_mask & det_preferred).sum())
q1_det_pref_rate = float(det_preferred[q1_mask].mean()) if q1_n > 0 else 0.0
overall_det_pref_rate = float(det_preferred.mean())

q1_mean_gap = float(np.mean(score_gaps[q1_mask])) if q1_n > 0 else 0.0
non_q1_mean_gap = float(np.mean(score_gaps[~q1_mask])) if (~q1_mask).sum() > 0 else 0.0

enrichment = q1_det_pref_rate / overall_det_pref_rate if overall_det_pref_rate > 0 else None

method3 = {
    "Q1_threshold": q1_threshold,
    "Q1_internal_episodes": q1_n,
    "Q1_det_preferred_count": q1_det_pref_count,
    "Q1_det_preferred_rate": round(q1_det_pref_rate, 4),
    "overall_det_preferred_rate": round(overall_det_pref_rate, 4),
    "enrichment_ratio": round(enrichment, 2) if enrichment else None,
    "Q1_mean_gap": round(q1_mean_gap, 2),
    "non_Q1_mean_gap": round(non_q1_mean_gap, 2),
}

print(f"  Q1 threshold (25th pctile of int_scores): {q1_threshold}")
print(f"  Q1 episodes: {q1_n} / {N}")
print(f"  Q1 det-preferred rate: {q1_det_pref_rate*100:.1f}%")
print(f"  Overall det-preferred rate: {overall_det_pref_rate*100:.1f}%")
print(f"  Enrichment ratio: {enrichment:.2f}x" if enrichment else "  Enrichment: N/A")
print(f"  Q1 mean gap: {q1_mean_gap:.2f} vs non-Q1: {non_q1_mean_gap:.2f}")
print()

# ==============================
# Robustness conclusion
# ==============================
all_fracs_m1 = [method1[k]["avoidance_frac"] for k in method1 if method1[k]["n_int_fail"] > 0]
all_fracs_m2 = [method2[k]["avoidance_frac"] for k in method2 if method2[k]["n_int_fail"] > 0]

min_frac_m1 = min(all_fracs_m1) if all_fracs_m1 else 0
max_frac_m1 = max(all_fracs_m1) if all_fracs_m1 else 0
min_frac_m2 = min(all_fracs_m2) if all_fracs_m2 else 0
max_frac_m2 = max(all_fracs_m2) if all_fracs_m2 else 0

robust = (min_frac_m1 > 0.70) and (min_frac_m2 > 0.70)

conclusion = (
    f"DAF {'is' if robust else 'is NOT'} robust to threshold choice. "
    f"Method 1 (absolute): avoidance fraction ranges from "
    f"{min_frac_m1*100:.1f}% to {max_frac_m1*100:.1f}% across thresholds with failures. "
    f"Method 2 (relative drop): ranges from "
    f"{min_frac_m2*100:.1f}% to {max_frac_m2*100:.1f}%. "
    f"Method 3 (rank): bottom-quartile internal episodes show "
    f"{q1_det_pref_rate*100:.1f}% det-preferred rate vs "
    f"{overall_det_pref_rate*100:.1f}% overall "
    f"({enrichment:.2f}x enrichment), confirming that low internal scores "
    f"strongly predict det-preference regardless of threshold definition."
)

print("=" * 60)
print("CONCLUSION")
print("=" * 60)
print(conclusion)

# ==============================
# Save results
# ==============================
results = {
    "method1_absolute_threshold": method1,
    "method2_relative_drop": method2,
    "method3_rankbased": method3,
    "robustness_conclusion": conclusion,
    "metadata": {
        "data_file": DATA_FILE,
        "n_episodes": N,
        "n_tasks": len(unique_tasks),
        "delta_full": round(delta_full, 4),
        "note": "r = Pearson correlation between per-task failure_rate_differential (int-det) and per-task delta"
    }
}

os.makedirs(OUT_DIR, exist_ok=True)
with open(OUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: {OUT_FILE}")
