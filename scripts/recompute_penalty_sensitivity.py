#!/usr/bin/env python3
"""Recompute Appendix K penalty sensitivity table from 245 episode-level data."""

import json
import numpy as np
from scipy import stats
from collections import defaultdict

DATA_FILE = "./results/baseline_245eps/episode_details.jsonl"
OUTPUT_FILE = "./results/penalty_sensitivity_recomputed.json"

# Load 245 unique episodes (filter one condition)
episodes = []
with open(DATA_FILE) as f:
    for line in f:
        d = json.loads(line)
        if d["condition"] == "always-deterministic":
            episodes.append(d)

assert len(episodes) == 245, f"Expected 245, got {len(episodes)}"

int_scores = np.array([e["internal_final"] for e in episodes])
det_scores = np.array([e["det_final"] for e in episodes])
task_types = np.array([e["task_type"] for e in episodes])

# Death = score of -100
death_int = (int_scores == -100).astype(int)
death_det = (det_scores == -100).astype(int)

# Undo P=100 penalty to get raw scores
raw_int = np.where(death_int, int_scores + 100, int_scores)
raw_det = np.where(death_det, det_scores + 100, det_scores)

assert np.all(raw_int >= 0), "Negative raw int scores"
assert np.all(raw_det >= 0), "Negative raw det scores"

# Both-alive mask and delta_nd (P-invariant)
both_alive = (death_int == 0) & (death_det == 0)
delta_nd = float((det_scores[both_alive] - int_scores[both_alive]).mean())

# Per-task grouping
unique_tasks = sorted(set(task_types))
task_idx = {t: np.where(task_types == t)[0] for t in unique_tasks}

# Per-task delta_DR (P-invariant)
task_delta_dr = np.array([
    death_int[task_idx[t]].mean() - death_det[task_idx[t]].mean()
    for t in unique_tasks
])

P_VALUES = [0, 1, 5, 10, 25, 50, 100, 200, 500]
B = 10000
rng = np.random.default_rng(42)
n = len(episodes)

print(f"Data: {n} episodes, {len(unique_tasks)} tasks")
print(f"Deaths: int={death_int.sum()}, det={death_det.sum()}, both_alive={both_alive.sum()}")
print(f"Delta_nd = {delta_nd:.4f} (P-invariant)\n")

results = []
for P in P_VALUES:
    adj_int = raw_int - P * death_int
    adj_det = raw_det - P * death_det

    mean_int_val = float(adj_int.mean())
    mean_det_val = float(adj_det.mean())
    delta_full = mean_det_val - mean_int_val

    daf = (1.0 - delta_nd / delta_full) if abs(delta_full) > 1e-10 else float('nan')

    # Bootstrap 95% CI for DAF
    boot_dafs = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        b_delta_full = adj_det[idx].mean() - adj_int[idx].mean()
        b_ba = both_alive[idx]
        if b_ba.sum() > 0:
            b_delta_nd = float((raw_det[idx][b_ba] - raw_int[idx][b_ba]).mean())
        else:
            b_delta_nd = 0.0
        if abs(b_delta_full) > 1e-10:
            boot_dafs[b] = 1.0 - b_delta_nd / b_delta_full
        else:
            boot_dafs[b] = np.nan

    valid = boot_dafs[~np.isnan(boot_dafs)]
    ci_lo = float(np.percentile(valid, 2.5)) if len(valid) > 0 else float('nan')
    ci_hi = float(np.percentile(valid, 97.5)) if len(valid) > 0 else float('nan')

    # Per-task correlation
    task_delta_full_p = np.array([
        adj_det[task_idx[t]].mean() - adj_int[task_idx[t]].mean()
        for t in unique_tasks
    ])

    if np.std(task_delta_dr) > 0 and np.std(task_delta_full_p) > 0:
        r_val, p_val = stats.pearsonr(task_delta_dr, task_delta_full_p)
        r_val = float(r_val)
        p_val = float(p_val)
    else:
        r_val, p_val = float('nan'), float('nan')

    # Label distribution
    D = int(np.sum(adj_det > adj_int))
    I = int(np.sum(adj_int > adj_det))
    N = int(np.sum(adj_det == adj_int))

    row = {
        "P": P,
        "DAF": round(daf * 100, 1),
        "CI_lo": round(ci_lo * 100, 1),
        "CI_hi": round(ci_hi * 100, 1),
        "Delta_full": round(delta_full, 2),
        "Delta_nd": round(delta_nd, 2),
        "r": round(r_val, 2),
        "p_value": p_val,
        "D": D, "I": I, "N": N,
        "mean_det": round(mean_det_val, 2),
        "mean_int": round(mean_int_val, 2),
    }
    results.append(row)

    p_str = f"{p_val:.2e}" if not np.isnan(p_val) else "nan"
    print(f"P={P:>3d} | DAF={row['DAF']:>6.1f}% [{row['CI_lo']:>5.1f}, {row['CI_hi']:>5.1f}] | "
          f"Delta_full={row['Delta_full']:>7.2f} | Delta_nd={row['Delta_nd']:>5.2f} | "
          f"r={row['r']:>5.2f} (p={p_str}) | D/I/N={D}/{I}/{N}")

# P=100 verification
p100 = next(r for r in results if r["P"] == 100)
print("\n=== P=100 Verification ===")
checks = [
    ("DAF", p100["DAF"], 90.4),
    ("r", p100["r"], 0.97),
    ("Delta_full", p100["Delta_full"], 29.60),
    ("D", p100["D"], 86),
    ("I", p100["I"], 9),
    ("N", p100["N"], 150),
]
all_pass = True
for name, actual, expected in checks:
    ok = abs(actual - expected) < 0.01 if isinstance(expected, float) else actual == expected
    mark = "PASS" if ok else "FAIL"
    if not ok:
        all_pass = False
    print(f"  [{mark}] {name}: {actual} (expected {expected})")
print(f"  Delta_nd: {p100['Delta_nd']} (expected ~2.84)")
print(f"\nAll P=100 checks passed: {all_pass}")

# Save
output = {
    "P_values": P_VALUES,
    "results": results,
    "n_episodes": n,
    "n_bootstrap": B,
    "n_both_alive": int(both_alive.sum()),
    "delta_nd": round(delta_nd, 4),
    "data_file": DATA_FILE,
    "task_counts": {t: len(task_idx[t]) for t in unique_tasks},
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved to {OUTPUT_FILE}")
