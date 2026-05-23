import json
import numpy as np
from scipy import stats
from collections import defaultdict

DATA_PATH = "./results/alfworld_counterfactual/episodes.jsonl"
OUT_PATH = "./artifacts/alfworld-daf/alfworld_daf_results.json"

def load_episodes(path):
    episodes = []
    with open(path) as f:
        for line in f:
            ep = json.loads(line)
            episodes.append({
                "task_type": ep["task_type"],
                "det_score": ep["deterministic"]["score"],
                "int_score": ep["internal"]["score"],
                "det_success": ep["deterministic"]["success"],
                "int_success": ep["internal"]["success"],
                "label": ep["label"],
            })
    return episodes

def compute_daf(det_scores, int_scores):
    det = np.array(det_scores, dtype=float)
    intn = np.array(int_scores, dtype=float)
    delta_full = det.mean() - intn.mean()
    both_success = (det == 1.0) & (intn == 1.0)
    if both_success.sum() > 0:
        delta_non_failure = det[both_success].mean() - intn[both_success].mean()
    else:
        delta_non_failure = 0.0
    if abs(delta_full) < 1e-12:
        daf = float('nan')
    else:
        daf = (delta_full - delta_non_failure) / delta_full
    return delta_full, delta_non_failure, daf

def bootstrap_ci(det_scores, int_scores, metric_fn, n_boot=10000, alpha=0.05):
    rng = np.random.RandomState(42)
    det = np.array(det_scores, dtype=float)
    intn = np.array(int_scores, dtype=float)
    n = len(det)
    boot_vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_vals.append(metric_fn(det[idx], intn[idx]))
    boot_vals = np.array(boot_vals)
    lo = float(np.nanpercentile(boot_vals, 100 * alpha / 2))
    hi = float(np.nanpercentile(boot_vals, 100 * (1 - alpha / 2)))
    return [lo, hi]

def delta_fn(det, intn):
    return det.mean() - intn.mean()

def daf_fn(det, intn):
    delta_full = det.mean() - intn.mean()
    both_success = (det == 1.0) & (intn == 1.0)
    if both_success.sum() > 0:
        delta_nf = det[both_success].mean() - intn[both_success].mean()
    else:
        delta_nf = 0.0
    if abs(delta_full) < 1e-12:
        return float('nan')
    return (delta_full - delta_nf) / delta_full

def main():
    episodes = load_episodes(DATA_PATH)
    N = len(episodes)
    assert N == 135, f"Expected 135 episodes, got {N}"

    by_type = defaultdict(list)
    for ep in episodes:
        by_type[ep["task_type"]].append(ep)

    all_det = [ep["det_score"] for ep in episodes]
    all_int = [ep["int_score"] for ep in episodes]
    all_labels = [ep["label"] for ep in episodes]

    det_sr = np.mean(all_det)
    int_sr = np.mean(all_int)
    delta_full, delta_nf, daf_equiv = compute_daf(all_det, all_int)
    daf_ci = bootstrap_ci(all_det, all_int, daf_fn)
    delta_ci = bootstrap_ci(all_det, all_int, delta_fn)

    n_det_pref = sum(1 for l in all_labels if l == "det-preferred")
    n_int_pref = sum(1 for l in all_labels if l == "internal-preferred")
    n_no_pref = sum(1 for l in all_labels if l == "no-preference")

    print(f"N = {N}")
    print(f"Labels: det-preferred={n_det_pref} ({n_det_pref/N*100:.1f}%), "
          f"internal-preferred={n_int_pref} ({n_int_pref/N*100:.1f}%), "
          f"no-preference={n_no_pref} ({n_no_pref/N*100:.1f}%)")
    print(f"Overall det SR: {det_sr:.4f}, int SR: {int_sr:.4f}")
    print(f"Δ_full: {delta_full:.4f}, Δ_non-failure: {delta_nf:.4f}, DAF: {daf_equiv:.4f}")
    print(f"DAF 95% CI: {daf_ci}")
    print(f"Delta 95% CI: {delta_ci}")

    per_task = {}
    task_deltas = []
    task_failure_diffs = []

    for tt in sorted(by_type.keys()):
        eps = by_type[tt]
        n_t = len(eps)
        det_s = [e["det_score"] for e in eps]
        int_s = [e["int_score"] for e in eps]
        labels = [e["label"] for e in eps]

        det_sr_t = np.mean(det_s)
        int_sr_t = np.mean(int_s)
        delta_t = det_sr_t - int_sr_t
        det_fail_rate = 1.0 - det_sr_t
        int_fail_rate = 1.0 - int_sr_t
        failure_diff = int_fail_rate - det_fail_rate

        n_det_p = sum(1 for l in labels if l == "det-preferred")
        n_int_p = sum(1 for l in labels if l == "internal-preferred")
        n_no_p = sum(1 for l in labels if l == "no-preference")

        d_ci = bootstrap_ci(det_s, int_s, delta_fn) if n_t >= 5 else [float('nan'), float('nan')]

        per_task[tt] = {
            "n": n_t,
            "det_sr": round(det_sr_t, 4),
            "int_sr": round(int_sr_t, 4),
            "delta": round(delta_t, 4),
            "det_fail_rate": round(det_fail_rate, 4),
            "int_fail_rate": round(int_fail_rate, 4),
            "failure_diff": round(failure_diff, 4),
            "det_preferred": n_det_p,
            "int_preferred": n_int_p,
            "no_preference": n_no_p,
            "delta_CI": [round(d_ci[0], 4), round(d_ci[1], 4)],
        }

        task_deltas.append(delta_t)
        task_failure_diffs.append(failure_diff)

        print(f"\n{tt} (n={n_t}):")
        print(f"  det SR: {det_sr_t:.4f}, int SR: {int_sr_t:.4f}, Δ: {delta_t:.4f}")
        print(f"  det fail: {det_fail_rate:.4f}, int fail: {int_fail_rate:.4f}, fail diff: {failure_diff:.4f}")
        print(f"  labels: det={n_det_p}, int={n_int_p}, no={n_no_p}")
        print(f"  Δ CI: {d_ci}")

    if len(task_deltas) >= 3:
        r, p = stats.pearsonr(task_failure_diffs, task_deltas)
    else:
        r, p = float('nan'), float('nan')

    print(f"\nPearson r (failure_diff vs delta): r={r:.4f}, p={p:.4f}")

    conclusion_parts = []
    if daf_equiv > 0.5:
        conclusion_parts.append(f"DAF={daf_equiv:.2f} indicates failure-rate differential dominates the det-internal gap")
    elif daf_equiv > 0:
        conclusion_parts.append(f"DAF={daf_equiv:.2f} indicates partial contribution from failure-rate differential")
    else:
        conclusion_parts.append(f"DAF={daf_equiv:.2f} indicates failure-rate differential does not explain the gap")

    if not np.isnan(r):
        if r > 0.5 and p < 0.1:
            conclusion_parts.append(f"Pearson r={r:.2f} (p={p:.3f}) confirms failure diff predicts per-task delta")
        elif r > 0:
            conclusion_parts.append(f"Pearson r={r:.2f} (p={p:.3f}) shows weak positive trend")
        else:
            conclusion_parts.append(f"Pearson r={r:.2f} (p={p:.3f}) shows no clear trend")

    conclusion = "; ".join(conclusion_parts)

    result = {
        "N": N,
        "task_types": len(by_type),
        "label_counts": {
            "det_preferred": n_det_pref,
            "internal_preferred": n_int_pref,
            "no_preference": n_no_pref,
        },
        "overall": {
            "det_success_rate": round(det_sr, 4),
            "int_success_rate": round(int_sr, 4),
            "delta_full": round(delta_full, 4),
            "delta_non_failure": round(delta_nf, 4),
            "DAF_equiv": round(daf_equiv, 4) if not np.isnan(daf_equiv) else None,
            "DAF_CI": [round(daf_ci[0], 4), round(daf_ci[1], 4)],
            "delta_CI": [round(delta_ci[0], 4), round(delta_ci[1], 4)],
        },
        "per_task": per_task,
        "pearson_r": round(r, 4) if not np.isnan(r) else None,
        "pearson_p": round(p, 4) if not np.isnan(p) else None,
        "conclusion": conclusion,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved to {OUT_PATH}")

if __name__ == "__main__":
    main()
