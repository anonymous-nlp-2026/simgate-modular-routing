"""Plan-012 Death Decomposition Analysis.

Quantifies how much of the Det-Internal gap comes from death avoidance
vs genuine score improvement. Core evidence for the paper.

Input:  results/{prescreening,research,plan_c}/{task_name}/episodes.jsonl
Output: JSON report, paper table JSON, CSV summary (all in --output dir)

Dependencies: Python 3.8+ stdlib only (scipy optional for exact p-values)
"""

import argparse
import csv
import json
import math
import os
import random
import statistics
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

DEATH_THRESHOLD = -100
BOOTSTRAP_N = 10000
BOOTSTRAP_CI = 0.95
BOOTSTRAP_SEED = 42


def is_dead(score: float) -> bool:
    return score <= DEATH_THRESHOLD


def classify_episode(int_score: float, det_score: float) -> Tuple[str, str, str]:
    """Returns (label, reason, signal_source)."""
    int_died = is_dead(int_score)
    det_died = is_dead(det_score)

    if int_died and not det_died:
        return "det-preferred", "int_died_det_alive", "death_asymmetry"
    if det_died and not int_died:
        return "internal-preferred", "det_died_int_alive", "death_asymmetry"
    if int_died and det_died:
        return "no-preference", "both_died", "both_dead"
    if det_score > int_score:
        return "det-preferred", "both_alive_det_wins", "score_comparison"
    if int_score > det_score:
        return "internal-preferred", "both_alive_int_wins", "score_comparison"
    return "no-preference", "tie", "tied_alive"


def safe_mean(values: List[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for incomplete beta (Lentz's method)."""
    max_iter = 200
    eps = 3e-12
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x < 0 or x > 1:
        return float("nan")
    if x == 0 or x == 1:
        return x
    ln_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - ln_beta)
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x) / a
    else:
        return 1 - front * _betacf(b, a, 1 - x) / b


def _t_sf(t_val: float, df: float) -> float:
    """Survival function for Student's t-distribution (pure Python)."""
    x = df / (df + t_val * t_val)
    p = 0.5 * _betai(df / 2.0, 0.5, x)
    return p


def pearsonr(x: List[float], y: List[float]) -> Tuple[float, float]:
    """Pearson correlation with two-tailed p-value."""
    try:
        from scipy.stats import pearsonr as sp_pearsonr
        r, p = sp_pearsonr(x, y)
        return float(r), float(p)
    except ImportError:
        pass
    n = len(x)
    if n < 3:
        return float("nan"), float("nan")
    mx, my = statistics.mean(x), statistics.mean(y)
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in x) / (n - 1))
    sy = math.sqrt(sum((yi - my) ** 2 for yi in y) / (n - 1))
    if sx == 0 or sy == 0:
        return float("nan"), float("nan")
    r = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / ((n - 1) * sx * sy)
    r = max(-1.0, min(1.0, r))
    t = r * math.sqrt((n - 2) / (1 - r * r + 1e-15))
    p = 2 * _t_sf(abs(t), n - 2)
    return r, p


def bootstrap_ci(values: List[float], stat_fn=safe_mean,
                 n_boot: int = BOOTSTRAP_N, ci: float = BOOTSTRAP_CI,
                 seed: int = BOOTSTRAP_SEED) -> Tuple[float, float]:
    """Bootstrap confidence interval for a statistic."""
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(values)
    boots = []
    for _ in range(n_boot):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        boots.append(stat_fn(sample))
    boots.sort()
    alpha = 1 - ci
    lo = boots[int(n_boot * alpha / 2)]
    hi = boots[int(n_boot * (1 - alpha / 2))]
    return lo, hi


def bootstrap_delta_ci(int_scores: List[float], det_scores: List[float],
                       n_boot: int = BOOTSTRAP_N, ci: float = BOOTSTRAP_CI,
                       seed: int = BOOTSTRAP_SEED) -> Tuple[float, float]:
    """Bootstrap CI for mean(det) - mean(int), resampling paired episodes."""
    if len(int_scores) < 2:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(int_scores)
    boots = []
    for _ in range(n_boot):
        idxs = [rng.randint(0, n - 1) for _ in range(n)]
        d = statistics.mean([det_scores[i] for i in idxs]) - statistics.mean([int_scores[i] for i in idxs])
        boots.append(d)
    boots.sort()
    alpha = 1 - ci
    return boots[int(n_boot * alpha / 2)], boots[int(n_boot * (1 - alpha / 2))]


def load_episodes(data_dirs: List[str], task_filter: Optional[List[str]] = None) -> Dict[str, List[dict]]:
    """Load episodes grouped by task_name from multiple data directories."""
    task_episodes: Dict[str, List[dict]] = defaultdict(list)

    for data_dir in data_dirs:
        if not os.path.isdir(data_dir):
            print(f"  WARN: {data_dir} not found, skipping")
            continue
        for task_name in sorted(os.listdir(data_dir)):
            task_dir = os.path.join(data_dir, task_name)
            ep_file = os.path.join(task_dir, "episodes.jsonl")
            if not os.path.isfile(ep_file):
                continue
            if task_filter and task_name not in task_filter:
                continue
            with open(ep_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    ep = json.loads(line)
                    if "always_internal" not in ep or "always_deterministic" not in ep:
                        continue
                    task_episodes[task_name].append(ep)

    return dict(task_episodes)


def analyze_task(task_name: str, episodes: List[dict]) -> dict:
    """Full death decomposition for one task."""
    n = len(episodes)
    int_scores = [ep["always_internal"]["final_score"] for ep in episodes]
    det_scores = [ep["always_deterministic"]["final_score"] for ep in episodes]

    int_deaths = sum(1 for s in int_scores if is_dead(s))
    det_deaths = sum(1 for s in det_scores if is_dead(s))
    int_death_rate = int_deaths / n
    det_death_rate = det_deaths / n

    int_mean = safe_mean(int_scores)
    det_mean = safe_mean(det_scores)
    delta_full = det_mean - int_mean

    # No-death subset: exclude episodes where EITHER mode died
    nd_int, nd_det = [], []
    # Death subset: episodes where at least one mode died
    d_int, d_det = [], []
    for i_s, d_s in zip(int_scores, det_scores):
        if not is_dead(i_s) and not is_dead(d_s):
            nd_int.append(i_s)
            nd_det.append(d_s)
        else:
            d_int.append(i_s)
            d_det.append(d_s)

    delta_no_death = (safe_mean(nd_det) - safe_mean(nd_int)) if nd_int else float("nan")
    delta_death_only = (safe_mean(d_det) - safe_mean(d_int)) if d_int else float("nan")

    # Penalty-removed: replace -100 with 0
    int_scores_pr = [0.0 if is_dead(s) else s for s in int_scores]
    det_scores_pr = [0.0 if is_dead(s) else s for s in det_scores]
    delta_penalty_removed = safe_mean(det_scores_pr) - safe_mean(int_scores_pr)

    # Death avoidance fraction
    if not math.isnan(delta_no_death) and delta_full != 0:
        death_avoidance_frac = (delta_full - delta_no_death) / delta_full
    else:
        death_avoidance_frac = float("nan")

    # Label distribution
    label_counts = defaultdict(int)
    signal_counts = defaultdict(int)
    label_signal_cross = defaultdict(lambda: defaultdict(int))
    for i_s, d_s in zip(int_scores, det_scores):
        label, reason, signal_source = classify_episode(i_s, d_s)
        label_counts[label] += 1
        signal_counts[signal_source] += 1
        label_signal_cross[label][signal_source] += 1

    # Bootstrap CIs
    ci_delta_full = bootstrap_delta_ci(int_scores, det_scores)
    ci_delta_nd = bootstrap_delta_ci(nd_int, nd_det) if nd_int else (float("nan"), float("nan"))

    def _round_or_none(v, d=2):
        return round(v, d) if not math.isnan(v) else None

    return {
        "task": task_name,
        "n_episodes": n,
        "int_death_count": int_deaths,
        "det_death_count": det_deaths,
        "int_death_rate": round(int_death_rate, 4),
        "det_death_rate": round(det_death_rate, 4),
        "int_mean": round(int_mean, 2),
        "det_mean": round(det_mean, 2),
        "delta_full": round(delta_full, 2),
        "delta_full_ci": [_round_or_none(ci_delta_full[0]), _round_or_none(ci_delta_full[1])],
        "delta_no_death": _round_or_none(delta_no_death),
        "delta_no_death_ci": [_round_or_none(ci_delta_nd[0]), _round_or_none(ci_delta_nd[1])],
        "n_no_death_episodes": len(nd_int),
        "delta_death_only": _round_or_none(delta_death_only),
        "n_death_episodes": len(d_int),
        "death_avoidance_fraction": _round_or_none(death_avoidance_frac, 4),
        "delta_penalty_removed": round(delta_penalty_removed, 2),
        "label_distribution": {
            "det-preferred": label_counts.get("det-preferred", 0),
            "internal-preferred": label_counts.get("internal-preferred", 0),
            "no-preference": label_counts.get("no-preference", 0),
        },
        "signal_source_distribution": dict(signal_counts),
        "label_signal_cross": {k: dict(v) for k, v in label_signal_cross.items()},
    }


def aggregate_from_raw(data_dirs: List[str], task_filter: Optional[List[str]] = None) -> dict:
    """Aggregate from raw episode scores for exact results."""
    task_episodes = load_episodes(data_dirs, task_filter)
    all_int, all_det = [], []
    all_int_nd, all_det_nd = [], []
    all_int_d, all_det_d = [], []

    task_death_rates = []
    task_deltas = []
    total_label = defaultdict(int)
    total_signal = defaultdict(int)
    total_label_signal = defaultdict(lambda: defaultdict(int))

    for task_name, episodes in task_episodes.items():
        int_scores = [ep["always_internal"]["final_score"] for ep in episodes]
        det_scores = [ep["always_deterministic"]["final_score"] for ep in episodes]
        all_int.extend(int_scores)
        all_det.extend(det_scores)

        n = len(int_scores)
        int_dr = sum(1 for s in int_scores if is_dead(s)) / n
        det_dr = sum(1 for s in det_scores if is_dead(s)) / n
        delta = safe_mean(det_scores) - safe_mean(int_scores)
        task_death_rates.append(int_dr - det_dr)
        task_deltas.append(delta)

        for i_s, d_s in zip(int_scores, det_scores):
            if not is_dead(i_s) and not is_dead(d_s):
                all_int_nd.append(i_s)
                all_det_nd.append(d_s)
            else:
                all_int_d.append(i_s)
                all_det_d.append(d_s)
            label, _, signal_source = classify_episode(i_s, d_s)
            total_label[label] += 1
            total_signal[signal_source] += 1
            total_label_signal[label][signal_source] += 1

    n_total = len(all_int)
    if n_total == 0:
        return {"error": "no episodes"}

    int_mean = safe_mean(all_int)
    det_mean = safe_mean(all_det)
    delta_full = det_mean - int_mean

    int_nd_mean = safe_mean(all_int_nd) if all_int_nd else float("nan")
    det_nd_mean = safe_mean(all_det_nd) if all_det_nd else float("nan")
    delta_nd = (det_nd_mean - int_nd_mean) if all_int_nd else float("nan")

    delta_death_only = (safe_mean(all_det_d) - safe_mean(all_int_d)) if all_int_d else float("nan")

    int_pr = safe_mean([0.0 if is_dead(s) else s for s in all_int])
    det_pr = safe_mean([0.0 if is_dead(s) else s for s in all_det])
    delta_pr = det_pr - int_pr

    if not math.isnan(delta_nd) and delta_full != 0:
        death_frac = (delta_full - delta_nd) / delta_full
    else:
        death_frac = float("nan")

    r, p = pearsonr(task_death_rates, task_deltas)

    # Bootstrap CIs on aggregate
    ci_delta_full = bootstrap_delta_ci(all_int, all_det)
    ci_delta_nd = bootstrap_delta_ci(all_int_nd, all_det_nd) if all_int_nd else (float("nan"), float("nan"))

    # Bootstrap CI for death_avoidance_fraction
    if all_int_nd and delta_full != 0:
        rng = random.Random(BOOTSTRAP_SEED)
        n_ep = len(all_int)
        daf_boots = []
        for _ in range(BOOTSTRAP_N):
            idxs = [rng.randint(0, n_ep - 1) for _ in range(n_ep)]
            b_int = [all_int[i] for i in idxs]
            b_det = [all_det[i] for i in idxs]
            b_df = safe_mean(b_det) - safe_mean(b_int)
            b_int_nd = [b_int[j] for j in range(n_ep) if not is_dead(b_int[j]) and not is_dead(b_det[j])]
            b_det_nd = [b_det[j] for j in range(n_ep) if not is_dead(b_int[j]) and not is_dead(b_det[j])]
            b_dnd = (safe_mean(b_det_nd) - safe_mean(b_int_nd)) if b_int_nd else 0
            daf = (b_df - b_dnd) / b_df if b_df != 0 else float("nan")
            if not math.isnan(daf):
                daf_boots.append(daf)
        daf_boots.sort()
        alpha = 1 - BOOTSTRAP_CI
        ci_daf = (
            daf_boots[int(len(daf_boots) * alpha / 2)] if daf_boots else float("nan"),
            daf_boots[int(len(daf_boots) * (1 - alpha / 2))] if daf_boots else float("nan"),
        )
    else:
        ci_daf = (float("nan"), float("nan"))

    def _round_or_none(v, d=2):
        return round(v, d) if not math.isnan(v) else None

    return {
        "n_tasks": len(task_episodes),
        "n_episodes": n_total,
        "int_death_rate": round(sum(1 for s in all_int if is_dead(s)) / n_total, 4),
        "det_death_rate": round(sum(1 for s in all_det if is_dead(s)) / n_total, 4),
        "int_mean": round(int_mean, 2),
        "det_mean": round(det_mean, 2),
        "delta_full": round(delta_full, 2),
        "delta_full_ci": [_round_or_none(ci_delta_full[0]), _round_or_none(ci_delta_full[1])],
        "delta_no_death": _round_or_none(delta_nd),
        "delta_no_death_ci": [_round_or_none(ci_delta_nd[0]), _round_or_none(ci_delta_nd[1])],
        "delta_death_only": _round_or_none(delta_death_only),
        "n_death_episodes": len(all_int_d),
        "n_no_death_episodes": len(all_int_nd),
        "death_avoidance_fraction": _round_or_none(death_frac, 4),
        "death_avoidance_fraction_ci": [_round_or_none(ci_daf[0], 4), _round_or_none(ci_daf[1], 4)],
        "delta_penalty_removed": round(delta_pr, 2),
        "correlation_death_rate_diff_vs_delta": {
            "pearson_r": round(r, 4) if not math.isnan(r) else None,
            "p_value": round(p, 6) if not math.isnan(p) else None,
            "n_tasks": len(task_episodes),
        },
        "label_distribution": dict(total_label),
        "signal_source_distribution": dict(total_signal),
        "label_signal_cross": {k: dict(v) for k, v in total_label_signal.items()},
    }



def compute_task_weighted(task_results: List[dict]) -> dict:
    """Task-weighted aggregation: each task contributes equally regardless of episode count."""
    valid_tasks = [tr for tr in task_results if tr["death_avoidance_fraction"] is not None]
    all_tasks_delta = [tr["delta_full"] for tr in task_results]
    all_tasks_delta_nd = [tr["delta_no_death"] for tr in task_results if tr["delta_no_death"] is not None]

    tw_delta_full = safe_mean(all_tasks_delta)
    tw_delta_nd = safe_mean(all_tasks_delta_nd) if all_tasks_delta_nd else float("nan")

    task_dafs = [tr["death_avoidance_fraction"] for tr in valid_tasks]
    tw_daf = safe_mean(task_dafs) if task_dafs else float("nan")

    # Bootstrap CI: resample tasks (not episodes) for robustness to unequal sample sizes
    if len(task_dafs) >= 2 and BOOTSTRAP_N > 0:
        rng = random.Random(BOOTSTRAP_SEED)
        n_tasks = len(task_dafs)
        boots = []
        for _ in range(BOOTSTRAP_N):
            idxs = [rng.randint(0, n_tasks - 1) for _ in range(n_tasks)]
            boots.append(safe_mean([task_dafs[i] for i in idxs]))
        boots.sort()
        alpha = 1 - BOOTSTRAP_CI
        ci_daf = (boots[int(len(boots) * alpha / 2)], boots[int(len(boots) * (1 - alpha / 2))])
    else:
        ci_daf = (float("nan"), float("nan"))

    def _round_or_none(v, d=2):
        return round(v, d) if not math.isnan(v) else None

    return {
        "task_weighted_daf": _round_or_none(tw_daf, 4),
        "task_weighted_daf_ci": [_round_or_none(ci_daf[0], 4), _round_or_none(ci_daf[1], 4)],
        "task_weighted_delta_full": _round_or_none(tw_delta_full),
        "task_weighted_delta_nd": _round_or_none(tw_delta_nd),
        "n_tasks_with_daf": len(valid_tasks),
    }

def print_table(task_results: List[dict], aggregate: dict):
    """Print terminal-format summary table."""
    hdr = (f"| {'Task':<42} | {'N':>4} | {'Int':>7} | {'Det':>7} | {'D':>7} "
           f"| {'I_d%':>5} | {'D_d%':>5} "
           f"| {'D_nd':>7} | {'D_frac':>7} | {'D_pr':>7} "
           f"| {'det':>4} {'int':>4} {'no':>4} |")
    sep = "-" * len(hdr)

    print(sep)
    print(hdr)
    print(sep)

    for tr in task_results:
        dnd = f"{tr['delta_no_death']:+.1f}" if tr["delta_no_death"] is not None else "N/A"
        dfr = f"{tr['death_avoidance_fraction']:.0%}" if tr["death_avoidance_fraction"] is not None else "N/A"
        dpr = f"{tr['delta_penalty_removed']:+.1f}"
        ld = tr["label_distribution"]
        print(f"| {tr['task']:<42} | {tr['n_episodes']:>4} "
              f"| {tr['int_mean']:>7.1f} | {tr['det_mean']:>7.1f} | {tr['delta_full']:>+7.1f} "
              f"| {tr['int_death_rate']*100:>4.0f}% | {tr['det_death_rate']*100:>4.0f}% "
              f"| {dnd:>7s} | {dfr:>7s} | {dpr:>7s} "
              f"| {ld.get('det-preferred',0):>4} {ld.get('internal-preferred',0):>4} {ld.get('no-preference',0):>4} |")

    print(sep)
    agg = aggregate
    dnd = f"{agg['delta_no_death']:+.1f}" if agg.get("delta_no_death") is not None else "N/A"
    dfr = f"{agg['death_avoidance_fraction']:.0%}" if agg.get("death_avoidance_fraction") is not None else "N/A"
    dpr = f"{agg['delta_penalty_removed']:+.1f}"
    ld = agg["label_distribution"]
    print(f"| {'AGGREGATE':<42} | {agg['n_episodes']:>4} "
          f"| {agg['int_mean']:>7.1f} | {agg['det_mean']:>7.1f} | {agg['delta_full']:>+7.1f} "
          f"| {agg['int_death_rate']*100:>4.0f}% | {agg['det_death_rate']*100:>4.0f}% "
          f"| {dnd:>7s} | {dfr:>7s} | {dpr:>7s} "
          f"| {ld.get('det-preferred',0):>4} {ld.get('internal-preferred',0):>4} {ld.get('no-preference',0):>4} |")
    print(sep)

    # Bootstrap CIs
    ci_f = agg.get("delta_full_ci", [None, None])
    ci_nd = agg.get("delta_no_death_ci", [None, None])
    ci_daf = agg.get("death_avoidance_fraction_ci", [None, None])
    print(f"\nDelta_full = {agg['delta_full']:+.2f}  95% CI [{ci_f[0]}, {ci_f[1]}]")
    print(f"Delta_nd   = {agg.get('delta_no_death', 'N/A')}  95% CI [{ci_nd[0]}, {ci_nd[1]}]")
    print(f"Delta_death= {agg.get('delta_death_only', 'N/A')}")
    print(f"Death frac = {dfr}  95% CI [{ci_daf[0]}, {ci_daf[1]}]")
    print(f"Delta_pr   = {agg['delta_penalty_removed']:+.2f}")
    tw_daf = agg.get("task_weighted_daf")
    tw_ci = agg.get("task_weighted_daf_ci", [None, None])
    if tw_daf is not None:
        print(f"\nTask-weighted (each task equal weight, n={agg.get('n_tasks_with_daf', '?')}):")
        print(f"  DAF        = {tw_daf:.1%}  95% CI [{tw_ci[0]}, {tw_ci[1]}]")
        print(f"  Delta_full = {agg.get('task_weighted_delta_full')}")
        print(f"  Delta_nd   = {agg.get('task_weighted_delta_nd')}")

    # Correlation
    corr = agg.get("correlation_death_rate_diff_vs_delta", {})
    r = corr.get("pearson_r")
    p = corr.get("p_value")
    r_str = f"{r:.3f}" if r is not None else "N/A"
    p_str = f"{p:.6f}" if p is not None else "N/A"
    print(f"\nCorrelation (death_rate_diff vs Delta): r={r_str}, p={p_str}, n={corr.get('n_tasks', 0)} tasks")

    # Signal source breakdown
    print(f"\nSignal source distribution:")
    for src in ["death_asymmetry", "both_dead", "score_comparison", "tied_alive"]:
        cnt = agg["signal_source_distribution"].get(src, 0)
        pct = cnt / agg["n_episodes"] * 100 if agg["n_episodes"] else 0
        print(f"  {src}: {cnt} ({pct:.1f}%)")

    # Cross table
    print(f"\nLabel x Signal cross table:")
    print(f"  {'Label':<20} | {'death_asym':>10} | {'both_dead':>10} | {'score_cmp':>10} | {'tied_alive':>10} |")
    cross = agg.get("label_signal_cross", {})
    for lbl in ["det-preferred", "internal-preferred", "no-preference"]:
        row = cross.get(lbl, {})
        print(f"  {lbl:<20} | {row.get('death_asymmetry',0):>10} | {row.get('both_dead',0):>10} "
              f"| {row.get('score_comparison',0):>10} | {row.get('tied_alive',0):>10} |")


def generate_paper_table_data(task_results: List[dict], aggregate: dict) -> dict:
    """Generate data structured for LaTeX table generation."""
    rows = []
    for tr in task_results:
        rows.append({
            "task": tr["task"],
            "n": tr["n_episodes"],
            "int_mean": tr["int_mean"],
            "det_mean": tr["det_mean"],
            "delta": tr["delta_full"],
            "delta_ci": tr.get("delta_full_ci"),
            "int_death_pct": round(tr["int_death_rate"] * 100, 1),
            "det_death_pct": round(tr["det_death_rate"] * 100, 1),
            "delta_no_death": tr["delta_no_death"],
            "delta_death_only": tr.get("delta_death_only"),
            "death_avoidance_pct": round(tr["death_avoidance_fraction"] * 100, 1) if tr["death_avoidance_fraction"] is not None else None,
            "delta_penalty_removed": tr["delta_penalty_removed"],
            "det_preferred": tr["label_distribution"].get("det-preferred", 0),
            "internal_preferred": tr["label_distribution"].get("internal-preferred", 0),
            "no_preference": tr["label_distribution"].get("no-preference", 0),
        })

    agg_row = {
        "task": "Overall",
        "n": aggregate["n_episodes"],
        "int_mean": aggregate["int_mean"],
        "det_mean": aggregate["det_mean"],
        "delta": aggregate["delta_full"],
        "delta_ci": aggregate.get("delta_full_ci"),
        "int_death_pct": round(aggregate["int_death_rate"] * 100, 1),
        "det_death_pct": round(aggregate["det_death_rate"] * 100, 1),
        "delta_no_death": aggregate.get("delta_no_death"),
        "delta_death_only": aggregate.get("delta_death_only"),
        "death_avoidance_pct": round(aggregate["death_avoidance_fraction"] * 100, 1) if aggregate.get("death_avoidance_fraction") is not None else None,
        "death_avoidance_fraction_ci": aggregate.get("death_avoidance_fraction_ci"),
        "delta_penalty_removed": aggregate["delta_penalty_removed"],
        "det_preferred": aggregate["label_distribution"].get("det-preferred", 0),
        "internal_preferred": aggregate["label_distribution"].get("internal-preferred", 0),
        "no_preference": aggregate["label_distribution"].get("no-preference", 0),
    }

    corr = aggregate.get("correlation_death_rate_diff_vs_delta", {})

    return {
        "rows": rows,
        "aggregate": agg_row,
        "correlation": {
            "pearson_r": corr.get("pearson_r"),
            "p_value": corr.get("p_value"),
            "n": corr.get("n_tasks"),
        },
        "task_weighted": {
            "daf": aggregate.get("task_weighted_daf"),
            "daf_ci": aggregate.get("task_weighted_daf_ci"),
            "delta_full": aggregate.get("task_weighted_delta_full"),
            "delta_nd": aggregate.get("task_weighted_delta_nd"),
            "n_tasks_with_daf": aggregate.get("n_tasks_with_daf"),
        },
    }


def write_csv(task_results: List[dict], aggregate: dict, path: str):
    """Write per-task summary CSV."""
    fields = [
        "task", "n_episodes", "int_mean", "det_mean", "delta_full",
        "int_death_rate", "det_death_rate",
        "delta_no_death", "delta_death_only", "death_avoidance_fraction",
        "delta_penalty_removed",
        "det_preferred", "internal_preferred", "no_preference",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for tr in task_results:
            row = {k: tr.get(k) for k in fields}
            row["det_preferred"] = tr["label_distribution"].get("det-preferred", 0)
            row["internal_preferred"] = tr["label_distribution"].get("internal-preferred", 0)
            row["no_preference"] = tr["label_distribution"].get("no-preference", 0)
            w.writerow(row)
        # Aggregate row
        agg_row = {
            "task": "AGGREGATE",
            "n_episodes": aggregate["n_episodes"],
            "int_mean": aggregate["int_mean"],
            "det_mean": aggregate["det_mean"],
            "delta_full": aggregate["delta_full"],
            "int_death_rate": aggregate["int_death_rate"],
            "det_death_rate": aggregate["det_death_rate"],
            "delta_no_death": aggregate.get("delta_no_death"),
            "delta_death_only": aggregate.get("delta_death_only"),
            "death_avoidance_fraction": aggregate.get("death_avoidance_fraction"),
            "delta_penalty_removed": aggregate["delta_penalty_removed"],
            "det_preferred": aggregate["label_distribution"].get("det-preferred", 0),
            "internal_preferred": aggregate["label_distribution"].get("internal-preferred", 0),
            "no_preference": aggregate["label_distribution"].get("no-preference", 0),
        }
        w.writerow(agg_row)
        if aggregate.get("task_weighted_daf") is not None:
            tw_row = {
                "task": "TASK_WEIGHTED",
                "n_episodes": aggregate.get("n_tasks_with_daf"),
                "delta_full": aggregate.get("task_weighted_delta_full"),
                "delta_no_death": aggregate.get("task_weighted_delta_nd"),
                "death_avoidance_fraction": aggregate.get("task_weighted_daf"),
            }
            w.writerow(tw_row)


def main():
    parser = argparse.ArgumentParser(description="Plan-012 Death Decomposition Analysis")
    parser.add_argument("--data_dirs", nargs="+", required=True,
                        help="Data directories (e.g. results/prescreening results/plan_c)")
    parser.add_argument("--output", default="results/decomposition_final",
                        help="Output directory for all result files")
    parser.add_argument("--tasks", default=None,
                        help="Comma-separated task types to filter (default: all)")
    parser.add_argument("--no-bootstrap", action="store_true",
                        help="Skip bootstrap CI computation (faster)")
    args = parser.parse_args()

    if args.no_bootstrap:
        global BOOTSTRAP_N
        BOOTSTRAP_N = 0

    task_filter = [t.strip() for t in args.tasks.split(",")] if args.tasks else None

    task_episodes = load_episodes(args.data_dirs, task_filter)
    if not task_episodes:
        print("ERROR: No episodes found in specified directories")
        return

    print(f"Loaded {sum(len(v) for v in task_episodes.values())} episodes "
          f"across {len(task_episodes)} tasks from {len(args.data_dirs)} directories")

    # Per-task analysis
    task_results = []
    for task_name in sorted(task_episodes.keys()):
        result = analyze_task(task_name, task_episodes[task_name])
        task_results.append(result)

    # Aggregate from raw scores (exact)
    aggregate = aggregate_from_raw(args.data_dirs, task_filter)

    # Task-weighted aggregation
    task_weighted = compute_task_weighted(task_results)
    aggregate.update(task_weighted)

    # Terminal output
    print_table(task_results, aggregate)

    # Save outputs
    os.makedirs(args.output, exist_ok=True)

    report_path = os.path.join(args.output, "report.json")
    report = {"per_task": task_results, "aggregate": aggregate}
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[Saved report to {report_path}]")

    paper_path = os.path.join(args.output, "paper_table.json")
    paper_data = generate_paper_table_data(task_results, aggregate)
    with open(paper_path, "w") as f:
        json.dump(paper_data, f, indent=2)
    print(f"[Saved paper table to {paper_path}]")

    csv_path = os.path.join(args.output, "summary.csv")
    write_csv(task_results, aggregate, csv_path)
    print(f"[Saved CSV to {csv_path}]")


if __name__ == "__main__":
    main()
