#!/usr/bin/env python3
"""Gemma-2-9B ScienceWorld counterfactual analysis."""
import argparse
import json
import os
from collections import defaultdict

import numpy as np

INCLUDED_TASKS = [
    "identify-life-stages-1",
    "identify-life-stages-2",
    "freeze",
    "grow-fruit",
    "measure-melting-point-unknown-substance",
    "lifespan-longest-lived",
    "lifespan-longest-lived-then-shortest-lived",
    "chemistry-mix",
    "inclined-plane-determine-angle",
    "find-animal",
    "find-non-living-thing",
]

QWEN = {
    "model": "Qwen2.5-7B",
    "int_dr": 42.9, "int_dr_ci": [36.7, 49.0],
    "det_dr": 16.3, "det_dr_ci": [11.8, 21.2],
    "delta_dr": 26.5, "delta_dr_ci": [20.4, 32.7],
    "daf": 90.4, "daf_ci": [81.3, 99.1],
}
LLAMA = {
    "model": "Llama-3.1-8B",
    "int_dr": 41.4, "int_dr_ci": [35.0, 47.7],
    "det_dr": 11.4, "det_dr_ci": [7.3, 15.9],
    "delta_dr": 30.0, "delta_dr_ci": [23.6, 36.4],
    "daf": 88.2, "daf_ci": [80.2, 95.1],
}


def load_episodes(input_dir):
    all_episodes = []
    for task in INCLUDED_TASKS:
        task_dir = os.path.join(input_dir, task)
        if not os.path.isdir(task_dir):
            print(f"  WARNING: missing task dir {task}")
            continue
        summary_path = os.path.join(task_dir, "episode_summary.jsonl")
        if os.path.exists(summary_path):
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
                    ep["task_type"] = task
                    ep["internal_score"] = ep.get("internal_final", ep.get("internal_score", 0.0))
                    ep["deterministic_score"] = ep.get("det_final", ep.get("deterministic_score", 0.0))
                    all_episodes.append(ep)
        else:
            for fname in sorted(os.listdir(task_dir)):
                if fname.endswith(".json") and fname != "collection_meta.json":
                    with open(os.path.join(task_dir, fname)) as f:
                        ep = json.load(f)
                        ep.setdefault("task_type", task)
                        all_episodes.append(ep)
    return all_episodes


def per_task_breakdown(episodes):
    by_task = defaultdict(list)
    for ep in episodes:
        by_task[ep["task_type"]].append(ep)
    rows = []
    for task in INCLUDED_TASKS:
        eps = by_task.get(task, [])
        n = len(eps)
        if n == 0:
            rows.append({"task": task, "n": 0})
            continue
        det_pref = sum(1 for e in eps if e["label"] == "det-preferred")
        int_pref = sum(1 for e in eps if e["label"] == "internal-preferred")
        no_pref = sum(1 for e in eps if e["label"] == "no-preference")
        int_deaths = sum(1 for e in eps if e["internal_score"] <= -100)
        det_deaths = sum(1 for e in eps if e["deterministic_score"] <= -100)
        rows.append({
            "task": task,
            "n": n,
            "det_pref": det_pref, "det_pref_pct": det_pref / n * 100,
            "int_pref": int_pref, "int_pref_pct": int_pref / n * 100,
            "no_pref": no_pref, "no_pref_pct": no_pref / n * 100,
            "int_dr": int_deaths / n * 100,
            "det_dr": det_deaths / n * 100,
        })
    return rows


def compute_aggregate(episodes):
    n = len(episodes)
    det_pref = sum(1 for e in episodes if e["label"] == "det-preferred")
    int_pref = sum(1 for e in episodes if e["label"] == "internal-preferred")
    no_pref = sum(1 for e in episodes if e["label"] == "no-preference")

    int_scores = np.array([e["internal_score"] for e in episodes])
    det_scores = np.array([e["deterministic_score"] for e in episodes])

    int_dr = np.mean(int_scores <= -100) * 100
    det_dr = np.mean(det_scores <= -100) * 100
    delta_dr = int_dr - det_dr

    delta_full = np.mean(int_scores) - np.mean(det_scores)
    nd_mask = (int_scores > -100) & (det_scores > -100)
    n_nd = np.sum(nd_mask)
    if n_nd > 0:
        delta_nd = np.mean(int_scores[nd_mask]) - np.mean(det_scores[nd_mask])
    else:
        delta_nd = 0.0
    if delta_full != 0:
        daf = (1 - delta_nd / delta_full) * 100
    else:
        daf = None

    return {
        "n": n,
        "det_pref": det_pref, "det_pref_pct": det_pref / n * 100,
        "int_pref": int_pref, "int_pref_pct": int_pref / n * 100,
        "no_pref": no_pref, "no_pref_pct": no_pref / n * 100,
        "int_dr": int_dr, "det_dr": det_dr, "delta_dr": delta_dr,
        "delta_full": delta_full, "delta_nd": delta_nd, "n_nd": int(n_nd),
        "daf": daf,
        "mean_int_score": float(np.mean(int_scores)),
        "mean_det_score": float(np.mean(det_scores)),
    }


def cluster_bootstrap(episodes, stat_fn, B=10000, seed=42):
    rng = np.random.RandomState(seed)
    by_task = defaultdict(list)
    for ep in episodes:
        by_task[ep["task_type"]].append(ep)
    task_keys = sorted(by_task.keys())
    K = len(task_keys)
    task_eps = [by_task[k] for k in task_keys]

    stats = np.empty(B)
    for i in range(B):
        idxs = rng.randint(0, K, size=K)
        resampled = []
        for idx in idxs:
            resampled.extend(task_eps[idx])
        stats[i] = stat_fn(resampled)
    lo = float(np.percentile(stats, 2.5))
    hi = float(np.percentile(stats, 97.5))
    return lo, hi


def stat_int_dr(eps):
    scores = np.array([e["internal_score"] for e in eps])
    return np.mean(scores <= -100) * 100

def stat_det_dr(eps):
    scores = np.array([e["deterministic_score"] for e in eps])
    return np.mean(scores <= -100) * 100

def stat_delta_dr(eps):
    return stat_int_dr(eps) - stat_det_dr(eps)

def stat_daf(eps):
    int_s = np.array([e["internal_score"] for e in eps])
    det_s = np.array([e["deterministic_score"] for e in eps])
    delta_full = np.mean(int_s) - np.mean(det_s)
    nd = (int_s > -100) & (det_s > -100)
    if np.sum(nd) == 0 or delta_full == 0:
        return np.nan
    delta_nd = np.mean(int_s[nd]) - np.mean(det_s[nd])
    return (1 - delta_nd / delta_full) * 100


def fmt_ci(val, lo, hi, unit="%", decimals=1):
    f = f"{{:.{decimals}f}}"
    if unit == "pp":
        sign = "+" if val >= 0 else ""
        return f"{sign}{f.format(val)}pp [{f.format(lo)},{f.format(hi)}]"
    return f"{f.format(val)}{unit} [{f.format(lo)},{f.format(hi)}]"


def print_per_task(rows):
    print("=" * 110)
    print("PER-TASK BREAKDOWN")
    print("=" * 110)
    hdr = f"{'Task':<45s} {'N':>3s}  {'Det-Pref':>10s}  {'Int-Pref':>10s}  {'No-Pref':>10s}  {'IntDR':>6s}  {'DetDR':>6s}"
    print(hdr)
    print("-" * 110)
    for r in rows:
        if r["n"] == 0:
            print(f"{r['task']:<45s} {'0':>3s}  {'—':>10s}  {'—':>10s}  {'—':>10s}  {'—':>6s}  {'—':>6s}")
            continue
        print(
            f"{r['task']:<45s} {r['n']:3d}  "
            f"{r['det_pref']:2d} ({r['det_pref_pct']:4.1f}%)  "
            f"{r['int_pref']:2d} ({r['int_pref_pct']:4.1f}%)  "
            f"{r['no_pref']:2d} ({r['no_pref_pct']:4.1f}%)  "
            f"{r['int_dr']:5.1f}%  {r['det_dr']:5.1f}%"
        )


def print_aggregate(agg):
    print("\n" + "=" * 60)
    print("AGGREGATE METRICS")
    print("=" * 60)
    print(f"  Total episodes:         {agg['n']}")
    print(f"  Det-preferred:          {agg['det_pref']:3d}  ({agg['det_pref_pct']:.1f}%)")
    print(f"  Int-preferred:          {agg['int_pref']:3d}  ({agg['int_pref_pct']:.1f}%)")
    print(f"  No-preference:          {agg['no_pref']:3d}  ({agg['no_pref_pct']:.1f}%)")
    print(f"  Mean internal score:    {agg['mean_int_score']:.2f}")
    print(f"  Mean det score:         {agg['mean_det_score']:.2f}")
    print(f"  Internal DR:            {agg['int_dr']:.1f}%")
    print(f"  Deterministic DR:       {agg['det_dr']:.1f}%")
    print(f"  ΔDR (int - det):        {agg['delta_dr']:+.1f}pp")
    print(f"  Δ_full:                 {agg['delta_full']:.4f}")
    print(f"  Δ_nd (n={agg['n_nd']}):          {agg['delta_nd']:.4f}")
    if agg["daf"] is not None:
        print(f"  DAF:                    {agg['daf']:.1f}%")
    else:
        print(f"  DAF:                    undefined (Δ_full=0)")


def print_bootstrap(agg, cis):
    print("\n" + "=" * 60)
    print(f"BOOTSTRAP 95% CIs (cluster, K=11)")
    print("=" * 60)
    print(f"  Internal DR:   {fmt_ci(agg['int_dr'], cis['int_dr'][0], cis['int_dr'][1])}")
    print(f"  Det DR:        {fmt_ci(agg['det_dr'], cis['det_dr'][0], cis['det_dr'][1])}")
    print(f"  ΔDR:           {fmt_ci(agg['delta_dr'], cis['delta_dr'][0], cis['delta_dr'][1], unit='pp')}")
    if agg["daf"] is not None:
        print(f"  DAF:           {fmt_ci(agg['daf'], cis['daf'][0], cis['daf'][1])}")
    else:
        print(f"  DAF:           undefined")


def print_cross_model(agg, cis):
    print("\n" + "=" * 90)
    print("CROSS-MODEL COMPARISON (ScienceWorld)")
    print("=" * 90)
    hdr = f"{'Model':<18s}  {'Int DR':>22s}  {'Det DR':>22s}  {'ΔDR':>24s}  {'DAF':>22s}"
    print(hdr)
    print("-" * 90)
    for m in [QWEN, LLAMA]:
        print(
            f"{m['model']:<18s}  "
            f"{m['int_dr']:.1f}% [{m['int_dr_ci'][0]:.1f},{m['int_dr_ci'][1]:.1f}]  "
            f"{m['det_dr']:.1f}% [{m['det_dr_ci'][0]:.1f},{m['det_dr_ci'][1]:.1f}]  "
            f"+{m['delta_dr']:.1f}pp [{m['delta_dr_ci'][0]:.1f},{m['delta_dr_ci'][1]:.1f}]  "
            f"{m['daf']:.1f}% [{m['daf_ci'][0]:.1f},{m['daf_ci'][1]:.1f}]"
        )
    g_int_dr = fmt_ci(agg["int_dr"], cis["int_dr"][0], cis["int_dr"][1])
    g_det_dr = fmt_ci(agg["det_dr"], cis["det_dr"][0], cis["det_dr"][1])
    g_delta = fmt_ci(agg["delta_dr"], cis["delta_dr"][0], cis["delta_dr"][1], unit="pp")
    if agg["daf"] is not None:
        g_daf = fmt_ci(agg["daf"], cis["daf"][0], cis["daf"][1])
    else:
        g_daf = "undef"
    print(f"{'Gemma-2-9B':<18s}  {g_int_dr:>22s}  {g_det_dr:>22s}  {g_delta:>24s}  {g_daf:>22s}")


def save_json(agg, cis, rows, out_dir):
    data = {
        "aggregate": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in agg.items()},
        "bootstrap_95ci": {k: [round(v[0], 2), round(v[1], 2)] for k, v in cis.items()},
        "per_task": rows,
    }
    path = os.path.join(out_dir, "analysis.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {path}")


def save_latex(agg, cis, out_dir):
    def row(name, int_dr, int_ci, det_dr, det_ci, ddr, ddr_ci, daf, daf_ci):
        return (
            f"  {name} & "
            f"{int_dr:.1f}\\% [{int_ci[0]:.1f},{int_ci[1]:.1f}] & "
            f"{det_dr:.1f}\\% [{det_ci[0]:.1f},{det_ci[1]:.1f}] & "
            f"+{ddr:.1f}pp [{ddr_ci[0]:.1f},{ddr_ci[1]:.1f}] & "
            f"{daf:.1f}\\% [{daf_ci[0]:.1f},{daf_ci[1]:.1f}] \\\\"
        )
    lines = [
        "% Cross-model ScienceWorld counterfactual comparison",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Model & Int DR & Det DR & $\\Delta$DR & DAF \\\\",
        "\\midrule",
        row("Qwen2.5-7B", QWEN["int_dr"], QWEN["int_dr_ci"], QWEN["det_dr"], QWEN["det_dr_ci"],
            QWEN["delta_dr"], QWEN["delta_dr_ci"], QWEN["daf"], QWEN["daf_ci"]),
        row("Llama-3.1-8B", LLAMA["int_dr"], LLAMA["int_dr_ci"], LLAMA["det_dr"], LLAMA["det_dr_ci"],
            LLAMA["delta_dr"], LLAMA["delta_dr_ci"], LLAMA["daf"], LLAMA["daf_ci"]),
    ]
    if agg["daf"] is not None:
        lines.append(row("Gemma-2-9B", agg["int_dr"], cis["int_dr"], agg["det_dr"], cis["det_dr"],
                         agg["delta_dr"], cis["delta_dr"], agg["daf"], cis["daf"]))
    else:
        lines.append(
            f"  Gemma-2-9B & "
            f"{agg['int_dr']:.1f}\\% [{cis['int_dr'][0]:.1f},{cis['int_dr'][1]:.1f}] & "
            f"{agg['det_dr']:.1f}\\% [{cis['det_dr'][0]:.1f},{cis['det_dr'][1]:.1f}] & "
            f"+{agg['delta_dr']:.1f}pp [{cis['delta_dr'][0]:.1f},{cis['delta_dr'][1]:.1f}] & "
            f"--- \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    path = os.path.join(out_dir, "cross_model_table.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Gemma-2-9B ScienceWorld counterfactual analysis")
    parser.add_argument("--input", default="results/gemma-9b-counterfactual/",
                        help="Results directory (default: results/gemma-9b-counterfactual/)")
    parser.add_argument("--bootstrap_samples", type=int, default=10000, help="Bootstrap iterations (default: 10000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    episodes = load_episodes(args.input)
    print(f"Loaded {len(episodes)} episodes from {args.input}")
    if len(episodes) == 0:
        print("ERROR: no episodes found. Check --input path.")
        return

    by_task = defaultdict(int)
    for ep in episodes:
        by_task[ep["task_type"]] += 1
    for task in INCLUDED_TASKS:
        cnt = by_task.get(task, 0)
        if cnt != 20:
            print(f"  WARNING: {task} has {cnt}/20 episodes")

    rows = per_task_breakdown(episodes)
    print_per_task(rows)

    agg = compute_aggregate(episodes)
    print_aggregate(agg)

    print(f"\nRunning cluster bootstrap (B={args.bootstrap_samples}, seed={args.seed})...")
    cis = {}
    cis["int_dr"] = cluster_bootstrap(episodes, stat_int_dr, B=args.bootstrap_samples, seed=args.seed)
    cis["det_dr"] = cluster_bootstrap(episodes, stat_det_dr, B=args.bootstrap_samples, seed=args.seed)
    cis["delta_dr"] = cluster_bootstrap(episodes, stat_delta_dr, B=args.bootstrap_samples, seed=args.seed)
    cis["daf"] = cluster_bootstrap(episodes, stat_daf, B=args.bootstrap_samples, seed=args.seed)
    print_bootstrap(agg, cis)

    print_cross_model(agg, cis)

    save_json(agg, cis, rows, args.input)
    save_latex(agg, cis, args.input)


if __name__ == "__main__":
    main()
