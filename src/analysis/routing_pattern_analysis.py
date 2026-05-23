"""Plan-005: Routing pattern analysis.

Analyzes oracle routing decisions from plan-003 eval and plan_c raw data.
Reports per-task patterns, death correlations, and router-vs-oracle comparison.
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = BASE / "results" / "plan_003_eval"
RAW_DIR = BASE / "results" / "plan_c"
OUT_DIR = BASE / "results" / "plan_005_analysis"


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_eval_episodes(eval_dir):
    return load_jsonl(eval_dir / "episode_details.jsonl")


def load_raw_summaries(raw_dir):
    summaries = {}
    for task_dir in sorted(raw_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        summary_path = task_dir / "episode_summary.jsonl"
        if summary_path.exists():
            summaries[task_dir.name] = load_jsonl(summary_path)
    return summaries


def load_raw_steps(raw_dir):
    steps = {}
    for task_dir in sorted(raw_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        steps_path = task_dir / "labeled_steps.jsonl"
        if steps_path.exists():
            steps[task_dir.name] = load_jsonl(steps_path)
    return steps


# ── Analysis functions ──────────────────────────────────────────────


def analyze_oracle_decisions(eval_episodes):
    """Analyze oracle label distribution overall and per task."""
    oracle_rows = [e for e in eval_episodes if e["condition"] == "oracle"]
    if not oracle_rows:
        oracle_rows = eval_episodes

    overall = Counter(e["oracle_label"] for e in oracle_rows)
    n = len(oracle_rows)

    per_task = defaultdict(lambda: Counter())
    for e in oracle_rows:
        per_task[e["task_type"]][e["oracle_label"]] += 1

    per_task_rates = {}
    for task, counts in sorted(per_task.items()):
        t = sum(counts.values())
        per_task_rates[task] = {
            label: {"count": c, "rate": round(c / t, 4)}
            for label, c in counts.most_common()
        }
        per_task_rates[task]["_n"] = t

    return {
        "total_episodes": n,
        "overall_distribution": {
            label: {"count": c, "rate": round(c / n, 4)}
            for label, c in overall.most_common()
        },
        "per_task": per_task_rates,
    }


def analyze_death_vs_decisions(eval_episodes):
    """Split episodes by death status and compare oracle decisions."""
    oracle_rows = [e for e in eval_episodes if e["condition"] == "oracle"]
    if not oracle_rows:
        oracle_rows = eval_episodes

    death_eps = [e for e in oracle_rows if e["internal_final"] == -100 or e["det_final"] == -100]
    alive_eps = [e for e in oracle_rows if e["internal_final"] != -100 and e["det_final"] != -100]

    def dist(rows):
        c = Counter(e["oracle_label"] for e in rows)
        n = len(rows)
        if n == 0:
            return {"n": 0}
        return {
            "n": n,
            "distribution": {
                label: {"count": cnt, "rate": round(cnt / n, 4)}
                for label, cnt in c.most_common()
            },
        }

    int_died = [e for e in oracle_rows if e["internal_final"] == -100 and e["det_final"] != -100]
    det_died = [e for e in oracle_rows if e["det_final"] == -100 and e["internal_final"] != -100]
    both_died = [e for e in oracle_rows if e["internal_final"] == -100 and e["det_final"] == -100]

    return {
        "any_death": dist(death_eps),
        "no_death": dist(alive_eps),
        "internal_died_only": dist(int_died),
        "det_died_only": dist(det_died),
        "both_died": dist(both_died),
    }


def analyze_signal_sources(raw_summaries):
    """Analyze signal_source and reason distributions from raw episode summaries."""
    all_signal = Counter()
    all_reason = Counter()
    per_task_signal = {}

    for task, eps in sorted(raw_summaries.items()):
        task_signal = Counter()
        task_reason = Counter()
        for ep in eps:
            src = ep.get("signal_source", "unknown")
            reason = ep.get("reason", "unknown")
            all_signal[src] += 1
            all_reason[reason] += 1
            task_signal[src] += 1
            task_reason[reason] += 1

        n = len(eps)
        per_task_signal[task] = {
            "n_episodes": n,
            "signal_source": {k: round(v / n, 4) for k, v in task_signal.most_common()},
            "reason": {k: round(v / n, 4) for k, v in task_reason.most_common()},
        }

    total = sum(all_signal.values())
    return {
        "overall_signal_source": {k: {"count": v, "rate": round(v / total, 4)} for k, v in all_signal.most_common()},
        "overall_reason": {k: {"count": v, "rate": round(v / total, 4)} for k, v in all_reason.most_common()},
        "per_task": per_task_signal,
    }


def analyze_step_patterns(raw_steps):
    """Analyze step-level label distribution and step position patterns."""
    all_labels = Counter()
    all_signals = Counter()
    step_position_labels = defaultdict(lambda: Counter())

    per_task = {}
    for task, steps in sorted(raw_steps.items()):
        task_labels = Counter()
        task_signals = Counter()
        step_counts = []
        ep_ids = set()

        for s in steps:
            label = s.get("label", "unknown")
            sig = s.get("signal_source", "unknown")
            task_labels[label] += 1
            task_signals[sig] += 1
            all_labels[label] += 1
            all_signals[sig] += 1
            ep_ids.add(s.get("episode_id", s.get("episode", -1)))

            idx = s.get("step_idx", 0)
            bucket = "early" if idx < 5 else "mid" if idx < 15 else "late"
            step_position_labels[bucket][label] += 1

        n_steps = len(steps)
        per_task[task] = {
            "n_steps": n_steps,
            "n_episodes": len(ep_ids),
            "label_dist": {k: round(v / n_steps, 4) for k, v in task_labels.most_common()},
            "signal_dist": {k: round(v / n_steps, 4) for k, v in task_signals.most_common()},
        }

    total = sum(all_labels.values())
    position_summary = {}
    for bucket, counts in step_position_labels.items():
        bt = sum(counts.values())
        position_summary[bucket] = {k: round(v / bt, 4) for k, v in counts.most_common()}

    return {
        "total_steps": total,
        "overall_label_dist": {k: {"count": v, "rate": round(v / total, 4)} for k, v in all_labels.most_common()},
        "step_position_patterns": position_summary,
        "per_task": per_task,
    }


def analyze_death_step_timing(raw_summaries):
    """Analyze when death occurs (step count) by mode."""
    int_death_steps = []
    det_death_steps = []

    for task, eps in raw_summaries.items():
        for ep in eps:
            if ep.get("internal_final", 0) == -100:
                int_death_steps.append(ep.get("internal_steps", 0))
            if ep.get("det_final", 0) == -100:
                det_death_steps.append(ep.get("det_steps", 0))

    def stats(vals):
        if not vals:
            return {"n": 0}
        import statistics
        return {
            "n": len(vals),
            "mean": round(statistics.mean(vals), 2),
            "median": round(statistics.median(vals), 2),
            "min": min(vals),
            "max": max(vals),
        }

    return {
        "internal_death_steps": stats(int_death_steps),
        "det_death_steps": stats(det_death_steps),
    }


def analyze_router_vs_oracle(eval_episodes):
    """Compare trained-router decisions against oracle labels."""
    oracle_map = {}
    for e in eval_episodes:
        if e["condition"] == "oracle":
            key = (e["task_type"], e["episode_idx"])
            oracle_map[key] = e

    router_rows = [e for e in eval_episodes if e["condition"] == "trained-router"]
    if not router_rows:
        return {"available": False}

    agree = 0
    disagree_details = defaultdict(int)
    n = 0
    per_task_agreement = defaultdict(lambda: {"agree": 0, "total": 0})

    for r in router_rows:
        key = (r["task_type"], r["episode_idx"])
        o = oracle_map.get(key)
        if o is None:
            continue
        n += 1
        router_decision = r["decision"]
        oracle_label = o["oracle_label"]

        oracle_decision = "deterministic"
        if oracle_label == "internal-preferred":
            oracle_decision = "internal"
        elif oracle_label == "no-preference":
            oracle_decision = router_decision

        if router_decision == oracle_decision:
            agree += 1
            per_task_agreement[r["task_type"]]["agree"] += 1
        else:
            disagree_details[f"router={router_decision},oracle={oracle_label}"] += 1
        per_task_agreement[r["task_type"]]["total"] += 1

    per_task_rates = {}
    for task, v in sorted(per_task_agreement.items()):
        per_task_rates[task] = {
            "agreement_rate": round(v["agree"] / v["total"], 4) if v["total"] > 0 else None,
            "n": v["total"],
        }

    return {
        "available": True,
        "n_compared": n,
        "agreement_rate": round(agree / n, 4) if n > 0 else None,
        "disagreements": dict(disagree_details),
        "per_task_agreement": per_task_rates,
    }


def compute_score_gap_analysis(raw_summaries):
    """Analyze score gap distribution for episodes with a preference."""
    gaps = []
    per_task = {}

    for task, eps in sorted(raw_summaries.items()):
        task_gaps = []
        for ep in eps:
            if ep.get("label") != "no-preference":
                task_gaps.append(abs(ep.get("score_gap", 0)))
        gaps.extend(task_gaps)

        if task_gaps:
            import statistics
            per_task[task] = {
                "n": len(task_gaps),
                "mean_gap": round(statistics.mean(task_gaps), 2),
                "median_gap": round(statistics.median(task_gaps), 2),
            }

    if gaps:
        import statistics
        overall = {
            "n": len(gaps),
            "mean_gap": round(statistics.mean(gaps), 2),
            "median_gap": round(statistics.median(gaps), 2),
            "pct_gap_100": round(sum(1 for g in gaps if g >= 100) / len(gaps), 4),
        }
    else:
        overall = {"n": 0}

    return {"overall": overall, "per_task": per_task}


# ── Text summary ────────────────────────────────────────────────────


def generate_text_summary(report):
    lines = []
    lines.append("=" * 60)
    lines.append("Plan-005 Routing Pattern Analysis Summary")
    lines.append("=" * 60)

    od = report.get("oracle_decisions", {})
    lines.append(f"\nTotal episodes analyzed: {od.get('total_episodes', '?')}")
    lines.append("\nOracle decision distribution (overall):")
    for label, info in od.get("overall_distribution", {}).items():
        lines.append(f"  {label}: {info['count']} ({info['rate']:.1%})")

    lines.append("\nOracle decisions per task:")
    for task, info in sorted(od.get("per_task", {}).items()):
        parts = []
        for label in ["det-preferred", "internal-preferred", "no-preference"]:
            if label in info:
                parts.append(f"{label}={info[label]['rate']:.0%}")
        lines.append(f"  {task}: {', '.join(parts)}")

    dv = report.get("death_vs_decisions", {})
    lines.append("\nDeath-related decision breakdown:")
    for category in ["internal_died_only", "det_died_only", "both_died", "no_death"]:
        info = dv.get(category, {})
        n = info.get("n", 0)
        if n > 0:
            dist_str = ", ".join(
                f"{l}={d['rate']:.0%}" for l, d in info.get("distribution", {}).items()
            )
            lines.append(f"  {category} (n={n}): {dist_str}")

    ss = report.get("signal_sources", {})
    lines.append("\nSignal source distribution (from raw episodes):")
    for src, info in ss.get("overall_signal_source", {}).items():
        lines.append(f"  {src}: {info['count']} ({info['rate']:.1%})")

    dt = report.get("death_step_timing", {})
    for mode in ["internal_death_steps", "det_death_steps"]:
        info = dt.get(mode, {})
        if info.get("n", 0) > 0:
            lines.append(f"\n{mode}: mean={info['mean']}, median={info['median']}, range=[{info['min']}, {info['max']}]")

    sg = report.get("score_gap_analysis", {}).get("overall", {})
    if sg.get("n", 0) > 0:
        lines.append(f"\nScore gap (episodes with preference): mean={sg['mean_gap']}, median={sg['median_gap']}, %>=100: {sg['pct_gap_100']:.1%}")

    rv = report.get("router_vs_oracle", {})
    if rv.get("available"):
        lines.append(f"\nRouter vs Oracle: agreement={rv['agreement_rate']:.1%} (n={rv['n_compared']})")
        if rv.get("disagreements"):
            lines.append("  Disagreements:")
            for k, v in rv["disagreements"].items():
                lines.append(f"    {k}: {v}")
    else:
        lines.append("\nTrained router data: not available for comparison")

    sp = report.get("step_patterns", {})
    lines.append("\nStep position patterns:")
    for bucket in ["early", "mid", "late"]:
        info = sp.get("step_position_patterns", {}).get(bucket, {})
        if info:
            parts = ", ".join(f"{l}={r:.0%}" for l, r in info.items())
            lines.append(f"  {bucket}: {parts}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


# ── Main ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Plan-005 routing pattern analysis")
    parser.add_argument("--eval-dir", type=str, default=str(EVAL_DIR),
                        help="Path to plan_003_eval results directory")
    parser.add_argument("--raw-dir", type=str, default=str(RAW_DIR),
                        help="Path to plan_c raw episode data directory")
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR),
                        help="Output directory for analysis results")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print summary only, don't write files")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    # Load data
    eval_episodes = []
    if (eval_dir / "episode_details.jsonl").exists():
        eval_episodes = load_eval_episodes(eval_dir)
        print(f"Loaded {len(eval_episodes)} eval episode records")
    else:
        print(f"WARNING: {eval_dir / 'episode_details.jsonl'} not found")

    raw_summaries = {}
    raw_steps = {}
    if raw_dir.exists():
        raw_summaries = load_raw_summaries(raw_dir)
        raw_steps = load_raw_steps(raw_dir)
        print(f"Loaded raw data for {len(raw_summaries)} tasks")
    else:
        print(f"WARNING: {raw_dir} not found")

    # Run analyses
    report = {}

    if eval_episodes:
        report["oracle_decisions"] = analyze_oracle_decisions(eval_episodes)
        report["death_vs_decisions"] = analyze_death_vs_decisions(eval_episodes)
        report["router_vs_oracle"] = analyze_router_vs_oracle(eval_episodes)

    if raw_summaries:
        report["signal_sources"] = analyze_signal_sources(raw_summaries)
        report["death_step_timing"] = analyze_death_step_timing(raw_summaries)
        report["score_gap_analysis"] = compute_score_gap_analysis(raw_summaries)

    if raw_steps:
        report["step_patterns"] = analyze_step_patterns(raw_steps)

    # Generate summary
    summary = generate_text_summary(report)
    print("\n" + summary)

    if args.dry_run:
        print("\n[dry-run] Skipping file output.")
        return

    # Write outputs
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "routing_patterns.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] {report_path}")

    summary_path = out_dir / "routing_patterns_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"[Saved] {summary_path}")


if __name__ == "__main__":
    main()
