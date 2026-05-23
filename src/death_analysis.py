import json, os, statistics, math
from collections import Counter


def classify_signal(delta, delta_no_death, i_death_pct, d_death_pct):
    if i_death_pct <= 5 and d_death_pct <= 5 and abs(delta) < 2:
        return "no-effect"
    death_diff = i_death_pct - d_death_pct
    if math.isnan(delta_no_death):
        if death_diff > 10:
            return "death-dominated"
        return "internal-preferred" if delta < 0 else "mixed"
    if death_diff > 10 and delta_no_death < 0:
        return "death-dominated"
    if death_diff > 10 and delta_no_death > 0:
        return "mixed"
    if abs(death_diff) <= 10 and delta > 5:
        return "genuine-det"
    if abs(death_diff) <= 10 and delta < -5:
        return "internal-preferred"
    return "mixed"


def analyze_dir(base_dir, label):
    tasks = sorted([d for d in os.listdir(base_dir)
                    if os.path.isdir(os.path.join(base_dir, d))])

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    header = (f"| {'Task':<40} | {'Int':>6} | {'Det':>6} | {'Δ':>6} "
              f"| {'I_death%':>8} | {'D_death%':>8} "
              f"| {'Int_nd':>7} | {'Det_nd':>7} | {'Δ_nd':>6} "
              f"| {'Signal_type':<18} |")
    sep = (f"|{'-'*42}|{'-'*8}|{'-'*8}|{'-'*8}"
           f"|{'-'*10}|{'-'*10}"
           f"|{'-'*9}|{'-'*9}|{'-'*8}"
           f"|{'-'*20}|")
    print(header)
    print(sep)

    all_i, all_d = [], []
    all_i_nd, all_d_nd = [], []
    signal_counts = Counter()
    task_results = []

    for task in tasks:
        ep_file = os.path.join(base_dir, task, "episodes.jsonl")
        if not os.path.exists(ep_file):
            continue

        with open(ep_file) as f:
            data = [json.loads(line) for line in f if line.strip()]

        i_scores, d_scores = [], []
        for ep in data:
            ai = ep.get("always_internal", {})
            ad = ep.get("always_deterministic", {})
            if ai:
                i_scores.append(ai.get("final_score", 0))
            if ad:
                d_scores.append(ad.get("final_score", 0))

        if not i_scores or not d_scores:
            continue

        n = len(i_scores)
        i_death = sum(1 for s in i_scores if s <= -100)
        d_death = sum(1 for s in d_scores if s <= -100)
        i_death_pct = 100 * i_death / n
        d_death_pct = 100 * d_death / n

        i_mean = statistics.mean(i_scores)
        d_mean = statistics.mean(d_scores)
        delta = d_mean - i_mean

        i_nd = [s for s in i_scores if s > -100]
        d_nd = [s for s in d_scores if s > -100]
        i_nd_mean = statistics.mean(i_nd) if i_nd else float('nan')
        d_nd_mean = statistics.mean(d_nd) if d_nd else float('nan')
        delta_nd = (d_nd_mean - i_nd_mean) if i_nd and d_nd else float('nan')

        all_i.extend(i_scores)
        all_d.extend(d_scores)
        all_i_nd.extend(i_nd)
        all_d_nd.extend(d_nd)

        signal = classify_signal(delta, delta_nd, i_death_pct, d_death_pct)
        signal_counts[signal] += 1

        delta_nd_str = f"{delta_nd:+.1f}" if not math.isnan(delta_nd) else "N/A"
        i_nd_str = f"{i_nd_mean:.1f}" if not math.isnan(i_nd_mean) else "N/A"
        d_nd_str = f"{d_nd_mean:.1f}" if not math.isnan(d_nd_mean) else "N/A"

        print(f"| {task:<40} | {i_mean:6.1f} | {d_mean:6.1f} | {delta:+6.1f} "
              f"| {i_death_pct:7.0f}% | {d_death_pct:7.0f}% "
              f"| {i_nd_str:>7s} | {d_nd_str:>7s} | {delta_nd_str:>6s} "
              f"| {signal:<18s} |")

        task_results.append({
            "task": task,
            "int_mean": round(i_mean, 2),
            "det_mean": round(d_mean, 2),
            "delta": round(delta, 2),
            "int_death_pct": round(i_death_pct, 1),
            "det_death_pct": round(d_death_pct, 1),
            "int_nd_mean": round(i_nd_mean, 2) if not math.isnan(i_nd_mean) else None,
            "det_nd_mean": round(d_nd_mean, 2) if not math.isnan(d_nd_mean) else None,
            "delta_nd": round(delta_nd, 2) if not math.isnan(delta_nd) else None,
            "signal_type": signal,
            "n_episodes": n,
        })

    # Aggregates
    if all_i and all_d:
        n_total = len(all_i)
        i_death_total = sum(1 for s in all_i if s <= -100)
        d_death_total = sum(1 for s in all_d if s <= -100)
        i_mean_all = statistics.mean(all_i)
        d_mean_all = statistics.mean(all_d)
        delta_all = d_mean_all - i_mean_all

        i_nd_mean_all = statistics.mean(all_i_nd) if all_i_nd else float('nan')
        d_nd_mean_all = statistics.mean(all_d_nd) if all_d_nd else float('nan')
        delta_nd_all = (d_nd_mean_all - i_nd_mean_all) if all_i_nd and all_d_nd else float('nan')

        delta_nd_str = f"{delta_nd_all:+.1f}" if not math.isnan(delta_nd_all) else "N/A"
        i_nd_str = f"{i_nd_mean_all:.1f}" if not math.isnan(i_nd_mean_all) else "N/A"
        d_nd_str = f"{d_nd_mean_all:.1f}" if not math.isnan(d_nd_mean_all) else "N/A"

        print(sep)
        print(f"| {'AGGREGATE':<40} | {i_mean_all:6.1f} | {d_mean_all:6.1f} | {delta_all:+6.1f} "
              f"| {100*i_death_total/n_total:7.0f}% | {100*d_death_total/n_total:7.0f}% "
              f"| {i_nd_str:>7s} | {d_nd_str:>7s} | {delta_nd_str:>6s} "
              f"| {'---':<18s} |")

        print(f"\n  Death rate: Internal={i_death_total}/{n_total} ({100*i_death_total/n_total:.1f}%)  "
              f"Det={d_death_total}/{n_total} ({100*d_death_total/n_total:.1f}%)")
        print(f"  Overall Δ(D-I)={delta_all:+.1f}  |  No-death Δ(D-I)={delta_nd_str}")

    print(f"\n  Signal type distribution:")
    for st in ["no-effect", "death-dominated", "genuine-det", "internal-preferred", "mixed"]:
        print(f"    {st}: {signal_counts.get(st, 0)} tasks")

    return {
        "label": label,
        "tasks": task_results,
        "signal_distribution": {st: signal_counts.get(st, 0)
                                for st in ["no-effect", "death-dominated", "genuine-det", "internal-preferred", "mixed"]},
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="Single data directory to analyze")
    parser.add_argument("--output", help="Output JSON path")
    args = parser.parse_args()

    os.chdir(".")

    if args.data_dir:
        label = os.path.basename(args.data_dir.rstrip("/")).upper() + " SIGNAL TYPE ANALYSIS"
        result = analyze_dir(args.data_dir, label)
        out_path = args.output or os.path.join(args.data_dir, "signal_type_summary.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n[Saved summary to {out_path}]")
    else:
        results = {}
        results["prescreening"] = analyze_dir("results/prescreening",
                                              "PRESCREENING SIGNAL TYPE ANALYSIS")
        results["mvp"] = analyze_dir("results/mvp",
                                     "MVP SIGNAL TYPE ANALYSIS")
        out_path = "results/prescreening/signal_type_summary.json"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[Saved summary to {out_path}]")
