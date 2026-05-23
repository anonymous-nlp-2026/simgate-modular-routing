"""
Death-risk threshold routing analysis for ScienceWorld baseline_245eps.

Computes per-task-type death rates for internal vs deterministic agents,
then sweeps a threshold: if internal_death_rate > threshold, route the
entire task to deterministic; otherwise keep internal. Reports mean score,
death rate, and delta vs always-det for each threshold, finds the optimal
threshold, and bootstraps a 95% CI on its delta vs always-det (unpaired).

Input:  ./results/baseline_245eps/episode_details.jsonl
Output: Printed table of threshold sweep results, optimal threshold, bootstrap CI.
Run:    python scripts/death_risk_routing.py
"""

import json
import random
import statistics
from collections import defaultdict

DATA_PATH = "./results/baseline_245eps/episode_details.jsonl"
THRESHOLDS = [round(t * 0.1, 1) for t in range(11)]
N_BOOTSTRAP = 10000
SEED = 42


def load_episodes(path):
    with open(path) as f:
        data = [json.loads(line) for line in f if line.strip()]
    return [r for r in data if r["condition"] == "always-internal"]


def per_task_stats(episodes):
    tasks = defaultdict(list)
    for r in episodes:
        tasks[r["task_type"]].append(r)
    stats = {}
    for task, recs in sorted(tasks.items()):
        n = len(recs)
        i_deaths = sum(1 for r in recs if r["internal_final"] <= -100)
        d_deaths = sum(1 for r in recs if r["det_final"] <= -100)
        stats[task] = {
            "n": n,
            "internal_death_rate": i_deaths / n,
            "det_death_rate": d_deaths / n,
            "internal_mean": statistics.mean([r["internal_final"] for r in recs]),
            "det_mean": statistics.mean([r["det_final"] for r in recs]),
            "episodes": recs,
        }
    return stats


def route_score(episode, task_death_rates, threshold):
    task = episode["task_type"]
    if task_death_rates[task] > threshold:
        return episode["det_final"]
    else:
        return episode["internal_final"]


def evaluate_threshold(episodes, task_death_rates, threshold):
    scores = [route_score(ep, task_death_rates, threshold) for ep in episodes]
    mean_score = statistics.mean(scores)
    death_rate = sum(1 for s in scores if s <= -100) / len(scores)
    return mean_score, death_rate, scores


def bootstrap_ci_unpaired(routed_scores, det_scores, n_resamples=10000,
                          confidence=0.95, seed=42):
    rng = random.Random(seed)
    n = len(routed_scores)
    deltas = []
    for _ in range(n_resamples):
        r_sample = [routed_scores[rng.randint(0, n - 1)] for _ in range(n)]
        d_sample = [det_scores[rng.randint(0, n - 1)] for _ in range(n)]
        deltas.append(statistics.mean(r_sample) - statistics.mean(d_sample))
    deltas.sort()
    alpha = 1 - confidence
    lo = deltas[int(n_resamples * alpha / 2)]
    hi = deltas[int(n_resamples * (1 - alpha / 2))]
    return lo, hi


def main():
    episodes = load_episodes(DATA_PATH)
    task_stats = per_task_stats(episodes)

    always_det = statistics.mean([r["det_final"] for r in episodes])
    always_int = statistics.mean([r["internal_final"] for r in episodes])

    print(f"Episodes: {len(episodes)}")
    print(f"Tasks:    {len(task_stats)}")
    print(f"always-det mean:      {always_det:+.2f}")
    print(f"always-internal mean: {always_int:+.2f}")

    print(f"\n{'Task':<50s} {'N':>4s} {'I_death%':>9s} {'D_death%':>9s} "
          f"{'I_mean':>8s} {'D_mean':>8s}")
    print("-" * 92)
    for task, s in task_stats.items():
        print(f"{task:<50s} {s['n']:4d} "
              f"{s['internal_death_rate']*100:8.0f}% "
              f"{s['det_death_rate']*100:8.0f}% "
              f"{s['internal_mean']:+8.1f} {s['det_mean']:+8.1f}")

    task_death_rates = {t: s["internal_death_rate"] for t, s in task_stats.items()}

    print(f"\n{'Thresh':>6s} {'Routed':>7s} {'Mean':>8s} {'Death%':>7s} "
          f"{'Delta_det':>10s} {'Routing decisions'}")
    print("-" * 100)

    best_thresh = None
    best_score = float("-inf")
    best_routed_scores = None

    for thresh in THRESHOLDS:
        mean_score, death_rate, scores = evaluate_threshold(
            episodes, task_death_rates, thresh)
        delta_det = mean_score - always_det

        routed_to_det = [t for t, dr in task_death_rates.items() if dr > thresh]
        routed_to_int = [t for t, dr in task_death_rates.items() if dr <= thresh]
        n_det = sum(task_stats[t]["n"] for t in routed_to_det)
        n_int = sum(task_stats[t]["n"] for t in routed_to_int)
        routing_str = (f"{len(routed_to_det)} det, {len(routed_to_int)} int "
                       f"({n_det}+{n_int} eps)")

        marker = ""
        if mean_score > best_score:
            best_score = mean_score
            best_thresh = thresh
            best_routed_scores = scores
            marker = " <-- best"

        print(f"{thresh:6.1f} {n_det:3d}+{n_int:3d} {mean_score:+8.2f} "
              f"{death_rate*100:6.1f}% {delta_det:+10.2f}   {routing_str}{marker}")

    print(f"\nOptimal threshold: {best_thresh}")
    print(f"Optimal mean score: {best_score:+.2f}")
    print(f"Delta vs always-det: {best_score - always_det:+.2f}")

    print(f"\nRouting at threshold={best_thresh}:")
    for task in sorted(task_death_rates):
        dr = task_death_rates[task]
        route = "det" if dr > best_thresh else "INTERNAL"
        print(f"  {task:<50s} death_rate={dr:.2f}  -> {route}")

    det_scores = [r["det_final"] for r in episodes]
    lo, hi = bootstrap_ci_unpaired(best_routed_scores, det_scores,
                                   N_BOOTSTRAP, seed=SEED)
    print(f"\nBootstrap 95% CI on delta vs always-det "
          f"({N_BOOTSTRAP} resamples, unpaired): [{lo:+.2f}, {hi:+.2f}]")


if __name__ == "__main__":
    main()
