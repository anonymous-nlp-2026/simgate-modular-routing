#!/usr/bin/env python3
"""Dose-response curve for reflexion baseline on death-prone tasks.

Runs reflexion (with self-reflection) with varying N trials per episode.
"""

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from reflexion_baseline import APIAgent, run_trial_loop, resolve_task_type

DEATH_PRONE_TASKS = [
    "chemistry-mix", "find-animal", "freeze",
    "grow-fruit", "identify-life-stages-1", "identify-life-stages-2",
    "inclined-plane-determine-angle", "lifespan-longest-lived",
    "lifespan-longest-lived-then-shortest-lived", "measure-melting-point-unknown-substance",
]


def run_dose_condition(agent, n_trials, tasks, resolved_tasks, episodes_per_task,
                       max_steps, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    condition_stats = {"n_trials": n_trials, "tasks": {}}

    for task_short, task_resolved in zip(tasks, resolved_tasks):
        print(f"\n{'='*60}")
        print(f"[reflexion N={n_trials}] Task: {task_short}")
        print(f"{'='*60}")

        deaths = 0
        successes = 0
        total_trials_used = 0

        for ep_idx in range(episodes_per_task):
            variation_idx = ep_idx % 100
            print(f"  Ep {ep_idx+1}/{episodes_per_task} (var={variation_idx}) ",
                  end="", flush=True)

            result = run_trial_loop(
                agent, task_resolved, variation_idx,
                n_trials, max_steps, "reflexion",
            )

            status = (
                "SUCCESS" if result["final_is_success"]
                else ("DEATH" if result["final_is_death"] else "FAIL")
            )
            print(f"-> {status} (score={result['final_score']:.0f}, "
                  f"trials={result['num_trials']})")

            if result["final_is_death"]:
                deaths += 1
            if result["final_is_success"]:
                successes += 1
            total_trials_used += result["num_trials"]

            ep_file = os.path.join(output_dir, f"{task_short}_ep{ep_idx:03d}.json")
            with open(ep_file, "w") as f:
                json.dump(result, f, indent=2)

        n = episodes_per_task
        task_stats = {
            "task_short": task_short,
            "n_episodes": n,
            "deaths": deaths,
            "successes": successes,
            "death_rate": round(deaths / n, 4) if n > 0 else 0,
            "success_rate": round(successes / n, 4) if n > 0 else 0,
            "avg_trials": round(total_trials_used / n, 2) if n > 0 else 0,
        }
        condition_stats["tasks"][task_short] = task_stats
        print(f"  DR={task_stats['death_rate']:.1%}, SR={task_stats['success_rate']:.1%}")

    total_eps = sum(s["n_episodes"] for s in condition_stats["tasks"].values())
    total_deaths = sum(s["deaths"] for s in condition_stats["tasks"].values())
    total_successes = sum(s["successes"] for s in condition_stats["tasks"].values())
    condition_stats["aggregate"] = {
        "total_episodes": total_eps,
        "total_deaths": total_deaths,
        "total_successes": total_successes,
        "death_rate": round(total_deaths / total_eps, 4) if total_eps > 0 else 0,
        "success_rate": round(total_successes / total_eps, 4) if total_eps > 0 else 0,
    }
    return condition_stats


def main():
    parser = argparse.ArgumentParser(
        description="Reflexion dose-response curve on death-prone tasks")
    parser.add_argument("--n-values", type=str, default="1,2,3,4,5",
                        help="Comma-separated reflexion N values (default: 1,2,3,4,5)")
    parser.add_argument("--episodes-per-task", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--model", type=str, default="Qwen2.5-7B-Instruct")
    parser.add_argument("--api-base", type=str, default="http://localhost:8001/v1")
    parser.add_argument("--output-dir", type=str,
                        default="results/reflexion_dose_response/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    n_values = [int(x.strip()) for x in args.n_values.split(",")]
    tasks = list(DEATH_PRONE_TASKS)
    resolved_tasks = [resolve_task_type(t) for t in tasks]

    total_episodes = len(tasks) * args.episodes_per_task * len(n_values)
    print(f"Reflexion dose-response plan: N={n_values}, {len(tasks)} tasks, "
          f"{args.episodes_per_task} eps/task")
    print(f"Total episodes: {total_episodes}")

    if args.dry_run:
        for n in n_values:
            print(f"  N={n}: {len(tasks) * args.episodes_per_task} episodes, "
                  f"up to {len(tasks) * args.episodes_per_task * n} trials")
        return

    random.seed(args.seed)
    agent = APIAgent(model=args.model, api_base=args.api_base)

    dose_response = {
        "n_values": n_values,
        "seed": args.seed,
        "model": args.model,
        "max_steps": args.max_steps,
        "episodes_per_task": args.episodes_per_task,
        "tasks": list(tasks),
        "conditions": {},
    }

    for n in n_values:
        print(f"\n{'#'*60}")
        print(f"# CONDITION: reflexion N={n}")
        print(f"{'#'*60}")

        cond_dir = os.path.join(args.output_dir, f"reflexion_n{n}")
        stats = run_dose_condition(
            agent, n, tasks, resolved_tasks,
            args.episodes_per_task, args.max_steps, cond_dir,
        )
        dose_response["conditions"][f"n{n}"] = stats

        cond_summary = os.path.join(cond_dir, "condition_summary.json")
        with open(cond_summary, "w") as f:
            json.dump(stats, f, indent=2)

        print(f"\n[reflexion N={n}] Aggregate: DR={stats['aggregate']['death_rate']:.1%}")

    summary_file = os.path.join(args.output_dir, "dose_response_summary.json")
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    with open(summary_file, "w") as f:
        json.dump(dose_response, f, indent=2)

    print(f"\n{'='*60}")
    print("REFLEXION DOSE-RESPONSE SUMMARY")
    print(f"{'='*60}")
    for n in n_values:
        agg = dose_response["conditions"][f"n{n}"]["aggregate"]
        print(f"  N={n:2d}: DR={agg['death_rate']:.1%} "
              f"({agg['total_deaths']}/{agg['total_episodes']})")
    print(f"\nSummary saved to {summary_file}")


if __name__ == "__main__":
    main()
