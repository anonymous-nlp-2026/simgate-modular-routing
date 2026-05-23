"""Step-level tie-rate analysis for plan_c (245 episodes / 11 tasks).

Compares score_delta at each step index between always_internal and
always_deterministic runs within each episode. A 'tie' means both
modalities produce the same score_delta at that step.
"""
import json, os
from collections import defaultdict

DATA_DIR = "./results/plan_c"

def load_episodes(task):
    path = os.path.join(DATA_DIR, task, "episodes.jsonl")
    eps = []
    with open(path) as f:
        for line in f:
            eps.append(json.loads(line))
    return eps

def pct(num, denom):
    return 100.0 * num / denom if denom > 0 else 0.0

def main():
    tasks = sorted(d for d in os.listdir(DATA_DIR)
                   if os.path.isdir(os.path.join(DATA_DIR, d)))

    results = {}
    grand_total = grand_tie = grand_det = grand_int = 0
    grand_both_zero = 0
    grand_episodes = 0

    for task in tasks:
        episodes = load_episodes(task)
        task_total = task_tie = task_det = task_int = 0
        task_both_zero = 0

        for ep in episodes:
            int_steps = ep["always_internal"]["steps"]
            det_steps = ep["always_deterministic"]["steps"]
            n = min(len(int_steps), len(det_steps))

            for i in range(n):
                id_ = int_steps[i]["score_delta"]
                dd = det_steps[i]["score_delta"]
                task_total += 1
                if id_ == dd:
                    task_tie += 1
                    if id_ == 0.0 and dd == 0.0:
                        task_both_zero += 1
                elif dd > id_:
                    task_det += 1
                else:
                    task_int += 1

        results[task] = {
            "episodes": len(episodes),
            "steps": task_total,
            "ties": task_tie,
            "tie_rate": round(pct(task_tie, task_total), 4),
            "both_zero": task_both_zero,
            "det_preferred": task_det,
            "int_preferred": task_int,
        }
        grand_total += task_total
        grand_tie += task_tie
        grand_det += task_det
        grand_int += task_int
        grand_both_zero += task_both_zero
        grand_episodes += len(episodes)

    overall = {
        "total_steps": grand_total,
        "tie_steps": grand_tie,
        "tie_rate": round(pct(grand_tie, grand_total), 4),
        "both_zero_ties": grand_both_zero,
        "nonzero_equal_ties": grand_tie - grand_both_zero,
        "det_preferred_steps": grand_det,
        "int_preferred_steps": grand_int,
        "episodes": grand_episodes,
        "tasks": len(tasks),
    }

    output = {
        "overall": overall,
        "per_task": results,
        "old_mvp_comparison": {
            "old_rate": 98.17,
            "old_steps": 2948,
            "old_ties": 2894,
            "old_scope": "5 MVP tasks (oracle routing mode)"
        },
        "methodology_note": (
            "plan_c uses independent always-internal/always-deterministic "
            "trajectories (not oracle routing). Tie = score_delta equal at "
            "same step index across both runs."
        )
    }

    # Print summary
    print("=" * 60)
    print("STEP-LEVEL TIE RATE: plan_c (245 eps / 11 tasks)")
    print("=" * 60)
    print(f"Total episodes: {grand_episodes}")
    print(f"Total steps:    {grand_total}")
    print(f"Tie steps:      {grand_tie}")
    print(f"Tie rate:       {pct(grand_tie, grand_total):.2f}% ({grand_tie}/{grand_total})")
    print(f"  both_zero:    {grand_both_zero}")
    print(f"  nonzero_eq:   {grand_tie - grand_both_zero}")
    print(f"Det preferred:  {grand_det}")
    print(f"Int preferred:  {grand_int}")
    print()

    for task in tasks:
        r = results[task]
        print(f"{task:50s}  eps={r['episodes']:3d}  steps={r['steps']:5d}  "
              f"tie={r['tie_rate']:6.2f}% ({r['ties']}/{r['steps']})  "
              f"both0={r['both_zero']}  det={r['det_preferred']}  int={r['int_preferred']}")
    print()

    # Old comparison
    print(f"Old MVP tie rate: 98.2% (2894/2948, 5 tasks, oracle routing)")
    delta = abs(pct(grand_tie, grand_total) - 98.17)
    print(f"Delta from old:   {delta:.2f}pp")

    # Save JSON
    out_path = "./results/step_tie_rate_full.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
