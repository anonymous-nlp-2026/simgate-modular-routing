"""Main entry point for Decomposition-Guided Router evaluation.

Usage:
  python -m guided_router.run_guided_router --variant oracle
  python -m guided_router.run_guided_router --variant learned
  python -m guided_router.run_guided_router --variant oracle --verbose

Output: results/guided_router/{variant}_results.json

Baselines for comparison:
  - always-deterministic: episode-level bal_acc = 50%
  - trained 1.7B router: episode-level bal_acc = 47.1%
  - trained 4B router: episode-level bal_acc = 58.96%
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guided_router.router import GuidedRouter
from guided_router.eval_episode import load_episodes, evaluate_router
from guided_router.config import (
    PROJECT_ROOT, COUNTERFACTUAL_DATA,
    DEATH_RISK_TASKS, NON_DEATH_RISK_TASKS, DEATH_THRESHOLD,
)


BASELINES = {
    "always-deterministic": 0.50,
    "trained-1.7B": 0.471,
    "trained-4B": 0.5896,
}


def main():
    parser = argparse.ArgumentParser(
        description="Decomposition-Guided Router: task-type death risk routing")
    parser.add_argument("--variant", choices=["oracle", "learned"], default="oracle",
                        help="Router variant (default: oracle)")
    parser.add_argument("--data_path", type=str, default=None,
                        help=f"Path to episodes.jsonl (default: {COUNTERFACTUAL_DATA})")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: PROJECT_ROOT/results/guided_router/)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-episode misclassifications")
    parser.add_argument("--dry_run", action="store_true",
                        help="Just verify imports and config, don't run eval")
    args = parser.parse_args()

    if args.dry_run:
        print("=== DRY RUN: Decomposition-Guided Router ===")
        print(f"Variant: {args.variant}")
        print(f"Death-risk tasks ({len(DEATH_RISK_TASKS)}): {list(DEATH_RISK_TASKS.keys())}")
        print(f"Non-death-risk tasks ({len(NON_DEATH_RISK_TASKS)}): {list(NON_DEATH_RISK_TASKS.keys())}")
        print(f"Death threshold: {DEATH_THRESHOLD}")
        print(f"Data path: {args.data_path or COUNTERFACTUAL_DATA}")
        print(f"Data exists: {os.path.isfile(args.data_path or COUNTERFACTUAL_DATA)}")
        router = GuidedRouter(variant=args.variant)
        print(f"Router initialized: variant={router.variant}")
        test_decision = router.route("chemistry-mix", "You see a thermometer")
        print(f"Test route('chemistry-mix'): {test_decision}")
        test_decision2 = router.route("find-non-living-thing", "You see rocks")
        print(f"Test route('find-non-living-thing'): {test_decision2}")
        print("DRY RUN OK")
        return

    # Load data
    data_path = args.data_path or COUNTERFACTUAL_DATA
    if not os.path.isfile(data_path):
        print(f"ERROR: Data file not found: {data_path}")
        sys.exit(1)

    print(f"Loading episodes from {data_path}...")
    episodes = load_episodes(data_path)
    print(f"Loaded {len(episodes)} episodes")

    # Initialize router
    router = GuidedRouter(variant=args.variant)
    print(f"Router: variant={args.variant}")

    # Evaluate
    print(f"\nEvaluating...")
    results = evaluate_router(router, episodes, verbose=args.verbose)

    # Print summary
    print(f"\n{'='*60}")
    print(f"GUIDED ROUTER ({args.variant}) RESULTS")
    print(f"{'='*60}")
    print(f"Episodes: {results['n_episodes']} ({results['n_tasks']} task types)")
    print(f"Evaluated (excl. no-preference): {results['n_evaluated']}")
    print(f"No-preference (excluded): {results['n_no_preference']}")
    print(f"\nBalanced Accuracy: {results['balanced_accuracy']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Det-preferred recall: {results['det_recall']:.4f}")
    print(f"Internal-preferred recall: {results['internal_recall']:.4f}")

    # Compare with baselines
    print(f"\n--- Comparison with baselines ---")
    for name, acc in BASELINES.items():
        delta = results['balanced_accuracy'] - acc
        print(f"  vs {name}: {delta:+.4f} ({results['balanced_accuracy']:.4f} vs {acc:.4f})")

    # Per-task breakdown
    print(f"\n--- Per-task breakdown ---")
    print(f"{'Task':<45} {'bal_acc':>8} {'det_r':>6} {'int_r':>6} {'n':>4}")
    for task, metrics in sorted(results["per_task"].items()):
        print(f"{task:<45} {metrics['balanced_accuracy']:>8.4f} "
              f"{metrics['det_recall']:>6.4f} {metrics['internal_recall']:>6.4f} "
              f"{metrics['n_evaluated']:>4}")

    # Save results
    output_dir = args.output_dir or os.path.join(PROJECT_ROOT, "results/guided_router")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{args.variant}_results.json")

    output = {
        "variant": args.variant,
        "data_path": data_path,
        "baselines": BASELINES,
        "results": results,
        "config": {
            "death_risk_tasks": DEATH_RISK_TASKS,
            "non_death_risk_tasks": NON_DEATH_RISK_TASKS,
            "death_threshold": DEATH_THRESHOLD,
        },
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
