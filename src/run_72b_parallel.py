#!/usr/bin/env python3
"""72B episode collection with auto-resume and task-level parallelism.

Auto-detects completed tasks/episodes in the output directory and only
collects what's missing. Supports running a subset of tasks for
multi-machine distribution.

Usage:
    # Run all remaining tasks (auto-skip completed)
    python src/run_72b_parallel.py

    # Run specific tasks only (for distributing across machines)
    python src/run_72b_parallel.py --tasks "freeze,grow-fruit,find-animal"

    # Dry run: show what needs to be done
    python src/run_72b_parallel.py --dry-run

    # Single-GPU mode for new 96GB cards
    python src/run_72b_parallel.py --tp 1 --gpu-id 0 --tasks "freeze,grow-fruit"
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from episode_level_collection import (
    run_task_collection,
    cleanup_gpu_processes,
    resolve_task_type,
)
from llm_agent import LLMAgent

ALL_TASKS = [
    "chemistry-mix",
    "find-animal",
    "find-non-living-thing",
    "freeze",
    "grow-fruit",
    "identify-life-stages-1",
    "identify-life-stages-2",
    "inclined-plane-determine-angle",
    "lifespan-longest-lived",
    "lifespan-longest-lived-then-shortest-lived",
    "measure-melting-point-unknown-substance",
]

DEFAULT_MODEL = "./models/Qwen/Qwen2.5-72B-Instruct-AWQ"
DEFAULT_OUTPUT = "results/backbone_72b/"


def scan_progress(output_dir):
    """Scan output directory for completed episodes per task."""
    progress = {}
    if not os.path.isdir(output_dir):
        return progress
    for entry in os.listdir(output_dir):
        task_path = os.path.join(output_dir, entry)
        if not os.path.isdir(task_path):
            continue
        episodes_file = os.path.join(task_path, "episodes.jsonl")
        if os.path.exists(episodes_file):
            with open(episodes_file) as f:
                count = sum(1 for line in f if line.strip())
            progress[entry] = count
        else:
            progress[entry] = 0
    return progress


def main():
    parser = argparse.ArgumentParser(
        description="72B episode collection with auto-resume and task splitting"
    )
    parser.add_argument(
        "--tasks", type=str, default=None,
        help='Comma-separated task types to run (default: all 11 tasks). '
             'Example: --tasks "freeze,grow-fruit,find-animal"'
    )
    parser.add_argument(
        "--episodes-per-task", type=int, default=20,
        help="Episodes per task (default: 20)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Model path (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--tp", type=int, default=2,
        help="Tensor parallel size (default: 2 for dual-GPU, use 1 for single 96GB card)"
    )
    parser.add_argument(
        "--gpu-id", type=int, default=0,
        help="Starting GPU ID (default: 0)"
    )
    parser.add_argument(
        "--max-steps", type=int, default=30,
        help="Max steps per episode (default: 30)"
    )
    parser.add_argument(
        "--n-candidates", type=int, default=3,
        help="Candidate actions for deterministic mode (default: 3)"
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="vLLM server port (reserved for future API server mode, currently unused)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show progress and plan without running collection"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    else:
        tasks = list(ALL_TASKS)

    tasks = [resolve_task_type(t) for t in tasks]

    progress = scan_progress(args.output_dir)

    print("=" * 78)
    print("72B Episode Collection (auto-resume)")
    print("=" * 78)
    print(f"Model:  {args.model}")
    print(f"Output: {args.output_dir}")
    print(f"Target: {args.episodes_per_task} episodes/task")
    print(f"TP={args.tp}, GPU={args.gpu_id}, max_steps={args.max_steps}")
    print()
    print(f"  {'Task':<50} {'Done':>5} {'Target':>6} {'Status':>10}")
    print("  " + "-" * 73)

    remaining_tasks = []
    for task in tasks:
        task_dir_name = task.replace(" ", "_")
        done = progress.get(task_dir_name, 0)
        need = max(0, args.episodes_per_task - done)
        status = "DONE" if need == 0 else f"{need} left"
        print(f"  {task:<50} {done:>5} {args.episodes_per_task:>6} {status:>10}")
        if need > 0:
            remaining_tasks.append((task, done, need))

    if not remaining_tasks:
        print("\nAll tasks complete!")
        return

    total_ep = sum(r for _, _, r in remaining_tasks)
    print(f"\nPlan: {len(remaining_tasks)} tasks, {total_ep} episodes to collect")

    if args.dry_run:
        print("\n[DRY RUN] Would run:")
        for task, start, need in remaining_tasks:
            print(f"  {task}: start_episode={start}, collect={need}")
        print("\nTo execute, remove --dry-run flag.")
        return

    cleanup_gpu_processes(args.gpu_id)
    print("\nInitializing LLM agent...")
    agent = LLMAgent(
        model_path=args.model,
        use_vllm=True,
        gpu_id=args.gpu_id,
        tensor_parallel_size=args.tp,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}

    for i, (task, start_ep, n_episodes) in enumerate(remaining_tasks):
        print(f"\n[{i+1}/{len(remaining_tasks)}] Starting: {task} "
              f"(episodes {start_ep}..{start_ep + n_episodes - 1})")

        result = run_task_collection(
            agent, task, n_episodes, start_ep,
            args.max_steps, args.output_dir, args.n_candidates,
        )
        all_results[task] = result

        task_dir = os.path.join(args.output_dir, task.replace(" ", "_"))
        meta = {
            "task_name": task,
            "model": os.path.basename(args.model),
            "target_episodes": args.episodes_per_task,
            "collected_episodes": n_episodes,
            "start_episode": start_ep,
            "tp": args.tp,
            "seed": args.seed,
        }
        with open(os.path.join(task_dir, "collection_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    print(f"\n{'=' * 60}")
    print("COLLECTION SUMMARY")
    print(f"{'=' * 60}")
    for task, result in all_results.items():
        lc = result["label_counts"]
        print(f"  {task}: det={lc['det-preferred']} int={lc['internal-preferred']} "
              f"tie={lc['no-preference']} | steps={result['total_labeled_steps']}")
    print("\nDone.")


if __name__ == "__main__":
    main()
