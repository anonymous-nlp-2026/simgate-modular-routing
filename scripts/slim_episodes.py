# slim_episodes.py — Trim episodes.jsonl to analysis-essential fields.
# Drops: llm_output, candidates, observation. Keeps: scores, actions, metadata, lethal flags.
# Single-pass: reads original once, writes slim + gzip backup simultaneously.

import argparse
import gzip
import json
import os
import subprocess
import sys

STEP_KEEP_KEYS = {"step", "action", "modality", "score_before", "score_after", "score_delta"}
MODE_META_KEYS = {"mode", "task_name", "variation_idx", "final_score", "initial_score", "num_steps", "total_time_s"}


def slim_step(step, mode_key):
    out = {k: step[k] for k in STEP_KEEP_KEYS if k in step}
    if mode_key == "always_deterministic":
        candidates = step.get("candidates", [])
        action = step.get("action")
        for c in candidates:
            if c.get("action") == action:
                out["lethal"] = c.get("lethal", False)
                break
    return out


def slim_mode(mode_data, mode_key):
    if not isinstance(mode_data, dict):
        return mode_data
    out = {k: mode_data[k] for k in MODE_META_KEYS if k in mode_data}
    if "steps" in mode_data:
        out["steps"] = [slim_step(s, mode_key) for s in mode_data["steps"]]
    if "action_history" in mode_data:
        out["action_history"] = mode_data["action_history"]
    return out


def slim_record(record):
    out = {
        "episode_idx": record.get("episode_idx"),
        "variation_idx": record.get("variation_idx"),
    }
    for mode_key in ("always_internal", "always_deterministic"):
        if mode_key in record:
            out[mode_key] = slim_mode(record[mode_key], mode_key)
    return out


def is_file_open(path):
    try:
        result = subprocess.run(["lsof", path], capture_output=True, text=True, timeout=5)
        return bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def process_task_dir(task_dir, dry_run=False):
    src = os.path.join(task_dir, "episodes.jsonl")
    if not os.path.isfile(src):
        print(f"SKIP {task_dir}: no episodes.jsonl")
        return

    if is_file_open(src):
        print(f"SKIP {task_dir}: file is open by another process")
        return

    src_size = os.path.getsize(src)
    if src_size < 1024 * 1024:
        print(f"SKIP {task_dir}: only {src_size/1024:.0f} KB")
        return

    # Estimate by sampling first 1000 lines
    sample_orig = 0
    sample_slim = 0
    sample_count = 0
    with open(src, "r") as f:
        for i, line in enumerate(f):
            if i >= 1000:
                break
            record = json.loads(line)
            sample_orig += len(line)
            slimmed = json.dumps(slim_record(record), separators=(",", ":"))
            sample_slim += len(slimmed) + 1
            sample_count += 1

    if sample_count == 0:
        print(f"SKIP {task_dir}: empty file")
        return

    ratio = sample_slim / sample_orig
    est_slim_size = src_size * ratio
    savings = src_size - est_slim_size

    print(f"\n{'[DRY RUN] ' if dry_run else ''}{os.path.basename(task_dir)}")
    print(f"  Original:  {src_size/1024**3:.2f} GB")
    print(f"  Estimated: {est_slim_size/1024**3:.2f} GB ({ratio:.1%} of original)")
    print(f"  Savings:   {savings/1024**3:.2f} GB")

    if dry_run:
        return

    # Single-pass: write slim + gzip backup simultaneously
    slim_tmp = src + ".slim.tmp"
    gz_path = os.path.join(task_dir, "episodes_full.jsonl.gz")

    line_count = 0
    print(f"  Processing (single pass: slim + gzip)...")
    with open(src, "rb") as fin, \
         open(slim_tmp, "w") as fout_slim, \
         gzip.open(gz_path, "wb", compresslevel=6) as fout_gz:
        for raw_line in fin:
            fout_gz.write(raw_line)
            record = json.loads(raw_line)
            fout_slim.write(json.dumps(slim_record(record), separators=(",", ":")) + "\n")
            line_count += 1
            if line_count % 500000 == 0:
                print(f"    ...{line_count:,} records", flush=True)

    # Verify slim line count
    verify_count = 0
    with open(slim_tmp, "r") as f:
        for _ in f:
            verify_count += 1

    if verify_count != line_count:
        print(f"  ERROR: line count mismatch! expected={line_count}, got={verify_count}. Aborting.")
        os.remove(slim_tmp)
        os.remove(gz_path)
        return

    slim_size = os.path.getsize(slim_tmp)
    gz_size = os.path.getsize(gz_path)
    print(f"  Slim:   {slim_size/1024**3:.2f} GB ({line_count:,} records)")
    print(f"  Backup: {gz_size/1024**3:.2f} GB (gzip)")

    # Replace original
    os.remove(src)
    os.rename(slim_tmp, src)
    freed = src_size - slim_size
    print(f"  DONE: freed {freed/1024**3:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="Trim episodes.jsonl to analysis-essential fields")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task-dir", help="Path to a single task directory")
    group.add_argument("--all", action="store_true", help="Process all task dirs under results/plan_c/")
    parser.add_argument("--dry-run", action="store_true", help="Only estimate savings")
    parser.add_argument("--results-root", default="./results/plan_c",
                        help="Root of results directories (used with --all)")
    args = parser.parse_args()

    if args.task_dir:
        process_task_dir(args.task_dir, dry_run=args.dry_run)
    else:
        for name in sorted(os.listdir(args.results_root)):
            task_dir = os.path.join(args.results_root, name)
            if os.path.isdir(task_dir):
                process_task_dir(task_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
