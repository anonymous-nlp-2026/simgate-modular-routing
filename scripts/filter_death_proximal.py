# Death-proximal data filtering for router SFT experiment.
# Reads plan_c labeled_steps + episodes.jsonl, extracts fatal step info,
# filters to death-proximal windows, converts to implicit SFT format.

import argparse
import json
import os
import glob
import random
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional, Tuple

DEATH_THRESHOLD = -100
SIMULATE_PREFIX = "[SIMULATE]"
IMPLICIT_SYSTEM_PROMPT = (
    "You are an agent in a ScienceWorld text environment. "
    "Given the current observation and action history, choose the next action. "
    "If you believe a deterministic simulation would produce a better outcome, "
    "prefix your action with [SIMULATE]. Otherwise, output the action directly."
)
RAW_LABEL_MAP = {
    "internal-preferred": "internal",
    "det-preferred": "deterministic",
}


def format_action_history(history: List[str], max_actions: int = 5) -> str:
    recent = history[-max_actions:] if history else []
    return " -> ".join(recent) if recent else "(none)"


def record_to_chat(record: Dict) -> Optional[Dict]:
    raw_label = record.get("label", "")
    route = RAW_LABEL_MAP.get(raw_label)
    if route is None:
        return None

    obs = record["observation"]
    action = record["action"]
    history = record.get("action_history", [])
    user_content = f"Observation: {obs}\nAction history: {format_action_history(history)}"

    if route == "deterministic":
        assistant_content = f"{SIMULATE_PREFIX} {action}"
    else:
        assistant_content = action

    return {
        "messages": [
            {"role": "system", "content": IMPLICIT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "task_type": record.get("task_type", ""),
        "variation": record.get("variation", -1),
        "step_idx": record.get("step_idx", -1),
        "route_label": route,
        "episode_id": record.get("episode_id", -1),
        "signal_source": record.get("signal_source", ""),
    }


def get_fatal_steps(data_dir: str) -> Dict[Tuple, int]:
    """Extract fatal step index for each death_asymmetry episode."""
    fatal_map = {}
    tasks = sorted(os.listdir(data_dir))
    for task in tasks:
        ep_file = os.path.join(data_dir, task, "episodes.jsonl")
        if not os.path.exists(ep_file):
            continue
        with open(ep_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ep = json.loads(line)
                ai = ep.get("always_internal", {})
                ad = ep.get("always_deterministic", {})
                int_score = ai.get("final_score", 0)
                det_score = ad.get("final_score", 0)
                int_died = int_score <= DEATH_THRESHOLD
                det_died = det_score <= DEATH_THRESHOLD

                if (int_died and not det_died) or (det_died and not int_died):
                    ep_idx = ep.get("episode_idx", 0)
                    var_idx = ep.get("variation_idx", 0)
                    dying = ai if int_died else ad
                    dying_steps = dying.get("steps", [])
                    fatal_step = len(dying_steps) - 1 if dying_steps else 29
                    fatal_map[(task, ep_idx, var_idx)] = fatal_step
    return fatal_map


def load_labeled_steps(data_dir: str) -> List[Dict]:
    records = []
    for fpath in sorted(glob.glob(os.path.join(data_dir, "*", "labeled_steps.jsonl"))):
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def split_episodes(records: List[Dict], eval_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    """Split at episode level."""
    ep_groups = defaultdict(list)
    for r in records:
        key = (r["task_type"], r["episode_id"], r["variation"])
        ep_groups[key].append(r)

    keys = sorted(ep_groups.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)
    n_eval = max(1, int(len(keys) * eval_ratio))
    eval_keys = set(keys[:n_eval])

    train_recs, eval_recs = [], []
    for key in keys:
        if key in eval_keys:
            eval_recs.extend(ep_groups[key])
        else:
            train_recs.extend(ep_groups[key])
    return train_recs, eval_recs


def filter_death_proximal(records: List[Dict], fatal_map: Dict, k: int) -> List[Dict]:
    """Keep death-proximal steps for death_asymmetry, all steps for score_comparison."""
    filtered = []
    for r in records:
        if r["signal_source"] == "score_comparison":
            filtered.append(r)
        elif r["signal_source"] == "death_asymmetry":
            key = (r["task_type"], r["episode_id"], r["variation"])
            fatal_step = fatal_map.get(key)
            if fatal_step is None:
                filtered.append(r)
                continue
            if fatal_step - k <= r["step_idx"] <= fatal_step:
                filtered.append(r)
    return filtered


def write_implicit(records: List[Dict], output_path: str) -> Dict:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    stats = Counter()
    with open(output_path, "w") as f:
        for r in records:
            chat = record_to_chat(r)
            if chat is None:
                stats["skipped"] += 1
                continue
            f.write(json.dumps(chat, ensure_ascii=False) + "\n")
            stats[chat["route_label"]] += 1
            stats["total"] += 1
    return dict(stats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="results/plan_c")
    parser.add_argument("--output_dir", default="data/death-proximal")
    parser.add_argument("--k_values", default="1,3,5,10")
    parser.add_argument("--eval_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.chdir(".")

    fatal_map = get_fatal_steps(args.data_dir)
    print(f"Fatal step info for {len(fatal_map)} episodes")

    records = load_labeled_steps(args.data_dir)
    print(f"Total labeled steps: {len(records)}")

    train_recs, eval_recs = split_episodes(records, args.eval_split, args.seed)

    ep_train = set((r["task_type"], r["episode_id"], r["variation"]) for r in train_recs)
    ep_eval = set((r["task_type"], r["episode_id"], r["variation"]) for r in eval_recs)
    print(f"Train: {len(train_recs)} steps from {len(ep_train)} episodes")
    print(f"Eval:  {len(eval_recs)} steps from {len(ep_eval)} episodes")

    os.makedirs(args.output_dir, exist_ok=True)

    # Write full training data
    full_stats = write_implicit(train_recs, os.path.join(args.output_dir, "train_full.jsonl"))
    print(f"\nFull train: {full_stats}")

    # Write eval data (unfiltered, always the same)
    eval_stats = write_implicit(eval_recs, os.path.join(args.output_dir, "eval.jsonl"))
    print(f"Eval: {eval_stats}")

    # Save eval metadata for later analysis
    eval_meta = []
    for r in eval_recs:
        key = (r["task_type"], r["episode_id"], r["variation"])
        eval_meta.append({
            "task_type": r["task_type"],
            "episode_id": r["episode_id"],
            "variation": r["variation"],
            "step_idx": r["step_idx"],
            "signal_source": r["signal_source"],
            "label": r["label"],
            "fatal_step": fatal_map.get(key),
        })
    with open(os.path.join(args.output_dir, "eval_meta.jsonl"), "w") as f:
        for m in eval_meta:
            f.write(json.dumps(m) + "\n")

    # Write filtered training data for each k
    k_values = [int(x) for x in args.k_values.split(",")]
    summary = {"full": full_stats, "eval": eval_stats, "k_variants": {}}

    for k in k_values:
        filtered = filter_death_proximal(train_recs, fatal_map, k)
        out_path = os.path.join(args.output_dir, f"train_k{k}.jsonl")
        k_stats = write_implicit(filtered, out_path)
        print(f"k={k}: {k_stats}")

        # Detailed breakdown
        death_steps = sum(1 for r in filtered if r["signal_source"] == "death_asymmetry")
        score_steps = sum(1 for r in filtered if r["signal_source"] == "score_comparison")
        print(f"  death_asymmetry: {death_steps}, score_comparison: {score_steps}")

        summary["k_variants"][str(k)] = {
            "stats": k_stats,
            "death_steps": death_steps,
            "score_steps": score_steps,
        }

    with open(os.path.join(args.output_dir, "filter_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {args.output_dir}/filter_summary.json")


if __name__ == "__main__":
    main()
