"""Convert ScienceWorld counterfactual data to unified multi-environment format."""
import json
import glob
import os
from collections import Counter

INPUT_DIR = "results/plan_c"
OUTPUT_DIR = "data/scienceworld-counterfactual"

def convert_steps(steps):
    return [
        {
            "step": s["step"],
            "observation": s.get("observation", ""),
            "action": s["action"],
            "score": s["score_after"],
        }
        for s in steps
    ]

def make_label(int_score, det_score):
    if det_score > int_score:
        return "det-preferred"
    elif int_score > det_score:
        return "internal-preferred"
    return "no-preference"

def death_category(int_dead, det_dead):
    if int_dead and det_dead:
        return "both_die"
    if int_dead:
        return "only_int_dies"
    if det_dead:
        return "only_det_dies"
    return "neither_die"

def is_dead(branch):
    if branch["final_score"] == -100.0:
        return True
    steps = branch["steps"]
    if steps:
        last = steps[-1]
        if last.get("is_dead", False):
            return True
        if last.get("lethal", False):
            return True
    return False

def convert_episode(raw, task_type, global_idx):
    ai = raw["always_internal"]
    ad = raw["always_deterministic"]
    variation = raw["variation_idx"]

    int_score = ai["final_score"]
    det_score = ad["final_score"]
    int_dead = is_dead(ai)
    det_dead = is_dead(ad)

    return {
        "episode_id": f"scienceworld_{task_type}_ep{global_idx:03d}",
        "environment": "scienceworld",
        "task_type": task_type,
        "task_id": f"{task_type}/variation_{variation}",
        "task_desc": f"ScienceWorld task: {task_type.replace('-', ' ')} (variation {variation})",
        "internal": {
            "trajectory": convert_steps(ai["steps"]),
            "final_score": int_score,
            "success": int_score > 0,
            "death": int_dead,
            "n_steps": ai["num_steps"],
        },
        "deterministic": {
            "trajectory": convert_steps(ad["steps"]),
            "final_score": det_score,
            "success": det_score > 0,
            "death": det_dead,
            "n_steps": ad["num_steps"],
        },
        "label": make_label(int_score, det_score),
        "score_gap": abs(det_score - int_score),
        "metadata": {
            "death_category": death_category(int_dead, det_dead),
            "variation": variation,
        },
    }

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_episodes = []
    task_files = sorted(glob.glob(f"{INPUT_DIR}/*/episodes.jsonl"))

    for fpath in task_files:
        task_type = fpath.split("/")[-2]
        with open(fpath) as f:
            for line in f:
                raw = json.loads(line.strip())
                ep = convert_episode(raw, task_type, raw["episode_idx"])
                all_episodes.append(ep)

    out_path = os.path.join(OUTPUT_DIR, "episodes.jsonl")
    with open(out_path, "w") as f:
        for ep in all_episodes:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")

    labels = Counter(e["label"] for e in all_episodes)
    deaths = Counter(e["metadata"]["death_category"] for e in all_episodes)
    tasks = sorted(set(e["task_type"] for e in all_episodes))

    summary = {
        "total_episodes": len(all_episodes),
        "num_tasks": len(tasks),
        "tasks": tasks,
        "label_distribution": dict(labels),
        "death_distribution": dict(deaths),
        "episodes_per_task": {
            t: sum(1 for e in all_episodes if e["task_type"] == t) for t in tasks
        },
    }
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Total episodes: {len(all_episodes)}")
    print(f"Tasks: {len(tasks)}")
    print(f"Labels: {dict(labels)}")
    print(f"Deaths: {dict(deaths)}")
    print(f"Written to: {out_path}")

if __name__ == "__main__":
    main()
