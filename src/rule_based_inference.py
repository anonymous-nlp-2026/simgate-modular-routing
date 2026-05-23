"""Plan-011: Rule-based keyword routing inference and score simulation.

Given episode data with always-internal and always-deterministic trajectories,
simulates what score a keyword-based router would achieve by selecting
deterministic mode for keyword-matching episodes and internal otherwise.
Compares against always-det, always-internal, and oracle baselines.

Per D014: v3 keywords are science-specific state-change verbs only.
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


DEATH_THRESHOLD = -100


def load_keywords(path: str) -> List[str]:
    with open(path) as f:
        data = json.load(f)
    if "keywords" in data:
        return data["keywords"]
    return data["route_to_simulator"]


def keyword_match(text: str, keywords: List[str]) -> bool:
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def classify_episode_gt(int_score: float, det_score: float) -> str:
    int_died = int_score <= DEATH_THRESHOLD
    det_died = det_score <= DEATH_THRESHOLD
    if int_died and not det_died:
        return "det-preferred"
    if det_died and not int_died:
        return "internal-preferred"
    if int_died and det_died:
        return "no-preference"
    if det_score > int_score:
        return "det-preferred"
    if int_score > det_score:
        return "internal-preferred"
    return "no-preference"


def get_steps(episode: Dict) -> List[Dict]:
    for key in ["oracle", "always_internal"]:
        if key in episode and episode[key].get("steps"):
            return episode[key]["steps"]
    return []


def route_episode(episode: Dict, keywords: List[str],
                  strategy: str = "majority",
                  threshold: float = 0.5) -> Dict:
    """Apply keyword routing to an episode, return routing decision + simulated score."""
    int_ep = episode["always_internal"]
    det_ep = episode["always_deterministic"]
    int_score = int_ep["final_score"]
    det_score = det_ep["final_score"]
    oracle_score = episode.get("oracle", {}).get("final_score")

    steps = get_steps(episode)
    n_steps = len(steps)
    n_matched = sum(1 for s in steps if keyword_match(s.get("action", ""), keywords))
    match_rate = n_matched / n_steps if n_steps else 0.0

    if strategy == "any":
        route_to_det = n_matched > 0
    else:
        route_to_det = match_rate > threshold

    routed_label = "det-preferred" if route_to_det else "internal-preferred"
    simulated_score = det_score if route_to_det else int_score
    gt_label = classify_episode_gt(int_score, det_score)

    return {
        "episode_idx": episode.get("episode_idx", -1),
        "variation_idx": episode.get("variation_idx", -1),
        "n_steps": n_steps,
        "n_keyword_matched": n_matched,
        "keyword_match_rate": round(match_rate, 4),
        "routed_label": routed_label,
        "gt_label": gt_label,
        "correct": routed_label == gt_label or gt_label == "no-preference",
        "int_score": int_score,
        "det_score": det_score,
        "oracle_score": oracle_score,
        "simulated_score": simulated_score,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Plan-011: Keyword routing inference + score simulation")
    parser.add_argument("--data_dirs", type=str, required=True,
                        help="Comma-separated data directories")
    parser.add_argument("--keywords", type=str,
                        default="src/rule_based_keywords_v3.json")
    parser.add_argument("--output", type=str,
                        default="results/rule_based_inference_report.json")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--strategy", choices=["majority", "any"],
                        default="majority")
    args = parser.parse_args()

    keywords = load_keywords(args.keywords)
    print(f"Keywords ({len(keywords)}): {keywords}")
    print(f"Strategy: {args.strategy}  |  Threshold: {args.threshold}")

    data_dirs = [d.strip() for d in args.data_dirs.split(",") if d.strip()]
    all_results = []
    per_task = {}

    for data_dir in data_dirs:
        if not os.path.isdir(data_dir):
            print(f"WARN: {data_dir} not found, skipping")
            continue
        for task_name in sorted(os.listdir(data_dir)):
            ep_file = os.path.join(data_dir, task_name, "episodes.jsonl")
            if not os.path.isfile(ep_file):
                continue
            task_results = []
            with open(ep_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    episode = json.loads(line)
                    result = route_episode(episode, keywords,
                                           strategy=args.strategy,
                                           threshold=args.threshold)
                    result["task"] = task_name
                    task_results.append(result)

            if not task_results:
                continue

            n_eps = len(task_results)
            avg_match_rate = sum(r["keyword_match_rate"] for r in task_results) / n_eps

            scores_int = [r["int_score"] for r in task_results]
            scores_det = [r["det_score"] for r in task_results]
            scores_sim = [r["simulated_score"] for r in task_results]
            scores_oracle = [r["oracle_score"] for r in task_results
                             if r["oracle_score"] is not None]

            avg_int = sum(scores_int) / n_eps
            avg_det = sum(scores_det) / n_eps
            avg_sim = sum(scores_sim) / n_eps
            avg_oracle = sum(scores_oracle) / len(scores_oracle) if scores_oracle else None

            scorable = [r for r in task_results if r["gt_label"] != "no-preference"]
            n_correct = sum(1 for r in scorable if r["routed_label"] == r["gt_label"])
            accuracy = n_correct / len(scorable) if scorable else 0.0

            per_task[task_name] = {
                "n_episodes": n_eps,
                "avg_keyword_match_rate": round(avg_match_rate, 4),
                "routing_accuracy": round(accuracy, 4),
                "n_scorable": len(scorable),
                "n_correct": n_correct,
                "avg_score_always_internal": round(avg_int, 2),
                "avg_score_always_det": round(avg_det, 2),
                "avg_score_keyword_router": round(avg_sim, 2),
                "avg_score_oracle": round(avg_oracle, 2) if avg_oracle is not None else None,
            }
            all_results.extend(task_results)

    if not all_results:
        print("No episodes found.")
        return

    n_total = len(all_results)
    overall_int = sum(r["int_score"] for r in all_results) / n_total
    overall_det = sum(r["det_score"] for r in all_results) / n_total
    overall_sim = sum(r["simulated_score"] for r in all_results) / n_total
    oracle_scores = [r["oracle_score"] for r in all_results if r["oracle_score"] is not None]
    overall_oracle = sum(oracle_scores) / len(oracle_scores) if oracle_scores else None
    overall_match = sum(r["keyword_match_rate"] for r in all_results) / n_total

    scorable_all = [r for r in all_results if r["gt_label"] != "no-preference"]
    n_correct_all = sum(1 for r in scorable_all if r["routed_label"] == r["gt_label"])
    accuracy_all = n_correct_all / len(scorable_all) if scorable_all else 0.0

    report = {
        "config": {
            "keywords": keywords,
            "strategy": args.strategy,
            "threshold": args.threshold,
        },
        "overall": {
            "n_episodes": n_total,
            "avg_keyword_match_rate": round(overall_match, 4),
            "routing_accuracy": round(accuracy_all, 4),
            "avg_score_always_internal": round(overall_int, 2),
            "avg_score_always_det": round(overall_det, 2),
            "avg_score_keyword_router": round(overall_sim, 2),
            "avg_score_oracle": round(overall_oracle, 2) if overall_oracle is not None else None,
        },
        "per_task": per_task,
        "episodes": all_results,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {args.output}")

    # --- Terminal summary ---
    print(f"\n{'='*80}")
    print("KEYWORD ROUTING INFERENCE — SCORE SIMULATION")
    print(f"{'='*80}")
    print(f"Episodes: {n_total}  |  Avg keyword match rate: {overall_match:.1%}")
    print(f"Routing accuracy (excl no-pref): {accuracy_all:.1%} "
          f"({n_correct_all}/{len(scorable_all)})")
    print()

    print(f"{'Task':<45} {'Eps':>4} {'Match%':>7} "
          f"{'Int':>7} {'Det':>7} {'KwRtr':>7} {'Oracle':>7} {'Acc':>6}")
    print("-" * 100)
    for task_name in sorted(per_task.keys()):
        t = per_task[task_name]
        orc = f"{t['avg_score_oracle']:>7.1f}" if t["avg_score_oracle"] is not None else "    N/A"
        print(f"{task_name:<45} {t['n_episodes']:>4} "
              f"{t['avg_keyword_match_rate']:>6.1%} "
              f"{t['avg_score_always_internal']:>7.1f} "
              f"{t['avg_score_always_det']:>7.1f} "
              f"{t['avg_score_keyword_router']:>7.1f} "
              f"{orc} "
              f"{t['routing_accuracy']:>5.1%}")

    print()
    orc_str = f"{overall_oracle:.1f}" if overall_oracle is not None else "N/A"
    print(f"Overall avg scores:  Internal={overall_int:.1f}  "
          f"Det={overall_det:.1f}  "
          f"KeywordRouter={overall_sim:.1f}  "
          f"Oracle={orc_str}")

    delta_vs_int = overall_sim - overall_int
    delta_vs_det = overall_sim - overall_det
    print(f"Keyword router delta:  vs Internal={delta_vs_int:+.1f}  "
          f"vs Det={delta_vs_det:+.1f}")


if __name__ == "__main__":
    main()
