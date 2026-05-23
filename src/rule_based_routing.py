"""Plan-011: Rule-based routing baseline.

Pre-registered keyword list decides routing per step:
  - keyword match in obs/actions → deterministic (fork-based simulator evaluation)
  - no match → internal (LLM ReAct reasoning)

Usage:
  python src/rule_based_routing.py --task_types "boil,melt" --episodes 20 --gpu_id 0
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scienceworld_utils import (
    create_env, reset_env, get_valid_actions, get_score,
    step_env, close_env, fork_and_lookahead, get_task_types,
    get_max_variations,
    STRONG_TASKS, MEDIUM_TASKS, WEAK_TASKS, ALL_MVP_TASKS,
)
from llm_agent import LLMAgent, find_model_path


# Disk space optimization: strip verbose per-step fields (llm_output, per-step
# action_history, candidates). Set SLIM_EPISODES=0 for full debug output.
SLIM_EPISODES = os.environ.get("SLIM_EPISODES", "1") == "1"


def _slim_episode_data(ep):
    """Strip per-step debug fields when SLIM_EPISODES is enabled."""
    if not SLIM_EPISODES:
        return ep
    result = {k: v for k, v in ep.items() if k != "steps"}
    if "steps" in ep:
        result["steps"] = [
            {k: v for k, v in s.items()
             if k not in ("llm_output", "action_history", "candidates",
                          "available_actions")}
            for s in ep["steps"]
        ]
    return result


def resolve_task_type(name: str) -> str:
    try:
        available = get_task_types()
    except Exception:
        return name
    if name in available:
        return name
    matches = [t for t in available if name.lower() in t.lower()]
    if matches:
        return matches[0]
    return name


def is_lethal(fr: Dict[str, Any]) -> bool:
    if fr["done"] and fr["score"] < 0:
        return True
    if fr["score"] <= -100:
        return True
    return False


def load_keywords(keyword_file: str) -> List[str]:
    with open(keyword_file, "r") as f:
        data = json.load(f)
    if "keywords" in data:
        return data["keywords"]
    return data["route_to_simulator"]


def get_keywords_version(keyword_file: str) -> str:
    with open(keyword_file, "r") as f:
        data = json.load(f)
    if "_meta" in data and "version" in data["_meta"]:
        return data["_meta"]["version"]
    return "v1"


def keyword_file_md5(keyword_file: str) -> str:
    with open(keyword_file, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def route_by_keywords(
    observation: str,
    action_candidates: List[str],
    keywords: List[str],
) -> Tuple[str, Optional[str]]:
    """Return (route_decision, matched_keyword_or_None)."""
    text = (observation + " " + " ".join(action_candidates)).lower()
    for kw in keywords:
        if kw.lower() in text:
            return "deterministic", kw
    return "internal", None


def run_episode_rule_based(
    agent: LLMAgent,
    task_name: str,
    variation_idx: int,
    max_steps: int,
    keywords: List[str],
    n_candidates: int = 3,
) -> Dict[str, Any]:
    env = create_env(task_name, variation_idx)
    obs, _ = reset_env(env)
    initial_score = get_score(env)

    steps = []
    action_history = []
    history_dicts = []
    routing_counts = Counter()
    keyword_hits = Counter()
    t0 = time.time()

    for step_idx in range(max_steps):
        valid_actions = get_valid_actions(env)
        if not valid_actions:
            break

        route, matched_kw = route_by_keywords(obs, valid_actions, keywords)
        routing_counts[route] += 1
        if matched_kw:
            keyword_hits[matched_kw] += 1

        score_before = get_score(env)

        if route == "deterministic":
            candidates, raw_output = agent.generate_candidate_actions(
                obs, valid_actions, history_dicts, n=n_candidates
            )
            best_action = None
            best_delta = float("-inf")
            best_action_fallback = None
            best_delta_fallback = float("-inf")
            fork_results = []
            for cand in candidates:
                fr = fork_and_lookahead(task_name, variation_idx, action_history, cand)
                lethal = is_lethal(fr)
                fork_results.append({
                    "action": cand,
                    "score_delta": fr["score_delta"],
                    "score": fr["score"],
                    "lethal": lethal,
                })
                if fr["score_delta"] > best_delta_fallback:
                    best_delta_fallback = fr["score_delta"]
                    best_action_fallback = cand
                if not lethal and fr["score_delta"] > best_delta:
                    best_delta = fr["score_delta"]
                    best_action = cand

            if best_action is None:
                best_action = best_action_fallback if best_action_fallback else valid_actions[0]

            action = best_action
            llm_output = raw_output[:300]
        else:
            action, raw_output = agent.select_action_react(obs, valid_actions, history_dicts)
            llm_output = raw_output[:300]

        obs_new, reward, done, info = step_env(env, action)
        score_after = get_score(env)

        step_record = {
            "step": step_idx,
            "observation": obs[:500],
            "action": action,
            "route_decision": route,
            "keyword_matched": matched_kw,
            "score_before": score_before,
            "score_after": score_after,
            "score_delta": score_after - score_before,
            "llm_output": llm_output,
        }
        steps.append(step_record)
        action_history.append(action)
        history_dicts.append({
            "action": action,
            "observation": obs_new[:200],
            "score": score_after,
        })
        obs = obs_new

        if done:
            break

    final_score = get_score(env)
    close_env(env)

    return {
        "mode": "rule_based_routing",
        "task_name": task_name,
        "variation_idx": variation_idx,
        "final_score": final_score,
        "initial_score": initial_score,
        "num_steps": len(steps),
        "total_time_s": time.time() - t0,
        "routing_stats": dict(routing_counts),
        "keyword_hits": dict(keyword_hits),
        "steps": steps,
        "action_history": action_history,
    }


def run_task_type(
    agent: LLMAgent,
    task_name: str,
    n_episodes: int,
    max_steps: int,
    keywords: List[str],
    output_dir: str,
    n_candidates: int = 3,
) -> Dict[str, Any]:
    task_dir = os.path.join(output_dir, task_name.replace(" ", "_"))
    os.makedirs(task_dir, exist_ok=True)
    episodes_file = os.path.join(task_dir, "episodes.jsonl")

    all_episodes = []
    max_var = get_max_variations(task_name)
    actual_unique = min(n_episodes, max_var)

    print(f"\n{'='*60}")
    print(f"RULE-BASED ROUTING: {task_name}")
    print(f"Episodes: {n_episodes} | Max steps: {max_steps}")
    print(f"Max variations: {max_var} | Unique: {actual_unique}")
    print(f"Keywords: {len(keywords)}")
    if actual_unique < n_episodes:
        print(f"  WARNING: {n_episodes} episodes > {max_var} variations, some will repeat")
    print(f"{'='*60}")

    total_routing = Counter()
    total_kw_hits = Counter()

    for ep_idx in range(n_episodes):
        variation_idx = ep_idx % max_var
        print(f"\n--- Episode {ep_idx}/{n_episodes} (variation={variation_idx}) ---", flush=True)

        t = time.time()
        result = run_episode_rule_based(
            agent, task_name, variation_idx, max_steps, keywords, n_candidates
        )
        rs = result["routing_stats"]
        det_n = rs.get("deterministic", 0)
        int_n = rs.get("internal", 0)
        print(f"  score={result['final_score']:.1f} det={det_n} int={int_n} ({time.time()-t:.0f}s)")

        episode_record = {
            "episode": ep_idx,
            "variation": variation_idx,
            **_slim_episode_data(result),
        }
        all_episodes.append(episode_record)

        with open(episodes_file, "a") as f:
            f.write(json.dumps(episode_record, default=str) + "\n")

        total_routing.update(result["routing_stats"])
        total_kw_hits.update(result["keyword_hits"])

    scores = [e["final_score"] for e in all_episodes]
    total_steps_all = sum(e["num_steps"] for e in all_episodes)
    det_total = total_routing.get("deterministic", 0)
    int_total = total_routing.get("internal", 0)
    step_total = det_total + int_total

    summary = {
        "task_name": task_name,
        "n_episodes": n_episodes,
        "max_variations": max_var,
        "unique_variations": actual_unique,
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "scores": scores,
        "routing_distribution": {
            "deterministic": det_total,
            "internal": int_total,
            "det_pct": det_total / step_total * 100 if step_total > 0 else 0,
            "int_pct": int_total / step_total * 100 if step_total > 0 else 0,
        },
        "keyword_match_rate": det_total / step_total * 100 if step_total > 0 else 0,
        "top_matched_keywords": total_kw_hits.most_common(10),
    }

    summary_file = os.path.join(task_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== RULE-BASED ROUTING: {task_name} ===")
    print(f"Episodes: {n_episodes}")
    print(f"Mean score: {summary['mean_score']:.2f} ± {summary['std_score']:.2f}")
    rd = summary["routing_distribution"]
    print(f"Routing distribution: det={rd['det_pct']:.1f}%, int={rd['int_pct']:.1f}%")
    print(f"Keyword match rate: {summary['keyword_match_rate']:.1f}%")
    top_kw = ", ".join(f"{kw}({n})" for kw, n in total_kw_hits.most_common(5))
    print(f"Top matched keywords: {top_kw}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Plan-011: Rule-based routing baseline")
    parser.add_argument("--task_types", type=str, required=True,
                        help="Comma-separated task type names")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--use_vllm", action="store_true", default=True)
    parser.add_argument("--no_vllm", action="store_true", default=False,
                        help="Use HF transformers instead of vLLM")
    parser.add_argument("--output_dir", type=str, default="results/rule_based/")
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--n_candidates", type=int, default=3,
                        help="Number of candidate actions for deterministic mode")
    parser.add_argument("--keyword_file", type=str,
                        default="src/rule_based_keywords_v2.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.no_vllm:
        args.use_vllm = False

    random.seed(args.seed)
    np.random.seed(args.seed)

    keywords = load_keywords(args.keyword_file)
    kw_md5 = keyword_file_md5(args.keyword_file)
    kw_version = get_keywords_version(args.keyword_file)
    print(f"Keyword file: {args.keyword_file}")
    print(f"Keyword version: {kw_version}")
    print(f"Keyword file MD5: {kw_md5}")
    print(f"Keywords ({len(keywords)}): {keywords}")

    task_names_raw = [t.strip() for t in args.task_types.split(",") if t.strip()]
    task_names = [resolve_task_type(t) for t in task_names_raw]
    print(f"Task types: {task_names}")
    print(f"Episodes: {args.episodes}, Max steps: {args.max_steps}")
    print(f"GPU: {args.gpu_id}, vLLM: {args.use_vllm}")
    print(f"Random seed: {args.seed}")

    agent = LLMAgent(
        model_path=args.model_path,
        use_vllm=args.use_vllm,
        gpu_id=args.gpu_id,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    all_summaries = {}

    for task_name in task_names:
        summary = run_task_type(
            agent, task_name, args.episodes, args.max_steps,
            keywords, args.output_dir, args.n_candidates,
        )
        all_summaries[task_name] = summary

    global_summary_path = os.path.join(args.output_dir, "global_summary.json")
    with open(global_summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nGlobal summary saved to {global_summary_path}")


if __name__ == "__main__":
    main()
