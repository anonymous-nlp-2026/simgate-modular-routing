"""Lightweight 2-way prescreening: always-internal vs always-deterministic.

Simplified version of phase1_signal.py for quick signal checks.
Runs fewer episodes (default 5) on a single task type,
outputs per-episode JSONL and a terminal summary with Δ(D-I) direction.

Key differences from phase1_signal.py:
  - No oracle-routing mode (2-way only)
  - always-internal does NOT use death filter (pure LLM)
  - always-deterministic uses death filter via fork_and_lookahead
  - Single task type per invocation (--task_type, not comma-separated)

Usage:
  python src/signal_prescreening.py --task_type boil --episodes 5 --gpu_id 0
"""

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scienceworld_utils import (
    create_env, reset_env, get_valid_actions, get_score,
    step_env, close_env, fork_and_lookahead, get_task_types,
    get_max_variations,
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
    """Resolve short task name to ScienceWorld task name."""
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
    """Check if a fork result indicates a lethal action."""
    if fr["done"] and fr["score"] < 0:
        return True
    if fr["score"] <= -100:
        return True
    return False


def run_episode_internal(
    agent: LLMAgent,
    task_name: str,
    variation_idx: int,
    max_steps: int,
) -> Dict[str, Any]:
    """Always-internal: pure LLM ReAct, no death filter."""
    env = create_env(task_name, variation_idx)
    obs, _ = reset_env(env)
    initial_score = get_score(env)

    steps = []
    action_history = []
    history_dicts = []
    t0 = time.time()

    for step_idx in range(max_steps):
        valid_actions = get_valid_actions(env)
        if not valid_actions:
            break

        action, raw_output = agent.select_action_react(obs, valid_actions, history_dicts)

        score_before = get_score(env)
        obs_new, reward, done, info = step_env(env, action)
        score_after = get_score(env)

        steps.append({
            "step": step_idx,
            "action": action,
            "modality": "internal",
            "score_before": score_before,
            "score_after": score_after,
            "score_delta": score_after - score_before,
            "observation": obs_new[:500],
            "llm_output": raw_output[:300],
        })
        action_history.append(action)
        history_dicts.append({"action": action, "observation": obs_new[:200], "score": score_after})
        obs = obs_new

        if done:
            break

    final_score = get_score(env)
    close_env(env)

    return {
        "mode": "always-internal",
        "task_name": task_name,
        "variation_idx": variation_idx,
        "final_score": final_score,
        "initial_score": initial_score,
        "num_steps": len(steps),
        "total_time_s": time.time() - t0,
        "steps": steps,
        "action_history": action_history,
    }


def run_episode_deterministic(
    agent: LLMAgent,
    task_name: str,
    variation_idx: int,
    max_steps: int,
    n_candidates: int = 3,
) -> Dict[str, Any]:
    """Always-deterministic: LLM candidates + fork evaluation + death filter."""
    env = create_env(task_name, variation_idx)
    obs, _ = reset_env(env)
    initial_score = get_score(env)

    steps = []
    action_history = []
    history_dicts = []
    t0 = time.time()

    for step_idx in range(max_steps):
        valid_actions = get_valid_actions(env)
        if not valid_actions:
            break

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

        score_before = get_score(env)
        obs_new, reward, done, info = step_env(env, best_action)
        score_after = get_score(env)

        steps.append({
            "step": step_idx,
            "action": best_action,
            "modality": "deterministic",
            "score_before": score_before,
            "score_after": score_after,
            "score_delta": score_after - score_before,
            "candidates": fork_results,
            "observation": obs_new[:500],
        })
        action_history.append(best_action)
        history_dicts.append({"action": best_action, "observation": obs_new[:200], "score": score_after})
        obs = obs_new

        if done:
            break

    final_score = get_score(env)
    close_env(env)

    return {
        "mode": "always-deterministic",
        "task_name": task_name,
        "variation_idx": variation_idx,
        "final_score": final_score,
        "initial_score": initial_score,
        "num_steps": len(steps),
        "total_time_s": time.time() - t0,
        "steps": steps,
        "action_history": action_history,
    }


def run_prescreening(
    agent: LLMAgent,
    task_name: str,
    n_episodes: int,
    max_steps: int,
    output_dir: str,
    n_candidates: int = 3,
) -> Dict[str, Any]:
    """Run 2-way prescreening for a single task type."""
    task_dir = os.path.join(output_dir, task_name.replace(" ", "_"))
    os.makedirs(task_dir, exist_ok=True)

    episodes_file = os.path.join(task_dir, "episodes.jsonl")

    max_var = get_max_variations(task_name)
    actual_unique = min(n_episodes, max_var)
    print(f"\n{'='*60}")
    print(f"PRESCREENING: {task_name} | Episodes: {n_episodes} | Max steps: {max_steps}")
    print(f"Max variations: {max_var} | Unique: {actual_unique}")
    print(f"{'='*60}")

    scores_internal = []
    scores_det = []

    for ep_idx in range(n_episodes):
        variation_idx = ep_idx % max_var
        print(f"\n--- Episode {ep_idx}/{n_episodes} (variation={variation_idx}) ---")

        print(f"  [I] always-internal...", end=" ", flush=True)
        t = time.time()
        r_int = run_episode_internal(agent, task_name, variation_idx, max_steps)
        print(f"score={r_int['final_score']:.1f} ({time.time()-t:.0f}s)")

        print(f"  [D] always-deterministic...", end=" ", flush=True)
        t = time.time()
        r_det = run_episode_deterministic(
            agent, task_name, variation_idx, max_steps, n_candidates
        )
        print(f"score={r_det['final_score']:.1f} ({time.time()-t:.0f}s)")

        scores_internal.append(r_int["final_score"])
        scores_det.append(r_det["final_score"])

        episode_record = {
            "episode_idx": ep_idx,
            "variation_idx": variation_idx,
            "always_internal": _slim_episode_data(r_int),
            "always_deterministic": _slim_episode_data(r_det),
        }
        with open(episodes_file, "a") as f:
            f.write(json.dumps(episode_record, default=str) + "\n")

    import numpy as np
    mean_int = float(np.mean(scores_internal))
    mean_det = float(np.mean(scores_det))
    delta = mean_det - mean_int

    if delta > 1.0:
        direction = "POSITIVE"
    elif delta < -1.0:
        direction = "NEGATIVE"
    else:
        direction = "NEUTRAL"

    summary = {
        "task_name": task_name,
        "n_episodes": n_episodes,
        "max_variations": max_var,
        "internal_mean": mean_int,
        "internal_std": float(np.std(scores_internal)),
        "internal_scores": scores_internal,
        "deterministic_mean": mean_det,
        "deterministic_std": float(np.std(scores_det)),
        "deterministic_scores": scores_det,
        "delta_det_minus_int": delta,
        "direction": direction,
    }

    summary_file = os.path.join(task_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== PRESCREENING: {task_name} ===")
    print(f"Episodes: {n_episodes}")
    print(f"Internal mean: {mean_int:.2f}")
    print(f"Deterministic mean: {mean_det:.2f}")
    print(f"Δ(D-I): {delta:+.2f}")
    print(f"Direction: {direction}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Lightweight 2-way prescreening: always-internal vs always-deterministic"
    )
    parser.add_argument("--task_type", type=str, required=True,
                        help="Single task type name")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Episodes to run (default: 5)")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--use_vllm", action="store_true", default=True)
    parser.add_argument("--no_vllm", action="store_true", default=False,
                        help="Use HF transformers instead of vLLM")
    parser.add_argument("--output_dir", type=str, default="results/prescreening/")
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--n_candidates", type=int, default=3,
                        help="Number of candidate actions for deterministic mode")
    args = parser.parse_args()

    if args.no_vllm:
        args.use_vllm = False

    random.seed(42)

    task_name = resolve_task_type(args.task_type)
    print(f"Task: {task_name}")
    print(f"Episodes: {args.episodes}, Max steps: {args.max_steps}")
    print(f"GPU: {args.gpu_id}, vLLM: {args.use_vllm}")

    agent = LLMAgent(
        model_path=args.model_path,
        use_vllm=args.use_vllm,
        gpu_id=args.gpu_id,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    run_prescreening(
        agent, task_name, args.episodes, args.max_steps,
        args.output_dir, args.n_candidates,
    )


if __name__ == "__main__":
    main()
