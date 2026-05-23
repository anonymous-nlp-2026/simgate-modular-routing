"""Grounding Prompt Ablation (Condition A): Test whether explicit safety warnings
in the system prompt reduce death rate from 'focus on' misuse.

Adds a CRITICAL SAFETY RULE to the standard system prompt, then runs
always-internal episodes on death-prone tasks. Output format matches
episode_level_collection.py for downstream death analysis compatibility.

Usage:
  # Dry-run (1 task, 1 episode)
  python src/run_grounding_prompt_ablation.py \
    --task_types chemistry-mix --episodes 1 --gpu_id 0

  # Full run (10 tasks x 20 episodes)
  python src/run_grounding_prompt_ablation.py \
    --task_types "chemistry-mix,find-animal,freeze,grow-fruit,identify-life-stages-1,identify-life-stages-2,inclined-plane-determine-angle,lifespan-longest-lived,lifespan-longest-lived-then-shortest-lived,measure-melting-point-unknown-substance" \
    --episodes 20 --gpu_id 0
"""

import argparse
import json
import os
import random
import sys
import time
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Monkey-patch system prompt BEFORE importing functions that use it
import llm_agent

GROUNDING_WARNING = """
CRITICAL SAFETY RULE: The 'focus on [object]' action is ONLY for submitting your final answer. Do NOT use 'focus on' to examine, inspect, or interact with objects during exploration. Using 'focus on' on the wrong object will cause irreversible failure. Use 'look at', 'examine', or 'pick up' instead for exploration."""

ORIGINAL_SYSTEM_PROMPT = llm_agent.SYSTEM_PROMPT
PATCHED_SYSTEM_PROMPT = llm_agent.SYSTEM_PROMPT + GROUNDING_WARNING
llm_agent.SYSTEM_PROMPT = PATCHED_SYSTEM_PROMPT

from llm_agent import LLMAgent, find_model_path
from scienceworld_utils import (
    create_env, reset_env, get_valid_actions, get_score,
    step_env, close_env, get_task_types, get_max_variations,
)

DEATH_THRESHOLD = -100
SLIM_EPISODES = os.environ.get("SLIM_EPISODES", "1") == "1"


def _slim_step(step_data: dict) -> dict:
    if not SLIM_EPISODES:
        return step_data
    return {k: v for k, v in step_data.items()
            if k not in ("llm_output", "action_history", "candidates", "available_actions")}


def resolve_task_type(name: str) -> str:
    try:
        available = get_task_types()
    except Exception:
        return name
    if name in available:
        return name
    matches = [t for t in available if name.lower() in t.lower()]
    return matches[0] if matches else name


def run_episode_internal(
    agent: LLMAgent,
    task_name: str,
    variation_idx: int,
    max_steps: int,
) -> Dict[str, Any]:
    env = create_env(task_name, variation_idx)
    obs, _ = reset_env(env)
    initial_score = get_score(env)

    steps = []
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

        step_data = {
            "step": step_idx,
            "action": action,
            "modality": "internal",
            "score_before": score_before,
            "score_after": score_after,
            "score_delta": score_after - score_before,
            "observation": obs_new[:500],
            "llm_output": raw_output[:300],
            "action_history": [s["action"] for s in steps],
            "is_dead": score_after <= DEATH_THRESHOLD,
        }
        steps.append(step_data)
        history_dicts.append({
            "action": action,
            "observation": obs_new[:200],
            "score": score_after,
        })
        obs = obs_new

        if done:
            break

    final_score = get_score(env)
    total_time = time.time() - t0
    close_env(env)

    slim_steps = [_slim_step(s) for s in steps] if SLIM_EPISODES else steps
    return {
        "mode": "always-internal",
        "condition": "grounding-prompt",
        "task_name": task_name,
        "variation_idx": variation_idx,
        "final_score": final_score,
        "initial_score": initial_score,
        "num_steps": len(steps),
        "total_time_s": total_time,
        "death": final_score <= DEATH_THRESHOLD,
        "steps": slim_steps,
    }


def run_task(agent, task_name, n_episodes, max_steps, output_dir):
    task_dir = os.path.join(output_dir, task_name.replace(" ", "_"))
    os.makedirs(task_dir, exist_ok=True)

    episodes_file = os.path.join(task_dir, "episodes.jsonl")
    summary_file = os.path.join(task_dir, "episode_summary.jsonl")

    max_var = get_max_variations(task_name)
    deaths = 0
    scores = []

    for ep_idx in range(n_episodes):
        variation_idx = ep_idx % max(max_var, 1)
        print(f"  [{task_name}] ep {ep_idx}/{n_episodes} var={variation_idx} ...", end=" ", flush=True)

        t = time.time()
        result = run_episode_internal(agent, task_name, variation_idx, max_steps)
        elapsed = time.time() - t

        is_death = result["death"]
        if is_death:
            deaths += 1
        scores.append(result["final_score"])

        print(f"score={result['final_score']:.1f} steps={result['num_steps']} "
              f"death={'YES' if is_death else 'no'} ({elapsed:.0f}s)")

        ep_entry = {
            "episode_idx": ep_idx,
            "variation_idx": variation_idx,
            "always_internal": result,
        }
        with open(episodes_file, "a") as f:
            f.write(json.dumps(ep_entry) + "\n")

        ep_summary = {
            "episode": ep_idx,
            "variation": variation_idx,
            "condition": "grounding-prompt",
            "internal_final": result["final_score"],
            "internal_steps": result["num_steps"],
            "internal_time_s": result["total_time_s"],
            "death": is_death,
        }
        with open(summary_file, "a") as f:
            f.write(json.dumps(ep_summary) + "\n")

    death_rate = deaths / n_episodes if n_episodes > 0 else 0
    avg_score = sum(scores) / len(scores) if scores else 0

    meta = {
        "task_name": task_name,
        "condition": "grounding-prompt-ablation",
        "system_prompt_patch": "CRITICAL SAFETY RULE appended",
        "target_episodes": n_episodes,
        "collected_episodes": n_episodes,
        "deaths": deaths,
        "death_rate": death_rate,
        "avg_score": avg_score,
        "seed": GLOBAL_SEED,
        "max_steps": max_steps,
        "temperature": 0.3,
    }
    with open(os.path.join(task_dir, "collection_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  [{task_name}] DONE: death_rate={death_rate:.2%} avg_score={avg_score:.1f}")
    return meta


GLOBAL_SEED = 42


def main():
    global GLOBAL_SEED

    parser = argparse.ArgumentParser(
        description="Grounding Prompt Ablation (Condition A): test safety warnings vs death rate"
    )
    parser.add_argument("--task_types", type=str, required=True,
                        help="Comma-separated task type names")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of episodes per task (default: 20)")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir", type=str,
                        default="results/grounding_prompt_ablation/",
                        help="Output directory")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tp", type=int, default=1,
                        help="Tensor parallel size (default: 1)")
    args = parser.parse_args()

    GLOBAL_SEED = args.seed
    random.seed(args.seed)

    task_names_raw = [t.strip() for t in args.task_types.split(",") if t.strip()]
    task_names = [resolve_task_type(t) for t in task_names_raw]

    print("=" * 60)
    print("GROUNDING PROMPT ABLATION (Condition A)")
    print("=" * 60)
    print(f"Tasks: {task_names}")
    print(f"Episodes: {args.episodes} | Max steps: {args.max_steps}")
    print(f"GPU: {args.gpu_id} | Seed: {args.seed}")
    print(f"Output: {args.output_dir}")
    print()
    print("System prompt patch:")
    print(GROUNDING_WARNING.strip())
    print("=" * 60)

    # Verify patch is active
    assert llm_agent.SYSTEM_PROMPT == PATCHED_SYSTEM_PROMPT, "System prompt patch failed!"
    assert "CRITICAL SAFETY RULE" in llm_agent.SYSTEM_PROMPT

    agent = LLMAgent(
        model_path=args.model_path,
        use_vllm=True,
        gpu_id=args.gpu_id,
        tensor_parallel_size=args.tp,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}

    for task_name in task_names:
        result = run_task(agent, task_name, args.episodes, args.max_steps, args.output_dir)
        all_results[task_name] = result

    # Global summary
    total_eps = sum(r["collected_episodes"] for r in all_results.values())
    total_deaths = sum(r["deaths"] for r in all_results.values())
    global_death_rate = total_deaths / total_eps if total_eps > 0 else 0

    summary = {
        "condition": "grounding-prompt-ablation",
        "total_episodes": total_eps,
        "total_deaths": total_deaths,
        "global_death_rate": global_death_rate,
        "per_task": all_results,
    }
    summary_path = os.path.join(args.output_dir, "ablation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"GLOBAL SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total episodes: {total_eps}")
    print(f"Total deaths: {total_deaths}")
    print(f"Global death rate: {global_death_rate:.2%}")
    for tn, r in all_results.items():
        print(f"  {tn}: death_rate={r['death_rate']:.2%} avg_score={r['avg_score']:.1f}")
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
