#!/usr/bin/env python3
"""Condition B: Grounding filter ablation on death-prone tasks.

Hard-blocks 'focus on' actions in all steps except the last 3,
forcing the agent to re-select. Tests whether death vulnerability
is structural (agent always re-picks focus on) or correctable.

Uses vLLM OpenAI-compatible API (same as reflexion_baseline).
"""

import argparse
import json
import os
import random
import re
import sys
import time
from typing import List, Dict, Any, Tuple

from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from scienceworld_utils import (
    create_env, reset_env, get_valid_actions, get_score,
    step_env, close_env, get_task_types,
)
from llm_agent import parse_action, _format_valid_actions, _format_history

DEATH_THRESHOLD = -100
MAX_FILTER_RETRIES = 3
FOCUS_ON_WINDOW = 3  # allow focus on in last N steps

DEATH_PRONE_TASKS = [
    "chemistry-mix", "find-animal", "freeze",
    "grow-fruit", "identify-life-stages-1", "identify-life-stages-2",
    "inclined-plane-determine-angle", "lifespan-longest-lived",
    "lifespan-longest-lived-then-shortest-lived",
    "measure-melting-point-unknown-substance",
]

SYSTEM_PROMPT = (
    "You are an agent interacting with a ScienceWorld text environment. "
    "Your goal is to complete the given task by choosing actions from the available action list.\n\n"
    "Use the following format for each step:\n"
    "Thought: <your reasoning about what to do next>\n"
    "Action: <exact action string from the valid actions list>\n\n"
    "Rules:\n"
    "- You MUST choose an action from the valid actions list. Do not invent actions.\n"
    "- Think step by step about what physical/scientific process is needed.\n"
    "- Pay attention to the current observation for clues about object states."
)

REACT_USER_TEMPLATE = (
    "Current observation:\n{observation}\n\n"
    "Valid actions:\n{valid_actions_str}\n\n"
    "Previous actions taken: {history_str}\n\n"
    "Choose your next action."
)

FILTER_FEEDBACK = (
    "Invalid action. 'focus on' can only be used to submit your final answer "
    "in the last few steps. Please choose a different action."
)


def resolve_task_type(name: str) -> str:
    try:
        available = get_task_types()
    except Exception:
        return name
    if name in available:
        return name
    matches = [t for t in available if name.lower() in t.lower()]
    return matches[0] if matches else name


def is_focus_on(action: str) -> bool:
    return action.strip().lower().startswith("focus on")


class FilterAPIAgent:
    """ReAct agent with focus-on filtering via vLLM OpenAI API."""

    def __init__(self, model: str, api_base: str, temperature: float = 0.3,
                 max_tokens: int = 256):
        self.client = OpenAI(base_url=api_base, api_key="dummy")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _chat(self, messages: List[dict], temperature: float = None,
              max_tokens: int = None) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            stop=["Observation:", "\nObservation"],
        )
        return resp.choices[0].message.content.strip()

    def select_action(self, observation: str, valid_actions: List[str],
                      history: List[dict],
                      extra_context: str = "") -> Tuple[str, str]:
        system_content = SYSTEM_PROMPT
        if extra_context:
            system_content += "\n\n" + extra_context

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": REACT_USER_TEMPLATE.format(
                observation=observation,
                valid_actions_str=_format_valid_actions(valid_actions),
                history_str=_format_history(history),
            )},
        ]
        raw = self._chat(messages)
        action = parse_action(raw, valid_actions)
        if action is None:
            action = valid_actions[0]
        return action, raw


def run_episode_with_filter(
    agent: FilterAPIAgent,
    task_name: str,
    variation_idx: int,
    max_steps: int = 30,
) -> Dict[str, Any]:
    """Run one episode with focus-on hard filter."""
    env = create_env(task_name, variation_idx)
    obs, _ = reset_env(env)

    steps = []
    action_history = []
    history_dicts = []
    filter_log = []
    t0 = time.time()

    for step_idx in range(max_steps):
        valid_actions = get_valid_actions(env)
        if not valid_actions:
            break

        remaining_steps = max_steps - step_idx
        allow_focus = remaining_steps <= FOCUS_ON_WINDOW

        action, raw_output = agent.select_action(obs, valid_actions, history_dicts)

        retries = 0
        while is_focus_on(action) and not allow_focus and retries < MAX_FILTER_RETRIES:
            filter_log.append({
                "step_idx": step_idx,
                "retry": retries,
                "blocked_action": action,
                "remaining_steps": remaining_steps,
            })
            print(f"    [FILTER] step={step_idx} blocked: {action} (retry {retries+1})")

            non_focus_actions = [a for a in valid_actions if not is_focus_on(a)]
            if not non_focus_actions:
                break

            action, raw_output = agent.select_action(
                obs, non_focus_actions, history_dicts,
                extra_context=FILTER_FEEDBACK,
            )
            retries += 1

        if is_focus_on(action) and not allow_focus:
            filter_log.append({
                "step_idx": step_idx,
                "retry": retries,
                "blocked_action": action,
                "forced_passthrough": True,
                "remaining_steps": remaining_steps,
            })
            print(f"    [FILTER] step={step_idx} passthrough after {retries} retries: {action}")

        score_before = get_score(env)
        obs_new, reward, done, info = step_env(env, action)
        score_after = get_score(env)

        steps.append({
            "step_idx": step_idx,
            "action": action,
            "observation": obs_new[:300],
            "score_before": score_before,
            "score_after": score_after,
            "done": done,
            "was_filtered": retries > 0,
            "filter_retries": retries,
        })
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
    total_time = time.time() - t0
    close_env(env)

    return {
        "task_name": task_name,
        "variation_idx": variation_idx,
        "final_score": final_score,
        "is_death": final_score <= DEATH_THRESHOLD,
        "num_steps": len(steps),
        "total_time_s": round(total_time, 1),
        "action_history": action_history,
        "steps": steps,
        "filter_log": filter_log,
        "total_blocks": len([e for e in filter_log if not e.get("forced_passthrough")]),
        "total_passthroughs": len([e for e in filter_log if e.get("forced_passthrough")]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Condition B: grounding filter ablation on death-prone tasks")
    parser.add_argument("--episodes-per-task", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--model", type=str, default="Qwen2.5-7B-Instruct")
    parser.add_argument("--api-base", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--output-dir", type=str,
                        default="results/grounding_filter_ablation/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks", type=str, default=None,
                        help="Comma-separated task subset (default: all 10 death-prone)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run 1 task, 1 episode only")
    args = parser.parse_args()

    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",")]
    else:
        tasks = list(DEATH_PRONE_TASKS)

    if args.dry_run:
        tasks = tasks[:1]
        args.episodes_per_task = 1

    resolved_tasks = [resolve_task_type(t) for t in tasks]
    total_episodes = len(tasks) * args.episodes_per_task

    print(f"Grounding filter ablation (Condition B)")
    print(f"Tasks: {tasks}")
    print(f"Episodes per task: {args.episodes_per_task}")
    print(f"Total episodes: {total_episodes}")
    print(f"Filter: block 'focus on' except last {FOCUS_ON_WINDOW} steps")
    print(f"Max retries per block: {MAX_FILTER_RETRIES}")
    print(f"Model: {args.model} @ {args.api_base}")
    print(f"Seed: {args.seed}")

    random.seed(args.seed)
    agent = FilterAPIAgent(model=args.model, api_base=args.api_base)
    os.makedirs(args.output_dir, exist_ok=True)

    all_stats = {
        "condition": "grounding_filter",
        "model": args.model,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "filter_window": FOCUS_ON_WINDOW,
        "max_filter_retries": MAX_FILTER_RETRIES,
        "dry_run": args.dry_run,
        "tasks": {},
    }

    for task_short, task_resolved in zip(tasks, resolved_tasks):
        print(f"\n{'=' * 60}")
        print(f"Task: {task_short} -> {task_resolved}")
        print(f"{'=' * 60}")

        deaths = 0
        successes = 0
        total_blocks = 0
        total_passthroughs = 0

        for ep_idx in range(args.episodes_per_task):
            variation_idx = ep_idx % 100
            print(f"  Ep {ep_idx+1}/{args.episodes_per_task} (var={variation_idx}) ",
                  end="", flush=True)

            result = run_episode_with_filter(
                agent, task_resolved, variation_idx, args.max_steps,
            )

            status = (
                "SUCCESS" if result["final_score"] > 0 and not result["is_death"]
                else ("DEATH" if result["is_death"] else "FAIL")
            )
            print(f"-> {status} (score={result['final_score']:.0f}, "
                  f"steps={result['num_steps']}, "
                  f"blocks={result['total_blocks']}, "
                  f"pass={result['total_passthroughs']})")

            if result["is_death"]:
                deaths += 1
            if result["final_score"] > 0 and not result["is_death"]:
                successes += 1
            total_blocks += result["total_blocks"]
            total_passthroughs += result["total_passthroughs"]

            ep_file = os.path.join(args.output_dir, f"{task_short}_ep{ep_idx:03d}.json")
            with open(ep_file, "w") as f:
                json.dump(result, f, indent=2)

        n = args.episodes_per_task
        task_stats = {
            "task_short": task_short,
            "task_resolved": task_resolved,
            "n_episodes": n,
            "deaths": deaths,
            "successes": successes,
            "death_rate": round(deaths / n, 4) if n > 0 else 0,
            "success_rate": round(successes / n, 4) if n > 0 else 0,
            "total_blocks": total_blocks,
            "total_passthroughs": total_passthroughs,
            "avg_blocks_per_ep": round(total_blocks / n, 2) if n > 0 else 0,
        }
        all_stats["tasks"][task_short] = task_stats
        print(f"  DR={task_stats['death_rate']:.1%}, "
              f"SR={task_stats['success_rate']:.1%}, "
              f"avg_blocks={task_stats['avg_blocks_per_ep']}")

    total_eps = sum(s["n_episodes"] for s in all_stats["tasks"].values())
    total_deaths = sum(s["deaths"] for s in all_stats["tasks"].values())
    total_successes = sum(s["successes"] for s in all_stats["tasks"].values())
    total_all_blocks = sum(s["total_blocks"] for s in all_stats["tasks"].values())

    all_stats["aggregate"] = {
        "total_episodes": total_eps,
        "total_deaths": total_deaths,
        "total_successes": total_successes,
        "death_rate": round(total_deaths / total_eps, 4) if total_eps > 0 else 0,
        "success_rate": round(total_successes / total_eps, 4) if total_eps > 0 else 0,
        "total_blocks": total_all_blocks,
        "baseline_comparison": {
            "baseline_death_rate": 0.429,
            "baseline_daf": 0.904,
            "baseline_episodes": 245,
        },
    }

    summary_file = os.path.join(args.output_dir, "filter_ablation_summary.json")
    with open(summary_file, "w") as f:
        json.dump(all_stats, f, indent=2)

    print(f"\n{'=' * 60}")
    print("GROUNDING FILTER ABLATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total episodes: {total_eps}")
    print(f"Deaths: {total_deaths} ({all_stats['aggregate']['death_rate']:.1%})")
    print(f"Successes: {total_successes} ({all_stats['aggregate']['success_rate']:.1%})")
    print(f"Total blocks: {total_all_blocks}")
    print(f"Baseline death rate: 42.9% (245 episodes, no filter)")
    print(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    main()
