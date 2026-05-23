"""Reflexion baseline for ScienceWorld modular routing diagnostic.

Tests whether self-correction (Reflexion) can reduce internal modality death rate.
Baseline: 245 episodes, 42.9% death rate, DAF 90.4%.
Hypothesis: if Reflexion drops death rate below 20%, self-correction may rival routing.

Modes:
  --mode reflexion: failed episodes get LLM-generated reflection injected into next trial
  --mode retry: failed episodes get retried without reflection (controls for retry effect)
"""

import argparse
import json
import os
import random
import sys
import time
from typing import List, Dict, Any, Optional, Tuple

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scienceworld_utils import (
    create_env, reset_env, get_valid_actions, get_score,
    step_env, close_env, get_task_types,
)
from llm_agent import parse_action, _format_valid_actions, _format_history

TASKS = [
    "chemistry-mix", "find-animal", "find-non-living-thing", "freeze",
    "grow-fruit", "identify-life-stages-1", "identify-life-stages-2",
    "inclined-plane-determine-angle", "lifespan-longest-lived",
    "lifespan-longest-lived-then-shortest-lived", "measure-melting-point-unknown-substance",
]

DEATH_THRESHOLD = -100

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

REFLECTION_PROMPT = (
    "You just completed an episode in a ScienceWorld text environment but failed.\n\n"
    "Task: {task_name}\n"
    "Final score: {final_score}\n"
    "Outcome: {outcome}\n"
    "Number of steps taken: {num_steps}\n\n"
    "Here is a summary of your actions:\n{action_summary}\n\n"
    "Analyze why this episode failed and provide a short strategy for the next attempt. Focus on:\n"
    "1. What went wrong (wrong sequence, missed key step, dangerous action)?\n"
    "2. What should be done differently next time?\n\n"
    "Keep your reflection under 200 words."
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


class APIAgent:
    """ReAct agent using vLLM OpenAI-compatible API."""

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
                      extra_system: str = "") -> Tuple[str, str]:
        system_content = SYSTEM_PROMPT
        if extra_system:
            system_content += "\n\n" + extra_system
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

    def generate_reflection(self, task_name: str, final_score: float,
                            is_death: bool, num_steps: int,
                            action_history: List[str]) -> str:
        outcome = "death (score <= -100)" if is_death else f"low score ({final_score})"
        recent = action_history[-15:]
        action_summary = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(recent))
        if len(action_history) > 15:
            action_summary = (
                f"  ... ({len(action_history)-15} earlier actions omitted)\n"
                + action_summary
            )
        messages = [
            {"role": "system",
             "content": "You are a reflective agent analyzing your past performance."},
            {"role": "user", "content": REFLECTION_PROMPT.format(
                task_name=task_name,
                final_score=final_score,
                outcome=outcome,
                num_steps=num_steps,
                action_summary=action_summary,
            )},
        ]
        return self._chat(messages, temperature=0.5, max_tokens=300)


def run_episode(agent: APIAgent, task_name: str, variation_idx: int,
                max_steps: int = 30,
                reflection: str = "") -> Dict[str, Any]:
    """Run one episode with optional reflection injection."""
    env = create_env(task_name, variation_idx)
    obs, _ = reset_env(env)

    steps = []
    action_history = []
    history_dicts = []
    t0 = time.time()

    extra_system = ""
    if reflection:
        extra_system = (
            "REFLECTION FROM PREVIOUS ATTEMPT:\n"
            + reflection
            + "\n\nUse this reflection to avoid repeating past mistakes."
        )

    for step_idx in range(max_steps):
        valid_actions = get_valid_actions(env)
        if not valid_actions:
            break

        action, raw_output = agent.select_action(
            obs, valid_actions, history_dicts, extra_system=extra_system,
        )

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
    }


def run_trial_loop(agent: APIAgent, task_name: str, variation_idx: int,
                   max_trials: int, max_steps: int,
                   mode: str) -> Dict[str, Any]:
    """Run up to max_trials attempts with or without reflection."""
    trials = []
    accumulated_reflections: List[str] = []

    for trial_idx in range(max_trials):
        reflection_text = (
            "\n\n".join(accumulated_reflections) if accumulated_reflections else ""
        )

        ep = run_episode(
            agent, task_name, variation_idx, max_steps,
            reflection=reflection_text if mode == "reflexion" else "",
        )

        is_success = ep["final_score"] > 0 and not ep["is_death"]

        trial_record = {
            "trial_idx": trial_idx,
            "score": ep["final_score"],
            "is_death": ep["is_death"],
            "is_success": is_success,
            "num_steps": ep["num_steps"],
            "time_s": ep["total_time_s"],
            "actions": ep["action_history"],
            "reflection_text": None,
        }

        if not is_success and trial_idx < max_trials - 1 and mode == "reflexion":
            refl = agent.generate_reflection(
                task_name, ep["final_score"], ep["is_death"],
                ep["num_steps"], ep["action_history"],
            )
            trial_record["reflection_text"] = refl
            accumulated_reflections.append(f"[Trial {trial_idx + 1}] {refl}")

        trials.append(trial_record)

        if is_success:
            break

    final_trial = trials[-1]
    return {
        "task_name": task_name,
        "variation_idx": variation_idx,
        "mode": mode,
        "trials": trials,
        "num_trials": len(trials),
        "final_score": final_trial["score"],
        "final_is_death": final_trial["is_death"],
        "final_is_success": final_trial["is_success"],
        "success_trial": next(
            (t["trial_idx"] for t in trials if t["is_success"]), None
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Reflexion baseline for ScienceWorld modular routing diagnostic",
    )
    parser.add_argument(
        "--tasks", type=str, default=None,
        help="Comma-separated task subset (default: all 11)",
    )
    parser.add_argument(
        "--episodes-per-task", type=int, default=20,
        help="Episodes per task (default: 20)",
    )
    parser.add_argument(
        "--max-trials", type=int, default=3,
        help="Max retry trials per episode (default: 3)",
    )
    parser.add_argument(
        "--mode", type=str, choices=["reflexion", "retry"], required=True,
        help="reflexion or retry",
    )
    parser.add_argument(
        "--model", type=str, default="Qwen2.5-7B-Instruct",
        help="Model name for vLLM API",
    )
    parser.add_argument(
        "--api-base", type=str, default="http://localhost:8000/v1",
        help="vLLM OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: results/{mode}_baseline/)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=30,
        help="Max steps per episode (default: 30)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print plan without executing",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"results/{args.mode}_baseline/"

    task_list = list(TASKS)
    if args.tasks:
        task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
        for t in task_list:
            if t not in TASKS:
                print(f"Warning: '{t}' not in hardcoded task list")

    resolved_tasks = [resolve_task_type(t) for t in task_list]
    total_episodes = len(resolved_tasks) * args.episodes_per_task

    print(f"Mode: {args.mode}")
    print(f"Tasks: {len(resolved_tasks)} ({', '.join(task_list)})")
    print(f"Episodes per task: {args.episodes_per_task}")
    print(f"Max trials per episode: {args.max_trials}")
    print(f"Max steps per episode: {args.max_steps}")
    print(f"Total episodes: {total_episodes}")
    print(f"Model: {args.model}")
    print(f"API base: {args.api_base}")
    print(f"Output: {args.output_dir}")
    print(f"Seed: {args.seed}")

    if args.dry_run:
        print("\n[DRY RUN] Plan:")
        for t in task_list:
            print(f"  - {t}: {args.episodes_per_task} episodes x {args.max_trials} max trials")
        print(f"  Total: {total_episodes} episodes, up to {total_episodes * args.max_trials} trials")
        return

    random.seed(args.seed)
    agent = APIAgent(model=args.model, api_base=args.api_base)
    os.makedirs(args.output_dir, exist_ok=True)

    all_stats = {
        "mode": args.mode,
        "model": args.model,
        "max_trials": args.max_trials,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "tasks": {},
    }

    for task_short, task_resolved in zip(task_list, resolved_tasks):
        print(f"\n{'=' * 60}")
        print(f"Task: {task_short} -> {task_resolved}")
        print(f"{'=' * 60}")

        deaths = 0
        successes = 0
        total_trials_used = 0

        for ep_idx in range(args.episodes_per_task):
            variation_idx = ep_idx % 100
            print(
                f"  Episode {ep_idx + 1}/{args.episodes_per_task} (var={variation_idx}) ",
                end="", flush=True,
            )

            result = run_trial_loop(
                agent, task_resolved, variation_idx,
                args.max_trials, args.max_steps, args.mode,
            )

            status = (
                "SUCCESS" if result["final_is_success"]
                else ("DEATH" if result["final_is_death"] else "FAIL")
            )
            print(f"-> {status} (score={result['final_score']:.0f}, trials={result['num_trials']})")

            if result["final_is_death"]:
                deaths += 1
            if result["final_is_success"]:
                successes += 1
            total_trials_used += result["num_trials"]

            ep_file = os.path.join(
                args.output_dir, f"{task_short}_ep{ep_idx:03d}.json",
            )
            with open(ep_file, "w") as f:
                json.dump(result, f, indent=2)

        n = args.episodes_per_task
        death_rate = deaths / n if n > 0 else 0
        success_rate = successes / n if n > 0 else 0
        avg_trials = total_trials_used / n if n > 0 else 0

        task_stats = {
            "task_short": task_short,
            "task_resolved": task_resolved,
            "n_episodes": n,
            "deaths": deaths,
            "successes": successes,
            "death_rate": round(death_rate, 4),
            "success_rate": round(success_rate, 4),
            "avg_trials": round(avg_trials, 2),
        }
        all_stats["tasks"][task_short] = task_stats

        print(
            f"  Summary: death_rate={death_rate:.1%}, "
            f"success_rate={success_rate:.1%}, avg_trials={avg_trials:.1f}"
        )

    total_eps = sum(s["n_episodes"] for s in all_stats["tasks"].values())
    total_deaths = sum(s["deaths"] for s in all_stats["tasks"].values())
    total_successes = sum(s["successes"] for s in all_stats["tasks"].values())

    all_stats["global"] = {
        "total_episodes": total_eps,
        "total_deaths": total_deaths,
        "total_successes": total_successes,
        "overall_death_rate": round(total_deaths / total_eps, 4) if total_eps > 0 else 0,
        "overall_success_rate": round(total_successes / total_eps, 4) if total_eps > 0 else 0,
        "baseline_comparison": {
            "baseline_death_rate": 0.429,
            "baseline_daf": 0.904,
            "baseline_episodes": 245,
        },
    }

    summary_file = os.path.normpath(
        os.path.join(args.output_dir, "..", "reflexion_summary.json")
    )
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    with open(summary_file, "w") as f:
        json.dump(all_stats, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"GLOBAL SUMMARY ({args.mode})")
    print(f"{'=' * 60}")
    print(f"Total episodes: {total_eps}")
    print(f"Deaths: {total_deaths} ({all_stats['global']['overall_death_rate']:.1%})")
    print(f"Successes: {total_successes} ({all_stats['global']['overall_success_rate']:.1%})")
    print(f"Baseline death rate: 42.9% (245 episodes)")
    print(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    main()
