#!/usr/bin/env python3
"""
MiniHack Episode Runner
Runs episodes with deterministic (heuristic) or internal (LLM) agents.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import gym

from minihack_det_agent import (
    DeterministicHeuristicAgent,
    ActionMapper,
    OBS_KEYS,
    TASK_TYPES,
    _env_reset,
    _env_step,
)


LANG_OBS_KEYS = (
    "glyphs", "blstats", "message", "screen_descriptions",
    "tty_chars", "tty_cursor", "inv_letters", "inv_strs",
    "inv_glyphs", "inv_oclasses",
)


class BaseAgent:
    def reset(self):
        raise NotImplementedError
    def act(self, obs):
        raise NotImplementedError


class RandomAgent(BaseAgent):
    def __init__(self, env):
        self.n = env.action_space.n
    def reset(self):
        pass
    def act(self, obs):
        import random
        return random.randint(0, self.n - 1)


class LLMAgent(BaseAgent):
    SYSTEM_PROMPT = (
        "You are playing MiniHack. Your ONLY goal is to reach the 'stairs down'. "
        "IGNORE 'stairs up', it is NOT your goal.\n\n"
        "Distance cues in the map description:\n"
        "- 'adjacent [dir]' = 1 step away, move that direction NOW\n"
        "- 'very near [dir]' = 2 steps away\n"
        "- 'near [dir]' = a few steps away\n\n"
        "Always move TOWARD 'stairs down'. "
        "Reply with ONLY the direction name (e.g. southeast). No other text."
    )

    def __init__(self, env, api_base="http://localhost:8000/v1", model=None):
        from openai import OpenAI
        self.client = OpenAI(base_url=api_base, api_key="dummy")
        self.model = model or "Qwen2.5-7B-Instruct"
        self.valid_actions = sorted(
            [a for a in env.action_str_enum_map.keys() if len(a) > 1]
        )
        self.all_valid = set(env.action_str_enum_map.keys())
        self.fallback_action = self.valid_actions[0] if self.valid_actions else "north"
        self.parse_failures = 0
        self.total_calls = 0

    def reset(self):
        self.parse_failures = 0
        self.total_calls = 0

    def act(self, obs):
        self.total_calls += 1
        prompt = self._build_prompt(obs)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=20,
            )
            action_text = response.choices[0].message.content.strip()
            return self._parse_action(action_text)
        except Exception as e:
            print(f"    LLM error: {e}", file=sys.stderr)
            self.parse_failures += 1
            return self.fallback_action

    def _build_prompt(self, obs):
        parts = []
        for key, label in [
            ("text_glyphs", "Map"),
            ("text_message", "Message"),
            ("text_blstats", "Stats"),
            ("text_inventory", "Inventory"),
        ]:
            val = obs.get(key, "")
            if isinstance(val, bytes):
                val = val.decode("utf-8", errors="replace")
            val = str(val).strip()
            if val:
                if key == "text_glyphs":
                    parts.append(f"{label}:\n{val}")
                else:
                    parts.append(f"{label}: {val}")
        if not parts:
            for key in sorted(obs.keys()):
                if "text" in key.lower():
                    val = obs[key]
                    if isinstance(val, bytes):
                        val = val.decode("utf-8", errors="replace")
                    val = str(val).strip()
                    if val:
                        parts.append(f"{key}: {val}")
        actions_str = ", ".join(self.valid_actions)
        parts.append(f"\nAvailable actions: {actions_str}")
        parts.append("Which direction has 'stairs down'? Move toward it.")
        return "\n".join(parts)

    def _parse_action(self, text):
        text = text.lower().strip().strip('"').strip("'").strip(".")
        if text in self.all_valid:
            return text
        for action in self.valid_actions:
            if action in text:
                return action
        self.parse_failures += 1
        return self.fallback_action


def run_episode(env, agent, seed, max_steps=500):
    obs = _env_reset(env, seed=seed)
    agent.reset()
    total_reward = 0.0
    actions = []
    t0 = time.time()
    for _ in range(max_steps):
        action = agent.act(obs)
        actions.append(str(action))
        obs, reward, done, info = _env_step(env, action)
        total_reward += reward
        if done:
            break
    elapsed = time.time() - t0
    end_status = info.get("end_status", -1) if isinstance(info, dict) else -1
    died = end_status == 1
    goal_reached = end_status in (2, 3)
    result = {
        "score": float(total_reward),
        "died": bool(died),
        "goal_reached": bool(goal_reached),
        "steps": len(actions),
        "wall_time_s": round(elapsed, 2),
        "end_status": int(end_status),
    }
    if hasattr(agent, "parse_failures"):
        result["llm_parse_failures"] = agent.parse_failures
        result["llm_total_calls"] = agent.total_calls
    return result


def run_batch(task, strategy, n_episodes, base_seed, max_steps,
              output_path=None, model_name=None):
    import minihack

    if strategy == "internal":
        from nle_language_wrapper import NLELanguageWrapper
        base_env = gym.make(task, observation_keys=LANG_OBS_KEYS)
        env = NLELanguageWrapper(base_env)
        agent = LLMAgent(env, model=model_name)
    elif strategy == "deterministic":
        env = gym.make(task, observation_keys=OBS_KEYS)
        agent = DeterministicHeuristicAgent(env)
    elif strategy == "random":
        env = gym.make(task, observation_keys=OBS_KEYS)
        agent = RandomAgent(env)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    results = []
    for ep in range(n_episodes):
        seed = base_seed + ep
        result = run_episode(env, agent, seed=seed, max_steps=max_steps)
        result.update({
            "episode_id": ep,
            "task": task,
            "strategy": strategy,
            "seed": seed,
        })
        results.append(result)
        status = "DIED" if result["died"] else ("GOAL" if result["goal_reached"] else "TIMEOUT")
        extra = ""
        if "llm_parse_failures" in result:
            extra = f"  parse_fail={result['llm_parse_failures']}/{result['llm_total_calls']}"
        print(f"  [{strategy}] ep={ep:3d}  seed={seed}  steps={result['steps']:4d}  "
              f"score={result['score']:+.1f}  {status}  ({result['wall_time_s']:.1f}s){extra}")

    env.close()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"Saved {len(results)} episodes to {output_path}")

    return results


def print_summary(results):
    from collections import defaultdict
    by_key = defaultdict(list)
    for r in results:
        key = (r["task"], r["strategy"])
        by_key[key].append(r)
    print("\n" + "=" * 70)
    print(f"{'Task':<35s} {'Strategy':<15s} {'Episodes':>8s} {'Wins':>6s} {'Deaths':>7s} {'Avg Steps':>10s}")
    print("-" * 70)
    for (task, strat), rlist in sorted(by_key.items()):
        n = len(rlist)
        wins = sum(1 for r in rlist if r["goal_reached"])
        deaths = sum(1 for r in rlist if r["died"])
        avg_steps = sum(r["steps"] for r in rlist) / n
        print(f"{task:<35s} {strat:<15s} {n:>8d} {wins:>6d} {deaths:>7d} {avg_steps:>10.1f}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="MiniHack episode runner")
    parser.add_argument("--task", default="MiniHack-Room-5x5-v0")
    parser.add_argument("--strategy", default="deterministic",
                        choices=["deterministic", "internal", "random"])
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--output", default=None)
    parser.add_argument("--model", default=None,
                        help="vLLM model name/path")
    parser.add_argument("--tasks-all", action="store_true")
    args = parser.parse_args()

    tasks = TASK_TYPES if args.tasks_all else [args.task]
    output = args.output

    all_results = []
    for task in tasks:
        if output is None:
            safe_name = task.replace("/", "_")
            out = f"results/episode_summary_{safe_name}_{args.strategy}.jsonl"
        else:
            out = output
        print(f"\n>>> {task} | strategy={args.strategy} | episodes={args.episodes} | seed={args.seed}")
        results = run_batch(
            task=task,
            strategy=args.strategy,
            n_episodes=args.episodes,
            base_seed=args.seed,
            max_steps=args.max_steps,
            output_path=out,
            model_name=args.model,
        )
        all_results.extend(results)

    print_summary(all_results)


if __name__ == "__main__":
    main()
