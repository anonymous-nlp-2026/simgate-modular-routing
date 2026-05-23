import argparse
import json
import os
import random
import time
from datetime import datetime

import numpy as np
import crafter
from text_wrapper import CrafterTextWrapper
from config import MAX_STEPS, RESULTS_DIR, ACTIONS
from seed_patch import apply_determinism_patch


def run_episode(agent, env_wrapper, max_steps=MAX_STEPS):
    agent.reset()
    obs_text = env_wrapper.reset()
    
    steps = []
    total_reward = 0.0
    
    for step_idx in range(max_steps):
        action_name = agent.act(obs_text)
        obs_text, reward, done, info = env_wrapper.step(action_name)
        total_reward += reward
        
        step_record = {
            "step": step_idx,
            "action": action_name,
            "reward": reward,
            "health": info["inventory"]["health"],
            "food": info["inventory"]["food"],
            "drink": info["inventory"]["drink"],
            "energy": info["inventory"]["energy"],
        }
        steps.append(step_record)
        
        if done:
            break
    
    is_dead = info["inventory"]["health"] <= 0
    unlocked = [k for k, v in info["achievements"].items() if v > 0]
    
    result = {
        "total_steps": len(steps),
        "total_reward": total_reward,
        "is_death": is_dead,
        "final_health": info["inventory"]["health"],
        "achievements_unlocked": unlocked,
        "num_achievements": len(unlocked),
        "steps": steps,
    }
    return result


def run_heuristic_episode(agent, env, max_steps=MAX_STEPS):
    agent.reset()
    obs = env.reset()
    _, _, _, info = env.step(0)

    steps = []
    total_reward = 0.0

    for step_idx in range(max_steps):
        action = agent.act(obs, info)
        obs, reward, done, info = env.step(action)
        total_reward += reward

        step_record = {
            "step": step_idx,
            "action": ACTIONS[action] if action < len(ACTIONS) else str(action),
            "reward": reward,
            "health": info["inventory"]["health"],
            "food": info["inventory"]["food"],
            "drink": info["inventory"]["drink"],
            "energy": info["inventory"]["energy"],
        }
        steps.append(step_record)

        if done:
            break

    is_dead = info["inventory"]["health"] <= 0
    unlocked = [k for k, v in info["achievements"].items() if v > 0]

    result = {
        "total_steps": len(steps),
        "total_reward": total_reward,
        "is_death": is_dead,
        "final_health": info["inventory"]["health"],
        "achievements_unlocked": unlocked,
        "num_achievements": len(unlocked),
        "steps": steps,
    }
    return result


def main():
    apply_determinism_patch()
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["llm", "random", "heuristic"], default="random")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    env = crafter.Env(seed=args.seed)
    use_heuristic = args.agent == "heuristic"

    if use_heuristic:
        from heuristic_agent import CrafterHeuristicAgent
        agent = CrafterHeuristicAgent(seed=args.seed)
    elif args.agent == "random":
        from llm_agent import RandomAgent
        agent = RandomAgent()
    else:
        from llm_agent import CrafterLLMAgent
        agent = CrafterLLMAgent()

    if not use_heuristic:
        wrapper = CrafterTextWrapper(env)

    all_results = []
    for ep in range(args.episodes):
        t0 = time.time()
        if use_heuristic:
            result = run_heuristic_episode(agent, env, max_steps=args.max_steps)
        else:
            result = run_episode(agent, wrapper, max_steps=args.max_steps)
        elapsed = time.time() - t0
        result["episode"] = ep
        result["elapsed_seconds"] = round(elapsed, 2)
        all_results.append(result)
        
        print(f"Episode {ep}: steps={result['total_steps']}, "
              f"reward={result['total_reward']:.2f}, "
              f"death={result['is_death']}, "
              f"achievements={result['num_achievements']}, "
              f"time={elapsed:.1f}s")

    if args.output:
        outfile = args.output
        os.makedirs(os.path.dirname(outfile) if os.path.dirname(outfile) else ".", exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outfile = os.path.join(RESULTS_DIR, f"{args.agent}_{timestamp}.json")
    with open(outfile, "w") as f:
        json.dump({
            "agent_type": args.agent,
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "results": all_results,
        }, f, indent=2)
    print(f"\nResults saved to: {outfile}")


if __name__ == "__main__":
    main()
