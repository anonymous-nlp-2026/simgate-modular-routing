#!/usr/bin/env python3
"""Death rate measurement for Qwen2.5-0.5B-Instruct on ScienceWorld."""
import argparse, json, os, sys, time, random

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from scienceworld_utils import (
    create_env, reset_env, get_valid_actions, get_score,
    step_env, close_env, get_task_types, get_max_variations,
)
from llm_agent import LLMAgent

DEATH_THRESHOLD = -100

def run_episode_internal(agent, task_name, variation_idx, max_steps):
    env = create_env(task_name, variation_idx)
    obs, _ = reset_env(env)
    initial_score = get_score(env)
    action_history = []
    history_dicts = []
    t0 = time.time()

    for step_idx in range(max_steps):
        valid_actions = get_valid_actions(env)
        if not valid_actions:
            break
        action, raw = agent.select_action_react(obs, valid_actions, history_dicts)
        score_before = get_score(env)
        obs_new, reward, done, info = step_env(env, action)
        score_after = get_score(env)
        action_history.append(action)
        history_dicts.append({"action": action, "observation": obs_new[:200], "score": score_after})
        obs = obs_new
        if done:
            break

    final_score = get_score(env)
    close_env(env)
    death = final_score <= DEATH_THRESHOLD
    return {
        "task_name": task_name,
        "variation_idx": variation_idx,
        "final_score": final_score,
        "num_steps": len(action_history),
        "death": death,
        "time_s": round(time.time() - t0, 1),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--task_types", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output", type=str, default="guided_router/scaling_death_rate_0.5b.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    task_names = [t.strip() for t in args.task_types.split(",")]

    available = get_task_types()
    resolved = []
    for t in task_names:
        if t in available:
            resolved.append(t)
        else:
            matches = [a for a in available if t.lower() in a.lower()]
            if matches:
                resolved.append(matches[0])
                print(f"Resolved '{t}' -> '{matches[0]}'")
            else:
                print(f"WARNING: task '{t}' not found, skipping")
    task_names = resolved
    print(f"Tasks to run: {task_names}")

    from episode_level_collection import cleanup_gpu_processes
    cleanup_gpu_processes(args.gpu_id)

    agent = LLMAgent(model_path=args.model_path, use_vllm=True, gpu_id=args.gpu_id)

    results = []
    for task_name in task_names:
        max_var = get_max_variations(task_name)
        for ep in range(args.episodes):
            var_idx = ep % max_var
            print(f"[{task_name}] Episode {ep+1}/{args.episodes} (var={var_idx})...", end=" ", flush=True)
            try:
                r = run_episode_internal(agent, task_name, var_idx, args.max_steps)
                r["episode"] = ep
                results.append(r)
                status = "DEATH" if r["death"] else f"score={r['final_score']:.1f}"
                print(f"{status} ({r['num_steps']} steps, {r['time_s']}s)")
            except Exception as e:
                print(f"ERROR: {e}")
                results.append({
                    "task_name": task_name, "episode": ep, "variation_idx": var_idx,
                    "final_score": -999, "death": True, "error": str(e),
                    "num_steps": 0, "time_s": 0,
                })

            os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print("DEATH RATE SUMMARY: Qwen2.5-0.5B-Instruct")
    print("="*60)
    for task_name in task_names:
        task_results = [r for r in results if r["task_name"] == task_name]
        deaths = sum(1 for r in task_results if r["death"])
        total = len(task_results)
        pct = 100 * deaths / total if total > 0 else 0
        print(f"  {task_name}: death {deaths}/{total} ({pct:.0f}%)")

    all_deaths = sum(1 for r in results if r["death"])
    all_total = len(results)
    overall_pct = 100 * all_deaths / all_total if all_total > 0 else 0
    print(f"\nOverall: death {all_deaths}/{all_total} ({overall_pct:.1f}%)")
    print(f"Comparison: Qwen2.5-7B death rate = 42.9%")
    delta = overall_pct - 42.9
    direction = "more" if delta > 0 else "less"
    print(f"Delta: {delta:+.1f}pp (0.5B {direction} death-prone)")
    hypothesis = "supports" if delta > 0 else "contradicts"
    print(f"Conclusion: {hypothesis} 'smaller model -> higher death rate' hypothesis")

if __name__ == "__main__":
    main()
