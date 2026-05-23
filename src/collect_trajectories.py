"""Episode-level counterfactual trajectory collection for router SFT training.

Plan C (D010): For each task type, run N independent episodes under always-internal
and N under always-deterministic. Compare episode total scores to assign modality
labels. Death episodes (score <= -100) automatically lose.

Input:
  - ScienceWorld task types (via --task_types)
  - Qwen2.5-7B-Instruct model (via --model_path or auto-detected)
  - vLLM or HF transformers backend

Output:
  - {output_dir}/{task}/raw_episodes.jsonl   — full episode data (both modalities)
  - {output_dir}/{task}/sft_triples.jsonl    — (observation, action, modality_label) for SFT
  - {output_dir}/collection_summary.json     — global statistics

Dependencies: scienceworld, vllm (or transformers), scienceworld_utils, llm_agent
"""

import argparse
import json
import os
import random
import sys
import time
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scienceworld_utils import (
    create_env, reset_env, get_valid_actions, get_score,
    step_env, close_env, fork_and_lookahead, get_task_types,
    get_max_variations,
)
from llm_agent import LLMAgent, find_model_path

DEATH_THRESHOLD = -100

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
    """Resolve short task name to ScienceWorld's full task name."""
    try:
        available = get_task_types()
    except Exception:
        return name
    if name in available:
        return name
    matches = [t for t in available if name.lower() in t.lower()]
    return matches[0] if matches else name


def is_lethal(fr: Dict[str, Any]) -> bool:
    if fr["done"] and fr["score"] < 0:
        return True
    return fr["score"] <= -100


def is_death_episode(final_score: float) -> bool:
    """Check if an episode ended in death (score <= -100)."""
    return final_score <= DEATH_THRESHOLD


def run_episode_internal(
    agent: LLMAgent,
    task_name: str,
    variation_idx: int,
    max_steps: int,
) -> Dict[str, Any]:
    """Run one always-internal episode. LLM ReAct picks every action."""
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
            "step_idx": step_idx,
            "observation_text": obs[:500],
            "action_text": action,
            "available_actions": valid_actions,
            "score_before": score_before,
            "score_after": score_after,
            "score_delta": score_after - score_before,
            "is_death": score_after <= DEATH_THRESHOLD,
            "done": done,
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
        "modality": "internal",
        "task_name": task_name,
        "variation_idx": variation_idx,
        "final_score": final_score,
        "initial_score": initial_score,
        "is_death": is_death_episode(final_score),
        "num_steps": len(steps),
        "total_time_s": round(time.time() - t0, 2),
        "steps": steps,
    }


def run_episode_deterministic(
    agent: LLMAgent,
    task_name: str,
    variation_idx: int,
    max_steps: int,
    n_candidates: int = 3,
) -> Dict[str, Any]:
    """Run one always-deterministic episode. LLM generates candidates, fork evaluates."""
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
            "step_idx": step_idx,
            "observation_text": obs[:500],
            "action_text": best_action,
            "available_actions": valid_actions,
            "score_before": score_before,
            "score_after": score_after,
            "score_delta": score_after - score_before,
            "is_death": score_after <= DEATH_THRESHOLD,
            "done": done,
            "candidates": fork_results,
            "llm_output": raw_output[:300],
        })
        action_history.append(best_action)
        history_dicts.append({"action": best_action, "observation": obs_new[:200], "score": score_after})
        obs = obs_new

        if done:
            break

    final_score = get_score(env)
    close_env(env)

    return {
        "modality": "deterministic",
        "task_name": task_name,
        "variation_idx": variation_idx,
        "final_score": final_score,
        "initial_score": initial_score,
        "is_death": is_death_episode(final_score),
        "num_steps": len(steps),
        "total_time_s": round(time.time() - t0, 2),
        "steps": steps,
    }


def label_episode_pair(internal_score: float, det_score: float) -> str:
    """Determine which modality wins the episode-level comparison.

    Returns "deterministic", "internal", or "tie".
    Death episodes (score <= -100) automatically lose.
    """
    int_death = internal_score <= DEATH_THRESHOLD
    det_death = det_score <= DEATH_THRESHOLD

    if int_death and det_death:
        return "tie"
    if int_death:
        return "deterministic"
    if det_death:
        return "internal"

    if det_score > internal_score:
        return "deterministic"
    elif internal_score > det_score:
        return "internal"
    return "tie"


def extract_sft_triples(episode: Dict[str, Any], label: str) -> List[Dict[str, Any]]:
    """Extract (observation, action, modality_label) triples from the winning episode."""
    triples = []
    for step in episode["steps"]:
        triples.append({
            "observation": step["observation_text"],
            "action": step["action_text"],
            "available_actions": step["available_actions"],
            "modality_label": label,
            "task_name": episode["task_name"],
            "variation_idx": episode["variation_idx"],
            "step_idx": step["step_idx"],
            "score_after": step["score_after"],
        })
    return triples


def run_task_collection(
    agent: LLMAgent,
    task_name: str,
    n_episodes: int,
    max_steps: int,
    output_dir: str,
    n_candidates: int = 3,
) -> Dict[str, Any]:
    """Collect episode-level trajectory data for one task type.

    For each of n_episodes variations:
      1. Run always-internal episode
      2. Run always-deterministic episode (same variation)
      3. Compare final scores → assign modality label
      4. Extract SFT triples from the winning episode

    Writes raw_episodes.jsonl and sft_triples.jsonl incrementally.
    """
    task_dir = os.path.join(output_dir, task_name.replace(" ", "_"))
    os.makedirs(task_dir, exist_ok=True)

    raw_file = os.path.join(task_dir, "raw_episodes.jsonl")
    sft_file = os.path.join(task_dir, "sft_triples.jsonl")

    max_var = get_max_variations(task_name)

    print(f"\n{'='*60}")
    print(f"TRAJECTORY COLLECTION: {task_name}")
    print(f"Episodes: {n_episodes} | Max steps: {max_steps} | Variations: {max_var}")
    print(f"{'='*60}")

    label_counts = {"deterministic": 0, "internal": 0, "tie": 0}
    death_counts = {"internal": 0, "det": 0, "both": 0}
    total_sft_triples = 0
    score_gaps = []

    for ep_idx in range(n_episodes):
        variation_idx = ep_idx % max_var
        print(f"\n--- Episode {ep_idx}/{n_episodes} (variation={variation_idx}) ---")

        # Run both modalities on the same variation
        t_ep = time.time()
        ep_internal = run_episode_internal(agent, task_name, variation_idx, max_steps)
        ep_det = run_episode_deterministic(agent, task_name, variation_idx, max_steps, n_candidates)
        ep_time = time.time() - t_ep

        int_score = ep_internal["final_score"]
        det_score = ep_det["final_score"]
        label = label_episode_pair(int_score, det_score)

        # Track deaths
        if ep_internal["is_death"] and ep_det["is_death"]:
            death_counts["both"] += 1
        elif ep_internal["is_death"]:
            death_counts["internal"] += 1
        elif ep_det["is_death"]:
            death_counts["det"] += 1

        label_counts[label] += 1
        score_gap = abs(det_score - int_score)
        if label != "tie":
            score_gaps.append(score_gap)

        print(f"  internal={int_score:.1f} | det={det_score:.1f} | label={label} | gap={score_gap:.1f} | {ep_time:.1f}s")

        # Save raw episode data (both modalities)
        for ep in [ep_internal, ep_det]:
            ep_record = {
                "episode_idx": ep_idx,
                "label": label,
                "score_gap": score_gap,
                **_slim_episode_data(ep),
            }
            with open(raw_file, "a") as f:
                f.write(json.dumps(ep_record) + "\n")

        # Extract and save SFT triples from the winning modality only
        if label == "tie":
            continue

        winning_ep = ep_det if label == "deterministic" else ep_internal
        triples = extract_sft_triples(winning_ep, label)
        with open(sft_file, "a") as f:
            for t in triples:
                f.write(json.dumps(t) + "\n")
        total_sft_triples += len(triples)

    avg_gap = sum(score_gaps) / len(score_gaps) if score_gaps else 0.0

    print(f"\n=== {task_name} SUMMARY ===")
    print(f"Labels: det={label_counts['deterministic']} int={label_counts['internal']} tie={label_counts['tie']}")
    print(f"Deaths: int={death_counts['internal']} det={death_counts['det']} both={death_counts['both']}")
    print(f"Avg score gap (non-tie): {avg_gap:.2f}")
    print(f"SFT triples: {total_sft_triples}")

    return {
        "task_name": task_name,
        "n_episodes": n_episodes,
        "label_counts": label_counts,
        "death_counts": death_counts,
        "total_sft_triples": total_sft_triples,
        "avg_score_gap": avg_gap,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Collect episode-level counterfactual trajectories for router SFT (Plan C / D010)"
    )
    parser.add_argument("--task_types", type=str, required=True,
                        help="Comma-separated task type names (e.g. 'boil,melt')")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of episodes per task type (default: 20)")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="results/trajectories/",
                        help="Output directory for trajectory data")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to Qwen2.5-7B-Instruct (auto-detected if omitted)")
    parser.add_argument("--use_vllm", action="store_true", default=True,
                        help="Use vLLM backend (default: True)")
    parser.add_argument("--no_vllm", action="store_true", default=False,
                        help="Use HF transformers instead of vLLM")
    parser.add_argument("--max_steps", type=int, default=30,
                        help="Max steps per episode (default: 30)")
    parser.add_argument("--n_candidates", type=int, default=3,
                        help="Candidate actions for deterministic mode (default: 3)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.no_vllm:
        args.use_vllm = False

    random.seed(args.seed)

    task_names_raw = [t.strip() for t in args.task_types.split(",") if t.strip()]
    task_names = [resolve_task_type(t) for t in task_names_raw]

    print(f"Task types: {task_names}")
    print(f"Episodes: {args.episodes} | Max steps: {args.max_steps}")
    print(f"GPU: {args.gpu_id} | vLLM: {args.use_vllm} | Seed: {args.seed}")
    print(f"Output: {args.output_dir}")

    agent = LLMAgent(
        model_path=args.model_path,
        use_vllm=args.use_vllm,
        gpu_id=args.gpu_id,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}

    for task_name in task_names:
        result = run_task_collection(
            agent, task_name, args.episodes, args.max_steps,
            args.output_dir, args.n_candidates,
        )
        all_results[task_name] = result

    # Global summary
    summary_path = os.path.join(args.output_dir, "collection_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    total_triples = sum(r["total_sft_triples"] for r in all_results.values())
    print(f"Total SFT triples across all tasks: {total_triples}")


if __name__ == "__main__":
    main()
