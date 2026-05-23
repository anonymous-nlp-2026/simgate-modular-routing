import json, os, glob
from collections import defaultdict

BASE = "./results/plan_c"
task_dirs = sorted(glob.glob(os.path.join(BASE, "*")))

all_labeled = []
all_episodes = {}

for td in task_dirs:
    task_type = os.path.basename(td)
    ls_path = os.path.join(td, "labeled_steps.jsonl")
    if os.path.exists(ls_path):
        with open(ls_path) as f:
            for line in f:
                rec = json.loads(line.strip())
                rec["task_type"] = task_type
                all_labeled.append(rec)
    ep_path = os.path.join(td, "episodes.jsonl")
    if os.path.exists(ep_path):
        with open(ep_path) as f:
            for line in f:
                ep = json.loads(line.strip())
                eidx = ep["episode_idx"]
                all_episodes[(task_type, eidx)] = ep

# Check episode keys
sample_keys = set()
for k, v in all_episodes.items():
    sample_keys.update(v.keys())
print("Episode keys in data:", sample_keys)

# Check signal sources
signal_sources = set(s["signal_source"] for s in all_labeled)
print("Signal sources:", signal_sources)
print()

det_steps = [s for s in all_labeled if s["label"] == "det-preferred"]

# Group by episode
episode_groups = defaultdict(list)
for s in det_steps:
    key = (s["task_type"], s["episode_id"])
    episode_groups[key].append(s)

# Mislabeling analysis with corrected signal source names
mislabel_stats = defaultdict(lambda: {"total_det_steps": 0, "mislabeled": 0,
                                        "death_total": 0, "death_mis": 0,
                                        "score_total": 0, "score_mis": 0})

total_mislabeled = 0
total_det = 0
debug_score_comp = []

for (task_type, ep_id), steps in episode_groups.items():
    steps_sorted = sorted(steps, key=lambda x: x["step_idx"])
    signal = steps_sorted[0]["signal_source"]
    stats = mislabel_stats[task_type]
    stats["total_det_steps"] += len(steps_sorted)
    total_det += len(steps_sorted)

    ep_data = all_episodes.get((task_type, ep_id))

    if signal == "death_asymmetry":
        stats["death_total"] += len(steps_sorted)
        if ep_data and "always_internal" in ep_data:
            internal_steps = ep_data["always_internal"].get("steps", [])
            # Find fatal step
            fatal_step_idx = None
            for ist in internal_steps:
                if ist.get("is_dead", False):
                    fatal_step_idx = ist["step"]
                    break
            
            if fatal_step_idx is not None:
                for s in steps_sorted:
                    if s["step_idx"] < fatal_step_idx:
                        matching = [ist for ist in internal_steps if ist["step"] == s["step_idx"]]
                        if matching and matching[0]["score_delta"] >= 0 and not matching[0].get("is_dead", False):
                            stats["mislabeled"] += 1
                            stats["death_mis"] += 1
                            total_mislabeled += 1
            else:
                # No is_dead flag, but internal_score=-100 → score dropped at some point
                # Find the step where score went negative
                bad_step_idx = len(internal_steps)  # default: all steps mislabeled if no drop found
                for ist in internal_steps:
                    if ist["score_after"] < 0 or ist["score_delta"] < 0:
                        bad_step_idx = ist["step"]
                        break
                for s in steps_sorted:
                    if s["step_idx"] < bad_step_idx:
                        stats["mislabeled"] += 1
                        stats["death_mis"] += 1
                        total_mislabeled += 1

    elif signal in ("score_comparison", "score_gap"):
        stats["score_total"] += len(steps_sorted)
        if ep_data:
            internal_steps = ep_data.get("always_internal", {}).get("steps", [])
            # Try multiple possible keys for det trajectory
            det_traj = None
            for key_name in ["always_det", "deterministic", "always_deterministic"]:
                if key_name in ep_data:
                    det_traj = ep_data[key_name]
                    break
            
            if det_traj is None:
                # Debug: what keys are available?
                debug_score_comp.append((task_type, ep_id, list(ep_data.keys())))
                continue
                
            det_steps_list = det_traj.get("steps", [])
            
            for s in steps_sorted:
                sidx = s["step_idx"]
                # Note: the labeled_steps come from the DET trajectory (they train model to mimic det)
                # But the INTERNAL trajectory at same step_idx is from a different run
                # Mislabeled = internal score_delta >= det score_delta at this step
                matching_internal = [ist for ist in internal_steps if ist["step"] == sidx]
                matching_det = [dst for dst in det_steps_list if dst["step"] == sidx]
                
                if matching_internal and matching_det:
                    ist = matching_internal[0]
                    dst = matching_det[0]
                    if ist["score_delta"] >= dst["score_delta"]:
                        stats["mislabeled"] += 1
                        stats["score_mis"] += 1
                        total_mislabeled += 1

if debug_score_comp:
    print("DEBUG: score_comparison episodes missing det trajectory:")
    for tt, eid, keys in debug_score_comp[:5]:
        print(f"  {tt} ep={eid} keys={keys}")
    print()

print(f"=== MISLABELING ANALYSIS (v2) ===")
print(f"Total det-preferred steps: {total_det}")
print(f"Total mislabeled steps: {total_mislabeled}")
print(f"Mislabel rate (of det-preferred): {total_mislabeled/total_det*100:.1f}%")
print(f"Mislabel rate (of all 2040 steps): {total_mislabeled/2040*100:.1f}%")
print()

print(f"{'Task Type':<45} {'Det':>5} {'Mis':>5} {'Rate':>7} {'Death':>6} {'D_Mis':>6} {'Score':>6} {'S_Mis':>6}")
print("-" * 100)
grand = {"det": 0, "mis": 0, "death": 0, "d_mis": 0, "score": 0, "s_mis": 0}
for task_type in sorted(mislabel_stats.keys()):
    s = mislabel_stats[task_type]
    rate = s["mislabeled"] / s["total_det_steps"] * 100 if s["total_det_steps"] > 0 else 0
    print(f"{task_type:<45} {s['total_det_steps']:>5} {s['mislabeled']:>5} {rate:>6.1f}% {s['death_total']:>6} {s['death_mis']:>6} {s['score_total']:>6} {s['score_mis']:>6}")
    grand["det"] += s["total_det_steps"]
    grand["mis"] += s["mislabeled"]
    grand["death"] += s["death_total"]
    grand["d_mis"] += s["death_mis"]
    grand["score"] += s["score_total"]
    grand["s_mis"] += s["score_mis"]

print("-" * 100)
print(f"{'TOTAL':<45} {grand['det']:>5} {grand['mis']:>5} {grand['mis']/grand['det']*100:>6.1f}% {grand['death']:>6} {grand['d_mis']:>6} {grand['score']:>6} {grand['s_mis']:>6}")

# Breakdown: death_asymmetry mislabeling by position
print()
print("=== DEATH_ASYMMETRY: MISLABELED STEPS ARE PRE-FATAL STEPS ===")
for (task_type, ep_id), steps in sorted(episode_groups.items()):
    steps_sorted = sorted(steps, key=lambda x: x["step_idx"])
    signal = steps_sorted[0]["signal_source"]
    if signal != "death_asymmetry":
        continue
    ep_data = all_episodes.get((task_type, ep_id))
    if not ep_data or "always_internal" not in ep_data:
        continue
    internal_steps = ep_data["always_internal"].get("steps", [])
    fatal_idx = None
    for ist in internal_steps:
        if ist.get("is_dead", False):
            fatal_idx = ist["step"]
            break
    if fatal_idx is None:
        # Check for score drop
        for ist in internal_steps:
            if ist["score_after"] < 0:
                fatal_idx = ist["step"]
                break
    ep_len = len(steps_sorted)
    pre_fatal = sum(1 for s in steps_sorted if fatal_idx is not None and s["step_idx"] < fatal_idx)
    print(f"  {task_type:<40} ep={ep_id:<3} steps={ep_len:<3} fatal_at={fatal_idx} pre_fatal_steps={pre_fatal}")

