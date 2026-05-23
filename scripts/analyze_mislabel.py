import json, os, glob
from collections import defaultdict

BASE = "./results/plan_c"
task_dirs = sorted(glob.glob(os.path.join(BASE, "*")))

# Collect all labeled steps and episode data
all_labeled = []
all_episodes = {}  # (task_type, episode_id) -> episode data

for td in task_dirs:
    task_type = os.path.basename(td)
    
    # Read labeled_steps.jsonl
    ls_path = os.path.join(td, "labeled_steps.jsonl")
    if os.path.exists(ls_path):
        with open(ls_path) as f:
            for line in f:
                rec = json.loads(line.strip())
                rec["task_type"] = task_type
                all_labeled.append(rec)
    
    # Read episodes.jsonl for step-level detail
    ep_path = os.path.join(td, "episodes.jsonl")
    if os.path.exists(ep_path):
        with open(ep_path) as f:
            for line in f:
                ep = json.loads(line.strip())
                eidx = ep["episode_idx"]
                all_episodes[(task_type, eidx)] = ep

print(f"Total labeled steps: {len(all_labeled)}")
print(f"Total episodes with trajectory data: {len(all_episodes)}")
print()

# Only analyze det-preferred steps (those that end up in training data)
det_steps = [s for s in all_labeled if s["label"] == "det-preferred"]
print(f"Det-preferred steps (in training): {len(det_steps)}")

# Also count internal-preferred and no-preference
int_steps = [s for s in all_labeled if s["label"] == "internal-preferred"]
nop_steps = [s for s in all_labeled if s["label"] == "no-preference"]
print(f"Internal-preferred steps: {len(int_steps)}")
print(f"No-preference steps: {len(nop_steps)}")
print()

# Group det-preferred steps by (task_type, episode_id) and analyze mislabeling
# For death_asymmetry: find fatal step from episodes.jsonl internal trajectory
# Steps BEFORE the fatal step where internal policy had score_delta=0 and was not dead = mislabeled

mislabel_stats = defaultdict(lambda: {"total_det_steps": 0, "mislabeled": 0, 
                                        "death_asym_total": 0, "death_asym_mislabeled": 0,
                                        "score_gap_total": 0, "score_gap_mislabeled": 0})

episode_groups = defaultdict(list)
for s in det_steps:
    key = (s["task_type"], s["episode_id"])
    episode_groups[key].append(s)

total_mislabeled = 0
total_det = 0

for (task_type, ep_id), steps in episode_groups.items():
    steps_sorted = sorted(steps, key=lambda x: x["step_idx"])
    signal = steps_sorted[0]["signal_source"]
    stats = mislabel_stats[task_type]
    stats["total_det_steps"] += len(steps_sorted)
    total_det += len(steps_sorted)
    
    ep_data = all_episodes.get((task_type, ep_id))
    
    if signal == "death_asymmetry":
        stats["death_asym_total"] += len(steps_sorted)
        
        if ep_data and "always_internal" in ep_data:
            internal_traj = ep_data["always_internal"]
            internal_steps = internal_traj.get("steps", [])
            
            # Find the fatal step (first step where is_dead=True)
            fatal_step_idx = None
            for ist in internal_steps:
                if ist.get("is_dead", False):
                    fatal_step_idx = ist["step"]
                    break
            
            if fatal_step_idx is not None:
                # Steps before fatal step where internal was doing fine (score_delta >= 0)
                for s in steps_sorted:
                    if s["step_idx"] < fatal_step_idx:
                        # This step is before the fatal action - internal policy was fine here
                        # Check if internal policy had non-negative score at this step
                        matching_internal = [ist for ist in internal_steps if ist["step"] == s["step_idx"]]
                        if matching_internal:
                            ist = matching_internal[0]
                            if ist["score_delta"] >= 0 and not ist.get("is_dead", False):
                                stats["mislabeled"] += 1
                                stats["death_asym_mislabeled"] += 1
                                total_mislabeled += 1
            else:
                # Internal got -100 but no is_dead flag? Check if score went to -100
                # In this case, the internal policy scored -100 (failed) but didn't "die" per se
                # Still, all steps before the failure point are potentially mislabeled
                # Use score trajectory: steps where internal score_delta == 0 before things go wrong
                if internal_steps:
                    # Find last step where score was still non-negative
                    for s in steps_sorted:
                        matching_internal = [ist for ist in internal_steps if ist["step"] == s["step_idx"]]
                        if matching_internal:
                            ist = matching_internal[0]
                            if ist["score_delta"] >= 0 and ist["score_after"] >= 0:
                                stats["mislabeled"] += 1
                                stats["death_asym_mislabeled"] += 1
                                total_mislabeled += 1
    
    elif signal == "score_gap":
        stats["score_gap_total"] += len(steps_sorted)
        
        if ep_data and "always_internal" in ep_data:
            internal_traj = ep_data["always_internal"]
            internal_steps = internal_traj.get("steps", [])
            det_traj = ep_data.get("always_det") or ep_data.get("deterministic")
            det_steps_list = det_traj.get("steps", []) if det_traj else []
            
            for s in steps_sorted:
                sidx = s["step_idx"]
                matching_internal = [ist for ist in internal_steps if ist["step"] == sidx]
                matching_det = [dst for dst in det_steps_list if dst["step"] == sidx]
                
                if matching_internal and matching_det:
                    ist = matching_internal[0]
                    dst = matching_det[0]
                    # If internal policy's score_delta >= det's score_delta at this step,
                    # then routing to det is not beneficial = mislabeled
                    if ist["score_delta"] >= dst["score_delta"]:
                        stats["mislabeled"] += 1
                        stats["score_gap_mislabeled"] += 1
                        total_mislabeled += 1

print(f"=== MISLABELING ANALYSIS ===")
print(f"Total det-preferred steps: {total_det}")
print(f"Total mislabeled steps: {total_mislabeled}")
print(f"Mislabel rate: {total_mislabeled/total_det*100:.1f}%")
print()

print(f"{'Task Type':<45} {'Det Steps':>10} {'Mislabel':>10} {'Rate':>8} {'DeathAsym':>10} {'DA_Mis':>8} {'ScoreGap':>10} {'SG_Mis':>8}")
print("-" * 140)

for task_type in sorted(mislabel_stats.keys()):
    s = mislabel_stats[task_type]
    rate = s["mislabeled"] / s["total_det_steps"] * 100 if s["total_det_steps"] > 0 else 0
    print(f"{task_type:<45} {s['total_det_steps']:>10} {s['mislabeled']:>10} {rate:>7.1f}% {s['death_asym_total']:>10} {s['death_asym_mislabeled']:>8} {s['score_gap_total']:>10} {s['score_gap_mislabeled']:>8}")

# Also output: total steps per task type (all labels)
print()
print("=== STEP DISTRIBUTION BY TASK TYPE (ALL LABELS) ===")
task_counts = defaultdict(lambda: {"det": 0, "internal": 0, "no_pref": 0, "total": 0})
for s in all_labeled:
    tt = s["task_type"]
    task_counts[tt]["total"] += 1
    if s["label"] == "det-preferred":
        task_counts[tt]["det"] += 1
    elif s["label"] == "internal-preferred":
        task_counts[tt]["internal"] += 1
    else:
        task_counts[tt]["no_pref"] += 1

print(f"{'Task Type':<45} {'Total':>8} {'Det':>8} {'Internal':>8} {'NoPref':>8}")
print("-" * 90)
for tt in sorted(task_counts.keys()):
    c = task_counts[tt]
    print(f"{tt:<45} {c['total']:>8} {c['det']:>8} {c['internal']:>8} {c['no_pref']:>8}")

# Signal source distribution
print()
print("=== SIGNAL SOURCE DISTRIBUTION (DET-PREFERRED ONLY) ===")
signal_counts = defaultdict(lambda: defaultdict(int))
for s in det_steps:
    signal_counts[s["task_type"]][s["signal_source"]] += 1

for tt in sorted(signal_counts.keys()):
    for sig, cnt in sorted(signal_counts[tt].items()):
        print(f"  {tt:<40} {sig:<20} {cnt}")

# Death position analysis for death_asymmetry episodes
print()
print("=== FATAL STEP POSITION IN DEATH_ASYMMETRY EPISODES ===")
fatal_positions = defaultdict(list)
for (task_type, ep_id), steps in episode_groups.items():
    steps_sorted = sorted(steps, key=lambda x: x["step_idx"])
    signal = steps_sorted[0]["signal_source"]
    if signal == "death_asymmetry":
        ep_data = all_episodes.get((task_type, ep_id))
        if ep_data and "always_internal" in ep_data:
            internal_steps = ep_data["always_internal"].get("steps", [])
            total_ep_steps = len(steps_sorted)
            for ist in internal_steps:
                if ist.get("is_dead", False):
                    fatal_positions[task_type].append((ist["step"], total_ep_steps))
                    break

for tt in sorted(fatal_positions.keys()):
    positions = fatal_positions[tt]
    avg_pos = sum(p[0] for p in positions) / len(positions) if positions else 0
    avg_total = sum(p[1] for p in positions) / len(positions) if positions else 0
    avg_ratio = sum(p[0]/p[1] for p in positions) / len(positions) if positions else 0
    print(f"  {tt:<40} episodes={len(positions):>3} avg_fatal_step={avg_pos:.1f}/{avg_total:.1f} avg_position_ratio={avg_ratio:.2f}")

# Also check: the SFT data only includes det-preferred and internal-preferred, skipping no-preference
print()
print("=== SFT DATA COMPOSITION CHECK ===")
sft_path = "./data/plan-004-implicit/implicit_sft.jsonl"
sft_tasks = defaultdict(lambda: {"det": 0, "int": 0})
with open(sft_path) as f:
    for line in f:
        rec = json.loads(line.strip())
        tt = rec["task_type"]
        if rec["route_label"] == "deterministic":
            sft_tasks[tt]["det"] += 1
        else:
            sft_tasks[tt]["int"] += 1

print(f"{'Task Type':<45} {'SFT_Det':>8} {'SFT_Int':>8} {'SFT_Total':>10}")
print("-" * 75)
total_sft_det = 0
total_sft_int = 0
for tt in sorted(sft_tasks.keys()):
    c = sft_tasks[tt]
    total_sft_det += c["det"]
    total_sft_int += c["int"]
    print(f"{tt:<45} {c['det']:>8} {c['int']:>8} {c['det']+c['int']:>10}")
print(f"{'TOTAL':<45} {total_sft_det:>8} {total_sft_int:>8} {total_sft_det+total_sft_int:>10}")

