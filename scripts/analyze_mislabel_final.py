import json, os, glob
from collections import defaultdict, Counter

BASE = "./results/plan_c"
task_dirs = sorted(glob.glob(os.path.join(BASE, "*")))

all_labeled = []
all_episodes = {}
all_ep_summaries = {}

for td in task_dirs:
    task_type = os.path.basename(td)
    # labeled_steps
    ls_path = os.path.join(td, "labeled_steps.jsonl")
    if os.path.exists(ls_path):
        with open(ls_path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line.strip())
                    rec["task_type"] = task_type
                    all_labeled.append(rec)
    # episodes
    ep_path = os.path.join(td, "episodes.jsonl")
    if os.path.exists(ep_path):
        with open(ep_path) as f:
            for line in f:
                if line.strip():
                    ep = json.loads(line.strip())
                    all_episodes[(task_type, ep["episode_idx"])] = ep
    # episode_summary
    es_path = os.path.join(td, "episode_summary.jsonl")
    if os.path.exists(es_path):
        with open(es_path) as f:
            for line in f:
                if line.strip():
                    es = json.loads(line.strip())
                    all_ep_summaries[(task_type, es["episode"])] = es

# ====== DEFINITION A: Pre-fatal steps in death_asymmetry ======
# All steps before the internal policy's fatal step are mislabeled
# For episodes without is_dead, use score_after < 0 as proxy

# ====== DEFINITION B: Step-level score trajectory mismatch ======
# At each step, if internal's score_delta >= det's score_delta, the det label is wrong
# Only possible when both trajectories exist

# ====== DEFINITION C: Uniform-label penalty ======
# For death_asymmetry: N_pre_fatal / N_episode_steps steps are mislabeled
# For score_comparison: compare cumulative scores — steps before the score gap opens are mislabeled

# Group labeled steps by episode
ep_groups = defaultdict(list)
for s in all_labeled:
    ep_groups[(s["task_type"], s["episode_id"])].append(s)

# ====== DEFINITION A ======
defA = defaultdict(lambda: {"total": 0, "mislabeled": 0, "by_signal": defaultdict(lambda: {"total": 0, "mis": 0})})
total_A_mis = 0
total_A = 0

for (tt, eid), steps in ep_groups.items():
    steps_sorted = sorted(steps, key=lambda x: x["step_idx"])
    label = steps_sorted[0]["label"]
    signal = steps_sorted[0]["signal_source"]
    n = len(steps_sorted)
    
    defA[tt]["total"] += n
    defA[tt]["by_signal"][signal]["total"] += n
    total_A += n
    
    if label != "det-preferred":
        continue  # internal-preferred steps: no mislabeling in defA
    
    if signal == "death_asymmetry":
        ep_data = all_episodes.get((tt, eid))
        if not ep_data:
            continue
        
        # Try always_internal first, then always_deep_internal
        int_traj = ep_data.get("always_internal") or ep_data.get("always_deep_internal")
        if not int_traj:
            continue
        int_steps = int_traj.get("steps", [])
        
        # Find fatal step
        fatal_idx = None
        for ist in int_steps:
            if ist.get("is_dead", False):
                fatal_idx = ist["step"]
                break
        if fatal_idx is None:
            for ist in int_steps:
                if ist.get("score_after", 0) < 0 or ist.get("score_delta", 0) < 0:
                    fatal_idx = ist["step"]
                    break
        if fatal_idx is None:
            # Internal scored -100 but no specific step shows it? Default: no mislabeled
            # Actually check: did internal die? (episode_score_internal = -100)
            if steps_sorted[0].get("episode_score_internal", 0) == -100:
                # Internal died but no step-level is_dead flag
                # Use total internal steps as proxy (internal ran fewer steps = died early)
                fatal_idx = len(int_steps)
        
        if fatal_idx is not None:
            for s in steps_sorted:
                if s["step_idx"] < fatal_idx:
                    defA[tt]["mislabeled"] += 1
                    defA[tt]["by_signal"][signal]["mis"] += 1
                    total_A_mis += 1
    
    elif signal == "score_comparison":
        ep_data = all_episodes.get((tt, eid))
        if not ep_data:
            continue
        int_traj = ep_data.get("always_internal") or ep_data.get("always_deep_internal")
        det_traj = ep_data.get("always_deterministic")
        if not int_traj or not det_traj:
            continue
        
        int_steps = {s["step"]: s for s in int_traj.get("steps", [])}
        det_steps = {s["step"]: s for s in det_traj.get("steps", [])}
        
        # Find the step where det pulls ahead
        # Cumulative: if det_cumul > int_cumul, then det is ahead from this point
        int_cumul = 0
        det_cumul = 0
        diverge_step = None
        for sidx in range(max(len(int_steps), len(det_steps))):
            ist = int_steps.get(sidx)
            dst = det_steps.get(sidx)
            if ist:
                int_cumul += ist["score_delta"]
            if dst:
                det_cumul += dst["score_delta"]
            if det_cumul > int_cumul and diverge_step is None:
                diverge_step = sidx
        
        if diverge_step is not None:
            for s in steps_sorted:
                if s["step_idx"] < diverge_step:
                    defA[tt]["mislabeled"] += 1
                    defA[tt]["by_signal"][signal]["mis"] += 1
                    total_A_mis += 1

print("=" * 80)
print("DEFINITION A: Pre-event mislabeling")
print("  death_asymmetry → steps before fatal step")
print("  score_comparison → steps before det pulls ahead in cumulative score")
print("=" * 80)
print(f"\nTotal training steps: {total_A}")
print(f"Mislabeled (Def A): {total_A_mis}")
print(f"Rate: {total_A_mis/total_A*100:.1f}%")
print()

print(f"{'Task Type':<42} {'Total':>6} {'Mis':>6} {'Rate':>7} | {'DA_tot':>7} {'DA_mis':>7} {'SC_tot':>7} {'SC_mis':>7}")
print("-" * 105)
for tt in sorted(defA.keys()):
    s = defA[tt]
    rate = s["mislabeled"] / s["total"] * 100 if s["total"] > 0 else 0
    da = s["by_signal"]["death_asymmetry"]
    sc = s["by_signal"]["score_comparison"]
    print(f"{tt:<42} {s['total']:>6} {s['mislabeled']:>6} {rate:>6.1f}% | {da['total']:>7} {da['mis']:>7} {sc['total']:>7} {sc['mis']:>7}")

# Also: detailed death position stats
print()
print("=== DEATH_ASYMMETRY: FATAL STEP POSITIONS ===")
fatal_data = []
for (tt, eid), steps in ep_groups.items():
    steps_sorted = sorted(steps, key=lambda x: x["step_idx"])
    if steps_sorted[0]["signal_source"] != "death_asymmetry" or steps_sorted[0]["label"] != "det-preferred":
        continue
    ep_data = all_episodes.get((tt, eid))
    if not ep_data:
        continue
    int_traj = ep_data.get("always_internal") or ep_data.get("always_deep_internal")
    if not int_traj:
        continue
    int_steps = int_traj.get("steps", [])
    fatal_idx = None
    for ist in int_steps:
        if ist.get("is_dead", False):
            fatal_idx = ist["step"]
            break
    if fatal_idx is None:
        for ist in int_steps:
            if ist.get("score_after", 0) < 0 or ist.get("score_delta", 0) < 0:
                fatal_idx = ist["step"]
                break
    if fatal_idx is None and steps_sorted[0].get("episode_score_internal", 0) == -100:
        fatal_idx = len(int_steps)
    
    fatal_data.append({"task_type": tt, "ep_id": eid, "fatal_idx": fatal_idx, 
                        "ep_steps": len(steps_sorted), "pre_fatal": fatal_idx if fatal_idx else 0})

by_task_fatal = defaultdict(list)
for fd in fatal_data:
    by_task_fatal[fd["task_type"]].append(fd)

print(f"{'Task Type':<42} {'N_eps':>6} {'Avg Fatal':>10} {'Avg PreF':>10} {'Avg Ep':>8} {'PreF/Ep':>8}")
print("-" * 90)
total_pre_fatal = 0
total_ep_steps = 0
for tt in sorted(by_task_fatal.keys()):
    fds = by_task_fatal[tt]
    avg_fatal = sum(f["fatal_idx"] for f in fds if f["fatal_idx"]) / len(fds)
    avg_pre = sum(f["pre_fatal"] for f in fds) / len(fds)
    avg_ep = sum(f["ep_steps"] for f in fds) / len(fds)
    ratio = avg_pre / avg_ep if avg_ep > 0 else 0
    total_pre = sum(f["pre_fatal"] for f in fds)
    total_ep = sum(f["ep_steps"] for f in fds)
    total_pre_fatal += total_pre
    total_ep_steps += total_ep
    print(f"{tt:<42} {len(fds):>6} {avg_fatal:>10.1f} {avg_pre:>10.1f} {avg_ep:>8.0f} {ratio:>7.1f}%")

# SFT composition
print()
print("=== SFT TRAINING DATA COMPOSITION ===")
sft_path = "./data/plan-004-implicit/implicit_sft.jsonl"
sft_data = defaultdict(lambda: {"det": 0, "int": 0})
with open(sft_path) as f:
    for line in f:
        rec = json.loads(line.strip())
        tt = rec["task_type"]
        if rec["route_label"] == "deterministic":
            sft_data[tt]["det"] += 1
        else:
            sft_data[tt]["int"] += 1

print(f"{'Task Type':<42} {'Det':>6} {'Int':>6} {'Total':>7} {'Det%':>7} {'MisSteps':>9} {'Mis/Total':>10}")
print("-" * 95)
g_det = g_int = g_mis = 0
for tt in sorted(sft_data.keys()):
    c = sft_data[tt]
    total = c["det"] + c["int"]
    det_pct = c["det"] / total * 100 if total else 0
    mis = defA[tt]["mislabeled"]
    mis_pct = mis / total * 100 if total else 0
    g_det += c["det"]
    g_int += c["int"]
    g_mis += mis
    print(f"{tt:<42} {c['det']:>6} {c['int']:>6} {total:>7} {det_pct:>6.1f}% {mis:>9} {mis_pct:>9.1f}%")
print(f"{'TOTAL':<42} {g_det:>6} {g_int:>6} {g_det+g_int:>7} {g_det/(g_det+g_int)*100:>6.1f}% {g_mis:>9} {g_mis/(g_det+g_int)*100:>9.1f}%")

# Episode-level breakdown
print()
print("=== EPISODE-LEVEL BREAKDOWN ===")
ep_labels = Counter()
ep_signals = Counter()
for (tt, eid), steps in ep_groups.items():
    label = steps[0]["label"]
    signal = steps[0]["signal_source"]
    ep_labels[label] += 1
    ep_signals[(label, signal)] += 1

print(f"Episodes by label: {dict(ep_labels)}")
print(f"Episodes by (label, signal): {dict(ep_signals)}")
total_eps = sum(ep_labels.values())
print(f"Total episodes in training: {total_eps}")

