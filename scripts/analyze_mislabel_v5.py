import json, os, glob
from collections import defaultdict

BASE = "./results/plan_c"

all_labeled = []
all_episodes = {}
for td in sorted(glob.glob(os.path.join(BASE, "*"))):
    tt = os.path.basename(td)
    p = os.path.join(td, "labeled_steps.jsonl")
    if os.path.exists(p):
        with open(p) as f:
            for l in f:
                if l.strip():
                    r = json.loads(l.strip())
                    r["task_type"] = tt
                    all_labeled.append(r)
    p2 = os.path.join(td, "episodes.jsonl")
    if os.path.exists(p2):
        with open(p2) as f:
            for l in f:
                if l.strip():
                    ep = json.loads(l.strip())
                    all_episodes[(tt, ep["episode_idx"])] = ep

ep_groups = defaultdict(list)
for s in all_labeled:
    ep_groups[(s["task_type"], s["episode_id"])].append(s)

# Multiple definitions
defs = {
    "strict_conflict": 0,      # int_delta > det_delta (only when both > 0 or one < 0)
    "int_positive_det_zero": 0, # int made progress, det didn't
    "pre_fatal_with_int_progress": 0, # pre-fatal steps where int had score_delta > 0
    "pre_fatal_all": 0,        # all pre-fatal steps (death_asymmetry only)
    "pre_diverge_all": 0,      # all pre-divergence steps (all signals)
}

by_task = defaultdict(lambda: {k: 0 for k in list(defs.keys()) + ["total"]})

for (tt, eid), steps in ep_groups.items():
    steps_sorted = sorted(steps, key=lambda x: x["step_idx"])
    label = steps_sorted[0]["label"]
    signal = steps_sorted[0]["signal_source"]
    n = len(steps_sorted)
    by_task[tt]["total"] += n

    ep = all_episodes.get((tt, eid))
    if not ep:
        continue

    int_traj = ep.get("always_internal") or ep.get("always_deep_internal")
    det_traj = ep.get("always_deterministic")
    if not int_traj or not det_traj:
        # For death_asymmetry without det trajectory, skip step-level comparison
        if signal == "death_asymmetry" and label == "det-preferred" and int_traj:
            int_steps = int_traj.get("steps", [])
            fatal = None
            for ist in int_steps:
                if ist.get("is_dead", False):
                    fatal = ist["step"]; break
            if fatal is None:
                for ist in int_steps:
                    if ist.get("score_after", 0) < 0:
                        fatal = ist["step"]; break
            if fatal is None and steps_sorted[0].get("episode_score_internal", 0) == -100:
                fatal = len(int_steps)
            
            if fatal is not None:
                for s in steps_sorted:
                    if s["step_idx"] < fatal:
                        defs["pre_fatal_all"] += 1
                        by_task[tt]["pre_fatal_all"] += 1
                        defs["pre_diverge_all"] += 1
                        by_task[tt]["pre_diverge_all"] += 1
                        # Check if int had progress
                        ist = next((x for x in int_steps if x["step"] == s["step_idx"]), None)
                        if ist and ist["score_delta"] > 0:
                            defs["pre_fatal_with_int_progress"] += 1
                            by_task[tt]["pre_fatal_with_int_progress"] += 1
                            defs["int_positive_det_zero"] += 1
                            by_task[tt]["int_positive_det_zero"] += 1
        continue

    int_by_idx = {s["step"]: s for s in int_traj.get("steps", [])}
    det_by_idx = {s["step"]: s for s in det_traj.get("steps", [])}
    int_steps_list = int_traj.get("steps", [])

    if signal == "death_asymmetry" and label == "det-preferred":
        fatal = None
        for ist in int_steps_list:
            if ist.get("is_dead", False):
                fatal = ist["step"]; break
        if fatal is None:
            for ist in int_steps_list:
                if ist.get("score_after", 0) < 0:
                    fatal = ist["step"]; break
        if fatal is None and steps_sorted[0].get("episode_score_internal", 0) == -100:
            fatal = len(int_steps_list)

        for s in steps_sorted:
            sidx = s["step_idx"]
            ist = int_by_idx.get(sidx)
            dst = det_by_idx.get(sidx)

            if fatal is not None and sidx < fatal:
                defs["pre_fatal_all"] += 1
                by_task[tt]["pre_fatal_all"] += 1
                defs["pre_diverge_all"] += 1
                by_task[tt]["pre_diverge_all"] += 1
                if ist and ist["score_delta"] > 0:
                    defs["pre_fatal_with_int_progress"] += 1
                    by_task[tt]["pre_fatal_with_int_progress"] += 1

            if ist and dst:
                if ist["score_delta"] > dst["score_delta"]:
                    defs["strict_conflict"] += 1
                    by_task[tt]["strict_conflict"] += 1
                if ist["score_delta"] > 0 and dst["score_delta"] == 0:
                    defs["int_positive_det_zero"] += 1
                    by_task[tt]["int_positive_det_zero"] += 1

    elif signal == "score_comparison":
        # Cumulative score to find divergence point
        int_cum = 0; det_cum = 0; diverge = None
        max_idx = max(max(int_by_idx.keys(), default=0), max(det_by_idx.keys(), default=0))
        for i in range(max_idx + 1):
            if i in int_by_idx: int_cum += int_by_idx[i]["score_delta"]
            if i in det_by_idx: det_cum += det_by_idx[i]["score_delta"]
            if label == "det-preferred" and det_cum > int_cum and diverge is None:
                diverge = i
            elif label == "internal-preferred" and int_cum > det_cum and diverge is None:
                diverge = i

        for s in steps_sorted:
            sidx = s["step_idx"]
            ist = int_by_idx.get(sidx)
            dst = det_by_idx.get(sidx)

            if diverge is not None and sidx < diverge:
                defs["pre_diverge_all"] += 1
                by_task[tt]["pre_diverge_all"] += 1

            if ist and dst:
                if label == "det-preferred":
                    if ist["score_delta"] > dst["score_delta"]:
                        defs["strict_conflict"] += 1
                        by_task[tt]["strict_conflict"] += 1
                    if ist["score_delta"] > 0 and dst["score_delta"] == 0:
                        defs["int_positive_det_zero"] += 1
                        by_task[tt]["int_positive_det_zero"] += 1
                elif label == "internal-preferred":
                    if dst["score_delta"] > ist["score_delta"]:
                        defs["strict_conflict"] += 1
                        by_task[tt]["strict_conflict"] += 1

total = len(all_labeled)
print("=" * 80)
print("MULTI-DEFINITION MISLABELING ANALYSIS")
print("=" * 80)
print(f"Total training steps: {total}")
print()
for name, count in sorted(defs.items()):
    pct = count / total * 100
    print(f"  {name:<35} {count:>6} steps  ({pct:>5.1f}%)")

print()
print("BY TASK TYPE:")
print(f"{'Task':<42} {'Total':>5} | {'strict':>7} {'int>0':>7} {'preF+P':>7} {'preF':>7} {'preDiv':>7}")
print("-" * 95)
for tt in sorted(by_task.keys()):
    t = by_task[tt]
    print(f"{tt:<42} {t['total']:>5} | {t['strict_conflict']:>7} {t['int_positive_det_zero']:>7} {t['pre_fatal_with_int_progress']:>7} {t['pre_fatal_all']:>7} {t['pre_diverge_all']:>7}")

# Try to find what gives ~27.7% = ~565 steps
print()
print("=== SEARCHING FOR ~27.7% DEFINITION ===")
target = round(total * 0.277)
print(f"Target: ~{target} steps")
# Check: pre_fatal_with_int_progress + score_comparison strict conflict
combo1 = defs["pre_fatal_with_int_progress"] + sum(1 for (tt, eid), steps in ep_groups.items()
    for s in steps
    if steps[0]["signal_source"] == "score_comparison"
    and steps[0]["label"] == "det-preferred"
    and (tt, eid) in all_episodes
    and "always_internal" in (all_episodes[(tt, eid)])
    and "always_deterministic" in (all_episodes[(tt, eid)])
    and s["step_idx"] in {x["step"]: x for x in (all_episodes[(tt, eid)].get("always_internal") or {}).get("steps", [])}
    and {x["step"]: x for x in (all_episodes[(tt, eid)].get("always_internal") or {}).get("steps", [])}[s["step_idx"]]["score_delta"]
    >= {x["step"]: x for x in (all_episodes[(tt, eid)].get("always_deterministic") or {}).get("steps", [])}[s["step_idx"]]["score_delta"]
)
# That's too complex inline, skip

# Just report all numbers and let the paper agent decide
print(f"  strict_conflict = {defs['strict_conflict']} ({defs['strict_conflict']/total*100:.1f}%)")
print(f"  int_positive_det_zero = {defs['int_positive_det_zero']} ({defs['int_positive_det_zero']/total*100:.1f}%)")
print(f"  pre_fatal_with_int_progress = {defs['pre_fatal_with_int_progress']} ({defs['pre_fatal_with_int_progress']/total*100:.1f}%)")
print(f"  pre_fatal_all (death_asym only) = {defs['pre_fatal_all']} ({defs['pre_fatal_all']/total*100:.1f}%)")
print(f"  pre_diverge_all (all signals) = {defs['pre_diverge_all']} ({defs['pre_diverge_all']/total*100:.1f}%)")

# Compute: steps where int_delta >= det_delta (including both == 0)
# but ONLY in death_asymmetry episodes, ONLY pre-fatal
# This might be the "mislabeled" = pre-fatal steps where internal was at least as good
def_prefatal_int_geq = 0
for (tt, eid), steps in ep_groups.items():
    steps_sorted = sorted(steps, key=lambda x: x["step_idx"])
    if steps_sorted[0]["signal_source"] != "death_asymmetry" or steps_sorted[0]["label"] != "det-preferred":
        continue
    ep = all_episodes.get((tt, eid))
    if not ep: continue
    int_traj = ep.get("always_internal") or ep.get("always_deep_internal")
    if not int_traj: continue
    int_steps = int_traj.get("steps", [])
    fatal = None
    for ist in int_steps:
        if ist.get("is_dead", False):
            fatal = ist["step"]; break
    if fatal is None:
        for ist in int_steps:
            if ist.get("score_after", 0) < 0:
                fatal = ist["step"]; break
    if fatal is None and steps_sorted[0].get("episode_score_internal", 0) == -100:
        fatal = len(int_steps)
    if fatal is None:
        continue
    
    int_by = {s["step"]: s for s in int_steps}
    det_traj = ep.get("always_deterministic")
    det_by = {s["step"]: s for s in det_traj.get("steps", [])} if det_traj else {}
    
    for s in steps_sorted:
        if s["step_idx"] < fatal:
            ist = int_by.get(s["step_idx"])
            dst = det_by.get(s["step_idx"])
            if ist and dst:
                if ist["score_delta"] >= dst["score_delta"]:
                    def_prefatal_int_geq += 1
            elif ist:
                if ist["score_delta"] >= 0:
                    def_prefatal_int_geq += 1

print(f"\n  pre_fatal + int_delta >= det_delta = {def_prefatal_int_geq} ({def_prefatal_int_geq/total*100:.1f}%)")

# Also: int_positive_det_zero from death_asymmetry only
da_int_pos = sum(by_task[tt]["int_positive_det_zero"] for tt in by_task if tt != "find-non-living-thing")
sc_int_pos = by_task.get("find-non-living-thing", {}).get("int_positive_det_zero", 0)
print(f"  death_asym int>0,det==0 = {da_int_pos} ({da_int_pos/total*100:.1f}%)")
print(f"  score_comp int>0,det==0 = {sc_int_pos} ({sc_int_pos/total*100:.1f}%)")

