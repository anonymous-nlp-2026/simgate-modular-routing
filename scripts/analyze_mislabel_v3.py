import json, os, glob
from collections import defaultdict

BASE = "./results/plan_c"
task_dirs = sorted(glob.glob(os.path.join(BASE, "*")))

all_episodes = {}
all_labeled = []

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
                all_episodes[(task_type, ep["episode_idx"])] = ep

# For each labeled step, compute step-level oracle and check consistency
# Oracle: compare internal vs det score_delta at this step_idx
# Inconsistent = oracle says opposite direction of episode label
# Neutral (both score_delta=0) = NOT inconsistent

results = []  # (task_type, ep_id, step_idx, ep_label, signal, step_oracle, inconsistent)

ep_groups = defaultdict(list)
for s in all_labeled:
    ep_groups[(s["task_type"], s["episode_id"])].append(s)

for (task_type, ep_id), steps in ep_groups.items():
    ep_data = all_episodes.get((task_type, ep_id))
    if not ep_data:
        continue
    
    internal_steps = ep_data.get("always_internal", {}).get("steps", [])
    det_traj = ep_data.get("always_deterministic", {})
    det_steps = det_traj.get("steps", []) if det_traj else []
    
    int_by_idx = {s["step"]: s for s in internal_steps}
    det_by_idx = {s["step"]: s for s in det_steps}
    
    for s in steps:
        sidx = s["step_idx"]
        ep_label = s["label"]  # det-preferred or internal-preferred
        signal = s["signal_source"]
        
        ist = int_by_idx.get(sidx)
        dst = det_by_idx.get(sidx)
        
        if ist and dst:
            int_delta = ist["score_delta"]
            det_delta = dst["score_delta"]
            
            if det_delta > int_delta:
                step_oracle = "det-preferred"
            elif int_delta > det_delta:
                step_oracle = "internal-preferred"
            else:
                step_oracle = "neutral"
            
            # Inconsistent only when oracle says OPPOSITE of episode label
            if ep_label == "det-preferred" and step_oracle == "internal-preferred":
                inconsistent = True
            elif ep_label == "internal-preferred" and step_oracle == "det-preferred":
                inconsistent = True
            else:
                inconsistent = False
        else:
            step_oracle = "no-data"
            inconsistent = False
        
        results.append({
            "task_type": task_type, "ep_id": ep_id, "step_idx": sidx,
            "ep_label": ep_label, "signal": signal,
            "step_oracle": step_oracle, "inconsistent": inconsistent,
            "int_delta": ist["score_delta"] if ist else None,
            "det_delta": dst["score_delta"] if dst else None,
        })

total = len(results)
inconsistent_count = sum(1 for r in results if r["inconsistent"])
print(f"Total steps analyzed: {total}")
print(f"Inconsistent (strict direction conflict): {inconsistent_count}")
print(f"Rate: {inconsistent_count/total*100:.1f}%")
print()

# By oracle distribution
from collections import Counter
oracle_dist = Counter(r["step_oracle"] for r in results)
print(f"Step-level oracle distribution:")
for k, v in oracle_dist.most_common():
    print(f"  {k}: {v} ({v/total*100:.1f}%)")
print()

# By task type
print(f"{'Task Type':<45} {'Total':>6} {'Incons':>7} {'Rate':>7} | {'Neutral':>8} {'Det':>6} {'Int':>6} {'NoData':>7}")
print("-" * 110)
by_task = defaultdict(lambda: {"total": 0, "incons": 0, "neutral": 0, "det": 0, "int": 0, "nodata": 0})
for r in results:
    tt = r["task_type"]
    by_task[tt]["total"] += 1
    if r["inconsistent"]:
        by_task[tt]["incons"] += 1
    if r["step_oracle"] == "neutral":
        by_task[tt]["neutral"] += 1
    elif r["step_oracle"] == "det-preferred":
        by_task[tt]["det"] += 1
    elif r["step_oracle"] == "internal-preferred":
        by_task[tt]["int"] += 1
    else:
        by_task[tt]["nodata"] += 1

for tt in sorted(by_task.keys()):
    s = by_task[tt]
    rate = s["incons"] / s["total"] * 100 if s["total"] > 0 else 0
    print(f"{tt:<45} {s['total']:>6} {s['incons']:>7} {rate:>6.1f}% | {s['neutral']:>8} {s['det']:>6} {s['int']:>6} {s['nodata']:>7}")

# By signal source
print()
print(f"{'Signal Source':<25} {'Total':>6} {'Incons':>7} {'Rate':>7}")
print("-" * 50)
by_signal = defaultdict(lambda: {"total": 0, "incons": 0})
for r in results:
    sig = r["signal"]
    by_signal[sig]["total"] += 1
    if r["inconsistent"]:
        by_signal[sig]["incons"] += 1
for sig in sorted(by_signal.keys()):
    s = by_signal[sig]
    rate = s["incons"] / s["total"] * 100 if s["total"] > 0 else 0
    print(f"{sig:<25} {s['total']:>6} {s['incons']:>7} {rate:>6.1f}%")

# By ep_label direction
print()
print(f"{'Episode Label':<25} {'Total':>6} {'Incons':>7} {'Rate':>7}")
print("-" * 50)
by_label = defaultdict(lambda: {"total": 0, "incons": 0})
for r in results:
    lab = r["ep_label"]
    by_label[lab]["total"] += 1
    if r["inconsistent"]:
        by_label[lab]["incons"] += 1
for lab in sorted(by_label.keys()):
    s = by_label[lab]
    rate = s["incons"] / s["total"] * 100 if s["total"] > 0 else 0
    print(f"{lab:<25} {s['total']:>6} {s['incons']:>7} {rate:>6.1f}%")

# Detailed: for inconsistent steps, what are the score deltas?
print()
print("=== INCONSISTENT STEPS: SCORE DELTA DETAILS ===")
incons_steps = [r for r in results if r["inconsistent"]]
print(f"Total inconsistent: {len(incons_steps)}")
# Group by (task_type, ep_label, signal)
by_group = defaultdict(list)
for r in incons_steps:
    by_group[(r["task_type"], r["ep_label"], r["signal"])].append(r)

for (tt, label, sig), recs in sorted(by_group.items()):
    avg_int = sum(r["int_delta"] for r in recs) / len(recs)
    avg_det = sum(r["det_delta"] for r in recs) / len(recs)
    print(f"  {tt:<40} label={label:<20} signal={sig:<20} n={len(recs):<3} avg_int_delta={avg_int:>6.2f} avg_det_delta={avg_det:>6.2f}")

# Also check: are there death_asymmetry steps missing from episodes.jsonl det trajectory?
print()
print("=== DATA COVERAGE CHECK ===")
no_data = [r for r in results if r["step_oracle"] == "no-data"]
print(f"Steps with no trajectory data for comparison: {len(no_data)}")
if no_data:
    by_tt = Counter(r["task_type"] for r in no_data)
    for tt, cnt in by_tt.most_common():
        print(f"  {tt}: {cnt}")

# Check: for death_asymmetry, does always_deterministic trajectory exist?
print()
print("=== DEATH_ASYMMETRY TRAJECTORY AVAILABILITY ===")
for (tt, eid), ep in sorted(all_episodes.items()):
    has_int = "always_internal" in ep
    has_det = "always_deterministic" in ep
    has_di = "always_deep_internal" in ep
    # Only print a few
    ep_summary_label = None
    for s in all_labeled:
        if s["task_type"] == tt and s["episode_id"] == eid:
            ep_summary_label = s.get("signal_source")
            break
    if ep_summary_label == "death_asymmetry":
        int_steps_count = len(ep.get("always_internal", {}).get("steps", []))
        det_steps_count = len(ep.get("always_deterministic", {}).get("steps", []))
        if det_steps_count == 0 and not has_det:
            print(f"  {tt:<40} ep={eid:<3} has_int={has_int} has_det={has_det} has_di={has_di}")

