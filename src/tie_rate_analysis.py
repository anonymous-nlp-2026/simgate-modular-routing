"""Step-level tie-rate analysis for MVP oracle routing data."""
import json, os, sys
from collections import defaultdict

DATA_DIR = "./results/mvp"
TASKS = ["boil", "melt", "change-the-state-of-matter-of",
         "chemistry-mix-paint-secondary-color", "grow-plant"]

def load_oracle_steps(task):
    path = os.path.join(DATA_DIR, task, "episodes.jsonl")
    episodes = []
    with open(path) as f:
        for line in f:
            ep = json.loads(line)
            orc = ep.get("oracle_routing", {})
            steps = orc.get("steps", [])
            episodes.append(steps)
    return episodes

def analyze_steps(steps):
    tie = det_pref = int_pref = 0
    for s in steps:
        id_ = s["internal_delta"]
        dd = s["det_delta"]
        if id_ == dd:
            tie += 1
        elif dd > id_:
            det_pref += 1
        else:
            int_pref += 1
    return tie, det_pref, int_pref

def pct(num, denom):
    return 100.0 * num / denom if denom > 0 else 0.0

def phase_label(step_idx):
    if step_idx < 10:
        return "1-10"
    elif step_idx < 20:
        return "11-20"
    else:
        return "21-30"

def main():
    print("=" * 55)
    print("TIE RATE ANALYSIS (step-level: internal_delta vs det_delta)")
    print("=" * 55)
    print(f"Data granularity: step-level\n")

    global_tie = global_det = global_int = global_total = 0
    # Also track delta magnitude distribution
    all_nonzero_deltas = []

    for task in TASKS:
        episodes = load_oracle_steps(task)
        n_ep = len(episodes)
        all_steps = []
        phase_stats = defaultdict(lambda: [0, 0, 0])  # tie, det, int

        for ep_steps in episodes:
            all_steps.extend(ep_steps)
            for s in ep_steps:
                p = phase_label(s["step"])
                id_ = s["internal_delta"]
                dd = s["det_delta"]
                delta = abs(id_ - dd)
                if id_ == dd:
                    phase_stats[p][0] += 1
                elif dd > id_:
                    phase_stats[p][1] += 1
                    all_nonzero_deltas.append(delta)
                else:
                    phase_stats[p][2] += 1
                    all_nonzero_deltas.append(delta)

        n_steps = len(all_steps)
        tie, det, intn = analyze_steps(all_steps)

        global_tie += tie
        global_det += det
        global_int += intn
        global_total += n_steps

        print(f"Task: {task} ({n_ep} episodes, {n_steps} total steps)")
        print(f"  Tie rate (delta=0):       {pct(tie, n_steps):5.1f}%  ({tie}/{n_steps})")
        print(f"  Det preferred (det>int):  {pct(det, n_steps):5.1f}%  ({det}/{n_steps})")
        print(f"  Int preferred (int>det):  {pct(intn, n_steps):5.1f}%  ({intn}/{n_steps})")
        print(f"  By phase:")
        for phase in ["1-10", "11-20", "21-30"]:
            t, d, i = phase_stats[phase]
            total_p = t + d + i
            print(f"    Steps {phase:5s}: tie={pct(t,total_p):5.1f}%, det={pct(d,total_p):5.1f}%, int={pct(i,total_p):5.1f}%  (n={total_p})")

        # Breakdown: how many steps have both deltas = 0 vs both nonzero but equal
        both_zero = sum(1 for s in all_steps if s["internal_delta"] == 0 and s["det_delta"] == 0)
        both_eq_nonzero = tie - both_zero
        print(f"  Tie breakdown: both_zero={both_zero}, both_nonzero_equal={both_eq_nonzero}")

        # Score delta distribution for non-tie steps
        non_tie_steps = [s for s in all_steps if s["internal_delta"] != s["det_delta"]]
        if non_tie_steps:
            gaps = [abs(s["internal_delta"] - s["det_delta"]) for s in non_tie_steps]
            print(f"  Non-tie |gap| stats: min={min(gaps):.1f}, max={max(gaps):.1f}, mean={sum(gaps)/len(gaps):.1f}")
            # How many have gap >= 100 (catastrophic difference)
            big_gap = sum(1 for g in gaps if g >= 100)
            print(f"  Catastrophic gaps (|gap|>=100): {big_gap}/{len(gaps)} ({pct(big_gap,len(gaps)):.1f}%)")
        print()

    print("=" * 55)
    print("OVERALL")
    print("=" * 55)
    non_tie_rate = pct(global_det + global_int, global_total)
    tie_rate = pct(global_tie, global_total)
    print(f"Total steps:     {global_total}")
    print(f"Global tie rate: {tie_rate:.1f}%  ({global_tie}/{global_total})")
    print(f"  Det preferred: {pct(global_det, global_total):.1f}%  ({global_det})")
    print(f"  Int preferred: {pct(global_int, global_total):.1f}%  ({global_int})")
    print(f"Non-tie rate:    {non_tie_rate:.1f}%")
    print()

    if non_tie_rate < 10:
        print("*** CRITICAL: non-tie rate < 10% — router cannot learn meaningful signal ***")
    elif non_tie_rate < 20:
        print("** WARNING: non-tie rate < 20% — signal is weak, N-step rollout likely needed **")
    elif non_tie_rate < 30:
        print("* CAUTION: non-tie rate < 30% — consider N-step rollout for stronger signal *")
    else:
        print("OK: non-tie rate >= 30% — sufficient signal for router learning")

    # Per-task verdicts
    print()
    print("Per-task verdicts:")
    for task in TASKS:
        episodes = load_oracle_steps(task)
        all_steps = [s for ep in episodes for s in ep]
        tie, det, intn = analyze_steps(all_steps)
        n = len(all_steps)
        nt = pct(det + intn, n)
        status = "CRITICAL" if nt < 10 else "WARNING" if nt < 20 else "CAUTION" if nt < 30 else "OK"
        print(f"  {task:45s} non-tie={nt:5.1f}%  [{status}]")

if __name__ == "__main__":
    main()
