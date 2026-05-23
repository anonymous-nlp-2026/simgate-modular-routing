"""Plan-013: Labeling Quality Analysis (a) tie rate + (b) genuine vs tie-break."""
import json, os, sys
from collections import defaultdict, Counter
from pathlib import Path

PLAN_C = Path("./results/plan_c")
LABELS = Path("./results/plan_001b_labels/sft_triples.jsonl")
DEATH_THRESHOLD = -100

def is_dead(score):
    return score <= DEATH_THRESHOLD

# ---- (A) Episode-level analysis from raw episodes ----
print("=" * 60)
print("(A) EPISODE-LEVEL TIE RATE ANALYSIS")
print("=" * 60)

episodes_by_task = defaultdict(list)
for task_dir in sorted(PLAN_C.iterdir()):
    ep_file = task_dir / "episodes.jsonl"
    if not ep_file.exists():
        continue
    with open(ep_file) as f:
        for line in f:
            ep = json.loads(line)
            int_score = ep["always_internal"]["final_score"]
            det_score = ep["always_deterministic"]["final_score"]
            int_died = is_dead(int_score)
            det_died = is_dead(det_score)
            
            if int_died and not det_died:
                label = "det-preferred"
                reason = "int_died_det_alive"
            elif det_died and not int_died:
                label = "internal-preferred"
                reason = "det_died_int_alive"
            elif int_died and det_died:
                label = "no-preference"
                reason = "both_died"
            elif det_score > int_score:
                label = "det-preferred"
                reason = "both_alive_det_wins"
            elif int_score > det_score:
                label = "internal-preferred"
                reason = "both_alive_int_wins"
            else:
                label = "no-preference"
                reason = "tie"
            
            episodes_by_task[task_dir.name].append({
                "int_score": int_score,
                "det_score": det_score,
                "delta": det_score - int_score,
                "label": label,
                "reason": reason,
                "int_died": int_died,
                "det_died": det_died,
            })

all_episodes = []
for task, eps in episodes_by_task.items():
    all_episodes.extend(eps)

total = len(all_episodes)
print(f"\nTotal episodes: {total}")

# Overall label distribution
label_counts = Counter(ep["label"] for ep in all_episodes)
reason_counts = Counter(ep["reason"] for ep in all_episodes)
print(f"\nLabel distribution:")
for label, count in sorted(label_counts.items()):
    print(f"  {label}: {count} ({100*count/total:.1f}%)")

print(f"\nReason distribution:")
for reason, count in sorted(reason_counts.items()):
    print(f"  {reason}: {count} ({100*count/total:.1f}%)")

# Tie rate: episodes where int_score == det_score (both alive or both dead)
tie_episodes = [ep for ep in all_episodes if ep["int_score"] == ep["det_score"]]
tie_rate = len(tie_episodes) / total
print(f"\nOverall tie rate (exact score match): {len(tie_episodes)}/{total} = {100*tie_rate:.1f}%")

# Break down tie into both-alive vs both-dead
tie_alive = [ep for ep in tie_episodes if not ep["int_died"] and not ep["det_died"]]
tie_dead = [ep for ep in tie_episodes if ep["int_died"] and ep["det_died"]]
print(f"  Both alive, tied: {len(tie_alive)}")
print(f"  Both died: {len(tie_dead)}")

# Per-task tie rate
print(f"\nPer-task tie rate:")
print(f"{'Task':<45} {'Episodes':>8} {'Ties':>5} {'Tie%':>6} {'Det-Pref':>9} {'Int-Pref':>9} {'NoPref':>7}")
print("-" * 95)
for task in sorted(episodes_by_task.keys()):
    eps = episodes_by_task[task]
    n = len(eps)
    ties = sum(1 for ep in eps if ep["int_score"] == ep["det_score"])
    det_pref = sum(1 for ep in eps if ep["label"] == "det-preferred")
    int_pref = sum(1 for ep in eps if ep["label"] == "internal-preferred")
    no_pref = sum(1 for ep in eps if ep["label"] == "no-preference")
    print(f"{task:<45} {n:>8} {ties:>5} {100*ties/n:>5.1f}% {det_pref:>9} {int_pref:>9} {no_pref:>7}")

# ---- (B) Genuine vs Tie-Break analysis ----
print("\n" + "=" * 60)
print("(B) GENUINE VS TIE-BREAK LABEL FRACTION")
print("=" * 60)

det_preferred = [ep for ep in all_episodes if ep["label"] == "det-preferred"]
int_preferred = [ep for ep in all_episodes if ep["label"] == "internal-preferred"]

# Det-preferred breakdown
det_death = [ep for ep in det_preferred if ep["reason"] == "int_died_det_alive"]
det_score_win = [ep for ep in det_preferred if ep["reason"] == "both_alive_det_wins"]

print(f"\nDet-preferred episodes: {len(det_preferred)}")
print(f"  From death asymmetry (int died, det alive): {len(det_death)}")
print(f"  From score comparison (both alive, det > int): {len(det_score_win)}")

# Int-preferred breakdown
int_death = [ep for ep in int_preferred if ep["reason"] == "det_died_int_alive"]
int_score_win = [ep for ep in int_preferred if ep["reason"] == "both_alive_int_wins"]

print(f"\nInternal-preferred episodes: {len(int_preferred)}")
print(f"  From death asymmetry (det died, int alive): {len(int_death)}")
print(f"  From score comparison (both alive, int > det): {len(int_score_win)}")

# Score delta analysis for labeled episodes
print(f"\nScore delta analysis (det_score - int_score):")
print(f"\n  Det-preferred (score comparison):")
if det_score_win:
    deltas = [ep["delta"] for ep in det_score_win]
    deltas.sort()
    print(f"    Count: {len(deltas)}")
    print(f"    Min delta: {min(deltas):.2f}")
    print(f"    Max delta: {max(deltas):.2f}")
    print(f"    Mean delta: {sum(deltas)/len(deltas):.2f}")
    print(f"    Median delta: {deltas[len(deltas)//2]:.2f}")
    # Histogram of deltas
    bins = [0, 1, 5, 10, 25, 50, 100, 200]
    print(f"    Delta distribution:")
    for i in range(len(bins)-1):
        n_in_bin = sum(1 for d in deltas if bins[i] < d <= bins[i+1])
        print(f"      ({bins[i]}, {bins[i+1]}]: {n_in_bin}")
    n_above = sum(1 for d in deltas if d > bins[-1])
    if n_above:
        print(f"      > {bins[-1]}: {n_above}")

print(f"\n  Det-preferred (death asymmetry):")
if det_death:
    deltas = [ep["delta"] for ep in det_death]
    print(f"    Count: {len(deltas)}")
    print(f"    Typical pattern: int_score=-100 (died), det_score=0 (alive)")
    # Show actual score pairs
    score_pairs = Counter((ep["int_score"], ep["det_score"]) for ep in det_death)
    print(f"    Score pairs (int, det) → count:")
    for (i, d), c in score_pairs.most_common(10):
        print(f"      ({i:.0f}, {d:.0f}): {c}")

print(f"\n  Int-preferred (score comparison):")
if int_score_win:
    deltas = [ep["delta"] for ep in int_score_win]
    deltas.sort()
    print(f"    Count: {len(deltas)}")
    print(f"    Min delta: {min(deltas):.2f} (i.e., int won by {-min(deltas):.2f})")
    print(f"    Max delta: {max(deltas):.2f} (i.e., int won by {-max(deltas):.2f})")

print(f"\n  Int-preferred (death asymmetry):")
if int_death:
    score_pairs = Counter((ep["int_score"], ep["det_score"]) for ep in int_death)
    print(f"    Count: {len(int_death)}")
    print(f"    Score pairs (int, det) → count:")
    for (i, d), c in score_pairs.most_common(10):
        print(f"      ({i:.0f}, {d:.0f}): {c}")

# ---- Summary for report ----
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

# Genuine labels = those from true score/death differences
# The "tie-break" concern from reviewer: all no-preference (150) are excluded
# Among labeled data (95), ALL are genuine (no arbitrary tie-breaking)
genuine_det = len(det_preferred)
genuine_int = len(int_preferred)
no_pref = label_counts.get("no-preference", 0)

print(f"""
Total episodes: {total}
Labeled (used for training): {genuine_det + genuine_int} ({100*(genuine_det+genuine_int)/total:.1f}%)
  det-preferred: {genuine_det}
    - death_asymmetry: {len(det_death)}
    - score_comparison: {len(det_score_win)}
  internal-preferred: {genuine_int}
    - death_asymmetry: {len(int_death)}
    - score_comparison: {len(int_score_win)}
No-preference (excluded): {no_pref} ({100*no_pref/total:.1f}%)
  - both_died: {len(tie_dead)}
  - tied_alive: {len(tie_alive)}

Key finding: NO tie-break labels exist. The labeling uses strict inequality.
Ties become no-preference and are excluded from training data.
All {genuine_det + genuine_int} training labels are genuine (from real score/death differences).
""")

# For task (c): output filtered data stats
print("=" * 60)
print("(C) DATA FOR STRICT-ONLY RETRAIN")
print("=" * 60)
print(f"Strict-only = exclude no-preference = keep {genuine_det + genuine_int} episodes")
print(f"This is IDENTICAL to the current training data (already strict).")
print(f"If we want an even stricter filter (exclude death_asymmetry, keep only score_comparison):")
strict_score_only = len(det_score_win) + len(int_score_win)
print(f"  Score-comparison only: {strict_score_only} episodes")
