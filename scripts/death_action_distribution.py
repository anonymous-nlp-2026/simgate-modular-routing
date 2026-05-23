import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import os
import re
from collections import Counter, defaultdict
import numpy as np

# ── Config ──
RESULTS_DIR = "./results/prescreening"
OUT_DIR = "./artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Action categorization ──
def categorize_action(action):
    a = action.lower().strip()
    if a.startswith("focus on"):
        return "focus on [X]"
    if a.startswith("connect ") and " to " in a:
        return "connect [X] to [Y]"
    if a.startswith("disconnect"):
        return "disconnect [X]"
    if any(a.startswith(w) for w in ["go to ", "move to ", "teleport to ", "open door to ", "go through "]):
        return "movement"
    if any(a.startswith(w) for w in ["pick up ", "put ", "pour ", "use ", "activate ", "deactivate ",
                                      "mix ", "drop ", "eat ", "drink ", "turn on ", "turn off ",
                                      "close ", "open "]):
        return "object interaction"
    if any(a.startswith(w) for w in ["look around", "look at ", "read ", "examine ", "inventory",
                                      "task", "wait", "wait1"]):
        return "observation"
    if a.startswith("move ") and " to " in a:
        return "move [X] to [Y]"
    return "other"

# Category display order
CAT_ORDER = ["focus on [X]", "connect [X] to [Y]", "disconnect [X]",
             "move [X] to [Y]", "movement", "object interaction", "observation", "other"]

# ── Load episodes ──
tasks = sorted([d for d in os.listdir(RESULTS_DIR)
                if os.path.isdir(os.path.join(RESULTS_DIR, d))])

all_episodes = []
for task in tasks:
    ep_file = os.path.join(RESULTS_DIR, task, "episodes.jsonl")
    if not os.path.exists(ep_file):
        continue
    with open(ep_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ep = json.loads(line)
            ep["_task"] = task
            all_episodes.append(ep)

print(f"Loaded {len(all_episodes)} episodes from {len(tasks)} tasks")

# ── Extract actions for death vs non-death per strategy ──
strategies = ["always_internal", "always_deterministic"]
strat_labels = {"always_internal": "Internal (LLM)", "always_deterministic": "Deterministic"}

results = {}
for strat in strategies:
    death_actions = []
    alive_actions = []
    death_count = 0
    alive_count = 0
    death_episodes_data = []  # for timeline figure

    for ep in all_episodes:
        if strat not in ep:
            continue
        run = ep[strat]
        is_death = (run.get("final_score", 0) == -100)

        actions = []
        for step in run.get("steps", []):
            act = step.get("action", "")
            if act:
                actions.append(act)

        if not actions and "action_history" in run:
            actions = run["action_history"]

        if is_death:
            death_count += 1
            death_actions.extend(actions)
            death_episodes_data.append({
                "task": ep["_task"],
                "variation": ep.get("variation_idx", 0),
                "actions": actions,
                "num_steps": run.get("num_steps", len(actions)),
            })
        else:
            alive_count += 1
            alive_actions.extend(actions)

    # Categorize
    death_cats = Counter([categorize_action(a) for a in death_actions])
    alive_cats = Counter([categorize_action(a) for a in alive_actions])

    death_total = sum(death_cats.values())
    alive_total = sum(alive_cats.values())

    results[strat] = {
        "death_count": death_count,
        "alive_count": alive_count,
        "death_cats": death_cats,
        "alive_cats": alive_cats,
        "death_total": death_total,
        "alive_total": alive_total,
        "death_episodes_data": death_episodes_data,
    }

    print(f"\n{strat_labels[strat]}:")
    print(f"  Death episodes: {death_count}, Non-death: {alive_count}")
    print(f"  Death actions: {death_total}, Non-death actions: {alive_total}")
    print(f"  Death distribution:")
    for cat in CAT_ORDER:
        n = death_cats.get(cat, 0)
        pct = 100.0 * n / death_total if death_total else 0
        print(f"    {cat}: {n} ({pct:.1f}%)")
    print(f"  Non-death distribution:")
    for cat in CAT_ORDER:
        n = alive_cats.get(cat, 0)
        pct = 100.0 * n / alive_total if alive_total else 0
        print(f"    {cat}: {n} ({pct:.1f}%)")

# ── Also check: what do death episodes' "focus on" target? ──
strat = "always_internal"
focus_targets_death = []
for ep in all_episodes:
    if strat not in ep:
        continue
    run = ep[strat]
    if run.get("final_score", 0) != -100:
        continue
    for step in run.get("steps", []):
        act = step.get("action", "")
        if act.lower().startswith("focus on"):
            focus_targets_death.append(act)

focus_counter = Counter(focus_targets_death)
print(f"\n\nFocus-on targets in internal death episodes (top 15):")
for target, count in focus_counter.most_common(15):
    print(f"  {target}: {count}")

# ══════════════════════════════════════════════════════════
# FIGURE 1: Side-by-side grouped bar chart
# ══════════════════════════════════════════════════════════
# Colorblind-safe palette (Wong 2011)
COLORS = {
    "focus on [X]": "#D55E00",       # vermillion (danger/death)
    "connect [X] to [Y]": "#0072B2", # blue
    "disconnect [X]": "#56B4E9",     # sky blue
    "move [X] to [Y]": "#009E73",    # bluish green
    "movement": "#F0E442",           # yellow
    "object interaction": "#CC79A7", # reddish purple
    "observation": "#999999",        # grey
    "other": "#E69F00",              # orange
}

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

for idx, strat in enumerate(strategies):
    ax = axes[idx]
    r = results[strat]
    label = strat_labels[strat]

    # Compute proportions for death and non-death
    cats_used = [c for c in CAT_ORDER if r["death_cats"].get(c, 0) > 0 or r["alive_cats"].get(c, 0) > 0]

    death_pcts = [100.0 * r["death_cats"].get(c, 0) / r["death_total"] if r["death_total"] else 0 for c in cats_used]
    alive_pcts = [100.0 * r["alive_cats"].get(c, 0) / r["alive_total"] if r["alive_total"] else 0 for c in cats_used]

    x = np.arange(len(cats_used))
    w = 0.35

    bars_d = ax.bar(x - w/2, death_pcts, w, color="#D55E00", alpha=0.85, label=f"Death (n={r['death_count']})")
    bars_a = ax.bar(x + w/2, alive_pcts, w, color="#0072B2", alpha=0.85, label=f"Non-death (n={r['alive_count']})")

    ax.set_xticks(x)
    ax.set_xticklabels(cats_used, rotation=35, ha="right", fontsize=7.5)
    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.legend(fontsize=7.5, loc="upper right")

    # Value labels on top bars
    for bars in [bars_d, bars_a]:
        for bar in bars:
            h = bar.get_height()
            if h > 2:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.8, f"{h:.1f}",
                        ha="center", va="bottom", fontsize=6.5)

axes[0].set_ylabel("Proportion of actions (%)", fontsize=9)

fig.suptitle("Action Distribution: Death vs Non-Death Episodes", fontsize=11, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])

fig.savefig(os.path.join(OUT_DIR, "death_action_distribution.pdf"), bbox_inches="tight", dpi=300)
fig.savefig(os.path.join(OUT_DIR, "death_action_distribution.png"), bbox_inches="tight", dpi=300)
plt.close(fig)
print("\nFigure 1 saved.")

# ══════════════════════════════════════════════════════════
# FIGURE 2: Action sequence timeline for representative death episodes (internal)
# ══════════════════════════════════════════════════════════
# Pick representative death episodes: diverse tasks, varying lengths
death_eps = results["always_internal"]["death_episodes_data"]
# Sort by task diversity, pick 1 per task, prefer shorter episodes to fit
seen_tasks = set()
selected = []
# Sort by num_steps ascending
for ep in sorted(death_eps, key=lambda x: x["num_steps"]):
    if ep["task"] not in seen_tasks and len(selected) < 8:
        seen_tasks.add(ep["task"])
        selected.append(ep)

# Limit to 6 for readability
selected = selected[:6]

# Find last "focus on" step for each (the fatal one)
def find_fatal_step(actions):
    for i in range(len(actions)-1, -1, -1):
        if actions[i].lower().startswith("focus on"):
            return i
    return -1

fig2, ax2 = plt.subplots(figsize=(10, 3.5))

cat_colors = COLORS

y_positions = list(range(len(selected)))
max_steps = max(len(ep["actions"]) for ep in selected)

for y, ep in enumerate(selected):
    actions = ep["actions"]
    fatal_idx = find_fatal_step(actions)
    for x, act in enumerate(actions):
        cat = categorize_action(act)
        color = cat_colors.get(cat, "#CCCCCC")
        rect = plt.Rectangle((x, y - 0.35), 1, 0.7, facecolor=color, edgecolor="white", linewidth=0.3)
        ax2.add_patch(rect)
        # Mark fatal step
        if x == fatal_idx:
            ax2.plot(x + 0.5, y, marker="x", color="black", markersize=6, markeredgewidth=1.5, zorder=5)

ax2.set_xlim(-0.5, min(max_steps + 1, 32))
ax2.set_ylim(-0.6, len(selected) - 0.4)
ax2.set_yticks(y_positions)
ax2.set_yticklabels([f"{ep['task']}\n(v{ep['variation']})" for ep in selected], fontsize=7)
ax2.set_xlabel("Step", fontsize=9)
ax2.set_title("Action Sequences of Representative Death Episodes (Internal Strategy)", fontsize=10, fontweight="bold")

# Legend
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements = [Patch(facecolor=cat_colors[c], label=c) for c in CAT_ORDER if c in cat_colors]
legend_elements.append(Line2D([0], [0], marker="x", color="black", linestyle="None",
                               markersize=6, markeredgewidth=1.5, label="fatal focus on"))
ax2.legend(handles=legend_elements, fontsize=6.5, loc="upper right", ncol=3,
           framealpha=0.9, borderpad=0.4, handletextpad=0.4, columnspacing=0.8)

plt.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, "death_action_timeline.pdf"), bbox_inches="tight", dpi=300)
fig2.savefig(os.path.join(OUT_DIR, "death_action_timeline.png"), bbox_inches="tight", dpi=300)
plt.close(fig2)
print("Figure 2 saved.")

print("\nDone. Output files:")
for f in ["death_action_distribution.pdf", "death_action_distribution.png",
          "death_action_timeline.pdf", "death_action_timeline.png"]:
    path = os.path.join(OUT_DIR, f)
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f"  {path} ({size} bytes)")
