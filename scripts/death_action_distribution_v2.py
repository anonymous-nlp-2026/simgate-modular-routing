import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import json
import os
from collections import Counter
import numpy as np

RESULTS_DIR = "./results/prescreening"
OUT_DIR = "./artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

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

CAT_ORDER = ["focus on [X]", "connect [X] to [Y]", "disconnect [X]",
             "move [X] to [Y]", "movement", "object interaction", "observation", "other"]

# Shorter labels for figures
CAT_SHORT = {
    "focus on [X]": "focus on\n[X]",
    "connect [X] to [Y]": "connect\n[X] to [Y]",
    "disconnect [X]": "disconnect\n[X]",
    "move [X] to [Y]": "move [X]\nto [Y]",
    "movement": "movement",
    "object interaction": "object\ninteraction",
    "observation": "observation",
    "other": "other",
}

# Colorblind-safe palette
COLORS = {
    "focus on [X]": "#D55E00",
    "connect [X] to [Y]": "#0072B2",
    "disconnect [X]": "#56B4E9",
    "move [X] to [Y]": "#009E73",
    "movement": "#F0E442",
    "object interaction": "#CC79A7",
    "observation": "#999999",
    "other": "#E69F00",
}

# Load data
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

strategies = ["always_internal", "always_deterministic"]
strat_labels = {"always_internal": "Internal (LLM)", "always_deterministic": "Deterministic"}

results = {}
for strat in strategies:
    death_actions = []
    alive_actions = []
    death_count = 0
    alive_count = 0
    death_episodes_data = []

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


# ═══════════════════════════════════════════════════════════
# FIGURE 1: Focus on Internal strategy (where death problem is)
# Two-panel: left = full distribution (stacked bar), right = zoomed non-connect
# ═══════════════════════════════════════════════════════════

fig = plt.figure(figsize=(10, 4.2))
gs = gridspec.GridSpec(1, 2, width_ratios=[1.0, 1.6], wspace=0.35)

strat = "always_internal"
r = results[strat]

# Active categories (non-zero in either group)
cats_active = [c for c in CAT_ORDER
               if r["death_cats"].get(c, 0) > 0 or r["alive_cats"].get(c, 0) > 0]

death_pcts = {c: 100.0 * r["death_cats"].get(c, 0) / r["death_total"] for c in cats_active}
alive_pcts = {c: 100.0 * r["alive_cats"].get(c, 0) / r["alive_total"] for c in cats_active}

# Left panel: full stacked horizontal bars (death vs non-death)
ax_left = fig.add_subplot(gs[0])
bar_h = 0.35
# Stack order: connect first (biggest), then others
for i, (label, pcts, ypos) in enumerate([
    (f"Death\n(n={r['death_count']})", death_pcts, 1),
    (f"Non-death\n(n={r['alive_count']})", alive_pcts, 0),
]):
    left = 0
    for cat in cats_active:
        w = pcts.get(cat, 0)
        if w > 0:
            ax_left.barh(ypos, w, left=left, height=bar_h, color=COLORS[cat],
                        edgecolor="white", linewidth=0.5)
            if w > 4:
                ax_left.text(left + w/2, ypos, f"{w:.0f}%", ha="center", va="center",
                           fontsize=6.5, fontweight="bold", color="white" if w > 10 else "black")
            left += w

ax_left.set_yticks([0, 1])
ax_left.set_yticklabels([f"Non-death\n(n={r['alive_count']})", f"Death\n(n={r['death_count']})"], fontsize=8)
ax_left.set_xlabel("Proportion of actions (%)", fontsize=8)
ax_left.set_title("Full action distribution", fontsize=9, fontweight="bold")
ax_left.set_xlim(0, 105)
ax_left.spines["top"].set_visible(False)
ax_left.spines["right"].set_visible(False)

# Legend for left panel
legend_elements = [Patch(facecolor=COLORS[c], edgecolor="white", label=c)
                   for c in cats_active]
ax_left.legend(handles=legend_elements, fontsize=6, loc="lower right",
              ncol=1, framealpha=0.9, borderpad=0.3)

# Right panel: zoomed grouped bar for non-connect categories
ax_right = fig.add_subplot(gs[1])
cats_zoom = [c for c in cats_active if c != "connect [X] to [Y]"]

x = np.arange(len(cats_zoom))
w = 0.35

d_vals = [death_pcts.get(c, 0) for c in cats_zoom]
a_vals = [alive_pcts.get(c, 0) for c in cats_zoom]

bars_d = ax_right.bar(x - w/2, d_vals, w, color="#D55E00", alpha=0.9,
                       label=f"Death (n={r['death_count']})", edgecolor="white", linewidth=0.5)
bars_a = ax_right.bar(x + w/2, a_vals, w, color="#0072B2", alpha=0.9,
                       label=f"Non-death (n={r['alive_count']})", edgecolor="white", linewidth=0.5)

# Value labels
for bars in [bars_d, bars_a]:
    for bar in bars:
        h = bar.get_height()
        if h > 0.3:
            ax_right.text(bar.get_x() + bar.get_width()/2, h + 0.15,
                         f"{h:.1f}", ha="center", va="bottom", fontsize=7)

ax_right.set_xticks(x)
ax_right.set_xticklabels([CAT_SHORT[c] for c in cats_zoom], fontsize=7.5)
ax_right.set_ylabel("Proportion of actions (%)", fontsize=8)
ax_right.set_title("Non-connect actions (zoomed)", fontsize=9, fontweight="bold")
ax_right.legend(fontsize=7.5, loc="upper right")
ax_right.spines["top"].set_visible(False)
ax_right.spines["right"].set_visible(False)

# Highlight the focus-on difference
focus_idx = cats_zoom.index("focus on [X]")
ax_right.annotate("3x higher\nin death",
                  xy=(focus_idx - w/2, death_pcts["focus on [X]"]),
                  xytext=(focus_idx - w/2, death_pcts["focus on [X]"] + 2.5),
                  fontsize=6.5, ha="center", fontweight="bold", color="#D55E00",
                  arrowprops=dict(arrowstyle="-", color="#D55E00", lw=0.8))

fig.suptitle("Internal (LLM) Strategy: Action Distribution in Death vs Non-Death Episodes",
             fontsize=10.5, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "death_action_distribution.pdf"), bbox_inches="tight", dpi=300)
fig.savefig(os.path.join(OUT_DIR, "death_action_distribution.png"), bbox_inches="tight", dpi=300)
plt.close(fig)
print("Figure 1 (v2) saved.")


# ═══════════════════════════════════════════════════════════
# FIGURE 2: Action sequence timeline for representative death episodes
# ═══════════════════════════════════════════════════════════
death_eps = results["always_internal"]["death_episodes_data"]
seen_tasks = set()
selected = []
for ep in sorted(death_eps, key=lambda x: x["num_steps"]):
    if ep["task"] not in seen_tasks and len(selected) < 8:
        seen_tasks.add(ep["task"])
        selected.append(ep)
selected = selected[:6]

def find_fatal_step(actions):
    for i in range(len(actions)-1, -1, -1):
        if actions[i].lower().startswith("focus on"):
            return i
    return -1

fig2, ax2 = plt.subplots(figsize=(10, 3.2))

y_positions = list(range(len(selected)))
max_steps = max(len(ep["actions"]) for ep in selected)

for y, ep in enumerate(selected):
    actions = ep["actions"]
    fatal_idx = find_fatal_step(actions)
    for x, act in enumerate(actions):
        cat = categorize_action(act)
        color = COLORS.get(cat, "#CCCCCC")
        edgecolor = "black" if x == fatal_idx else "white"
        lw = 1.5 if x == fatal_idx else 0.3
        rect = plt.Rectangle((x, y - 0.35), 1, 0.7, facecolor=color,
                              edgecolor=edgecolor, linewidth=lw)
        ax2.add_patch(rect)
    # Arrow pointing to fatal step
    if fatal_idx >= 0:
        ax2.annotate("", xy=(fatal_idx + 0.5, y + 0.5),
                     xytext=(fatal_idx + 0.5, y + 0.85),
                     arrowprops=dict(arrowstyle="->", color="black", lw=1.2))

ax2.set_xlim(-0.5, min(max_steps + 1, 32))
ax2.set_ylim(-0.6, len(selected) + 0.2)
ax2.set_yticks(y_positions)
ax2.set_yticklabels([ep["task"].replace("-", " ") for ep in selected], fontsize=7)
ax2.set_xlabel("Step number", fontsize=9)
ax2.set_title("Action Sequences of Representative Death Episodes (Internal Strategy)",
              fontsize=10, fontweight="bold")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# Legend
legend_elements = [Patch(facecolor=COLORS[c], label=c) for c in CAT_ORDER if c in COLORS]
legend_elements.append(Patch(facecolor="white", edgecolor="black", linewidth=1.5,
                             label="fatal step"))
ax2.legend(handles=legend_elements, fontsize=6.5, loc="upper right", ncol=3,
           framealpha=0.95, borderpad=0.4, handletextpad=0.4, columnspacing=0.8)

plt.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, "death_action_timeline.pdf"), bbox_inches="tight", dpi=300)
fig2.savefig(os.path.join(OUT_DIR, "death_action_timeline.png"), bbox_inches="tight", dpi=300)
plt.close(fig2)
print("Figure 2 (v2) saved.")

# ═══════════════════════════════════════════════════════════
# Also: compare Internal vs Deterministic death rates in one figure
# ═══════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)

for idx, strat in enumerate(strategies):
    ax = axes3[idx]
    r = results[strat]
    cats_zoom = [c for c in CAT_ORDER
                 if c != "connect [X] to [Y]" and
                 (r["death_cats"].get(c, 0) > 0 or r["alive_cats"].get(c, 0) > 0)]

    d_vals = [100.0 * r["death_cats"].get(c, 0) / r["death_total"] if r["death_total"] else 0 for c in cats_zoom]
    a_vals = [100.0 * r["alive_cats"].get(c, 0) / r["alive_total"] if r["alive_total"] else 0 for c in cats_zoom]

    x = np.arange(len(cats_zoom))
    w = 0.35

    ax.bar(x - w/2, d_vals, w, color="#D55E00", alpha=0.9,
           label=f"Death (n={r['death_count']})", edgecolor="white", linewidth=0.5)
    ax.bar(x + w/2, a_vals, w, color="#0072B2", alpha=0.9,
           label=f"Non-death (n={r['alive_count']})", edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([CAT_SHORT[c] for c in cats_zoom], fontsize=7)
    ax.set_title(f"{strat_labels[strat]}", fontsize=9.5, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add connect % as text note
    d_connect = 100.0 * r["death_cats"].get("connect [X] to [Y]", 0) / r["death_total"] if r["death_total"] else 0
    a_connect = 100.0 * r["alive_cats"].get("connect [X] to [Y]", 0) / r["alive_total"] if r["alive_total"] else 0
    ax.text(0.02, 0.95, f"connect: {d_connect:.0f}% (death) / {a_connect:.0f}% (non-death)",
            transform=ax.transAxes, fontsize=6.5, va="top", fontstyle="italic", color="#555555")

axes3[0].set_ylabel("Proportion of actions (%)", fontsize=8)
fig3.suptitle("Non-Connect Action Distribution: Death vs Non-Death Episodes",
              fontsize=10.5, fontweight="bold", y=1.01)
plt.tight_layout()
fig3.savefig(os.path.join(OUT_DIR, "death_action_comparison.pdf"), bbox_inches="tight", dpi=300)
fig3.savefig(os.path.join(OUT_DIR, "death_action_comparison.png"), bbox_inches="tight", dpi=300)
plt.close(fig3)
print("Figure 3 (comparison) saved.")

print("\nAll output files:")
for f in sorted(os.listdir(OUT_DIR)):
    if f.startswith("death_action"):
        path = os.path.join(OUT_DIR, f)
        size = os.path.getsize(path)
        print(f"  {f} ({size:,} bytes)")
