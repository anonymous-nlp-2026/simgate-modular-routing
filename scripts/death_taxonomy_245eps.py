import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
import json
import os
from collections import Counter, defaultdict
import numpy as np

RESULTS_DIR = "./results/plan_c"
OUT_DIR = "./artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

# Episode limits per task to match baseline_245eps (245 total)
EPISODE_LIMITS = {
    "chemistry-mix": 20,
    "find-animal": 20,
    "find-non-living-thing": 40,
    "freeze": 20,
    "grow-fruit": 20,
    "identify-life-stages-1": 20,
    "identify-life-stages-2": 20,
    "inclined-plane-determine-angle": 20,
    "lifespan-longest-lived": 25,
    "lifespan-longest-lived-then-shortest-lived": 20,
    "measure-melting-point-unknown-substance": 20,
}

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

# ─────────────────────────────────────────────────
# Load data with episode limits
# ─────────────────────────────────────────────────
tasks = sorted([d for d in os.listdir(RESULTS_DIR)
                if os.path.isdir(os.path.join(RESULTS_DIR, d)) and d in EPISODE_LIMITS])
all_episodes = []
for task in tasks:
    ep_file = os.path.join(RESULTS_DIR, task, "episodes.jsonl")
    if not os.path.exists(ep_file):
        continue
    limit = EPISODE_LIMITS.get(task, 999)
    count = 0
    with open(ep_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if count >= limit:
                break
            ep = json.loads(line)
            ep["_task"] = task
            all_episodes.append(ep)
            count += 1

print(f"Loaded {len(all_episodes)} episodes from {len(tasks)} tasks")

# ─────────────────────────────────────────────────
# Analyze both strategies
# ─────────────────────────────────────────────────
strategies = ["always_internal", "always_deterministic"]
strat_labels = {"always_internal": "Internal (LLM)", "always_deterministic": "Deterministic"}

results = {}
for strat in strategies:
    death_actions = []
    alive_actions = []
    death_count = 0
    alive_count = 0
    death_episodes_data = []
    per_task_deaths = defaultdict(lambda: {"deaths": 0, "total": 0})

    for ep in all_episodes:
        if strat not in ep:
            continue
        run = ep[strat]
        task = ep["_task"]
        is_death = (run.get("final_score", 0) == -100)
        per_task_deaths[task]["total"] += 1

        actions = []
        for step in run.get("steps", []):
            act = step.get("action", "")
            if act:
                actions.append(act)
        if not actions and "action_history" in run:
            actions = run["action_history"]

        if is_death:
            death_count += 1
            per_task_deaths[task]["deaths"] += 1
            death_actions.extend(actions)
            death_episodes_data.append({
                "task": task,
                "variation": ep.get("variation_idx", 0),
                "episode_idx": ep.get("episode_idx", 0),
                "actions": actions,
                "num_steps": run.get("num_steps", len(actions)),
                "steps": run.get("steps", []),
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
        "death_actions": death_actions,
        "alive_actions": alive_actions,
        "death_cats": death_cats,
        "alive_cats": alive_cats,
        "death_total": death_total,
        "alive_total": alive_total,
        "death_episodes_data": death_episodes_data,
        "per_task_deaths": dict(per_task_deaths),
    }

# ─────────────────────────────────────────────────
# Death taxonomy analysis (Internal strategy only)
# ─────────────────────────────────────────────────
int_r = results["always_internal"]
det_r = results["always_deterministic"]
print(f"\n=== DEATH TAXONOMY (245 episodes) ===")
print(f"Internal: {int_r['death_count']} deaths / {int_r['death_count'] + int_r['alive_count']} episodes "
      f"({100*int_r['death_count']/(int_r['death_count']+int_r['alive_count']):.1f}%)")
print(f"Deterministic: {det_r['death_count']} deaths / {det_r['death_count'] + det_r['alive_count']} episodes "
      f"({100*det_r['death_count']/(det_r['death_count']+det_r['alive_count']):.1f}%)")

# Fatal action analysis
fatal_action_cats = Counter()
fatal_focus_targets = Counter()
non_focus_fatal = []

for dep in int_r["death_episodes_data"]:
    steps = dep["steps"]
    fatal_step = None
    for s in steps:
        if s.get("score_delta", 0) == -100 or s.get("is_dead", False):
            fatal_step = s
            break
        if "candidates" in s:
            for c in s.get("candidates", []):
                if c.get("lethal", False) and c.get("action") == s.get("action"):
                    fatal_step = s
                    break
            if fatal_step:
                break
    
    if fatal_step is None and steps:
        for s in reversed(steps):
            if s.get("score_after", 0) == -100 or s.get("score_delta", 0) <= -100:
                fatal_step = s
                break
    
    if fatal_step is None and steps:
        fatal_step = steps[-1]
    
    if fatal_step:
        action = fatal_step.get("action", "")
        cat = categorize_action(action)
        fatal_action_cats[cat] += 1
        if action.lower().startswith("focus on"):
            target = action[len("focus on "):].strip()
            fatal_focus_targets[target] += 1
        else:
            non_focus_fatal.append({
                "task": dep["task"],
                "action": action,
                "cat": cat,
            })

print(f"\n--- Fatal Action Categories ---")
total_deaths = sum(fatal_action_cats.values())
for cat, cnt in fatal_action_cats.most_common():
    print(f"  {cat}: {cnt} ({100*cnt/total_deaths:.1f}%)")

print(f"\n--- Focus-on Target Distribution (fatal actions only) ---")
for target, cnt in fatal_focus_targets.most_common(20):
    print(f"  focus on {target}: {cnt}")

if non_focus_fatal:
    print(f"\n--- Non-focus-on Fatal Actions ---")
    for nf in non_focus_fatal:
        print(f"  [{nf['task']}] {nf['action']} -> {nf['cat']}")

# Per-task death rate
print(f"\n--- Per-Task Death Rate (Internal) ---")
print(f"{'Task':<45} {'Deaths':>6} {'Total':>6} {'Rate':>8}")
for task in sorted(int_r["per_task_deaths"].keys()):
    d = int_r["per_task_deaths"][task]
    rate = 100 * d["deaths"] / d["total"] if d["total"] > 0 else 0
    print(f"  {task:<43} {d['deaths']:>6} {d['total']:>6} {rate:>7.1f}%")

# Per-task death rate (Deterministic)
print(f"\n--- Per-Task Death Rate (Deterministic) ---")
print(f"{'Task':<45} {'Deaths':>6} {'Total':>6} {'Rate':>8}")
for task in sorted(det_r["per_task_deaths"].keys()):
    d = det_r["per_task_deaths"][task]
    rate = 100 * d["deaths"] / d["total"] if d["total"] > 0 else 0
    print(f"  {task:<43} {d['deaths']:>6} {d['total']:>6} {rate:>7.1f}%")

# Action category distribution
print(f"\n--- Action Category Distribution (Internal) ---")
print(f"{'Category':<25} {'Death%':>8} {'Alive%':>8}")
for cat in CAT_ORDER:
    d_pct = 100.0 * int_r["death_cats"].get(cat, 0) / int_r["death_total"] if int_r["death_total"] else 0
    a_pct = 100.0 * int_r["alive_cats"].get(cat, 0) / int_r["alive_total"] if int_r["alive_total"] else 0
    print(f"  {cat:<23} {d_pct:>7.1f}% {a_pct:>7.1f}%")

# ─────────────────────────────────────────────────
# Write analysis report
# ─────────────────────────────────────────────────
report_lines = []
report_lines.append("# Death Failure Taxonomy (245 Episodes, 11 Tasks)")
report_lines.append("")
report_lines.append(f"**Data source**: `results/plan_c/` filtered to match `baseline_245eps/` (245 episodes, seed=42)")
report_lines.append(f"**Analysis date**: 2026-05-18")
report_lines.append("")
report_lines.append("## 1. Death Count Summary")
report_lines.append("")
int_total = int_r["death_count"] + int_r["alive_count"]
det_total = det_r["death_count"] + det_r["alive_count"]
report_lines.append(f"| Strategy | Deaths | Total | Death Rate |")
report_lines.append(f"|---|---|---|---|")
report_lines.append(f"| Internal (LLM) | {int_r['death_count']} | {int_total} | {100*int_r['death_count']/int_total:.1f}% |")
report_lines.append(f"| Deterministic | {det_r['death_count']} | {det_total} | {100*det_r['death_count']/det_total:.1f}% |")
report_lines.append("")
report_lines.append("## 2. Fatal Action Classification")
report_lines.append("")
report_lines.append(f"| Fatal Action Category | Count | % |")
report_lines.append(f"|---|---|---|")
for cat, cnt in fatal_action_cats.most_common():
    report_lines.append(f"| {cat} | {cnt} | {100*cnt/total_deaths:.1f}% |")
report_lines.append("")

focus_count = fatal_action_cats.get("focus on [X]", 0)
report_lines.append(f"**{100*focus_count/total_deaths:.1f}% of fatal actions are `focus on [X]`**")
report_lines.append("")

report_lines.append("## 3. Focus-on Target Distribution (Fatal Actions)")
report_lines.append("")
report_lines.append(f"| Target | Count | % of focus-on deaths |")
report_lines.append(f"|---|---|---|")
for target, cnt in fatal_focus_targets.most_common(20):
    pct = 100 * cnt / focus_count if focus_count > 0 else 0
    report_lines.append(f"| {target} | {cnt} | {pct:.1f}% |")
report_lines.append("")

report_lines.append("## 4. Per-Task Death Rate")
report_lines.append("")
report_lines.append(f"| Task | Int Deaths | Int Total | Int Rate | Det Deaths | Det Total | Det Rate |")
report_lines.append(f"|---|---|---|---|---|---|---|")
for task in sorted(EPISODE_LIMITS.keys()):
    id_ = int_r["per_task_deaths"].get(task, {"deaths": 0, "total": 0})
    dd_ = det_r["per_task_deaths"].get(task, {"deaths": 0, "total": 0})
    ir = 100 * id_["deaths"] / id_["total"] if id_["total"] > 0 else 0
    dr = 100 * dd_["deaths"] / dd_["total"] if dd_["total"] > 0 else 0
    report_lines.append(f"| {task} | {id_['deaths']} | {id_['total']} | {ir:.1f}% | {dd_['deaths']} | {dd_['total']} | {dr:.1f}% |")
report_lines.append("")

report_lines.append("## 5. Action Category Distribution (Internal)")
report_lines.append("")
report_lines.append(f"| Category | Death Episodes (%) | Non-Death Episodes (%) |")
report_lines.append(f"|---|---|---|")
for cat in CAT_ORDER:
    d_pct = 100.0 * int_r["death_cats"].get(cat, 0) / int_r["death_total"] if int_r["death_total"] else 0
    a_pct = 100.0 * int_r["alive_cats"].get(cat, 0) / int_r["alive_total"] if int_r["alive_total"] else 0
    report_lines.append(f"| {cat} | {d_pct:.1f}% | {a_pct:.1f}% |")
report_lines.append("")

if non_focus_fatal:
    report_lines.append("## 6. Non-Focus-On Fatal Actions")
    report_lines.append("")
    report_lines.append(f"| Task | Action | Category |")
    report_lines.append(f"|---|---|---|")
    for nf in non_focus_fatal:
        report_lines.append(f"| {nf['task']} | {nf['action']} | {nf['cat']} |")
    report_lines.append("")

report_lines.append("## Key Findings")
report_lines.append("")
report_lines.append(f"1. Internal strategy: {int_r['death_count']}/{int_total} episodes end in death ({100*int_r['death_count']/int_total:.1f}%), vs deterministic {det_r['death_count']}/{det_total} ({100*det_r['death_count']/det_total:.1f}%)")
report_lines.append(f"2. {focus_count}/{total_deaths} fatal actions ({100*focus_count/total_deaths:.1f}%) are `focus on [X]`")
if non_focus_fatal:
    report_lines.append(f"3. {len(non_focus_fatal)} death(s) caused by non-focus-on actions")
else:
    report_lines.append(f"3. ALL deaths caused by `focus on [X]` (100%)")
top_targets = fatal_focus_targets.most_common(3)
targets_str = ", ".join([f"`{t}` ({c})" for t, c in top_targets])
report_lines.append(f"4. Top focus-on targets: {targets_str}")
report_lines.append(f"5. `find-non-living-thing` has 0% death rate (both strategies)")

report_path = os.path.join(OUT_DIR, "death_failure_taxonomy_245eps.md")
with open(report_path, "w") as f:
    f.write("\n".join(report_lines))
print(f"\nReport written to {report_path}")

# ═══════════════════════════════════════════════════════════
# Figure 1: Death Action Distribution (dual panel)
# ═══════════════════════════════════════════════════════════
r = int_r
fig1, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(10, 4))

cats_all = [c for c in CAT_ORDER if r["death_cats"].get(c, 0) > 0 or r["alive_cats"].get(c, 0) > 0]
d_vals_all = [100.0 * r["death_cats"].get(c, 0) / r["death_total"] if r["death_total"] else 0 for c in cats_all]
a_vals_all = [100.0 * r["alive_cats"].get(c, 0) / r["alive_total"] if r["alive_total"] else 0 for c in cats_all]

x = np.arange(len(cats_all))
w = 0.35
ax_full.bar(x - w/2, d_vals_all, w, color="#D55E00", alpha=0.9,
            label=f"Death (n={r['death_count']})", edgecolor="white", linewidth=0.5)
ax_full.bar(x + w/2, a_vals_all, w, color="#0072B2", alpha=0.9,
            label=f"Non-death (n={r['alive_count']})", edgecolor="white", linewidth=0.5)
ax_full.set_xticks(x)
ax_full.set_xticklabels([CAT_SHORT[c] for c in cats_all], fontsize=7)
ax_full.set_ylabel("Proportion of actions (%)", fontsize=9)
ax_full.set_title("(a) Full Action Distribution", fontsize=10, fontweight="bold")
ax_full.legend(fontsize=7, loc="upper right")
ax_full.spines["top"].set_visible(False)
ax_full.spines["right"].set_visible(False)

cats_zoom = [c for c in cats_all if c != "connect [X] to [Y]"]
d_vals_zoom = [100.0 * r["death_cats"].get(c, 0) / r["death_total"] if r["death_total"] else 0 for c in cats_zoom]
a_vals_zoom = [100.0 * r["alive_cats"].get(c, 0) / r["alive_total"] if r["alive_total"] else 0 for c in cats_zoom]
x2 = np.arange(len(cats_zoom))
ax_zoom.bar(x2 - w/2, d_vals_zoom, w, color="#D55E00", alpha=0.9,
            label=f"Death (n={r['death_count']})", edgecolor="white", linewidth=0.5)
ax_zoom.bar(x2 + w/2, a_vals_zoom, w, color="#0072B2", alpha=0.9,
            label=f"Non-death (n={r['alive_count']})", edgecolor="white", linewidth=0.5)
ax_zoom.set_xticks(x2)
ax_zoom.set_xticklabels([CAT_SHORT[c] for c in cats_zoom], fontsize=7)
ax_zoom.set_title("(b) Non-Connect Actions (zoomed)", fontsize=10, fontweight="bold")
ax_zoom.legend(fontsize=7, loc="upper right")
ax_zoom.spines["top"].set_visible(False)
ax_zoom.spines["right"].set_visible(False)

d_connect = 100.0 * r["death_cats"].get("connect [X] to [Y]", 0) / r["death_total"] if r["death_total"] else 0
a_connect = 100.0 * r["alive_cats"].get("connect [X] to [Y]", 0) / r["alive_total"] if r["alive_total"] else 0
ax_zoom.text(0.02, 0.95, f"connect: {d_connect:.0f}% (death) / {a_connect:.0f}% (non-death)",
             transform=ax_zoom.transAxes, fontsize=6.5, va="top", fontstyle="italic", color="#555555")

fig1.suptitle("Internal Strategy: Action Distribution in Death vs Non-Death Episodes (n=245)",
              fontsize=10.5, fontweight="bold", y=1.01)
plt.tight_layout()
fig1.savefig(os.path.join(OUT_DIR, "death_action_distribution.pdf"), bbox_inches="tight", dpi=300)
fig1.savefig(os.path.join(OUT_DIR, "death_action_distribution.png"), bbox_inches="tight", dpi=300)
plt.close(fig1)
print("Figure 1 saved.")

# ═══════════════════════════════════════════════════════════
# Figure 2: Death Action Timeline (6 representative episodes)
# ═══════════════════════════════════════════════════════════
dep_data = int_r["death_episodes_data"]

# Select 6 representative episodes from different tasks
task_episodes = defaultdict(list)
for dep in dep_data:
    task_episodes[dep["task"]].append(dep)

selected = []
target_tasks = ["chemistry-mix", "freeze", "find-animal", "identify-life-stages-1",
                "inclined-plane-determine-angle", "lifespan-longest-lived"]
for t in target_tasks:
    if t in task_episodes and task_episodes[t]:
        selected.append(task_episodes[t][0])
    if len(selected) >= 6:
        break

# Fill remaining from other tasks if needed
if len(selected) < 6:
    for t in sorted(task_episodes.keys()):
        if t not in target_tasks and task_episodes[t]:
            selected.append(task_episodes[t][0])
        if len(selected) >= 6:
            break

max_steps = max(len(ep["actions"]) for ep in selected)

fig2, ax2 = plt.subplots(figsize=(10, 4))
y_positions = list(range(len(selected)))

for y, ep in zip(y_positions, selected):
    actions = ep["actions"]
    fatal_idx = -1
    steps = ep.get("steps", [])
    for i, s in enumerate(steps):
        if s.get("score_delta", 0) == -100 or s.get("is_dead", False):
            fatal_idx = i
            break
        if "candidates" in s:
            for c in s.get("candidates", []):
                if c.get("lethal", False) and c.get("action") == s.get("action"):
                    fatal_idx = i
                    break
            if fatal_idx >= 0:
                break
    if fatal_idx < 0 and steps:
        for i in range(len(steps)-1, -1, -1):
            if steps[i].get("score_after", 0) == -100 or steps[i].get("score_delta", 0) <= -100:
                fatal_idx = i
                break

    for x_pos, act in enumerate(actions):
        cat = categorize_action(act)
        color = COLORS.get(cat, "#E69F00")
        is_fatal = (x_pos == fatal_idx)
        edgecolor = "black" if is_fatal else "none"
        lw = 1.5 if is_fatal else 0
        rect = Rectangle((x_pos, y - 0.35), 1, 0.7, facecolor=color,
                          edgecolor=edgecolor, linewidth=lw)
        ax2.add_patch(rect)
    if fatal_idx >= 0:
        ax2.annotate("", xy=(fatal_idx + 0.5, y + 0.5),
                     xytext=(fatal_idx + 0.5, y + 0.85),
                     arrowprops=dict(arrowstyle="->", color="black", lw=1.2))

ax2.set_xlim(-0.5, min(max_steps + 1, 32))
ax2.set_ylim(-0.6, len(selected) + 0.2)
ax2.set_yticks(y_positions)
ax2.set_yticklabels([ep["task"].replace("-", " ") for ep in selected], fontsize=7)
ax2.set_xlabel("Step number", fontsize=9)
ax2.set_title("Action Sequences of Representative Death Episodes (Internal Strategy, n=245 dataset)",
              fontsize=10, fontweight="bold")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

legend_elements = [Patch(facecolor=COLORS[c], label=c) for c in CAT_ORDER if c in COLORS]
legend_elements.append(Patch(facecolor="white", edgecolor="black", linewidth=1.5,
                             label="fatal step"))
ax2.legend(handles=legend_elements, fontsize=6.5, loc="upper right", ncol=3,
           framealpha=0.95, borderpad=0.4, handletextpad=0.4, columnspacing=0.8)

plt.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, "death_action_timeline.pdf"), bbox_inches="tight", dpi=300)
fig2.savefig(os.path.join(OUT_DIR, "death_action_timeline.png"), bbox_inches="tight", dpi=300)
plt.close(fig2)
print("Figure 2 saved.")

# ═══════════════════════════════════════════════════════════
# Figure 3: Internal vs Deterministic comparison
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

    d_connect = 100.0 * r["death_cats"].get("connect [X] to [Y]", 0) / r["death_total"] if r["death_total"] else 0
    a_connect = 100.0 * r["alive_cats"].get("connect [X] to [Y]", 0) / r["alive_total"] if r["alive_total"] else 0
    ax.text(0.02, 0.95, f"connect: {d_connect:.0f}% (death) / {a_connect:.0f}% (non-death)",
            transform=ax.transAxes, fontsize=6.5, va="top", fontstyle="italic", color="#555555")

axes3[0].set_ylabel("Proportion of actions (%)", fontsize=8)
fig3.suptitle("Non-Connect Action Distribution: Death vs Non-Death Episodes (n=245)",
              fontsize=10.5, fontweight="bold", y=1.01)
plt.tight_layout()
fig3.savefig(os.path.join(OUT_DIR, "death_action_comparison.pdf"), bbox_inches="tight", dpi=300)
fig3.savefig(os.path.join(OUT_DIR, "death_action_comparison.png"), bbox_inches="tight", dpi=300)
plt.close(fig3)
print("Figure 3 saved.")

print("\nAll output files:")
for f in sorted(os.listdir(OUT_DIR)):
    if f.startswith("death_"):
        path = os.path.join(OUT_DIR, f)
        size = os.path.getsize(path)
        print(f"  {f} ({size:,} bytes)")
