"""Generate death_action_timeline figure (Figure 4): representative death episodes.

Horizontal stacked bar showing action sequences of 6 death episodes (Internal).
Data from death taxonomy analysis (245 episodes, 11 tasks).
Action sequences approximate from visual inspection of original figure.
"""
import paper_style; paper_style.apply_style()
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import os

COLORS = {
    "focus on [X]":        paper_style.C_DETERMINISTIC,
    "connect [X] to [Y]": paper_style.C_INTERNAL,
    "disconnect [X]":      paper_style.C_PURPLE,
    "observation":         paper_style.C_NEUTRAL,
    "object interaction":  paper_style.C_ACCENT,
    "movement":            paper_style.C_GREEN,
    "move [X] to [Y]":    paper_style.C_DEATH_SCORE,
    "other":               '#D9D9D9',
}

C = "connect [X] to [Y]"
F = "focus on [X]"
O = "observation"
D = "disconnect [X]"
OI = "object interaction"

# Representative death episodes (approximate from visual inspection)
EPISODES = [
    {
        "task": "chem-mix",
        "cats": [O] + [C]*13 + [O, F, F],
        "fatal_idx": 15,
    },
    {
        "task": "freeze",
        "cats": [O] + [C]*13 + [D, F, F],
        "fatal_idx": 16,
    },
    {
        "task": "find-animal",
        "cats": [O] + [C]*14 + [F, F],
        "fatal_idx": 16,
    },
    {
        "task": "life-stages-1",
        "cats": [O] + [C]*14 + [F, F],
        "fatal_idx": 16,
    },
    {
        "task": "incl-plane",
        "cats": [O] + [C]*14 + [F, F],
        "fatal_idx": 16,
    },
    {
        "task": "lifespan-long",
        "cats": [O] + [C]*15 + [F, F],
        "fatal_idx": 17,
    },
]

max_steps = max(len(ep["cats"]) for ep in EPISODES)

fig, ax = plt.subplots(figsize=(3.4, 2.0))

for y, ep in enumerate(EPISODES):
    for x_pos, cat in enumerate(ep["cats"]):
        is_fatal = (x_pos == ep["fatal_idx"])
        rect = Rectangle(
            (x_pos, y - 0.35), 1, 0.7,
            facecolor=COLORS[cat],
            edgecolor="#333333" if is_fatal else "none",
            linewidth=1.0 if is_fatal else 0,
        )
        ax.add_patch(rect)
    ax.annotate("", xy=(ep["fatal_idx"] + 0.5, y + 0.5),
                xytext=(ep["fatal_idx"] + 0.5, y + 0.85),
                arrowprops=dict(arrowstyle="->", color="#333333", lw=0.7))

ax.set_xlim(-0.5, max_steps + 0.5)
ax.set_ylim(-0.6, len(EPISODES) + 0.1)
ax.set_yticks(range(len(EPISODES)))
ax.set_yticklabels([ep["task"] for ep in EPISODES], fontsize=6)
ax.set_xlabel("Step number")

legend_cats = ["focus on [X]", "connect [X] to [Y]", "disconnect [X]",
               "observation", "object interaction"]
legend_elements = [Patch(facecolor=COLORS[c], label=c) for c in legend_cats]
legend_elements.append(Patch(facecolor="white", edgecolor="#333333", linewidth=1.0,
                             label="fatal step"))
leg = ax.legend(handles=legend_elements, fontsize=4.5, loc="upper right", ncol=2,
                borderpad=0.3, handletextpad=0.3, columnspacing=0.6)
paper_style.style_legend(leg)

plt.tight_layout(pad=0.3)

out_dir = os.path.dirname(os.path.abspath(__file__))
for ext in ("pdf", "png"):
    path = os.path.join(out_dir, f"death_action_timeline.{ext}")
    fig.savefig(path, format=ext if ext == "png" else None,
                dpi=300, bbox_inches="tight", pad_inches=0.08)
    print(f"Saved: {path} ({os.path.getsize(path)} bytes)")
plt.close()
