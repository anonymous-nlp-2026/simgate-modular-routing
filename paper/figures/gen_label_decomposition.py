"""Generate label_decomposition.pdf: episode-level label decomposition bar.

Fits ACL single-column width (3.25 in).
"""
import paper_style; paper_style.apply_style()
import matplotlib.pyplot as plt
import numpy as np
import os

# Data from plan-012-full (245 episodes)
categories = [
    ("No preference",              150, paper_style.C_NEUTRAL),
    ("Death-asym.:\nsole signal",   54, paper_style.C_DEATH_SOLE),
    ("Death-asym.:\ndeath + score", 21, paper_style.C_DEATH_SCORE),
    ("Non-death\nscore diff.",      20, paper_style.C_NON_DEATH),
]

names  = [c[0] for c in categories]
counts = np.array([c[1] for c in categories])
colors = [c[2] for c in categories]
total  = counts.sum()  # 245

fig, ax = plt.subplots(figsize=(3.25, 1.8))

lefts = np.zeros(1)
bars = []
for i, (name, count, color) in enumerate(categories):
    b = ax.barh(0, count, left=lefts, height=0.55, color=color,
                **paper_style.bar_edge_style())
    bars.append(b)
    cx = lefts[0] + count / 2
    fontcolor = 'white' if i > 0 else '#333333'
    fs = 9 if count >= 30 else 7.5
    ax.text(cx, 0, str(count), ha='center', va='center',
            fontsize=fs, fontweight='bold', color=fontcolor)
    lefts += count

ax.set_xlim(0, total)
ax.set_yticks([])
ax.set_xlabel(f'Episodes ($N = {total}$)', fontsize=8)
ax.spines['left'].set_visible(False)

# Legend below the bar
legend_handles = [plt.Rectangle((0,0),1,1, facecolor=c[2], **paper_style.bar_edge_style())
                  for c in categories]
legend_labels = [f"{c[0].replace(chr(10), ' ')} ({c[1]})" for c in categories]
leg = ax.legend(legend_handles, legend_labels, loc='upper center',
                bbox_to_anchor=(0.5, -0.75), ncol=2, fontsize=6.5,
                frameon=False, handlelength=1.2, handletextpad=0.4,
                columnspacing=1.0)

# Bracket over last 3 segments: 75/95 non-tie = 78.9%
non_tie_total = 54 + 21 + 20  # 95
death_dom = 54 + 21  # 75
bracket_left = 150
bracket_right = total
bracket_y = 0.38
tip = 0.04

ax.plot([bracket_left, bracket_left, bracket_right, bracket_right],
        [bracket_y - tip, bracket_y, bracket_y, bracket_y - tip],
        color='#333333', linewidth=0.7, clip_on=False)
ax.text((bracket_left + bracket_right) / 2, bracket_y + 0.06,
        f'{death_dom}/{non_tie_total} non-tie = 78.9%',
        ha='center', va='bottom', fontsize=6.5, style='italic')

ax.set_ylim(-0.4, 0.65)

plt.tight_layout(pad=0.2)

out_dir = os.path.dirname(os.path.abspath(__file__))
for ext in ['pdf', 'png']:
    path = os.path.join(out_dir, f'label_decomposition.{ext}')
    fig.savefig(path, bbox_inches='tight', pad_inches=0.08, dpi=300)
plt.close()
print(f"Saved: {os.path.join(out_dir, 'label_decomposition.pdf')} ({os.path.getsize(os.path.join(out_dir, 'label_decomposition.pdf'))} bytes)")
