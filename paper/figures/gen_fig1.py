"""Generate Figure 1: DAF scatter plot for EMNLP short paper.

Per-task death rate differential vs performance gap (Δ_full).
Data from plan-012-full: 11 ScienceWorld task types, 245 episodes.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os

tasks = [
    ("id-life-stages-1",   20, 47.50, 55, 10),
    ("lifespan-long",       25, 44.00, 56, 16),
    ("lifespan-l-then-s",   20, 40.80, 45, 10),
    ("inclined-plane",      20, 36.00, 55, 20),
    ("freeze",              20, 30.50, 55, 25),
    ("meas-melt-point",     20, 30.05, 55, 25),
    ("id-life-stages-2",    20, 30.00, 50, 20),
    ("find-animal",         20, 26.65, 50, 25),
    ("grow-fruit",          20, 25.10, 45, 20),
    ("chem-mix",            20, 21.80, 45, 25),
    ("find-non-living-thing†", 40,  9.58,  0,  0),
]

names = [t[0] for t in tasks]
delta_full = np.array([t[2] for t in tasks])
death_diff = np.array([t[3] - t[4] for t in tasks])

is_special = np.array([t[0].startswith("find-non-living") for t in tasks])

slope, intercept, r, p, se = stats.linregress(death_diff, delta_full)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 8,
    'axes.labelsize': 9,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'figure.dpi': 300,
})

fig, ax = plt.subplots(figsize=(3.25, 2.6))

ax.scatter(death_diff[~is_special], delta_full[~is_special],
           c='#2563EB', s=30, zorder=3, edgecolors='white', linewidths=0.5)
ax.scatter(death_diff[is_special], delta_full[is_special],
           c='#DC2626', s=50, marker='D', zorder=3, edgecolors='white', linewidths=0.5)

x_fit = np.linspace(-2, 50, 100)
y_fit = slope * x_fit + intercept
ax.plot(x_fit, y_fit, '--', color='#6B7280', linewidth=1, zorder=2)

ax.annotate(f'$r = {r:.4f}$\n$p = 8.12 \\times 10^{{-7}}$',
            xy=(0.97, 0.05), xycoords='axes fraction',
            ha='right', va='bottom', fontsize=7,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#D1D5DB', alpha=0.9))

ax.annotate('find-non-living-thing†\n(zero deaths)',
            xy=(death_diff[is_special][0], delta_full[is_special][0]),
            xytext=(8, 18), textcoords='offset points',
            fontsize=6.5, color='#DC2626',
            arrowprops=dict(arrowstyle='->', color='#DC2626', lw=0.8))

ax.set_xlabel('Death rate differential (int. $-$ det., pp)')
ax.set_ylabel('Performance gap ($\\Delta_{\\mathrm{full}}$, score difference)')
ax.set_xlim(-3, 50)
ax.set_ylim(5, 52)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout(pad=0.3)

out_path = os.path.join(os.path.dirname(__file__), 'fig1_daf_scatter.pdf')
fig.savefig(out_path, bbox_inches='tight', pad_inches=0.02)
plt.close()
print(f"Saved: {out_path} ({os.path.getsize(out_path)} bytes)")
