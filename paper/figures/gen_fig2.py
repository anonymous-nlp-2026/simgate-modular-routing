"""Generate Figure 2: Dose-response curves for Retry and Reflexion interventions.

Data: per_task_dose_response_data.json (10 death-prone tasks, N=1..5).
72B horizontal line updated to 30.6% (78/255, 11-task aggregate from backbone-scaling-70b).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

# Per-task death rates (%) — 10 death-prone tasks
RETRY = {
    "chemistry-mix":      [55, 55, 50, 50, 50],
    "find-animal":        [50, 55, 50, 50, 55],
    "freeze":             [55, 55, 55, 50, 50],
    "grow-fruit":         [45, 50, 45, 40, 50],
    "id-life-stages-1":   [35, 30, 25, 30, 25],
    "id-life-stages-2":   [25, 25, 25, 25, 25],
    "inclined-plane":     [55, 55, 45, 45, 45],
    "lifespan-long":      [50, 45, 45, 40, 45],
    "lifespan-l-then-s":  [55, 45, 45, 45, 35],
    "meas-melt-point":    [55, 55, 50, 55, 55],
}
REFLEXION = {
    "chemistry-mix":      [55, 45, 25, 35, 20],
    "find-animal":        [50, 40, 45, 30, 45],
    "freeze":             [55, 50, 50, 50, 60],
    "grow-fruit":         [50, 35, 30, 25, 25],
    "id-life-stages-1":   [35, 30, 35, 35, 35],
    "id-life-stages-2":   [25, 25, 25, 25, 30],
    "inclined-plane":     [50, 40, 20, 30, 35],
    "lifespan-long":      [50, 40, 45, 40, 40],
    "lifespan-l-then-s":  [55, 45, 30, 45, 35],
    "meas-melt-point":    [50, 50, 35, 50, 25],
}

NS = [1, 2, 3, 4, 5]

def compute_stats(data_dict):
    means, ci_lo, ci_hi = [], [], []
    for i in range(5):
        vals = np.array([v[i] for v in data_dict.values()])
        m = vals.mean()
        se = vals.std(ddof=1) / np.sqrt(len(vals))
        t_crit = 2.262  # t(0.025, df=9)
        means.append(m)
        ci_lo.append(m - t_crit * se)
        ci_hi.append(m + t_crit * se)
    return means, ci_lo, ci_hi

retry_m, retry_lo, retry_hi = compute_stats(RETRY)
reflex_m, reflex_lo, reflex_hi = compute_stats(REFLEXION)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'lines.linewidth': 1.8,
    'lines.markersize': 7,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

fig, ax = plt.subplots(figsize=(3.5, 2.8))

ax.fill_between(NS, retry_lo, retry_hi, color='#2166AC', alpha=0.12, zorder=1)
ax.plot(NS, retry_m, color='#2166AC', marker='o', label='Retry', zorder=3)

ax.fill_between(NS, reflex_lo, reflex_hi, color='#B2182B', alpha=0.12, zorder=1)
ax.plot(NS, reflex_m, color='#B2182B', marker='s', label='Reflexion', zorder=3)

ax.axhline(y=30.6, color='#666666', linestyle='--', linewidth=1.2, zorder=2)
ax.text(5.05, 30.6, '72B (30.6%)', va='center', ha='left', fontsize=7.5, color='#666666')

ax.axhline(y=16.3, color='#888888', linestyle=':', linewidth=1.0, zorder=2)
ax.text(5.05, 16.3, 'Deterministic (16.3%)', va='center', ha='left', fontsize=7.5, color='#888888')

ax.set_xlabel('Number of Trials (N)')
ax.set_ylabel('Death Rate (%)')
ax.set_xlim(0.7, 5.3)
ax.set_ylim(10, 60)
ax.xaxis.set_major_locator(ticker.FixedLocator([1, 2, 3, 4, 5]))
ax.yaxis.set_major_locator(ticker.MultipleLocator(10))

ax.grid(True, linestyle='--', linewidth=0.4, color='#CCCCCC', alpha=0.7, zorder=0)

ax.legend(loc='upper right', frameon=True, edgecolor='#CCCCCC',
          fancybox=False, framealpha=0.9)

fig.tight_layout(pad=0.4)
out_path = os.path.join(os.path.dirname(__file__), 'dose_response.pdf')
fig.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {out_path} ({os.path.getsize(out_path)} bytes)")
