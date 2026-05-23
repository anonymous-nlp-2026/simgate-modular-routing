"""Generate signal reversal figure: label distribution across three environments.

Three-panel grouped bar chart showing:
- ScienceWorld (N=245): strong det preference from death avoidance
- ALFWorld (N=135): moderate det preference from info gap (different mechanism)
- MATH (N=795): balanced labels, no routing signal (no catastrophic failure)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'figure.dpi': 300,
})

envs = [
    'ScienceWorld\n(N=245)',
    'ALFWorld\n(N=135)',
    'MATH\n(N=795)',
]
det_pref = [35.1, 25.9, 7.3]
int_pref = [3.7, 1.5, 8.3]
no_pref = [61.2, 72.6, 84.4]

x = np.arange(len(envs))
width = 0.22

fig, ax = plt.subplots(figsize=(5.5, 3.2))

bars1 = ax.bar(x - width, det_pref, width, label='Det-preferred',
               color='#c0392b', edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x, int_pref, width, label='Int-preferred',
               color='#27ae60', edgecolor='white', linewidth=0.5)
bars3 = ax.bar(x + width, no_pref, width, label='No-preference',
               color='#bdc3c7', edgecolor='white', linewidth=0.5, alpha=0.7)

ax.set_ylabel('Proportion (%)')
ax.set_xticks(x)
ax.set_xticklabels(envs)
ax.legend(loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.12))
ax.set_ylim(0, 95)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}%',
                    xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=7)

ax.axvline(x=1.5, color='#95a5a6', linestyle='--', linewidth=0.8, alpha=0.6)

ax.annotate('Catastrophic failure\n(routing signal)',
            xy=(0.5, 82), xycoords=('data', 'data'),
            fontsize=7.5, ha='center', va='bottom', color='#2c3e50', style='italic')
ax.annotate('No catastrophic failure\n(no signal)',
            xy=(2.0, 82), xycoords=('data', 'data'),
            fontsize=7.5, ha='center', va='bottom', color='#2c3e50', style='italic')

out_path = os.path.join(os.path.dirname(__file__), 'signal_reversal.pdf')
plt.tight_layout()
plt.savefig(out_path, bbox_inches='tight', dpi=300)
print(f"Saved: {out_path} ({os.path.getsize(out_path)} bytes)")
