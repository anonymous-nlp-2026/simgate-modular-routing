"""Generate pipeline overview figure for Section 3 (Method)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 8,
    'figure.dpi': 300,
})

fig, ax = plt.subplots(figsize=(3.25, 4.0))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis('off')

box_kw = dict(boxstyle='round,pad=0.3', facecolor='#F3F4F6', edgecolor='#4B5563', linewidth=0.8)
box_blue = dict(boxstyle='round,pad=0.3', facecolor='#DBEAFE', edgecolor='#2563EB', linewidth=0.8)
box_red = dict(boxstyle='round,pad=0.3', facecolor='#FEE2E2', edgecolor='#DC2626', linewidth=0.8)
box_green = dict(boxstyle='round,pad=0.3', facecolor='#D1FAE5', edgecolor='#059669', linewidth=0.8)
arrow_kw = dict(arrowstyle='->', color='#4B5563', lw=1.2,
                connectionstyle='arc3,rad=0')

# Task Instance
ax.text(5, 13.2, 'Task Instance $t$', ha='center', va='center', fontsize=9,
        fontweight='bold', bbox=box_kw)

# Fork arrow
ax.annotate('', xy=(3, 11.6), xytext=(5, 12.7),
            arrowprops=dict(arrowstyle='->', color='#4B5563', lw=1.0))
ax.annotate('', xy=(7, 11.6), xytext=(5, 12.7),
            arrowprops=dict(arrowstyle='->', color='#4B5563', lw=1.0))
ax.text(5, 12.35, 'fork', ha='center', va='center', fontsize=7, style='italic',
        color='#6B7280')

# Two strategy boxes
ax.text(3, 11.0, 'Pure Internal\nStrategy  $\\tau_{\\mathrm{int}}$', ha='center',
        va='center', fontsize=7.5, bbox=box_blue)
ax.text(7, 11.0, 'Pure Determ.\nStrategy  $\\tau_{\\mathrm{det}}$', ha='center',
        va='center', fontsize=7.5, bbox=box_red)

# Outputs
ax.text(3, 9.5, '$(s_{\\mathrm{int}},\\, d_{\\mathrm{int}})$', ha='center',
        va='center', fontsize=7.5, color='#2563EB')
ax.text(7, 9.5, '$(s_{\\mathrm{det}},\\, d_{\\mathrm{det}})$', ha='center',
        va='center', fontsize=7.5, color='#DC2626')

ax.annotate('', xy=(3, 9.9), xytext=(3, 10.4),
            arrowprops=dict(arrowstyle='->', color='#4B5563', lw=1.0))
ax.annotate('', xy=(7, 9.9), xytext=(7, 10.4),
            arrowprops=dict(arrowstyle='->', color='#4B5563', lw=1.0))

# Merge arrows
ax.annotate('', xy=(5, 8.6), xytext=(3, 9.1),
            arrowprops=dict(arrowstyle='->', color='#4B5563', lw=1.0))
ax.annotate('', xy=(5, 8.6), xytext=(7, 9.1),
            arrowprops=dict(arrowstyle='->', color='#4B5563', lw=1.0))

# Episode-Level Labeling
ax.text(5, 8.0, 'Episode-Level Labeling\n(Algorithm 1)', ha='center',
        va='center', fontsize=8, bbox=box_green)

# Arrow down
ax.annotate('', xy=(5, 6.7), xytext=(5, 7.3),
            arrowprops=dict(arrowstyle='->', color='#4B5563', lw=1.0))

# Label output
ax.text(5, 6.2, '$\\ell \\in \\{$det-pref, int-pref, no-pref$\\}$', ha='center',
        va='center', fontsize=7, bbox=box_kw)

# Arrow down
ax.annotate('', xy=(5, 5.0), xytext=(5, 5.6),
            arrowprops=dict(arrowstyle='->', color='#4B5563', lw=1.0))

# Death Decomposition
ax.text(5, 4.3, 'Death Decomposition\nAnalysis', ha='center',
        va='center', fontsize=8, bbox=box_green)

# Arrow down
ax.annotate('', xy=(5, 3.0), xytext=(5, 3.6),
            arrowprops=dict(arrowstyle='->', color='#4B5563', lw=1.0))

# DAF output
ax.text(5, 2.3, '$\\Delta_{\\mathrm{full}} \\approx \\delta P + \\Delta_{\\mathrm{nd}}$\n$\\longrightarrow$ DAF',
        ha='center', va='center', fontsize=7.5, bbox=box_kw)

plt.tight_layout(pad=0.2)

out_path = os.path.join(os.path.dirname(__file__), 'pipeline_overview.pdf')
fig.savefig(out_path, bbox_inches='tight', pad_inches=0.02)
plt.close()
print(f"Saved: {out_path} ({os.path.getsize(out_path)} bytes)")
