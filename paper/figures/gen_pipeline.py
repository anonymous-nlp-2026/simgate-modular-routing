"""Generate pipeline overview figure for Section 3 (Method) — polished v3."""
import paper_style; paper_style.apply_style()
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import os

plt.rcParams.update({
    'font.size': 8,
    'mathtext.fontset': 'cm',
})

# --- Muted academic color palette with semantic binding ---
C = {
    'int_fill':    '#E3EDF7',
    'int_edge':    '#2166AC',
    'int_text':    '#1A4E8A',
    'det_fill':    '#FAEAEA',
    'det_edge':    '#B2182B',
    'det_text':    '#8B1A1A',
    'proc_fill':   '#EEF4F0',
    'proc_edge':   '#5C8A6E',
    'neutral_fill': '#F0F1F3',
    'neutral_edge': '#4A5260',
    'result_fill': '#F8F9FB',
    'result_edge': '#8B95A0',
    'arrow':       '#6B7580',
    'shadow':      '#C0C6CD',
    'muted_text':  '#6B7580',
}

fig, ax = plt.subplots(figsize=(3.25, 3.75))
ax.set_xlim(0, 10)
ax.set_ylim(1.4, 14.0)
ax.axis('off')


def _bstyle(fill, edge, pad=0.3, lw=0.8):
    return dict(boxstyle=f'round,pad={pad}', facecolor=fill,
                edgecolor=edge, linewidth=lw)


def draw_box(x, y, text, fill, edge, fs=8, fw='normal',
             tc='#1a1a1a', pad=0.3, lw=0.8, shadow=True, **kw):
    if shadow:
        ax.text(x + 0.04, y - 0.05, text, ha='center', va='center',
                fontsize=fs, fontweight=fw, color='none',
                bbox=dict(boxstyle=f'round,pad={pad}', facecolor=C['shadow'],
                          edgecolor='none', linewidth=0, alpha=0.18),
                zorder=1, **kw)
    ax.text(x, y, text, ha='center', va='center', fontsize=fs,
            fontweight=fw, color=tc, bbox=_bstyle(fill, edge, pad, lw),
            zorder=2, **kw)


def draw_arrow(x1, y1, x2, y2, rad=0):
    style = f'arc3,rad={rad}' if rad else 'arc3,rad=0'
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->,head_length=2.5,head_width=1.5',
        color=C['arrow'], lw=0.6, zorder=1,
        mutation_scale=1, shrinkA=0, shrinkB=0,
        connectionstyle=style,
    )
    ax.add_patch(a)


# ═══════════════════════════════════════════
# Layout: compact top → bottom
# ═══════════════════════════════════════════

# 1) Task Instance
draw_box(5, 13.2, 'Task Instance $t$',
         C['neutral_fill'], C['neutral_edge'], fs=9, fw='bold')

# Fork label + branching arrows
ax.text(5, 12.35, 'fork', ha='center', va='center', fontsize=6.5,
        style='italic', color=C['muted_text'], zorder=3)
draw_arrow(4.3, 12.65, 3.2, 11.55)
draw_arrow(5.7, 12.65, 6.8, 11.55)

# 2) Pure strategy boxes (semantic blue / red)
draw_box(3, 10.9, 'Pure Internal\nStrategy  $\\tau_{\\mathrm{int}}$',
         C['int_fill'], C['int_edge'], fs=7.5, tc=C['int_text'])

draw_box(7, 10.9, 'Pure Determ.\nStrategy  $\\tau_{\\mathrm{det}}$',
         C['det_fill'], C['det_edge'], fs=7.5, tc=C['det_text'])

# Arrows → score/death outputs
draw_arrow(3, 10.25, 3, 9.8)
draw_arrow(7, 10.25, 7, 9.8)

# 3) Score/death outputs (unboxed colored text)
ax.text(3, 9.45, '$(s_{\\mathrm{int}},\\, d_{\\mathrm{int}})$',
        ha='center', va='center', fontsize=7.5,
        color=C['int_edge'], zorder=3)
ax.text(7, 9.45, '$(s_{\\mathrm{det}},\\, d_{\\mathrm{det}})$',
        ha='center', va='center', fontsize=7.5,
        color=C['det_edge'], zorder=3)

# Merge arrows → labeling (slight curve to avoid visual collision)
draw_arrow(3.5, 9.05, 4.4, 8.35, rad=0.08)
draw_arrow(6.5, 9.05, 5.6, 8.35, rad=-0.08)

# 4) Episode-Level Labeling (processing step)
draw_box(5, 7.7, 'Episode-Level Labeling\n(Algorithm 1)',
         C['proc_fill'], C['proc_edge'], fs=8, fw='medium', lw=0.9)

# Arrow → label output
draw_arrow(5, 6.95, 5, 6.55)

# 5) Label output (result — lighter, no shadow)
draw_box(5, 6.05,
         '$\\ell \\in \\{$det-pref, int-pref, no-pref$\\}$',
         C['result_fill'], C['result_edge'],
         fs=6.8, pad=0.25, lw=0.5, shadow=False)

# Arrow → decomposition
draw_arrow(5, 5.45, 5, 5.05)

# 6) Death Decomposition (processing step)
draw_box(5, 4.35, 'Death Decomposition\nAnalysis',
         C['proc_fill'], C['proc_edge'], fs=8, fw='medium', lw=0.9)

# Arrow → DAF
draw_arrow(5, 3.6, 5, 3.2)

# 7) DAF output (result — cleaner layout)
draw_box(5, 2.6,
         '$\\Delta_{\\mathrm{full}} \\approx \\delta P + \\Delta_{\\mathrm{nd}}$\n'
         'DAF $= 1 - \\Delta_{\\mathrm{nd}} / \\Delta_{\\mathrm{full}}$',
         C['result_fill'], C['result_edge'],
         fs=7, pad=0.25, lw=0.5, shadow=False)

# ═══════════════════════════════════════════

plt.tight_layout(pad=0.1)

out_dir = os.path.dirname(os.path.abspath(__file__))
out_pdf = os.path.join(out_dir, 'pipeline_overview.pdf')
out_png = os.path.join(out_dir, 'pipeline_overview.png')

fig.savefig(out_pdf, bbox_inches='tight', pad_inches=0.02, dpi=300)
fig.savefig(out_png, bbox_inches='tight', pad_inches=0.02, dpi=300)
plt.close()

print(f"Saved: {out_pdf} ({os.path.getsize(out_pdf)} bytes)")
print(f"Saved: {out_png} ({os.path.getsize(out_png)} bytes)")
