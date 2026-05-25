"""Unified style for all paper figures — Nature/Science aesthetic.

Color semantics:
- BLUE (#2166AC): internal strategy, regular tasks
- RED (#B2182B): deterministic strategy, death-related, outliers
- GREEN (#1B9E77): analysis/framework components
- AMBER (#D95F02): secondary categories (death+score)
- GRAY (#969696): neutral/no-preference
- PURPLE (#7570B3): tertiary
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# Semantic colors
C_INTERNAL = '#2166AC'
C_DETERMINISTIC = '#B2182B'
C_NEUTRAL = '#969696'
C_DEATH_SOLE = '#B2182B'
C_DEATH_SCORE = '#D95F02'
C_NON_DEATH = '#2166AC'
C_GREEN = '#1B9E77'
C_PURPLE = '#7570B3'
C_ACCENT = '#E7298A'

# Regression / trend lines
C_REGRESSION = '#636363'

# Environment colors (for signal_reversal)
C_SCIWORLD = '#2166AC'
C_ALFWORLD = '#D95F02'
C_MATH = '#7570B3'

# Spine / tick color
_SPINE_COLOR = '#333333'


def apply_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7.5,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.10,
        # Spines: only left + bottom, thin, dark gray
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.5,
        'axes.edgecolor': _SPINE_COLOR,
        # Ticks: inward, matching spine width
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.color': _SPINE_COLOR,
        'ytick.color': _SPINE_COLOR,
        # Lines and markers
        'lines.linewidth': 1.4,
        'lines.markersize': 5,
        # PDF export
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        # Background — subtle warm tint for premium feel
        'axes.grid': False,
        'figure.facecolor': 'white',
        'axes.facecolor': '#FAFAFA',
        'axes.labelcolor': _SPINE_COLOR,
        'text.color': _SPINE_COLOR,
    })


def style_legend(leg):
    """Style a legend to Nature/Science aesthetic."""
    frame = leg.get_frame()
    frame.set_linewidth(0.3)
    frame.set_edgecolor('#D0D0D0')
    frame.set_facecolor('#FAFAFA')
    frame.set_alpha(0.92)


def add_y_grid(ax, alpha=0.25, linewidth=0.35):
    """Add subtle y-axis major grid lines."""
    ax.yaxis.grid(True, linestyle='--', alpha=alpha, linewidth=linewidth,
                  color='#BBBBBB')
    ax.set_axisbelow(True)


def annotation_box(pad=0.40):
    """Return bbox dict for annotation boxes with rounded corners."""
    return dict(
        boxstyle=f'round,pad={pad}',
        facecolor='#FAFAFA',
        edgecolor='#C8C8C8',
        linewidth=0.4,
        alpha=0.95,
    )


def marker_shadow(offset=(0.5, -0.5), alpha=0.12):
    """Return path_effects list for subtle marker drop shadow."""
    return [pe.withSimplePatchShadow(offset=offset,
            shadow_rgbFace='#888888', alpha=alpha)]


# Calibrated scatter/line plot constants (from PB benchmark)
CI_ALPHA = 0.10
REGRESSION_ALPHA = 0.60
REGRESSION_LW = 1.0
MARKER_EDGE_WIDTH = 0.8


def bar_edge_style():
    """Return kwargs for bar edge styling."""
    return dict(edgecolor='white', linewidth=0.5)
