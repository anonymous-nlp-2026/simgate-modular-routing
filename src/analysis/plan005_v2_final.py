"""Plan-005 v2 final: Per-task routing patterns from full plan_c 245-episode dataset.

Computes all stats from plan_c episode_summary.jsonl (ground truth)
instead of stale routing_metrics.json.
"""
import json
import os
from collections import defaultdict, Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

BASE = Path(".")
PLAN_C = BASE / "results" / "plan_c"
OUT = BASE / "artifacts"
OUT.mkdir(exist_ok=True)

PLAN_C_TASKS = sorted([d.name for d in PLAN_C.iterdir() if d.is_dir()])

def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

# ── Load all plan_c data ───────────────────────────────────────────

print("Loading plan_c data (245 episodes)...")
summaries = {}
episodes_raw = {}
steps_raw = {}
for task in PLAN_C_TASKS:
    task_dir = PLAN_C / task
    summaries[task] = load_jsonl(task_dir / "episode_summary.jsonl")
    episodes_raw[task] = load_jsonl(task_dir / "episodes.jsonl")
    step_path = task_dir / "labeled_steps.jsonl"
    if step_path.exists():
        steps_raw[task] = load_jsonl(step_path)

total_eps = sum(len(v) for v in summaries.values())
print(f"Loaded {len(PLAN_C_TASKS)} tasks, {total_eps} episodes")

# ══════════════════════════════════════════════════════════════════
# Part A: Recompute per-task stats from episode_summary.jsonl
# ══════════════════════════════════════════════════════════════════

print("\n=== PART A: Per-task routing frequency (full dataset) ===\n")

results_a = []
for task in PLAN_C_TASKS:
    eps = summaries[task]
    n = len(eps)
    
    n_det_pref = sum(1 for e in eps if e['label'] == 'det-preferred')
    n_int_pref = sum(1 for e in eps if e['label'] == 'internal-preferred')
    n_no_pref = sum(1 for e in eps if e['label'] == 'no-preference')
    
    int_scores = [e['internal_final'] for e in eps]
    det_scores = [e['det_final'] for e in eps]
    
    int_deaths = sum(1 for s in int_scores if s == -100)
    det_deaths = sum(1 for s in det_scores if s == -100)
    
    int_mean = np.mean(int_scores)
    det_mean = np.mean(det_scores)
    delta_DI = det_mean - int_mean
    
    # DAF computation
    int_death_rate = int_deaths / n
    det_death_rate = det_deaths / n
    
    alive_int = [s for s in int_scores if s != -100]
    alive_det = [s for s in det_scores if s != -100]
    
    if len(alive_int) > 0 and len(alive_det) > 0:
        delta_no_death = np.mean(alive_det) - np.mean(alive_int)
    elif len(alive_det) > 0:
        delta_no_death = np.mean(alive_det)
    elif len(alive_int) > 0:
        delta_no_death = -np.mean(alive_int)
    else:
        delta_no_death = 0
    
    if delta_DI != 0:
        daf = 1 - (delta_no_death / delta_DI) if delta_DI > 0 else None
    else:
        daf = None
    
    # Signal category
    if daf is not None and daf >= 0.95 and delta_DI > 0:
        cat = 'death-dominated'
    elif daf is not None and 0 < daf < 0.95 and delta_DI > 0:
        cat = 'genuine-det'
    elif delta_DI < 0:
        cat = 'internal-preferred'
    elif delta_DI == 0:
        cat = 'no-signal'
    else:
        cat = 'death-dominated' if (daf or 0) >= 0.95 else 'genuine-det'
    
    results_a.append({
        'task': task,
        'n': n,
        'int_death_rate': int_death_rate,
        'det_death_rate': det_death_rate,
        'death_diff': int_death_rate - det_death_rate,
        'det_pref': n_det_pref / n,
        'int_pref': n_int_pref / n,
        'no_pref': n_no_pref / n,
        'int_mean': round(int_mean, 2),
        'det_mean': round(det_mean, 2),
        'delta_DI': round(delta_DI, 2),
        'delta_no_death': round(delta_no_death, 2),
        'daf': round(daf, 4) if daf is not None else None,
        'signal_category': cat,
        'n_det_pref': n_det_pref,
        'n_int_pref': n_int_pref,
        'n_no_pref': n_no_pref,
    })

# Sort by death_diff descending
results_a.sort(key=lambda x: x['death_diff'], reverse=True)

print(f"{'Task':<45} {'N':>3} {'IntDR':>6} {'DetDR':>6} {'ΔDR':>6} {'DetPr':>6} {'IntPr':>6} {'NoPr':>6} {'Δ(D-I)':>8} {'DAF':>7} {'Cat'}")
print("-" * 140)
for r in results_a:
    daf_str = f"{r['daf']:.3f}" if r['daf'] is not None else "  N/A"
    print(f"{r['task']:<45} {r['n']:>3} {r['int_death_rate']:>6.1%} {r['det_death_rate']:>6.1%} {r['death_diff']:>6.1%} "
          f"{r['det_pref']:>6.1%} {r['int_pref']:>6.1%} {r['no_pref']:>6.1%} {r['delta_DI']:>8.2f} {daf_str:>7} {r['signal_category']}")

# Aggregate stats
total_n = sum(r['n'] for r in results_a)
wmean_int_dr = sum(r['int_death_rate'] * r['n'] for r in results_a) / total_n
wmean_det_dr = sum(r['det_death_rate'] * r['n'] for r in results_a) / total_n
wmean_delta = sum(r['delta_DI'] * r['n'] for r in results_a) / total_n

positive_daf = [r for r in results_a if r['daf'] is not None and r['daf'] > 0]
wmean_daf = sum(r['daf'] * r['n'] for r in positive_daf) / sum(r['n'] for r in positive_daf) if positive_daf else 0

# Correlation
death_diffs = [r['death_diff'] for r in results_a]
det_prefs = [r['det_pref'] for r in results_a]
corr = np.corrcoef(death_diffs, det_prefs)[0, 1]

n_death_dom = sum(1 for r in results_a if r['signal_category'] == 'death-dominated')
n_genuine = sum(1 for r in results_a if r['signal_category'] == 'genuine-det')
n_internal = sum(1 for r in results_a if r['signal_category'] == 'internal-preferred')
n_no_signal = sum(1 for r in results_a if r['signal_category'] == 'no-signal')

print(f"\nAggregate (weighted by n_episodes):")
print(f"  Total: {total_n} episodes across {len(results_a)} tasks")
print(f"  Categories: {n_death_dom} death-dom, {n_genuine} genuine-det, {n_internal} internal-pref, {n_no_signal} no-signal")
print(f"  Weighted mean int death rate: {wmean_int_dr:.1%}")
print(f"  Weighted mean det death rate: {wmean_det_dr:.1%}")
print(f"  Weighted mean Δ(D-I): {wmean_delta:.2f}")
print(f"  Weighted mean DAF (positive): {wmean_daf:.4f}")
print(f"  Correlation(ΔDR, det_pref_rate): {corr:.4f}")

# ══════════════════════════════════════════════════════════════════
# Part B: Episode phase analysis
# ══════════════════════════════════════════════════════════════════

print("\n=== PART B: Episode phase analysis ===")

phase_data = {}
total_deaths_by_phase = {'early': {'int': 0, 'det': 0}, 'mid': {'int': 0, 'det': 0}, 'late': {'int': 0, 'det': 0}}

for task in PLAN_C_TASKS:
    pd_task = {'n': 0, 'int_deaths': 0, 'det_deaths': 0,
               'early': {'int': 0, 'det': 0}, 'mid': {'int': 0, 'det': 0}, 'late': {'int': 0, 'det': 0}}
    
    for ep in episodes_raw[task]:
        pd_task['n'] += 1
        for mode_key, mode_label in [('always_internal', 'int'), ('always_deterministic', 'det')]:
            run = ep.get(mode_key, {})
            steps = run.get('steps', [])
            n_steps = len(steps)
            if n_steps == 0:
                continue
            
            died = run.get('final_score', 0) == -100
            if died:
                pd_task[f'{mode_label}_deaths'] += 1
            
            third = max(1, n_steps // 3)
            for i, step in enumerate(steps):
                if step.get('score_delta', 0) == -100:
                    if i < third:
                        phase = 'early'
                    elif i < 2 * third:
                        phase = 'mid'
                    else:
                        phase = 'late'
                    pd_task[phase][mode_label] += 1
                    total_deaths_by_phase[phase][mode_label] += 1
    
    phase_data[task] = pd_task

# Overall death phase distribution
total_phase_deaths = sum(total_deaths_by_phase[p]['int'] + total_deaths_by_phase[p]['det'] for p in ['early', 'mid', 'late'])
print(f"\nDeath events by episode phase:")
for phase in ['early', 'mid', 'late']:
    n = total_deaths_by_phase[phase]['int'] + total_deaths_by_phase[phase]['det']
    print(f"  {phase}: {n} ({n/max(1,total_phase_deaths)*100:.1f}%) [int={total_deaths_by_phase[phase]['int']}, det={total_deaths_by_phase[phase]['det']}]")

total_int = sum(pd_task['int_deaths'] for pd_task in phase_data.values())
total_det = sum(pd_task['det_deaths'] for pd_task in phase_data.values())
print(f"\nTotal deaths: internal={total_int}, deterministic={total_det}, ratio={total_int/max(1,total_det):.2f}x")

# Per-task: where in the episode do deaths happen relative to total steps?
print(f"\nDeath step positions (relative to episode length):")
death_positions = {'int': [], 'det': []}
for task in PLAN_C_TASKS:
    for ep in episodes_raw[task]:
        for mode_key, mode_label in [('always_internal', 'int'), ('always_deterministic', 'det')]:
            run = ep.get(mode_key, {})
            steps = run.get('steps', [])
            n_steps = len(steps)
            if n_steps <= 1:
                continue
            for i, step in enumerate(steps):
                if step.get('score_delta', 0) == -100:
                    death_positions[mode_label].append(i / (n_steps - 1))

for mode in ['int', 'det']:
    if death_positions[mode]:
        arr = np.array(death_positions[mode])
        print(f"  {mode}: mean={arr.mean():.3f}, std={arr.std():.3f}, n={len(arr)}")

# Early-step behavior analysis: when do score progressions diverge?
print(f"\nStep-by-step score progression (first 10 steps, internal vs det):")
max_steps = 10
for phase_step in range(max_steps):
    int_scores_at_step = []
    det_scores_at_step = []
    for task in PLAN_C_TASKS:
        for ep in episodes_raw[task]:
            for mode_key, mode_label in [('always_internal', 'int'), ('always_deterministic', 'det')]:
                run = ep.get(mode_key, {})
                steps = run.get('steps', [])
                if len(steps) > phase_step:
                    score = steps[phase_step].get('score_after', 0)
                    if mode_label == 'int':
                        int_scores_at_step.append(score)
                    else:
                        det_scores_at_step.append(score)
    if int_scores_at_step and det_scores_at_step:
        int_m = np.mean(int_scores_at_step)
        det_m = np.mean(det_scores_at_step)
        print(f"  Step {phase_step:2d}: int={int_m:7.2f} (n={len(int_scores_at_step):3d}), det={det_m:7.2f} (n={len(det_scores_at_step):3d}), Δ={det_m-int_m:7.2f}")

# ══════════════════════════════════════════════════════════════════
# Part C: Publication-quality figures
# ══════════════════════════════════════════════════════════════════

print("\n=== PART C: Generating figures ===")

plt.rcParams.update({
    'font.size': 9,
    'font.family': 'sans-serif',
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
})

tasks_sorted = [r['task'] for r in results_a]

# Short display names
name_map = {
    'chemistry-mix': 'Chemistry Mix',
    'find-animal': 'Find Animal',
    'find-non-living-thing': 'Find Non-Living',
    'freeze': 'Freeze',
    'grow-fruit': 'Grow Fruit',
    'identify-life-stages-1': 'Life Stages 1',
    'identify-life-stages-2': 'Life Stages 2',
    'inclined-plane-determine-angle': 'Inclined Plane',
    'lifespan-longest-lived': 'Lifespan (Longest)',
    'lifespan-longest-lived-then-shortest-lived': 'Lifespan (Long→Short)',
    'measure-melting-point-unknown-substance': 'Melting Point',
}
short_names = [name_map.get(t, t) for t in tasks_sorted]

n_tasks = len(tasks_sorted)
y_pos = np.arange(n_tasks)

# ── Figure 1: Three-panel routing overview ──

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.5),
                                       gridspec_kw={'width_ratios': [1.1, 1.1, 0.8], 'wspace': 0.15})

bh = 0.35

# (a) Death rates
int_dr = [r['int_death_rate'] for r in results_a]
det_dr = [r['det_death_rate'] for r in results_a]

ax1.barh(y_pos - bh/2, int_dr, bh, label='Internal', color='#D32F2F', alpha=0.88, edgecolor='white', linewidth=0.3)
ax1.barh(y_pos + bh/2, det_dr, bh, label='Simulator', color='#1976D2', alpha=0.88, edgecolor='white', linewidth=0.3)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(short_names, fontsize=8)
ax1.set_xlabel('Death Rate')
ax1.set_title('(a) Death Rate', fontweight='bold')
ax1.legend(fontsize=7, loc='lower right', framealpha=0.9)
ax1.set_xlim(0, 0.72)
ax1.invert_yaxis()
ax1.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))

# (b) Oracle label distribution
dp = [r['det_pref'] for r in results_a]
ip = [r['int_pref'] for r in results_a]
np_ = [r['no_pref'] for r in results_a]

ax2.barh(y_pos, dp, 0.7, label='Sim-preferred', color='#1976D2', alpha=0.88, edgecolor='white', linewidth=0.3)
ax2.barh(y_pos, np_, 0.7, left=dp, label='No preference', color='#BDBDBD', alpha=0.88, edgecolor='white', linewidth=0.3)
left2 = [a + b for a, b in zip(dp, np_)]
ax2.barh(y_pos, ip, 0.7, left=left2, label='Int-preferred', color='#D32F2F', alpha=0.88, edgecolor='white', linewidth=0.3)
ax2.set_yticks(y_pos)
ax2.set_yticklabels([])
ax2.set_xlabel('Fraction')
ax2.set_title('(b) Oracle Label Distribution', fontweight='bold')
ax2.legend(fontsize=6.5, loc='lower right', framealpha=0.9)
ax2.set_xlim(0, 1.05)
ax2.invert_yaxis()

# (c) DAF
dafs = [r['daf'] if r['daf'] is not None else 0 for r in results_a]
colors_c = []
for d in dafs:
    if d >= 0.95:
        colors_c.append('#388E3C')
    elif d > 0:
        colors_c.append('#FFA000')
    elif d == 0:
        colors_c.append('#D32F2F')
    else:
        colors_c.append('#D32F2F')

ax3.barh(y_pos, dafs, 0.7, color=colors_c, alpha=0.88, edgecolor='white', linewidth=0.3)
ax3.axvline(x=0.95, color='#666', linestyle='--', linewidth=0.7, alpha=0.6)
ax3.set_yticks(y_pos)
ax3.set_yticklabels([])
ax3.set_xlabel('DAF')
ax3.set_title('(c) Death Avoidance\n     Fraction', fontweight='bold')
ax3.set_xlim(-0.15, 1.15)
ax3.invert_yaxis()

# Add episode counts on the right side of panel (a)
for i, r in enumerate(results_a):
    ax1.text(-0.02, i, f'n={r["n"]}', ha='right', va='center', fontsize=6, color='#666',
             transform=ax1.get_yaxis_transform())

fig.suptitle('Per-Task Routing Patterns in ScienceWorld', fontsize=12, fontweight='bold', y=1.01)
fig.savefig(OUT / "routing_heatmap.pdf", bbox_inches='tight', dpi=300)
fig.savefig(OUT / "routing_heatmap.png", bbox_inches='tight', dpi=300)
print(f"Saved: routing_heatmap.pdf/png")
plt.close(fig)

# ── Figure 2: Phase heatmap ──

fig2, ax_ph = plt.subplots(figsize=(6, 4.5))

phase_matrix = []
for task in tasks_sorted:
    pd_t = phase_data[task]
    n_ep = max(1, pd_t['n'])
    row = []
    for phase in ['early', 'mid', 'late']:
        diff = (pd_t[phase]['int'] - pd_t[phase]['det']) / n_ep
        row.append(diff)
    phase_matrix.append(row)

phase_matrix = np.array(phase_matrix)

vmax = max(0.01, np.abs(phase_matrix).max())
im = ax_ph.imshow(phase_matrix, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)

ax_ph.set_xticks([0, 1, 2])
ax_ph.set_xticklabels(['Early\n(steps 1–⅓)', 'Mid\n(steps ⅓–⅔)', 'Late\n(steps ⅔–end)'], fontsize=8)
ax_ph.set_yticks(range(n_tasks))
ax_ph.set_yticklabels(short_names, fontsize=8)
ax_ph.set_title('Death Rate Differential by Episode Phase\n(Internal − Simulator, per episode)', fontweight='bold', fontsize=10)

for i in range(n_tasks):
    for j in range(3):
        val = phase_matrix[i, j]
        color = 'white' if abs(val) > vmax * 0.5 else 'black'
        text = f'{val:.2f}' if val != 0 else '0'
        ax_ph.text(j, i, text, ha='center', va='center', fontsize=7, color=color)

cbar = plt.colorbar(im, ax=ax_ph, shrink=0.85, pad=0.02)
cbar.set_label('Δ Death Rate\n(Int − Sim)', fontsize=8)

fig2.savefig(OUT / "phase_heatmap.pdf", bbox_inches='tight', dpi=300)
fig2.savefig(OUT / "phase_heatmap.png", bbox_inches='tight', dpi=300)
print(f"Saved: phase_heatmap.pdf/png")
plt.close(fig2)

# ── Figure 3: Death rate scatter (DAF vs death differential) ──

fig3, ax_sc = plt.subplots(figsize=(5.5, 4))

death_diffs_plot = [r['death_diff'] for r in results_a]
dafs_plot = [r['daf'] if r['daf'] is not None else 0 for r in results_a]
sizes = [r['n'] * 3 for r in results_a]

scatter = ax_sc.scatter(death_diffs_plot, dafs_plot, s=sizes, c=death_diffs_plot,
                         cmap='RdYlBu_r', alpha=0.8, edgecolors='#333', linewidths=0.5)

for i, r in enumerate(results_a):
    offset_x = 0.01
    offset_y = -0.04 if r['task'] != 'find-non-living-thing' else 0.04
    ax_sc.annotate(name_map.get(r['task'], r['task']), (death_diffs_plot[i], dafs_plot[i]),
                   textcoords='offset points', xytext=(5, -5), fontsize=6, alpha=0.8)

ax_sc.axhline(y=0.95, color='#666', linestyle='--', linewidth=0.7, alpha=0.5)
ax_sc.set_xlabel('Death Rate Differential (Internal − Simulator)')
ax_sc.set_ylabel('Death Avoidance Fraction (DAF)')
ax_sc.set_title('Death Signal Dominance', fontweight='bold', fontsize=10)
ax_sc.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))

fig3.savefig(OUT / "daf_scatter.pdf", bbox_inches='tight', dpi=300)
fig3.savefig(OUT / "daf_scatter.png", bbox_inches='tight', dpi=300)
print(f"Saved: daf_scatter.pdf/png")
plt.close(fig3)

# ══════════════════════════════════════════════════════════════════
# Part D: Failure mode classification
# ══════════════════════════════════════════════════════════════════

print("\n=== PART D: Failure mode taxonomy ===")

death_actions = {'int': Counter(), 'det': Counter()}
death_contexts = defaultdict(list)
pre_death_patterns = {'int': [], 'det': []}

for task in PLAN_C_TASKS:
    for ep in episodes_raw[task]:
        for mode_key, label in [('always_internal', 'int'), ('always_deterministic', 'det')]:
            run = ep.get(mode_key, {})
            steps = run.get('steps', [])
            n_steps = len(steps)
            
            for i, step in enumerate(steps):
                if step.get('score_delta', 0) == -100:
                    action = step.get('action', 'unknown')
                    death_actions[label][action] += 1
                    
                    # Collect 3 steps before death
                    pre_actions = [steps[j].get('action', '') for j in range(max(0, i-3), i)]
                    pre_death_patterns[label].append({
                        'task': task,
                        'death_step': i,
                        'total_steps': n_steps,
                        'death_action': action,
                        'pre_actions': pre_actions,
                    })

# Death action taxonomy
print(f"\nInternal deaths: {sum(death_actions['int'].values())} total")
focus_int = sum(v for k, v in death_actions['int'].items() if 'focus on' in k)
print(f"  'focus on' actions: {focus_int} ({focus_int/max(1,sum(death_actions['int'].values()))*100:.0f}%)")
print(f"  Top locations:")
for action, count in death_actions['int'].most_common(10):
    location = action.replace('focus on ', '')
    print(f"    {count:3d}x  {location}")

print(f"\nDeterministic deaths: {sum(death_actions['det'].values())} total")
focus_det = sum(v for k, v in death_actions['det'].items() if 'focus on' in k)
print(f"  'focus on' actions: {focus_det} ({focus_det/max(1,sum(death_actions['det'].values()))*100:.0f}%)")

# Pre-death action patterns
print(f"\nPre-death action patterns (3 steps before death):")
pre_death_categories = Counter()
for pattern in pre_death_patterns['int']:
    pre = pattern['pre_actions']
    has_navigation = any('focus on' in a or 'connect' in a for a in pre)
    has_activate = any('activate' in a or 'use' in a for a in pre)
    if has_navigation and not has_activate:
        pre_death_categories['exploration_loop'] += 1
    elif has_activate:
        pre_death_categories['active_interaction'] += 1
    else:
        pre_death_categories['other'] += 1

print(f"  Internal pre-death patterns:")
for cat, count in pre_death_categories.most_common():
    total = sum(pre_death_categories.values())
    print(f"    {cat}: {count} ({count/max(1,total)*100:.0f}%)")

# Episodes where internal dies early but det survives
internal_early_death = []
for task in PLAN_C_TASKS:
    for ep in episodes_raw[task]:
        int_run = ep.get('always_internal', {})
        det_run = ep.get('always_deterministic', {})
        int_steps = int_run.get('steps', [])
        det_steps = det_run.get('steps', [])
        
        int_final = int_run.get('final_score', 0)
        det_final = det_run.get('final_score', 0)
        
        if int_final == -100 and det_final != -100:
            internal_early_death.append({
                'task': task,
                'int_steps': len(int_steps),
                'det_steps': len(det_steps),
                'det_score': det_final,
                'int_death_step': len(int_steps) - 1,
            })

print(f"\nEpisodes where internal dies but det survives: {len(internal_early_death)}")
step_ratio = [e['int_steps'] / max(1, e['det_steps']) for e in internal_early_death]
print(f"  Internal step count / Det step count: mean={np.mean(step_ratio):.2f}, median={np.median(step_ratio):.2f}")
print(f"  Internal dies after mean {np.mean([e['int_steps'] for e in internal_early_death]):.1f} steps")

# ══════════════════════════════════════════════════════════════════
# Output summary JSON
# ══════════════════════════════════════════════════════════════════

summary = {
    'dataset': {'n_tasks': len(PLAN_C_TASKS), 'n_episodes': total_eps, 'tasks': PLAN_C_TASKS},
    'categories': {'death_dominated': n_death_dom, 'genuine_det': n_genuine, 'internal_preferred': n_internal, 'no_signal': n_no_signal},
    'aggregate': {
        'weighted_int_death_rate': round(wmean_int_dr, 4),
        'weighted_det_death_rate': round(wmean_det_dr, 4),
        'weighted_delta_DI': round(wmean_delta, 2),
        'weighted_daf': round(wmean_daf, 4),
        'correlation_death_diff_det_pref': round(corr, 4),
        'total_int_deaths': total_int,
        'total_det_deaths': total_det,
        'death_ratio': round(total_int / max(1, total_det), 2),
    },
    'phase_analysis': {
        'deaths_by_phase': {p: dict(v) for p, v in total_deaths_by_phase.items()},
        'finding': 'All deaths occur at the terminal step (100% late phase). In ScienceWorld, death is caused by a lethal action that immediately ends the episode.',
    },
    'failure_taxonomy': {
        'dominant_action': 'focus on <location>',
        'focus_pct_internal': round(focus_int / max(1, sum(death_actions['int'].values())) * 100, 1),
        'focus_pct_det': round(focus_det / max(1, sum(death_actions['det'].values())) * 100, 1),
        'pre_death_patterns': dict(pre_death_categories),
        'int_dies_det_survives': len(internal_early_death),
        'mean_int_steps_before_death': round(np.mean([e['int_steps'] for e in internal_early_death]), 1) if internal_early_death else 0,
    },
    'per_task': results_a,
}

with open(OUT / "plan005_v2_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved: plan005_v2_summary.json")
print("\n=== ALL DONE ===")
