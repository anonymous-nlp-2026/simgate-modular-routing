"""Plan-005 v2: Per-task routing pattern analysis for paper Figure 1.

Generates:
1. Per-task routing frequency table (Part A)
2. Episode phase analysis (Part B)  
3. Routing heatmap figure (Part C)
4. Failure mode classification (Part D)
"""
import json
import os
import sys
from collections import defaultdict, Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

BASE = Path(".")
PLAN_C = BASE / "results" / "plan_c"
PLAN003 = BASE / "results" / "plan_003_eval"
PLAN005 = BASE / "results" / "plan005"
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

# ── Load data ──────────────────────────────────────────────────────

print("Loading data...")
routing_metrics = json.loads((PLAN005 / "routing_metrics.json").read_text())
episode_details = load_jsonl(PLAN003 / "episode_details.jsonl")
failure_taxonomy = json.loads((PLAN005 / "failure_taxonomy.json").read_text())

# Load plan_c step-level and episode-level data
plan_c_episodes = {}
plan_c_summaries = {}
plan_c_steps = {}
for task in PLAN_C_TASKS:
    task_dir = PLAN_C / task
    ep_path = task_dir / "episodes.jsonl"
    sum_path = task_dir / "episode_summary.jsonl"
    step_path = task_dir / "labeled_steps.jsonl"
    if ep_path.exists():
        plan_c_episodes[task] = load_jsonl(ep_path)
    if sum_path.exists():
        plan_c_summaries[task] = load_jsonl(sum_path)
    if step_path.exists():
        plan_c_steps[task] = load_jsonl(step_path)

print(f"Loaded: {len(routing_metrics)} tasks in routing_metrics, "
      f"{len(episode_details)} episode_details rows, "
      f"{len(plan_c_episodes)} plan_c tasks with episodes")

# Filter routing_metrics to plan_c 11 tasks
metrics_11 = [m for m in routing_metrics if m['task'] in PLAN_C_TASKS]
# Sort by death rate differential (int_death_rate - det_death_rate)
metrics_11.sort(key=lambda m: m['int_death_rate'] - m['det_death_rate'], reverse=True)

total_eps_11 = sum(m['n_episodes'] for m in metrics_11)
print(f"Plan C subset: {len(metrics_11)} tasks, {total_eps_11} episodes")

# ══════════════════════════════════════════════════════════════════
# Part A: Per-task deterministic routing frequency
# ══════════════════════════════════════════════════════════════════

print("\n=== PART A: Per-task routing frequency ===")

results_a = []
for m in metrics_11:
    task = m['task']
    n = m['n_episodes']
    det_pref = m['det_pref_rate']
    int_pref = m['int_pref_rate']
    no_pref = 1.0 - det_pref - int_pref
    death_diff = m['int_death_rate'] - m['det_death_rate']
    results_a.append({
        'task': task,
        'n': n,
        'int_death_rate': m['int_death_rate'],
        'det_death_rate': m['det_death_rate'],
        'death_diff': death_diff,
        'det_pref': det_pref,
        'int_pref': int_pref,
        'no_pref': no_pref,
        'delta_DI': m['delta_DI'],
        'daf': m.get('daf', 0),
        'signal_category': m['signal_category']
    })

# Sort by death_diff descending
results_a.sort(key=lambda x: x['death_diff'], reverse=True)

print(f"{'Task':<45} {'N':>3} {'IntDR':>6} {'DetDR':>6} {'ΔDR':>6} {'DetPref':>7} {'IntPref':>7} {'NoPref':>7} {'DAF':>6} {'Cat'}")
print("-" * 130)
for r in results_a:
    print(f"{r['task']:<45} {r['n']:>3} {r['int_death_rate']:>6.1%} {r['det_death_rate']:>6.1%} {r['death_diff']:>6.1%} "
          f"{r['det_pref']:>7.1%} {r['int_pref']:>7.1%} {r['no_pref']:>7.1%} {r['daf']:>6.2f} {r['signal_category']}")

# Correlation: death_diff vs det_pref_rate
death_diffs = [r['death_diff'] for r in results_a]
det_prefs = [r['det_pref'] for r in results_a]
if len(set(death_diffs)) > 1:
    corr = np.corrcoef(death_diffs, det_prefs)[0, 1]
    print(f"\nCorrelation(death_diff, det_pref_rate) = {corr:.4f}")

# ══════════════════════════════════════════════════════════════════
# Part B: Episode phase analysis
# ══════════════════════════════════════════════════════════════════

print("\n=== PART B: Episode phase analysis ===")

phase_data = defaultdict(lambda: {'early': {'deaths_int': 0, 'deaths_det': 0, 'steps_int': 0, 'steps_det': 0},
                                   'mid': {'deaths_int': 0, 'deaths_det': 0, 'steps_int': 0, 'steps_det': 0},
                                   'late': {'deaths_int': 0, 'deaths_det': 0, 'steps_int': 0, 'steps_det': 0},
                                   'n_episodes': 0, 'total_int_deaths': 0, 'total_det_deaths': 0})

death_step_distribution = {'early': 0, 'mid': 0, 'late': 0}
total_death_events = 0

for task, episodes in plan_c_episodes.items():
    for ep in episodes:
        phase_data[task]['n_episodes'] += 1
        for mode_key in ['always_internal', 'always_deterministic']:
            run = ep.get(mode_key, {})
            steps = run.get('steps', [])
            n_steps = len(steps)
            if n_steps == 0:
                continue
            
            is_internal = mode_key == 'always_internal'
            final_score = run.get('final_score', 0)
            died = (final_score == -100)
            
            if died:
                if is_internal:
                    phase_data[task]['total_int_deaths'] += 1
                else:
                    phase_data[task]['total_det_deaths'] += 1
            
            third = max(1, n_steps // 3)
            for i, step in enumerate(steps):
                if i < third:
                    phase = 'early'
                elif i < 2 * third:
                    phase = 'mid'
                else:
                    phase = 'late'
                
                if is_internal:
                    phase_data[task][phase]['steps_int'] += 1
                else:
                    phase_data[task][phase]['steps_det'] += 1
                
                score_delta = step.get('score_delta', 0)
                if score_delta == -100:
                    if is_internal:
                        phase_data[task][phase]['deaths_int'] += 1
                    else:
                        phase_data[task][phase]['deaths_det'] += 1
                    death_step_distribution[phase] += 1
                    total_death_events += 1

print(f"\nDeath events by phase (across all {len(plan_c_episodes)} tasks):")
for phase in ['early', 'mid', 'late']:
    n = death_step_distribution[phase]
    pct = n / max(1, total_death_events) * 100
    print(f"  {phase}: {n} ({pct:.1f}%)")
print(f"  total: {total_death_events}")

print(f"\nPer-task phase breakdown:")
print(f"{'Task':<45} {'N':>3} {'IntD':>4} {'DetD':>4} | {'Early_I':>7} {'Early_D':>7} {'Mid_I':>7} {'Mid_D':>7} {'Late_I':>7} {'Late_D':>7}")
print("-" * 130)
for task in sorted(phase_data.keys()):
    pd = phase_data[task]
    print(f"{task:<45} {pd['n_episodes']:>3} {pd['total_int_deaths']:>4} {pd['total_det_deaths']:>4} | "
          f"{pd['early']['deaths_int']:>7} {pd['early']['deaths_det']:>7} "
          f"{pd['mid']['deaths_int']:>7} {pd['mid']['deaths_det']:>7} "
          f"{pd['late']['deaths_int']:>7} {pd['late']['deaths_det']:>7}")

# ══════════════════════════════════════════════════════════════════
# Part C: Routing Heatmap Figure
# ══════════════════════════════════════════════════════════════════

print("\n=== PART C: Generating routing heatmap ===")

# Use all 30 tasks from routing_metrics, sorted by int_death_rate - det_death_rate
all_metrics = sorted(routing_metrics, key=lambda m: m.get('daf', 0) or 0, reverse=True)

# --- Figure 1a: Main heatmap (11 plan_c tasks) ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), gridspec_kw={'width_ratios': [1.2, 1.2, 1]})

tasks_sorted = [r['task'] for r in results_a]
tasks_display = [t.replace('-', ' ').replace('measure melting point unknown substance', 'melt-pt (unkn)')
                  .replace('lifespan longest lived then shortest lived', 'lifespan long→short')
                  .replace('inclined plane determine angle', 'inclined plane')
                  .replace('identify life stages 1', 'life stages 1')
                  .replace('identify life stages 2', 'life stages 2')
                  .replace('measure melting point unknown substance', 'melt-pt (unkn)')
                  .replace('find non living thing', 'find non-living')
                  for t in tasks_sorted]

# Shorten long names
short_names = []
for t in tasks_sorted:
    name = t.replace('-', ' ')
    if len(name) > 25:
        words = name.split()
        name = ' '.join(words[:3]) + '...'
    short_names.append(name)

n_tasks = len(tasks_sorted)

# Panel 1: Death rates (internal vs deterministic)
ax1 = axes[0]
y_pos = np.arange(n_tasks)
bar_height = 0.35
int_deaths = [next(r['int_death_rate'] for r in results_a if r['task'] == t) for t in tasks_sorted]
det_deaths = [next(r['det_death_rate'] for r in results_a if r['task'] == t) for t in tasks_sorted]

bars1 = ax1.barh(y_pos - bar_height/2, int_deaths, bar_height, label='Internal', color='#E74C3C', alpha=0.85)
bars2 = ax1.barh(y_pos + bar_height/2, det_deaths, bar_height, label='Deterministic', color='#3498DB', alpha=0.85)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(short_names, fontsize=8)
ax1.set_xlabel('Death Rate', fontsize=10)
ax1.set_title('(a) Death Rate by Mode', fontsize=11, fontweight='bold')
ax1.legend(fontsize=8, loc='lower right')
ax1.set_xlim(0, 0.75)
ax1.invert_yaxis()
ax1.tick_params(axis='x', labelsize=8)

# Panel 2: Label distribution (stacked bar)
ax2 = axes[1]
det_prefs_plot = [next(r['det_pref'] for r in results_a if r['task'] == t) for t in tasks_sorted]
int_prefs_plot = [next(r['int_pref'] for r in results_a if r['task'] == t) for t in tasks_sorted]
no_prefs_plot = [next(r['no_pref'] for r in results_a if r['task'] == t) for t in tasks_sorted]

ax2.barh(y_pos, det_prefs_plot, bar_height*2, label='det-preferred', color='#3498DB', alpha=0.85)
ax2.barh(y_pos, no_prefs_plot, bar_height*2, left=det_prefs_plot, label='no-preference', color='#95A5A6', alpha=0.85)
left2 = [d + n for d, n in zip(det_prefs_plot, no_prefs_plot)]
ax2.barh(y_pos, int_prefs_plot, bar_height*2, left=left2, label='int-preferred', color='#E74C3C', alpha=0.85)
ax2.set_yticks(y_pos)
ax2.set_yticklabels([])
ax2.set_xlabel('Oracle Label Distribution', fontsize=10)
ax2.set_title('(b) Routing Label Distribution', fontsize=11, fontweight='bold')
ax2.legend(fontsize=7, loc='lower right')
ax2.set_xlim(0, 1.05)
ax2.invert_yaxis()
ax2.tick_params(axis='x', labelsize=8)

# Panel 3: DAF + score delta
ax3 = axes[2]
dafs = [next(r['daf'] for r in results_a if r['task'] == t) for t in tasks_sorted]
deltas = [next(r['delta_DI'] for r in results_a if r['task'] == t) for t in tasks_sorted]

colors_daf = ['#27AE60' if d >= 0.95 else '#F39C12' if d >= 0.5 else '#E74C3C' for d in dafs]
ax3.barh(y_pos, dafs, bar_height*2, color=colors_daf, alpha=0.85)
ax3.axvline(x=0.95, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax3.set_yticks(y_pos)
ax3.set_yticklabels([])
ax3.set_xlabel('Death Avoidance Fraction (DAF)', fontsize=10)
ax3.set_title('(c) DAF', fontsize=11, fontweight='bold')
ax3.set_xlim(-0.1, 1.1)
ax3.invert_yaxis()
ax3.tick_params(axis='x', labelsize=8)

# Add DAF >= 0.95 threshold annotation
ax3.text(0.96, n_tasks - 0.5, 'DAF ≥ 0.95', fontsize=7, color='gray', va='top')

plt.suptitle('Per-Task Routing Patterns (11 ScienceWorld Tasks)', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUT / "routing_heatmap.pdf", bbox_inches='tight', dpi=300)
plt.savefig(OUT / "routing_heatmap.png", bbox_inches='tight', dpi=300)
print(f"Saved: {OUT / 'routing_heatmap.pdf'}")
print(f"Saved: {OUT / 'routing_heatmap.png'}")
plt.close()

# --- Figure 1b: Phase analysis heatmap ---
fig2, ax_phase = plt.subplots(figsize=(8, 5.5))

phase_matrix = []
phase_tasks = []
for task in tasks_sorted:
    if task not in phase_data:
        continue
    pd_t = phase_data[task]
    n_ep = max(1, pd_t['n_episodes'])
    row = []
    for phase in ['early', 'mid', 'late']:
        int_d = pd_t[phase]['deaths_int']
        det_d = pd_t[phase]['deaths_det']
        # Death rate differential per phase (normalized by episodes)
        row.append((int_d - det_d) / n_ep)
    phase_matrix.append(row)
    phase_tasks.append(task)

phase_matrix = np.array(phase_matrix)

# Short names for phase heatmap
phase_short = []
for t in phase_tasks:
    name = t.replace('-', ' ')
    if len(name) > 25:
        words = name.split()
        name = ' '.join(words[:3]) + '...'
    phase_short.append(name)

im = ax_phase.imshow(phase_matrix, cmap='RdBu_r', aspect='auto',
                      vmin=-max(0.01, abs(phase_matrix).max()),
                      vmax=max(0.01, abs(phase_matrix).max()))
ax_phase.set_xticks([0, 1, 2])
ax_phase.set_xticklabels(['Early (1/3)', 'Mid (1/3)', 'Late (1/3)'], fontsize=10)
ax_phase.set_yticks(range(len(phase_tasks)))
ax_phase.set_yticklabels(phase_short, fontsize=8)
ax_phase.set_title('Death Rate Differential by Episode Phase\n(Internal − Deterministic, per episode)', fontsize=11, fontweight='bold')

# Add text annotations
for i in range(len(phase_tasks)):
    for j in range(3):
        val = phase_matrix[i, j]
        color = 'white' if abs(val) > 0.15 else 'black'
        ax_phase.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7, color=color)

cbar = plt.colorbar(im, ax=ax_phase, shrink=0.8)
cbar.set_label('Δ Death Rate (Int − Det)', fontsize=9)

plt.tight_layout()
plt.savefig(OUT / "phase_heatmap.pdf", bbox_inches='tight', dpi=300)
plt.savefig(OUT / "phase_heatmap.png", bbox_inches='tight', dpi=300)
print(f"Saved: {OUT / 'phase_heatmap.pdf'}")
print(f"Saved: {OUT / 'phase_heatmap.png'}")
plt.close()

# ══════════════════════════════════════════════════════════════════
# Part D: Failure mode classification
# ══════════════════════════════════════════════════════════════════

print("\n=== PART D: Failure mode classification ===")

# Analyze death-causing actions from plan_c step-level data
death_actions_internal = Counter()
death_actions_det = Counter()
death_step_positions = {'internal': [], 'deterministic': []}
death_context = defaultdict(list)

for task, episodes in plan_c_episodes.items():
    for ep in episodes:
        for mode_key, mode_label in [('always_internal', 'internal'), ('always_deterministic', 'deterministic')]:
            run = ep.get(mode_key, {})
            steps = run.get('steps', [])
            n_steps = len(steps)
            if n_steps == 0:
                continue
            
            for i, step in enumerate(steps):
                if step.get('score_delta', 0) == -100:
                    action = step.get('action', 'unknown')
                    if mode_label == 'internal':
                        death_actions_internal[action] += 1
                    else:
                        death_actions_det[action] += 1
                    
                    death_step_positions[mode_label].append(i / max(1, n_steps - 1))
                    
                    # Categorize action
                    if action.startswith('focus on'):
                        action_cat = 'focus-action'
                    elif action.startswith('connect'):
                        action_cat = 'connect-action'
                    elif action.startswith('activate') or action.startswith('use'):
                        action_cat = 'activate-action'
                    else:
                        action_cat = 'other'
                    
                    death_context[mode_label].append({
                        'task': task,
                        'step': i,
                        'total_steps': n_steps,
                        'action': action,
                        'action_category': action_cat,
                        'relative_position': i / max(1, n_steps - 1)
                    })

print(f"\nInternal death-causing actions (top 15):")
for action, count in death_actions_internal.most_common(15):
    print(f"  {count:3d}x  {action}")

print(f"\nDeterministic death-causing actions (top 10):")
for action, count in death_actions_det.most_common(10):
    print(f"  {count:3d}x  {action}")

# Action category breakdown
int_cats = Counter(d['action_category'] for d in death_context['internal'])
det_cats = Counter(d['action_category'] for d in death_context['deterministic'])
print(f"\nAction category breakdown:")
print(f"  Internal: {dict(int_cats)}")
print(f"  Deterministic: {dict(det_cats)}")

# Death position analysis
if death_step_positions['internal']:
    int_pos = np.array(death_step_positions['internal'])
    det_pos = np.array(death_step_positions['deterministic']) if death_step_positions['deterministic'] else np.array([])
    print(f"\nDeath position (0=start, 1=end):")
    print(f"  Internal: mean={int_pos.mean():.3f}, median={np.median(int_pos):.3f}, n={len(int_pos)}")
    if len(det_pos) > 0:
        print(f"  Deterministic: mean={det_pos.mean():.3f}, median={np.median(det_pos):.3f}, n={len(det_pos)}")

# Task-independent failure pattern analysis
# Check if "focus on" actions dominate across all tasks
focus_by_task = defaultdict(lambda: {'focus': 0, 'other': 0})
for d in death_context['internal']:
    if d['action_category'] == 'focus-action':
        focus_by_task[d['task']]['focus'] += 1
    else:
        focus_by_task[d['task']]['other'] += 1

print(f"\nTask-independent pattern: 'focus on' actions in internal deaths:")
for task in sorted(focus_by_task.keys()):
    f = focus_by_task[task]
    total = f['focus'] + f['other']
    pct = f['focus'] / max(1, total) * 100
    print(f"  {task:<40} {f['focus']}/{total} ({pct:.0f}%)")

# ══════════════════════════════════════════════════════════════════
# Summary statistics for paper
# ══════════════════════════════════════════════════════════════════

print("\n=== SUMMARY STATISTICS (for paper) ===")

n_death_dom = sum(1 for r in results_a if r['signal_category'] == 'death-dominated')
n_genuine = sum(1 for r in results_a if r['signal_category'] == 'genuine-det')
n_internal = sum(1 for r in results_a if r['signal_category'] == 'internal-preferred')
n_no_signal = sum(1 for r in results_a if r['signal_category'] == 'no-signal')

mean_int_death = np.mean([r['int_death_rate'] for r in results_a])
mean_det_death = np.mean([r['det_death_rate'] for r in results_a])
mean_daf = np.mean([r['daf'] for r in results_a if r['daf'] is not None and r['daf'] >= 0])
mean_delta = np.mean([r['delta_DI'] for r in results_a])

# Weighted averages
total_n = sum(r['n'] for r in results_a)
wmean_int_death = sum(r['int_death_rate'] * r['n'] for r in results_a) / total_n
wmean_det_death = sum(r['det_death_rate'] * r['n'] for r in results_a) / total_n
wmean_delta = sum(r['delta_DI'] * r['n'] for r in results_a) / total_n

positive_daf = [r for r in results_a if r.get('daf', 0) > 0]
if positive_daf:
    wmean_daf = sum(r['daf'] * r['n'] for r in positive_daf) / sum(r['n'] for r in positive_daf)
else:
    wmean_daf = 0

print(f"Tasks: {len(results_a)} total, {n_death_dom} death-dominated, {n_genuine} genuine-det, {n_internal} internal-pref, {n_no_signal} no-signal")
print(f"Episodes: {total_eps_11}")
print(f"Mean death rates: internal={mean_int_death:.1%}, deterministic={mean_det_death:.1%}")
print(f"Weighted mean death rates: internal={wmean_int_death:.1%}, deterministic={wmean_det_death:.1%}")
print(f"Mean DAF (positive): {mean_daf:.4f}")
print(f"Weighted mean DAF: {wmean_daf:.4f}")
print(f"Mean Δ(D-I): {mean_delta:.2f}")
print(f"Weighted mean Δ(D-I): {wmean_delta:.2f}")

# Death phase summary
print(f"\nDeath by phase: early={death_step_distribution['early']}, mid={death_step_distribution['mid']}, late={death_step_distribution['late']}")
print(f"Total internal deaths: {sum(pd['total_int_deaths'] for pd in phase_data.values())}")
print(f"Total deterministic deaths: {sum(pd['total_det_deaths'] for pd in phase_data.values())}")
print(f"Death ratio (int/det): {sum(pd['total_int_deaths'] for pd in phase_data.values()) / max(1, sum(pd['total_det_deaths'] for pd in phase_data.values())):.2f}x")

# Save summary JSON
summary = {
    'n_tasks': len(results_a),
    'n_episodes': total_eps_11,
    'task_categories': {
        'death_dominated': n_death_dom,
        'genuine_det': n_genuine,
        'internal_preferred': n_internal,
        'no_signal': n_no_signal
    },
    'death_rates': {
        'mean_internal': round(mean_int_death, 4),
        'mean_deterministic': round(mean_det_death, 4),
        'weighted_internal': round(wmean_int_death, 4),
        'weighted_deterministic': round(wmean_det_death, 4),
    },
    'weighted_daf': round(wmean_daf, 4),
    'weighted_delta_DI': round(wmean_delta, 2),
    'death_phase_distribution': death_step_distribution,
    'death_ratio_int_over_det': round(sum(pd['total_int_deaths'] for pd in phase_data.values()) / max(1, sum(pd['total_det_deaths'] for pd in phase_data.values())), 2),
    'failure_dominant_action': 'focus on <location>' if int_cats.get('focus-action', 0) > 0 else 'other',
    'focus_action_pct_of_deaths': round(int_cats.get('focus-action', 0) / max(1, sum(int_cats.values())) * 100, 1),
    'per_task': results_a,
    'phase_data': {task: {
        'n_episodes': phase_data[task]['n_episodes'],
        'int_deaths': phase_data[task]['total_int_deaths'],
        'det_deaths': phase_data[task]['total_det_deaths'],
        'early_int': phase_data[task]['early']['deaths_int'],
        'early_det': phase_data[task]['early']['deaths_det'],
        'mid_int': phase_data[task]['mid']['deaths_int'],
        'mid_det': phase_data[task]['mid']['deaths_det'],
        'late_int': phase_data[task]['late']['deaths_int'],
        'late_det': phase_data[task]['late']['deaths_det'],
    } for task in sorted(phase_data.keys())}
}

with open(OUT / "plan005_v2_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved: {OUT / 'plan005_v2_summary.json'}")

print("\n=== DONE ===")
