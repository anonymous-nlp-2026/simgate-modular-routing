import json
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

DATA_DIR = "./results/plan_c"
OUT_DIR = "./results/routing-patterns"

TASK_SHORT = {
    'chemistry-mix': 'Chem-Mix',
    'find-animal': 'Find-Animal',
    'find-non-living-thing': 'Find-NL',
    'freeze': 'Freeze',
    'grow-fruit': 'Grow-Fruit',
    'identify-life-stages-1': 'Life-Stage-1',
    'identify-life-stages-2': 'Life-Stage-2',
    'inclined-plane-determine-angle': 'Incline',
    'lifespan-longest-lived': 'Lifespan-L',
    'lifespan-longest-lived-then-shortest-lived': 'Lifespan-LS',
    'measure-melting-point-unknown-substance': 'Melting-Pt',
}

# ── Load all episodes ──
all_eps = []
for task_dir in sorted(glob.glob(os.path.join(DATA_DIR, "*"))):
    task_name = os.path.basename(task_dir)
    ep_file = os.path.join(task_dir, "episode_summary.jsonl")
    if not os.path.exists(ep_file):
        continue
    with open(ep_file) as f:
        for line in f:
            rec = json.loads(line)
            rec['task'] = task_name
            all_eps.append(rec)

df = pd.DataFrame(all_eps)
print(f"Total episodes: {len(df)}, Tasks: {df['task'].nunique()}")

# ── A. Per-task routing pattern ──
print("\n" + "="*70)
print("A. PER-TASK ROUTING PATTERN")
print("="*70)

routing_table = []
for task in sorted(df['task'].unique()):
    sub = df[df['task'] == task]
    n = len(sub)
    det = (sub['label'] == 'det-preferred').sum()
    internal = (sub['label'] == 'internal-preferred').sum()
    nopref = (sub['label'] == 'no-preference').sum()
    non_tie = det + internal
    det_ratio = det / non_tie if non_tie > 0 else float('nan')
    routing_table.append({
        'task': task,
        'short': TASK_SHORT.get(task, task),
        'n': n,
        'det': det,
        'internal': internal,
        'no_pref': nopref,
        'det_ratio': det_ratio,
        'det_pct': det / n * 100,
        'int_pct': internal / n * 100,
        'nopref_pct': nopref / n * 100,
    })

rt = pd.DataFrame(routing_table).sort_values('det_ratio', ascending=False)
print(f"\n{'Task':<42} {'N':>4} {'Det':>4} {'Int':>4} {'NoPr':>4} {'Det/(D+I)':>10}")
print("-"*70)
for _, r in rt.iterrows():
    print(f"{r['task']:<42} {r['n']:>4} {r['det']:>4} {r['internal']:>4} {r['no_pref']:>4} {r['det_ratio']:>10.3f}")

# ── B. Death timing analysis ──
print("\n" + "="*70)
print("B. DEATH TIMING ANALYSIS")
print("="*70)

internal_deaths = df[df['internal_final'] == -100.0].copy()
det_deaths = df[df['det_final'] == -100.0].copy()

internal_deaths['death_step'] = internal_deaths['internal_steps']
det_deaths['death_step'] = det_deaths['det_steps']

def bucket(step):
    if step <= 10:
        return 'early (1-10)'
    elif step <= 20:
        return 'mid (11-20)'
    else:
        return 'late (21-30)'

internal_deaths['bucket'] = internal_deaths['death_step'].apply(bucket)
det_deaths['bucket'] = det_deaths['death_step'].apply(bucket)

print(f"\nInternal deaths: {len(internal_deaths)}")
print(f"  Mean death step: {internal_deaths['death_step'].mean():.1f}")
print(f"  Median death step: {internal_deaths['death_step'].median():.1f}")
for b in ['early (1-10)', 'mid (11-20)', 'late (21-30)']:
    c = (internal_deaths['bucket'] == b).sum()
    print(f"  {b}: {c} ({c/len(internal_deaths)*100:.1f}%)")

print(f"\nDeterministic deaths: {len(det_deaths)}")
print(f"  Mean death step: {det_deaths['death_step'].mean():.1f}")
print(f"  Median death step: {det_deaths['death_step'].median():.1f}")
for b in ['early (1-10)', 'mid (11-20)', 'late (21-30)']:
    c = (det_deaths['bucket'] == b).sum()
    print(f"  {b}: {c} ({c/len(det_deaths)*100:.1f}%)")

# Per-task death timing
print("\nPer-task internal death timing:")
for task in sorted(df['task'].unique()):
    sub = internal_deaths[internal_deaths['task'] == task]
    if len(sub) > 0:
        print(f"  {TASK_SHORT.get(task,task):<15}: n={len(sub):>2}, mean_step={sub['death_step'].mean():.1f}, "
              f"early={sum(sub['bucket']=='early (1-10)')}, mid={sum(sub['bucket']=='mid (11-20)')}, late={sum(sub['bucket']=='late (21-30)')}")

# ── C. Episode length comparison ──
print("\n" + "="*70)
print("C. EPISODE LENGTH COMPARISON")
print("="*70)

length_table = []
for task in sorted(df['task'].unique()):
    sub = df[df['task'] == task]
    length_table.append({
        'task': task,
        'short': TASK_SHORT.get(task, task),
        'n': len(sub),
        'int_mean': sub['internal_steps'].mean(),
        'int_std': sub['internal_steps'].std(),
        'det_mean': sub['det_steps'].mean(),
        'det_std': sub['det_steps'].std(),
        'diff': sub['det_steps'].mean() - sub['internal_steps'].mean(),
    })

lt = pd.DataFrame(length_table)
print(f"\n{'Task':<15} {'N':>4} {'Int-Mean':>9} {'Int-Std':>8} {'Det-Mean':>9} {'Det-Std':>8} {'Diff':>6}")
print("-"*60)
for _, r in lt.iterrows():
    print(f"{r['short']:<15} {r['n']:>4} {r['int_mean']:>9.1f} {r['int_std']:>8.1f} {r['det_mean']:>9.1f} {r['det_std']:>8.1f} {r['diff']:>6.1f}")

# ════════════════════════════════════════════════════════
# FIGURES
# ════════════════════════════════════════════════════════

# ── Figure 1: Routing Heatmap ──
fig, ax = plt.subplots(figsize=(5, 4.5))

rt_sorted = rt.sort_values('det_ratio', ascending=True)
tasks_sorted = rt_sorted['short'].values
heatmap_data = np.array([
    rt_sorted['det_pct'].values,
    rt_sorted['int_pct'].values,
    rt_sorted['nopref_pct'].values,
]).T

cmap = LinearSegmentedColormap.from_list('custom', ['#f7fbff', '#2171b5'])
im = ax.imshow(heatmap_data, cmap=cmap, aspect='auto', vmin=0, vmax=100)

ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Det-Preferred', 'Int-Preferred', 'No-Preference'])
ax.set_yticks(range(len(tasks_sorted)))
ax.set_yticklabels(tasks_sorted)

for i in range(len(tasks_sorted)):
    for j in range(3):
        val = heatmap_data[i, j]
        color = 'white' if val > 50 else 'black'
        ax.text(j, i, f'{val:.0f}%', ha='center', va='center', color=color, fontsize=8)

cbar = plt.colorbar(im, ax=ax, shrink=0.8, label='Percentage (%)')
ax.set_title('Routing Label Distribution per Task')
fig.savefig(os.path.join(OUT_DIR, 'routing_heatmap.pdf'))
print(f"\nSaved routing_heatmap.pdf")
plt.close()

# ── Figure 2: Death Timing ──
fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)

bins = np.arange(0.5, 31.5, 1)

axes[0].hist(internal_deaths['death_step'], bins=bins, color='#d95f02', alpha=0.85, edgecolor='white', linewidth=0.5)
axes[0].set_xlabel('Death Step')
axes[0].set_ylabel('Count')
axes[0].set_title(f'Internal Model (n={len(internal_deaths)})')
axes[0].axvline(x=10.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
axes[0].axvline(x=20.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
axes[0].set_xlim(0, 31)

axes[1].hist(det_deaths['death_step'], bins=bins, color='#1b9e77', alpha=0.85, edgecolor='white', linewidth=0.5)
axes[1].set_xlabel('Death Step')
axes[1].set_title(f'Deterministic Model (n={len(det_deaths)})')
axes[1].axvline(x=10.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
axes[1].axvline(x=20.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
axes[1].set_xlim(0, 31)

fig.suptitle('Death Timing Distribution', y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'death_timing.pdf'))
print(f"Saved death_timing.pdf")
plt.close()

# ── Figure 3: Episode Length Comparison ──
fig, ax = plt.subplots(figsize=(7, 4))

lt_sorted = lt.sort_values('diff', ascending=False)
x = np.arange(len(lt_sorted))
width = 0.35

bars1 = ax.bar(x - width/2, lt_sorted['int_mean'], width, 
               yerr=lt_sorted['int_std'], label='Internal', 
               color='#d95f02', alpha=0.85, capsize=3, error_kw={'linewidth': 0.8})
bars2 = ax.bar(x + width/2, lt_sorted['det_mean'], width,
               yerr=lt_sorted['det_std'], label='Deterministic',
               color='#1b9e77', alpha=0.85, capsize=3, error_kw={'linewidth': 0.8})

ax.set_xlabel('Task')
ax.set_ylabel('Mean Episode Length (steps)')
ax.set_title('Episode Length: Internal vs. Deterministic')
ax.set_xticks(x)
ax.set_xticklabels(lt_sorted['short'], rotation=45, ha='right')
ax.legend(loc='lower right')
ax.set_ylim(0, 35)
ax.axhline(y=30, color='gray', linestyle=':', linewidth=0.8, alpha=0.5, label='Max steps')

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'episode_length_comparison.pdf'))
print(f"Saved episode_length_comparison.pdf")
plt.close()

# ── Summary Stats ──
print("\n" + "="*70)
print("SUMMARY INSIGHTS")
print("="*70)

total_det = (df['label'] == 'det-preferred').sum()
total_int = (df['label'] == 'internal-preferred').sum()
total_nopref = (df['label'] == 'no-preference').sum()
print(f"\nOverall: {len(df)} episodes")
print(f"  Det-preferred: {total_det} ({total_det/len(df)*100:.1f}%)")
print(f"  Internal-preferred: {total_int} ({total_int/len(df)*100:.1f}%)")
print(f"  No-preference: {total_nopref} ({total_nopref/len(df)*100:.1f}%)")
print(f"  Det/(Det+Int): {total_det/(total_det+total_int)*100:.1f}%")

# Death-conditioned routing
death_eps = df[df['internal_final'] == -100.0]
no_death_eps = df[df['internal_final'] != -100.0]
print(f"\nInternal-death episodes: {len(death_eps)}")
print(f"  Det-preferred: {(death_eps['label']=='det-preferred').sum()} ({(death_eps['label']=='det-preferred').mean()*100:.1f}%)")
print(f"Non-death episodes: {len(no_death_eps)}")
print(f"  Det-preferred: {(no_death_eps['label']=='det-preferred').sum()} ({(no_death_eps['label']=='det-preferred').mean()*100:.1f}%)")

# Which tasks have internal actually better?
print("\nTasks where internal-preferred > 0:")
for _, r in rt.iterrows():
    if r['internal'] > 0:
        print(f"  {r['task']}: {r['internal']} episodes ({r['int_pct']:.1f}%)")

print("\nDone.")
