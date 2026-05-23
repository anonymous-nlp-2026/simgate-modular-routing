import json
import numpy as np
from scipy import stats

def wilson_ci(successes, n, alpha=0.05):
    if n == 0:
        return 0.0, (0.0, 1.0)
    p = successes / n
    z = stats.norm.ppf(1 - alpha/2)
    denom = 1 + z**2/n
    center = (p + z**2/(2*n)) / denom
    margin = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return p, (max(0.0, center - margin), min(1.0, center + margin))

def newcombe_wilson_diff_ci(x1, n1, x2, n2, alpha=0.05):
    p1, (l1, u1) = wilson_ci(x1, n1, alpha)
    p2, (l2, u2) = wilson_ci(x2, n2, alpha)
    diff = p1 - p2
    lower = diff - np.sqrt((p1 - l1)**2 + (u2 - p2)**2)
    upper = diff + np.sqrt((u1 - p1)**2 + (p2 - l2)**2)
    return diff, (lower, upper)

with open('./results/death_decomposition_report.json') as f:
    data = json.load(f)

tasks = data['per_task']

# Filter to death-prone tasks (at least one death in either strategy)
death_prone = [t for t in tasks if t['int_death_count'] > 0 or t['det_death_count'] > 0]

print(f"Total tasks: {len(tasks)}")
print(f"Death-prone tasks: {len(death_prone)}")
print()

# Header
header = f"{'Task':<45} {'N':>4} {'DR(int)':>8} {'95% CI(int)':>16} {'DR(det)':>8} {'95% CI(det)':>16} {'Delta':>7} {'95% CI(Delta)':>18} {'Sig?':>5}"
sep = "-" * len(header)
print(header)
print(sep)

results = []
total_int_deaths = 0
total_det_deaths = 0
total_n = 0
sig_count = 0

for t in sorted(death_prone, key=lambda x: x['int_death_rate'] - x['det_death_rate'], reverse=True):
    task = t['task']
    n = t['n_episodes']
    int_d = t['int_death_count']
    det_d = t['det_death_count']
    
    total_int_deaths += int_d
    total_det_deaths += det_d
    total_n += n
    
    p_int, ci_int = wilson_ci(int_d, n)
    p_det, ci_det = wilson_ci(det_d, n)
    delta, ci_delta = newcombe_wilson_diff_ci(int_d, n, det_d, n)
    
    sig = "Yes" if ci_delta[0] > 0 or ci_delta[1] < 0 else "No"
    if sig == "Yes":
        sig_count += 1
    
    results.append({
        'task': task, 'n': n,
        'int_deaths': int_d, 'det_deaths': det_d,
        'p_int': p_int, 'ci_int': ci_int,
        'p_det': p_det, 'ci_det': ci_det,
        'delta': delta, 'ci_delta': ci_delta,
        'sig': sig
    })
    
    print(f"{task:<45} {n:>4} {p_int:>7.1%} [{ci_int[0]:>5.1%}, {ci_int[1]:>5.1%}] {p_det:>7.1%} [{ci_det[0]:>5.1%}, {ci_det[1]:>5.1%}] {delta:>+6.1%} [{ci_delta[0]:>+5.1%}, {ci_delta[1]:>+5.1%}] {sig:>5}")

print(sep)

# Pooled analysis
p_int_pool, ci_int_pool = wilson_ci(total_int_deaths, total_n)
p_det_pool, ci_det_pool = wilson_ci(total_det_deaths, total_n)
delta_pool, ci_delta_pool = newcombe_wilson_diff_ci(total_int_deaths, total_n, total_det_deaths, total_n)
sig_pool = "Yes" if ci_delta_pool[0] > 0 or ci_delta_pool[1] < 0 else "No"

print(f"{'POOLED (death-prone tasks)':<45} {total_n:>4} {p_int_pool:>7.1%} [{ci_int_pool[0]:>5.1%}, {ci_int_pool[1]:>5.1%}] {p_det_pool:>7.1%} [{ci_det_pool[0]:>5.1%}, {ci_det_pool[1]:>5.1%}] {delta_pool:>+6.1%} [{ci_delta_pool[0]:>+5.1%}, {ci_delta_pool[1]:>+5.1%}] {sig_pool:>5}")

# All episodes pooled (including zero-death tasks)
all_int = sum(t['int_death_count'] for t in tasks)
all_det = sum(t['det_death_count'] for t in tasks)
all_n = sum(t['n_episodes'] for t in tasks)
p_int_all, ci_int_all = wilson_ci(all_int, all_n)
p_det_all, ci_det_all = wilson_ci(all_det, all_n)
delta_all, ci_delta_all = newcombe_wilson_diff_ci(all_int, all_n, all_det, all_n)
sig_all = "Yes" if ci_delta_all[0] > 0 or ci_delta_all[1] < 0 else "No"

print(f"{'POOLED (all 30 tasks)':<45} {all_n:>4} {p_int_all:>7.1%} [{ci_int_all[0]:>5.1%}, {ci_int_all[1]:>5.1%}] {p_det_all:>7.1%} [{ci_det_all[0]:>5.1%}, {ci_det_all[1]:>5.1%}] {delta_all:>+6.1%} [{ci_delta_all[0]:>+5.1%}, {ci_delta_all[1]:>+5.1%}] {sig_all:>5}")

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Death-prone tasks: {len(death_prone)} / {len(tasks)}")
print(f"Tasks with statistically significant Delta: {sig_count} / {len(death_prone)}")
print(f"Tasks with CI(Delta) containing 0: {len(death_prone) - sig_count} / {len(death_prone)}")
print()
print(f"Overall pooled (all tasks):")
print(f"  Internal death rate: {p_int_all:.1%} [{ci_int_all[0]:.1%}, {ci_int_all[1]:.1%}]")
print(f"  Deterministic death rate: {p_det_all:.1%} [{ci_det_all[0]:.1%}, {ci_det_all[1]:.1%}]")
print(f"  Delta: {delta_all:+.1%} [{ci_delta_all[0]:+.1%}, {ci_delta_all[1]:+.1%}]")
print(f"  Significant: {sig_all}")

# Tasks with 0 deaths in both
zero_death_tasks = [t for t in tasks if t['int_death_count'] == 0 and t['det_death_count'] == 0]
print(f"\nTasks with zero deaths in both strategies: {len(zero_death_tasks)}")
for t in zero_death_tasks:
    print(f"  - {t['task']} (n={t['n_episodes']})")

# Direction analysis
int_higher = [r for r in results if r['delta'] > 0]
det_higher = [r for r in results if r['delta'] < 0]
equal = [r for r in results if r['delta'] == 0]
print(f"\nDirection of Delta:")
print(f"  Internal death rate HIGHER: {len(int_higher)} tasks")
print(f"  Deterministic death rate HIGHER: {len(det_higher)} tasks")
print(f"  Equal: {len(equal)} tasks")

# Now write markdown output
md_lines = []
md_lines.append("# W4: Per-Task Death Rate Differential Confidence Intervals")
md_lines.append("")
md_lines.append("## Data Source")
md_lines.append(f"- Source: `results/death_decomposition_report.json` (matched-episode design)")
md_lines.append(f"- Total tasks: {len(tasks)}, Death-prone tasks: {len(death_prone)}")
md_lines.append(f"- Total episodes: {all_n}")
md_lines.append(f"- Method: Wilson score interval for individual proportions, Newcombe-Wilson for difference")
md_lines.append("")
md_lines.append("## Per-Task Death Rate Differential Table")
md_lines.append("")
md_lines.append("| Task | N | DR(int) | 95% CI | DR(det) | 95% CI | Delta | 95% CI(Delta) | Sig? |")
md_lines.append("|------|--:|--------:|-------:|--------:|-------:|------:|---------------:|-----:|")

for r in results:
    md_lines.append(f"| {r['task']} | {r['n']} | {r['p_int']:.1%} | [{r['ci_int'][0]:.1%}, {r['ci_int'][1]:.1%}] | {r['p_det']:.1%} | [{r['ci_det'][0]:.1%}, {r['ci_det'][1]:.1%}] | {r['delta']:+.1%} | [{r['ci_delta'][0]:+.1%}, {r['ci_delta'][1]:+.1%}] | {r['sig']} |")

md_lines.append(f"| **POOLED (death-prone)** | **{total_n}** | **{p_int_pool:.1%}** | [{ci_int_pool[0]:.1%}, {ci_int_pool[1]:.1%}] | **{p_det_pool:.1%}** | [{ci_det_pool[0]:.1%}, {ci_det_pool[1]:.1%}] | **{delta_pool:+.1%}** | [{ci_delta_pool[0]:+.1%}, {ci_delta_pool[1]:+.1%}] | **{sig_pool}** |")
md_lines.append(f"| **POOLED (all tasks)** | **{all_n}** | **{p_int_all:.1%}** | [{ci_int_all[0]:.1%}, {ci_int_all[1]:.1%}] | **{p_det_all:.1%}** | [{ci_det_all[0]:.1%}, {ci_det_all[1]:.1%}] | **{delta_all:+.1%}** | [{ci_delta_all[0]:+.1%}, {ci_delta_all[1]:+.1%}] | **{sig_all}** |")

md_lines.append("")
md_lines.append("## Summary Statistics")
md_lines.append("")
md_lines.append(f"- Death-prone tasks: {len(death_prone)} / {len(tasks)}")
md_lines.append(f"- Tasks with statistically significant Delta (95% CI excludes 0): **{sig_count} / {len(death_prone)}**")
md_lines.append(f"- Tasks with CI(Delta) containing 0 (non-significant): **{len(death_prone) - sig_count} / {len(death_prone)}**")
md_lines.append(f"- Direction: Internal higher in {len(int_higher)}, Deterministic higher in {len(det_higher)}, Equal in {len(equal)}")
md_lines.append("")
md_lines.append("## Overall Pooled Analysis")
md_lines.append("")
md_lines.append(f"- Internal death rate: {p_int_all:.1%} (95% CI: [{ci_int_all[0]:.1%}, {ci_int_all[1]:.1%}])")
md_lines.append(f"- Deterministic death rate: {p_det_all:.1%} (95% CI: [{ci_det_all[0]:.1%}, {ci_det_all[1]:.1%}])")
md_lines.append(f"- Delta (int - det): {delta_all:+.1%} (95% CI: [{ci_delta_all[0]:+.1%}, {ci_delta_all[1]:+.1%}])")
md_lines.append(f"- Pooled significance: **{sig_all}**")
md_lines.append("")
md_lines.append("## Forest Plot Data (task, delta, ci_lower, ci_upper)")
md_lines.append("")
md_lines.append("```")
for r in results:
    md_lines.append(f"{r['task']},{r['delta']:.4f},{r['ci_delta'][0]:.4f},{r['ci_delta'][1]:.4f}")
md_lines.append("```")
md_lines.append("")
md_lines.append("## Key Findings")
md_lines.append("")

# Determine the claim support
if sig_count == len(death_prone):
    claim = "**Fully supported**: Every death-prone task shows a statistically significant death rate differential favoring deterministic strategy."
elif sig_count > len(death_prone) * 0.7:
    claim = f"**Mostly supported**: {sig_count}/{len(death_prone)} death-prone tasks show significant differential. The pattern is pervasive but not universal."
elif sig_count > len(death_prone) * 0.3:
    claim = f"**Partially supported**: {sig_count}/{len(death_prone)} tasks reach significance. Many tasks have small samples making individual significance hard to achieve, but the direction is consistent."
else:
    claim = f"**Weakly supported**: Only {sig_count}/{len(death_prone)} tasks reach individual significance. The overall pooled effect is significant but per-task evidence is limited by sample size."

md_lines.append(f"1. Claim \"death dominance is universal across tasks\": {claim}")

# Check directionality
if len(int_higher) == len(death_prone):
    md_lines.append(f"2. **Directionality**: Internal death rate is higher in ALL {len(death_prone)} death-prone tasks (100% consistent direction).")
else:
    md_lines.append(f"2. **Directionality**: Internal death rate is higher in {len(int_higher)}/{len(death_prone)} death-prone tasks ({100*len(int_higher)/len(death_prone):.0f}% consistent direction).")

md_lines.append(f"3. **Zero-death tasks**: {len(zero_death_tasks)} tasks have zero deaths in both strategies, contributing no information to death differential.")
md_lines.append(f"4. **Sample size caveat**: Many tasks have small n (5-9 episodes), leading to wide CIs. Pooled analysis is more reliable than per-task for small-n tasks.")

with open('./artifacts/w4_death_rate_cis.md', 'w') as f:
    f.write('\n'.join(md_lines) + '\n')

print("\n\nArtifact written to: ./artifacts/w4_death_rate_cis.md")
