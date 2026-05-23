import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import os

DATA_PATH = "./results/baseline_245eps/episode_details.jsonl"
OUT_DIR = "./results/penalty_sensitivity_curve"
os.makedirs(OUT_DIR, exist_ok=True)

# Load episode data - use one condition to get all 245 paired scores
episodes = []
with open(DATA_PATH) as f:
    for line in f:
        ep = json.loads(line)
        if ep["condition"] == "always-deterministic":
            episodes.append(ep)

int_scores = np.array([ep["internal_final"] for ep in episodes])
det_scores = np.array([ep["det_final"] for ep in episodes])
N = len(int_scores)
print(f"N episodes: {N}")

# Identify death episodes
int_died = (int_scores == -100.0)
det_died = (det_scores == -100.0)
neither_died = ~int_died & ~det_died

print(f"Internal died: {int_died.sum()}, Det died: {det_died.sum()}")
print(f"Only int dies: {(int_died & ~det_died).sum()}")
print(f"Only det dies: {(~int_died & det_died).sum()}")
print(f"Both die: {(int_died & det_died).sum()}")
print(f"Neither dies: {neither_died.sum()}")

# Δ_nd: gap on non-death episodes only
delta_nd = np.mean(det_scores[neither_died]) - np.mean(int_scores[neither_died])
print(f"\nΔ_nd = {delta_nd:.4f}")
print(f"Δ_full(100) = {np.mean(det_scores) - np.mean(int_scores):.4f}")
print(f"Death rate internal = {int_died.mean():.4f}")
print(f"Death rate det = {det_died.mean():.4f}")

P_values = [0, 10, 25, 50, 100, 200, 500]

def compute_daf_for_P(int_sc, det_sc, P, int_d, det_d):
    int_adj = np.where(int_d, -P, int_sc)
    det_adj = np.where(det_d, -P, det_sc)
    delta_full = np.mean(det_adj) - np.mean(int_adj)
    nd = ~int_d & ~det_d
    d_nd = np.mean(det_sc[nd]) - np.mean(int_sc[nd]) if nd.sum() > 0 else 0.0
    if abs(delta_full) < 1e-10:
        return 0.0, delta_full, d_nd
    daf = (1.0 - d_nd / delta_full) * 100.0
    return daf, delta_full, d_nd

def bootstrap_daf(int_sc, det_sc, P, int_d, det_d, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(int_sc)
    dafs = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        i_s, d_s = int_sc[idx], det_sc[idx]
        i_d, d_d = int_d[idx], det_d[idx]
        nd = ~i_d & ~d_d
        i_adj = np.where(i_d, -P, i_s)
        d_adj = np.where(d_d, -P, d_s)
        delta_full = np.mean(d_adj) - np.mean(i_adj)
        if nd.sum() == 0 or abs(delta_full) < 1e-10:
            dafs.append(np.nan)
            continue
        d_nd = np.mean(d_s[nd]) - np.mean(i_s[nd])
        daf = (1.0 - d_nd / delta_full) * 100.0
        dafs.append(daf)
    dafs = np.array([d for d in dafs if not np.isnan(d)])
    return np.percentile(dafs, 2.5), np.percentile(dafs, 97.5)

results = []
print(f"\n{'P':>5} | {'Δ_full(P)':>10} | {'Δ_nd':>8} | {'DAF(P)':>8} | {'CI_low':>8} | {'CI_high':>8}")
print("-" * 60)

for P in P_values:
    daf, delta_full, d_nd = compute_daf_for_P(int_scores, det_scores, P, int_died, det_died)
    ci_low, ci_high = bootstrap_daf(int_scores, det_scores, P, int_died, det_died)
    results.append({
        "P": P,
        "delta_full": round(delta_full, 4),
        "delta_nd": round(d_nd, 4),
        "DAF": round(daf, 2),
        "DAF_CI_lower": round(ci_low, 2),
        "DAF_CI_upper": round(ci_high, 2)
    })
    print(f"{P:>5} | {delta_full:>10.4f} | {d_nd:>8.4f} | {daf:>8.2f} | {ci_low:>8.2f} | {ci_high:>8.2f}")

# Save data
with open(f"{OUT_DIR}/penalty_sensitivity_data.json", "w") as f:
    json.dump({
        "N": N,
        "delta_nd": round(delta_nd, 4),
        "n_internal_died": int(int_died.sum()),
        "n_det_died": int(det_died.sum()),
        "n_only_int_dies": int((int_died & ~det_died).sum()),
        "n_only_det_dies": int((~int_died & det_died).sum()),
        "n_both_die": int((int_died & det_died).sum()),
        "n_neither_dies": int(neither_died.sum()),
        "death_rate_internal": round(float(int_died.mean()), 4),
        "death_rate_det": round(float(det_died.mean()), 4),
        "results": results
    }, f, indent=2)

# === PLOT ===
fig, ax = plt.subplots(1, 1, figsize=(5.5, 4))

P_arr = np.array([r["P"] for r in results])
daf_arr = np.array([r["DAF"] for r in results])
ci_lo = np.array([r["DAF_CI_lower"] for r in results])
ci_hi = np.array([r["DAF_CI_upper"] for r in results])

P_plot = P_arr.copy().astype(float)
P_plot[0] = 1.0  # P=0 placeholder on log scale

ax.plot(P_plot[1:], daf_arr[1:], 'o-', color='#2563eb', linewidth=2, markersize=7, zorder=5)
ax.fill_between(P_plot[1:], ci_lo[1:], ci_hi[1:], alpha=0.15, color='#2563eb', zorder=3)

ax.plot(P_plot[0], daf_arr[0], 's', color='#dc2626', markersize=9, zorder=6, label=f'P=0 (DAF={daf_arr[0]:.1f}%)')
ax.errorbar(P_plot[0], daf_arr[0], yerr=[[daf_arr[0]-ci_lo[0]], [ci_hi[0]-daf_arr[0]]],
            color='#dc2626', capsize=4, capthick=1.5, zorder=6)

idx100 = list(P_arr).index(100)
ax.axvline(x=100, color='gray', linestyle='--', alpha=0.4, zorder=1)
ax.annotate(f'P=100\nDAF={daf_arr[idx100]:.1f}%', xy=(100, daf_arr[idx100]),
            xytext=(200, daf_arr[idx100]-15), fontsize=8,
            arrowprops=dict(arrowstyle='->', color='gray', lw=0.8),
            color='#374151')

ax.set_xscale('log')
ax.set_xticks([1, 10, 25, 50, 100, 200, 500])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xticklabels(['0', '10', '25', '50', '100', '200', '500'])
ax.set_xlabel('Death Penalty Magnitude (P)', fontsize=11)
ax.set_ylabel('DAF (%)', fontsize=11)
ax.set_ylim(-5, 105)
ax.set_title('DAF Sensitivity to Death Penalty Magnitude', fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(f"{OUT_DIR}/penalty_vs_daf.pdf", bbox_inches='tight', dpi=300)
fig.savefig(f"{OUT_DIR}/penalty_vs_daf.png", bbox_inches='tight', dpi=300)
print(f"\nPlot saved to {OUT_DIR}/")

# LaTeX table
with open(f"{OUT_DIR}/penalty_sensitivity_table.tex", "w") as f:
    f.write("\\begin{tabular}{rrrrr}\n")
    f.write("\\toprule\n")
    f.write("$P$ & $\\Delta_{\\text{full}}(P)$ & $\\Delta_{\\text{nd}}$ & DAF(\\%) & 95\\% CI \\\\\n")
    f.write("\\midrule\n")
    for r in results:
        f.write(f"{r['P']} & {r['delta_full']:.2f} & {r['delta_nd']:.2f} & {r['DAF']:.1f} & [{r['DAF_CI_lower']:.1f}, {r['DAF_CI_upper']:.1f}] \\\\\n")
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")

print("All done.")
