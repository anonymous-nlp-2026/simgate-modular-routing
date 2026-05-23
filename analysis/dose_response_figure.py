import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def wilson_ci(k, n, z=1.96):
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z / denom * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return round((center - margin) * 100, 2), round((center + margin) * 100, 2)

N = [1, 2, 3, 4, 5]

# Retry: final_dr = deaths/200 for cutoff k (from N=5 per-trial data)
retry_deaths = {1: 96, 2: 94, 3: 87, 4: 86, 5: 87}
retry_dr = [d / 200 * 100 for d in [retry_deaths[k] for k in N]]
retry_ci_lo = [wilson_ci(retry_deaths[k], 200)[0] for k in N]
retry_ci_hi = [wilson_ci(retry_deaths[k], 200)[1] for k in N]

# Reflexion: final_dr per independent condition
# N=1,2 from westb-13313; N=3,4,5 from westd-18266
refl_deaths = {1: 95, 2: 80, 3: 68, 4: 73, 5: 70}
refl_dr = [d / 200 * 100 for d in [refl_deaths[k] for k in N]]
refl_ci_lo = [wilson_ci(refl_deaths[k], 200)[0] for k in N]
refl_ci_hi = [wilson_ci(refl_deaths[k], 200)[1] for k in N]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.linewidth": 0.8,
})

fig, ax = plt.subplots(figsize=(3.5, 2.8))

ax.fill_between(N, retry_ci_lo, retry_ci_hi, color="tab:blue", alpha=0.15)
ax.plot(N, retry_dr, "o-", color="tab:blue", linewidth=1.5, markersize=5, label="Retry", zorder=3)

ax.fill_between(N, refl_ci_lo, refl_ci_hi, color="tab:red", alpha=0.15)
ax.plot(N, refl_dr, "s-", color="tab:red", linewidth=1.5, markersize=5, label="Reflexion", zorder=3)

ax.axhline(y=34.5, color="gray", linestyle="--", linewidth=1.0, zorder=1)
ax.text(5.05, 34.5, "72B (34.5%)", fontsize=7, va="center", color="gray")

ax.axhline(y=16.3, color="gray", linestyle=":", linewidth=1.0, zorder=1)
ax.text(5.05, 16.3, "Deterministic (16.3%)", fontsize=7, va="center", color="gray")

ax.set_xlabel("Number of Trials (N)")
ax.set_ylabel("Death Rate (%)")
ax.set_xticks(N)
ax.set_xlim(0.7, 5.3)
ax.set_ylim(10, 60)
ax.grid(True, linestyle="--", alpha=0.3, color="lightgray")
ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

plt.tight_layout()
out = "./analysis/dose_response_figure.pdf"
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved: {out}")

# Print data table for verification
print("\n--- Reflexion ---")
for k in N:
    lo, hi = wilson_ci(refl_deaths[k], 200)
    print(f"  N={k}: deaths={refl_deaths[k]}, DR={refl_deaths[k]/200*100:.1f}%, CI=[{lo}%, {hi}%]")
print("\n--- Retry ---")
for k in N:
    lo, hi = wilson_ci(retry_deaths[k], 200)
    print(f"  k={k}: deaths={retry_deaths[k]}, DR={retry_deaths[k]/200*100:.1f}%, CI=[{lo}%, {hi}%]")
