import json
import numpy as np
from scipy import stats
from pathlib import Path

DATA_PATH = "./results/baseline_245eps/episode_details.jsonl"
OUT_DIR = "./results/penalty_robustness"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

episodes = []
with open(DATA_PATH) as f:
    for line in f:
        ep = json.loads(line)
        if ep["condition"] == "always-deterministic":
            episodes.append(ep)

int_scores = np.array([ep["internal_final"] for ep in episodes])
det_scores = np.array([ep["det_final"] for ep in episodes])
N = len(int_scores)

int_died = (int_scores == -100.0)
det_died = (det_scores == -100.0)
neither_died = ~int_died & ~det_died

print(f"N={N}, int_died={int_died.sum()}, det_died={det_died.sum()}")
print(f"only_int_dies={(int_died & ~det_died).sum()}, only_det_dies={(~int_died & det_died).sum()}, "
      f"both_die={(int_died & det_died).sum()}, neither={neither_died.sum()}")

PENALTIES = [50, 100, 200]

def compute_metrics(int_sc, det_sc, P, int_d, det_d):
    int_adj = np.where(int_d, -P, int_sc)
    det_adj = np.where(det_d, -P, det_sc)
    delta_full = np.mean(det_adj) - np.mean(int_adj)
    nd = ~int_d & ~det_d
    delta_nd = np.mean(det_sc[nd]) - np.mean(int_sc[nd]) if nd.sum() > 0 else 0.0
    daf = (1.0 - delta_nd / delta_full) * 100.0 if abs(delta_full) > 1e-10 else 0.0
    delta_per_ep = det_adj - int_adj
    death_asym = int_d.astype(float) - det_d.astype(float)
    r, p = stats.pearsonr(delta_per_ep, death_asym)
    return delta_full, delta_nd, daf, r, p

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

results = {}
for P in PENALTIES:
    delta_full, delta_nd, daf, pearson_r, pearson_p = compute_metrics(
        int_scores, det_scores, P, int_died, det_died)
    ci_lo, ci_hi = bootstrap_daf(int_scores, det_scores, P, int_died, det_died)
    results[P] = {
        "P": P,
        "delta_full": round(delta_full, 4),
        "delta_nd": round(delta_nd, 4),
        "DAF": round(daf, 2),
        "DAF_CI_lower": round(ci_lo, 2),
        "DAF_CI_upper": round(ci_hi, 2),
        "pearson_r": round(pearson_r, 4),
        "pearson_p": pearson_p,
    }

r50, r100, r200 = results[50], results[100], results[200]

def ci_str(r):
    return f"[{r['DAF_CI_lower']:.1f}, {r['DAF_CI_upper']:.1f}]"

print(f"\n=== Penalty Robustness Analysis ({N} episodes) ===\n")
header = f"| {'Metric':<12} | {'P=-50':>14} | {'P=-100 (baseline)':>20} | {'P=-200':>14} |"
print(header)
print(f"|{'-'*14}|{'-'*16}|{'-'*22}|{'-'*16}|")
print(f"| {'delta_full':<12} | {r50['delta_full']:>14.2f} | {r100['delta_full']:>20.2f} | {r200['delta_full']:>14.2f} |")
print(f"| {'delta_nd':<12} | {r50['delta_nd']:>14.2f} | {r100['delta_nd']:>20.2f} | {r200['delta_nd']:>14.2f} |")
print(f"| {'DAF':<12} | {r50['DAF']:>13.1f}% | {r100['DAF']:>19.1f}% | {r200['DAF']:>13.1f}% |")
print(f"| {'DAF 95% CI':<12} | {ci_str(r50):>14} | {ci_str(r100):>20} | {ci_str(r200):>14} |")
print(f"| {'Pearson r':<12} | {r50['pearson_r']:>14.4f} | {r100['pearson_r']:>20.4f} | {r200['pearson_r']:>14.4f} |")

all_dafs = [r50['DAF'], r100['DAF'], r200['DAF']]
all_r_high = all(results[P]['pearson_r'] > 0.9 for P in PENALTIES)

print(f"\n--- Summary ---")
print(f"DAF range: {min(all_dafs):.1f}% - {max(all_dafs):.1f}% (spread: {max(all_dafs)-min(all_dafs):.1f}pp)")
print(f"All DAF > 80%: {all(d > 80 for d in all_dafs)}")
print(f"All Pearson r > 0.9: {all_r_high}")

if all(d > 80 for d in all_dafs) and all_r_high:
    conclusion = "ROBUST: DAF remains >80% across all penalty values; death avoidance dominates the score gap regardless of penalty magnitude."
else:
    conclusion = "SENSITIVE: DAF varies substantially with penalty magnitude."

print(f"\nConclusion: {conclusion}")

output = {
    "N": N,
    "n_only_int_dies": int((int_died & ~det_died).sum()),
    "n_only_det_dies": int((~int_died & det_died).sum()),
    "n_both_die": int((int_died & det_died).sum()),
    "n_neither_dies": int(neither_died.sum()),
    "penalty_results": {str(P): results[P] for P in PENALTIES},
    "conclusion": conclusion,
}

out_path = f"{OUT_DIR}/penalty_robustness_analysis.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {out_path}")
