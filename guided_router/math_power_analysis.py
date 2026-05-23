import numpy as np
from scipy import stats
import json

N = 300
p_det = 0.810
p_int = 0.767
effect = p_det - p_int

# Back-calculate discordant pairs from p=0.079
# b - c = 243 - 230 = 13
# chi2 = (b-c)^2 / (b+c), and p=0.079 -> chi2 ~ 3.08
chi2_from_p = stats.chi2.ppf(1 - 0.079, df=1)
discordant_total = round(13**2 / chi2_from_p)
b = (discordant_total + 13) / 2
c = (discordant_total - 13) / 2

# Verify
chi2_observed = (b - c)**2 / (b + c)
p_verify = 1 - stats.chi2.cdf(chi2_observed, df=1)

print("=== MATH Post-hoc Power Analysis ===")
print(f"\nSetup:")
print(f"  N = {N} paired samples")
print(f"  Effect: {effect*100:.1f}pp ({p_det*100:.1f}% vs {p_int*100:.1f}%)")
print(f"  Test: McNemar (chi-squared)")
print(f"  Observed: chi2 = {chi2_observed:.3f}, p = {p_verify:.4f}")
print(f"  Discordant pairs: {discordant_total} (b={b:.0f}, c={c:.0f})")

# Discordant pair proportions
p_b = b / N
p_c = c / N

# Post-hoc power via non-central chi-squared
# NCP = N * (p_b - p_c)^2 / (p_b + p_c)
ncp = N * (p_b - p_c)**2 / (p_b + p_c)
chi2_crit = stats.chi2.ppf(0.95, df=1)
power = 1 - stats.ncx2.cdf(chi2_crit, df=1, nc=ncp)

print(f"\nResults:")
print(f"  Post-hoc power: {power:.4f} ({power*100:.1f}%)")

# Required N for 80% and 90% power
results = {}
for target, label in [(0.8, "80%"), (0.9, "90%")]:
    for n_test in range(100, 10000, 5):
        ncp_test = n_test * (p_b - p_c)**2 / (p_b + p_c)
        pw = 1 - stats.ncx2.cdf(chi2_crit, df=1, nc=ncp_test)
        if pw >= target:
            print(f"  Required N for {label} power: {n_test}")
            results[label] = n_test
            break

# Interpretation
if power < 0.5:
    interp = "Severely underpowered — the study had little chance of detecting a true 4.3pp effect"
elif power < 0.8:
    interp = "Underpowered — below the conventional 80% threshold; non-significance is inconclusive"
else:
    interp = "Adequately powered"

print(f"\nInterpretation:")
print(f"  {interp}")

# Save JSON
out = {
    "experiment": "MATH",
    "N": N,
    "p_deterministic": p_det,
    "p_internal": p_int,
    "effect_pp": round(effect * 100, 1),
    "mcnemar_chi2": round(chi2_observed, 3),
    "mcnemar_p": round(p_verify, 4),
    "discordant_pairs": {
        "total": int(discordant_total),
        "b_det_only": int(b),
        "c_int_only": int(c)
    },
    "post_hoc_power": round(power, 4),
    "required_N_80pct_power": results.get("80%"),
    "required_N_90pct_power": results.get("90%"),
    "interpretation": interp
}

json_path = "./guided_router/math_power_results.json"
with open(json_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to {json_path}")
