import json, os, glob

DATA_DIR = "./results/plan_c"
OUT_DIR = "./results"
DEATH_SCORE = -100.0
PENALTIES = [10, 25, 50, 100, 200]

task_dirs = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])

# Collect per-task stats
task_stats = {}
global_counts = {"n_only_int_dies": 0, "n_only_det_dies": 0, "n_both_die": 0, "n_neither_die": 0,
                 "total_episodes": 0, "int_score_sum_nd": 0.0, "det_score_sum_nd": 0.0}

for task in task_dirs:
    fp = os.path.join(DATA_DIR, task, "episodes.jsonl")
    if not os.path.exists(fp):
        continue
    
    ts = {"n_only_int_dies": 0, "n_only_det_dies": 0, "n_both_die": 0, "n_neither_die": 0,
          "n_episodes": 0, "int_score_sum_nd": 0.0, "det_score_sum_nd": 0.0,
          "int_scores": [], "det_scores": []}
    
    with open(fp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ep = json.loads(line)
            int_score = ep["always_internal"]["final_score"]
            det_score = ep["always_deterministic"]["final_score"]
            int_dead = (int_score == DEATH_SCORE)
            det_dead = (det_score == DEATH_SCORE)
            
            ts["n_episodes"] += 1
            ts["int_scores"].append(int_score)
            ts["det_scores"].append(det_score)
            
            if int_dead and det_dead:
                ts["n_both_die"] += 1
            elif int_dead and not det_dead:
                ts["n_only_int_dies"] += 1
            elif not int_dead and det_dead:
                ts["n_only_det_dies"] += 1
            else:
                ts["n_neither_die"] += 1
            
            # Non-death scores: replace death with 0 to isolate non-death component
            if not int_dead:
                ts["int_score_sum_nd"] += int_score
            if not det_dead:
                ts["det_score_sum_nd"] += det_score
    
    task_stats[task] = ts
    for k in ["n_only_int_dies", "n_only_det_dies", "n_both_die", "n_neither_die"]:
        global_counts[k] += ts[k]
    global_counts["total_episodes"] += ts["n_episodes"]
    global_counts["int_score_sum_nd"] += ts["int_score_sum_nd"]
    global_counts["det_score_sum_nd"] += ts["det_score_sum_nd"]

N = global_counts["total_episodes"]
print(f"Total episodes: {N}")
print(f"n_only_int_dies: {global_counts['n_only_int_dies']}")
print(f"n_only_det_dies: {global_counts['n_only_det_dies']}")
print(f"n_both_die: {global_counts['n_both_die']}")
print(f"n_neither_die: {global_counts['n_neither_die']}")

# delta = (n_only_int_dies - n_only_det_dies) / N
delta = (global_counts["n_only_int_dies"] - global_counts["n_only_det_dies"]) / N
print(f"\ndelta (death rate differential): {delta:.4f}")

# Compute Δ_full at P=100 from raw data
int_mean = sum(sum(task_stats[t]["int_scores"]) for t in task_stats) / N
det_mean = sum(sum(task_stats[t]["det_scores"]) for t in task_stats) / N
delta_full_100 = det_mean - int_mean
print(f"Δ_full (P=100, from data): {delta_full_100:.2f}")

# Δ_nd: non-death component
# For non-death episodes, the score gap is the same regardless of penalty.
# Δ_nd = (1/N) * [ Σ(det_score for non-death det) - Σ(int_score for non-death int) ]
# But we need to be careful: death episodes contribute 0 to non-death component
# Actually: Δ_full(P) = δ*P + Δ_nd
# So Δ_nd = Δ_full(100) - δ*100
delta_nd = delta_full_100 - delta * 100
print(f"Δ_nd: {delta_nd:.2f}")
print(f"Verification: δ*100 + Δ_nd = {delta*100 + delta_nd:.2f} (should be {delta_full_100:.2f})")

# Overall sensitivity table
print("\n=== Overall Penalty Sensitivity ===")
print(f"{'P':>5} | {'Δ_full(P)':>10} | {'Δ_nd':>6} | {'DAF(P)':>8}")
print("-" * 40)

overall_table = []
for P in PENALTIES:
    df = delta * P + delta_nd
    daf = 1 - delta_nd / df if df != 0 else 0
    print(f"{P:>5} | {df:>10.2f} | {delta_nd:>6.2f} | {daf*100:>7.1f}%")
    overall_table.append({"P": P, "delta_full": round(df, 2), "delta_nd": round(delta_nd, 2), "DAF": round(daf * 100, 1)})

# Per-task sensitivity
print("\n=== Per-task Sensitivity ===")
print(f"{'Task':<45} | {'n':>3} | {'δ_i':>7} | {'Δ_nd_i':>7} | {'DAF(100)':>8} | {'DAF(50)':>8}")
print("-" * 95)

per_task_table = []
for task in task_dirs:
    if task not in task_stats:
        continue
    ts = task_stats[task]
    n_i = ts["n_episodes"]
    delta_i = (ts["n_only_int_dies"] - ts["n_only_det_dies"]) / n_i
    
    int_mean_i = sum(ts["int_scores"]) / n_i
    det_mean_i = sum(ts["det_scores"]) / n_i
    delta_full_i_100 = det_mean_i - int_mean_i
    delta_nd_i = delta_full_i_100 - delta_i * 100
    
    row = {"task": task, "n": n_i, "delta_i": round(delta_i, 4),
           "n_only_int_dies": ts["n_only_int_dies"], "n_only_det_dies": ts["n_only_det_dies"],
           "n_both_die": ts["n_both_die"], "n_neither_die": ts["n_neither_die"],
           "delta_nd_i": round(delta_nd_i, 2)}
    
    for P in PENALTIES:
        df_p = delta_i * P + delta_nd_i
        daf_p = (1 - delta_nd_i / df_p) * 100 if df_p != 0 else 0
        row[f"DAF_{P}"] = round(daf_p, 1)
        row[f"delta_full_{P}"] = round(df_p, 2)
    
    per_task_table.append(row)
    print(f"{task:<45} | {n_i:>3} | {delta_i:>7.4f} | {delta_nd_i:>7.2f} | {row['DAF_100']:>7.1f}% | {row['DAF_50']:>7.1f}%")

# Save JSON
results = {
    "global": {
        "N": N,
        "n_only_int_dies": global_counts["n_only_int_dies"],
        "n_only_det_dies": global_counts["n_only_det_dies"],
        "n_both_die": global_counts["n_both_die"],
        "n_neither_die": global_counts["n_neither_die"],
        "delta": round(delta, 4),
        "delta_nd": round(delta_nd, 2),
        "delta_full_100": round(delta_full_100, 2),
    },
    "overall_table": overall_table,
    "per_task_table": per_task_table,
}

with open(os.path.join(OUT_DIR, "penalty_sensitivity.json"), "w") as f:
    json.dump(results, f, indent=2)

# Generate LaTeX
latex_lines = []
latex_lines.append(r"% === Table A: Overall Penalty Sensitivity ===")
latex_lines.append(r"\begin{table}[t]")
latex_lines.append(r"\centering")
latex_lines.append(r"\caption{Sensitivity of $\Delta_\text{full}$ and DAF to death penalty magnitude $P$.}")
latex_lines.append(r"\label{tab:penalty-sensitivity}")
latex_lines.append(r"\begin{tabular}{rrrr}")
latex_lines.append(r"\toprule")
latex_lines.append(r"$P$ & $\Delta_\text{full}(P)$ & $\Delta_\text{nd}$ & DAF($P$) \\")
latex_lines.append(r"\midrule")
for row in overall_table:
    marker = r" \textbf{*}" if row["P"] == 100 else ""
    latex_lines.append(f"{row['P']} & {row['delta_full']:.2f} & {row['delta_nd']:.2f} & {row['DAF']:.1f}\\%{marker} \\\\")
latex_lines.append(r"\bottomrule")
latex_lines.append(r"\end{tabular}")
latex_lines.append(r"\vspace{0.5em}")
latex_lines.append(r"\parbox{0.9\linewidth}{\small\textit{Note:} $\Delta_\text{full}(P) = \delta \cdot P + \Delta_\text{nd}$, where $\delta = " + f"{delta:.4f}" + r"$ is the death-rate differential and $\Delta_\text{nd} = " + f"{delta_nd:.2f}" + r"$ is the penalty-independent score gap. * denotes the default setting used in all experiments.}")
latex_lines.append(r"\end{table}")
latex_lines.append("")
latex_lines.append(r"% === Table B: Per-task Sensitivity (P=100 vs P=50) ===")
latex_lines.append(r"\begin{table}[t]")
latex_lines.append(r"\centering")
latex_lines.append(r"\caption{Per-task penalty sensitivity analysis.}")
latex_lines.append(r"\label{tab:per-task-penalty}")
latex_lines.append(r"\begin{tabular}{lrrrr}")
latex_lines.append(r"\toprule")
latex_lines.append(r"Task & $n$ & $\delta_i$ & DAF(100) & DAF(50) \\")
latex_lines.append(r"\midrule")
for row in per_task_table:
    task_short = row["task"].replace("-", " ").replace("_", " ")
    if len(task_short) > 30:
        task_short = task_short[:27] + "..."
    latex_lines.append(f"{task_short} & {row['n']} & {row['delta_i']:.4f} & {row['DAF_100']:.1f}\\% & {row['DAF_50']:.1f}\\% \\\\")
latex_lines.append(r"\midrule")
latex_lines.append(f"Overall & {N} & {delta:.4f} & {overall_table[3]['DAF']:.1f}\\% & {overall_table[2]['DAF']:.1f}\\% \\\\")
latex_lines.append(r"\bottomrule")
latex_lines.append(r"\end{tabular}")
latex_lines.append(r"\end{table}")

latex_content = "\n".join(latex_lines)
with open(os.path.join(OUT_DIR, "penalty_sensitivity_table.tex"), "w") as f:
    f.write(latex_content)

print("\n\nFiles saved:")
print(f"  {os.path.join(OUT_DIR, 'penalty_sensitivity.json')}")
print(f"  {os.path.join(OUT_DIR, 'penalty_sensitivity_table.tex')}")
