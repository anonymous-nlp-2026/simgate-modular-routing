"""Leave-one-task-out DAF sensitivity analysis.

Removes each of 11 task types in turn, recomputes DAF, outputs table.
DAF = death_asymmetry_episodes / (death_asymmetry + score_comparison).
"""
import json
import os
from collections import defaultdict

DATA_PATH = "./results/baseline_245eps/episode_details.jsonl"
OUT_DIR = "./artifacts/appendix"
DEATH_SCORE = -100.0


def load_episodes():
    episodes = []
    with open(DATA_PATH) as f:
        for line in f:
            ep = json.loads(line)
            if ep["condition"] == "always-deterministic":
                episodes.append(ep)
    return episodes


def classify_episode(ep):
    int_s = ep["internal_final"]
    det_s = ep["det_final"]
    int_died = (int_s == DEATH_SCORE)
    det_died = (det_s == DEATH_SCORE)

    if (int_died and not det_died) or (det_died and not int_died):
        return "death_asymmetry"
    elif int_died and det_died:
        return "both_dead"
    elif det_s != int_s:
        return "score_comparison"
    else:
        return "tied_alive"


def compute_daf(episodes):
    death_asym = sum(1 for ep in episodes if classify_episode(ep) == "death_asymmetry")
    score_comp = sum(1 for ep in episodes if classify_episode(ep) == "score_comparison")
    decisive = death_asym + score_comp
    if decisive == 0:
        return None, death_asym, score_comp
    return death_asym / decisive, death_asym, score_comp


def main():
    episodes = load_episodes()
    print(f"Loaded {len(episodes)} episodes")

    tasks = sorted(set(ep["task_type"] for ep in episodes))
    print(f"Task types ({len(tasks)}): {tasks}")

    task_eps = defaultdict(list)
    for ep in episodes:
        task_eps[ep["task_type"]].append(ep)

    # Full DAF
    full_daf, full_da, full_sc = compute_daf(episodes)
    print(f"\nFull DAF: {full_da}/{full_da + full_sc} = {full_daf:.4f} ({full_daf*100:.1f}%)")

    # Leave-one-out
    results = []
    for task in tasks:
        remaining = [ep for ep in episodes if ep["task_type"] != task]
        n_remaining = len(remaining)
        n_removed = len(task_eps[task])
        daf, da, sc = compute_daf(remaining)
        delta = (daf - full_daf) * 100 if daf is not None else None
        results.append({
            "removed_task": task,
            "n_removed": n_removed,
            "n_remaining": n_remaining,
            "daf": daf,
            "daf_pct": daf * 100 if daf is not None else None,
            "delta_pp": delta,
            "death_asymmetry": da,
            "score_comparison": sc,
        })

    # Sort by absolute delta
    results.sort(key=lambda r: abs(r["delta_pp"]) if r["delta_pp"] is not None else 0, reverse=True)

    # Print table
    print(f"\n{'Removed Task':<48} {'N_rem':>5} {'DA':>4} {'SC':>4} {'DAF(%)':>7} {'Δ(pp)':>7}")
    print("-" * 80)
    print(f"{'(none - full)':<48} {len(episodes):>5} {full_da:>4} {full_sc:>4} {full_daf*100:>7.1f} {'---':>7}")
    for r in results:
        daf_str = f"{r['daf_pct']:.1f}" if r['daf_pct'] is not None else "N/A"
        delta_str = f"{r['delta_pp']:+.1f}" if r['delta_pp'] is not None else "N/A"
        print(f"{r['removed_task']:<48} {r['n_remaining']:>5} {r['death_asymmetry']:>4} {r['score_comparison']:>4} {daf_str:>7} {delta_str:>7}")

    daf_values = [r["daf_pct"] for r in results if r["daf_pct"] is not None]
    print(f"\nDAF ranges from {min(daf_values):.1f}% to {max(daf_values):.1f}% across all leave-one-out conditions")
    print(f"Max |Δ| = {max(abs(r['delta_pp']) for r in results if r['delta_pp'] is not None):.1f} pp")

    # Save JSON
    os.makedirs(OUT_DIR, exist_ok=True)
    output = {
        "description": "Leave-one-task-out DAF sensitivity analysis",
        "daf_definition": "DAF = death_asymmetry_episodes / (death_asymmetry + score_comparison)",
        "data_source": DATA_PATH,
        "full": {
            "n_episodes": len(episodes),
            "death_asymmetry": full_da,
            "score_comparison": full_sc,
            "daf_pct": round(full_daf * 100, 1),
        },
        "leave_one_out": [
            {
                "removed_task": r["removed_task"],
                "n_removed": r["n_removed"],
                "n_remaining": r["n_remaining"],
                "death_asymmetry": r["death_asymmetry"],
                "score_comparison": r["score_comparison"],
                "daf_pct": round(r["daf_pct"], 1) if r["daf_pct"] is not None else None,
                "delta_pp": round(r["delta_pp"], 1) if r["delta_pp"] is not None else None,
            }
            for r in results
        ],
        "summary": {
            "daf_min_pct": round(min(daf_values), 1),
            "daf_max_pct": round(max(daf_values), 1),
            "max_abs_delta_pp": round(max(abs(r["delta_pp"]) for r in results if r["delta_pp"] is not None), 1),
            "range_statement": f"DAF ranges from {min(daf_values):.1f}% to {max(daf_values):.1f}% across all leave-one-out conditions",
        },
    }
    json_path = os.path.join(OUT_DIR, "leave_one_task_out_daf_data.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nJSON saved to {json_path}")

    # Generate LaTeX table
    latex_lines = []
    latex_lines.append(r"\begin{table}[h]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\small")
    latex_lines.append(r"\caption{Leave-one-task-out DAF sensitivity. Removing any single task type changes DAF by at most " + f"{max(abs(r['delta_pp']) for r in results if r['delta_pp'] is not None):.1f}" + r"\,pp.}")
    latex_lines.append(r"\label{tab:loto-daf}")
    latex_lines.append(r"\begin{tabular}{lrrr}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"Removed Task & $N_\text{rem}$ & DAF (\%) & $\Delta$ (pp) \\")
    latex_lines.append(r"\midrule")

    # Full row
    latex_lines.append(f"(none, full) & {len(episodes)} & {full_daf*100:.1f} & --- \\\\")
    latex_lines.append(r"\midrule")

    # Per-task rows sorted by absolute delta
    for r in results:
        task_display = r["removed_task"].replace("-", " ").replace("_", " ")
        # Capitalize first letter of each word, but keep short
        task_display = task_display.title()
        # Shorten long names
        task_display = task_display.replace("Lifespan Longest Lived Then Shortest Lived", "Lifespan Long+Short")
        task_display = task_display.replace("Measure Melting Point Unknown Substance", "Melt Pt Unknown")
        task_display = task_display.replace("Inclined Plane Determine Angle", "Inclined Plane Angle")
        task_display = task_display.replace("Identify Life Stages 1", "Life Stages 1")
        task_display = task_display.replace("Identify Life Stages 2", "Life Stages 2")
        task_display = task_display.replace("Find Non Living Thing", "Find Non-Living")
        task_display = task_display.replace("Lifespan Longest Lived", "Lifespan Longest")
        task_display = task_display.replace("Chemistry Mix", "Chemistry Mix")
        task_display = task_display.replace("Find Animal", "Find Animal")
        task_display = task_display.replace("Grow Fruit", "Grow Fruit")

        daf_str = f"{r['daf_pct']:.1f}" if r["daf_pct"] is not None else "N/A"
        delta_str = f"{r['delta_pp']:+.1f}" if r["delta_pp"] is not None else "N/A"
        latex_lines.append(f"{task_display} & {r['n_remaining']} & {daf_str} & {delta_str} \\\\")

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table}")

    tex_path = os.path.join(OUT_DIR, "leave_one_task_out_daf_table.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(latex_lines) + "\n")
    print(f"LaTeX saved to {tex_path}")


if __name__ == "__main__":
    main()
