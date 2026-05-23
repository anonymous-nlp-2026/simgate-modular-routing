"""Task-type death risk configuration from plan-012-full analysis (245 episodes, 11 tasks).

Death risk = internal death rate per task type from counterfactual data.
Tasks with internal_death_rate > DEATH_THRESHOLD are classified as death-risk.
"""

PROJECT_ROOT = "."
COUNTERFACTUAL_DATA = f"{PROJECT_ROOT}/data/scienceworld-counterfactual/episodes.jsonl"
IMPLICIT_SFT_DATA = f"{PROJECT_ROOT}/data/plan-004-implicit/implicit_sft.jsonl"

# Internal death rates from plan-012-full counterfactual analysis
DEATH_RISK_TASKS = {
    "freeze": 0.55,
    "find-animal": 0.50,
    "identify-life-stages-1": 0.55,
    "identify-life-stages-2": 0.50,
    "inclined-plane-determine-angle": 0.55,
    "lifespan-longest-lived": 0.56,
    "lifespan-longest-lived-then-shortest-lived": 0.45,
    "measure-melting-point-unknown-substance": 0.55,
    "grow-fruit": 0.45,
    "chemistry-mix": 0.45,
}

NON_DEATH_RISK_TASKS = {
    "find-non-living-thing": 0.0,
}

# Task with internal_death_rate > this threshold = death-risk
DEATH_THRESHOLD = 0.10

ALL_TASKS = {**DEATH_RISK_TASKS, **NON_DEATH_RISK_TASKS}
