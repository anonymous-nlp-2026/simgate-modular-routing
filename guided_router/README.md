# Decomposition-Guided Router

Task-type death risk routing based on the 78.9% label fraction finding (DAF 88–90%).

## Core Insight

78.9% of non-tie routing labels arise from death asymmetry. Death risk is a
task-type level binary property. We route entire episodes based on
whether the task type has high internal death rate.

## Variants

- **oracle**: Uses ground-truth death risk labels from plan-012-full counterfactual data
- **learned**: Infers death risk from observation keywords (no ground truth needed)

## Usage

```bash
cd .

# Dry run (verify imports)
python -m guided_router.run_guided_router --dry_run

# Oracle evaluation
python -m guided_router.run_guided_router --variant oracle

# Learned evaluation
python -m guided_router.run_guided_router --variant learned --verbose
```

## Data

Input: `data/scienceworld-counterfactual/episodes.jsonl` (245 episodes, 11 tasks)

## Baselines

| Method | Episode bal_acc |
|--------|----------------|
| always-deterministic | 50.0% |
| trained 1.7B router | 47.1% |
| trained 4B router | 58.96% |
| **guided oracle** | TBD |
| **guided learned** | TBD |
