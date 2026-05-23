# Counterfactual Routing Signal Decomposition for LLM Agents

Code for the paper *"Death Avoidance as the Dominant Routing Signal: A Counterfactual Analysis on ScienceWorld"*.

This framework investigates **where routing signal comes from** when LLM agents choose between deterministic (tool-use) and internal (generative) reasoning strategies. Using episode-level counterfactual evaluation in ScienceWorld, we decompose the routing signal and find that **78.9% of non-tie routing labels arise from death asymmetry** — a finding replicated across three architecturally distinct 7–9B model families (Qwen2.5-7B, Llama-3.1-8B, Gemma-2-9B). Cross-environment probes on ALFWorld and MATH diagnose qualitatively different routing mechanisms.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (tested on A6000)
- [ScienceWorld](https://github.com/allenai/ScienceWorld) environment

### Installation

```bash
pip install -r requirements.txt
pip install scienceworld
```

Key dependencies: `transformers`, `peft`, `torch`, `scienceworld`, `numpy`, `scikit-learn`

## Project Structure

```
src/                          # Core source code
  counterfactual_labeling.py  # Episode-level counterfactual data collection
  death_aware_labeling.py     # Death-aware 3-label classification
  episode_level_collection.py # Episode-level trajectory collection
  eval_end_to_end.py          # End-to-end evaluation (5 routing conditions)
  llm_agent.py                # LLM agent (ReAct) for ScienceWorld
  scienceworld_utils.py       # ScienceWorld environment utilities
  training/
    router_sft.py             # Router SFT training (LoRA, classification head)
    generative_router_sft.py  # Generative router variant
    data_loader.py            # Data loading utilities

guided_router/                # Decomposition-guided router
  router.py                   # Task-type death risk routing
  config.py                   # Death risk task configuration
  
scripts/                      # Experiment scripts and analyses
paper/                        # Paper source (LaTeX)
```

## Running Experiments

### 1. Trajectory Collection

Collect counterfactual trajectories by running both strategies per episode:

```bash
python src/episode_level_collection.py \
  --task_types "boil,melt,freeze,chemistry-mix,grow-fruit,find-animal" \
  --episodes 25 --gpu_id 0
```

### 2. Death-Aware Labeling

Generate training labels from collected episodes:

```bash
python src/death_aware_labeling.py \
  --input_dir results/plan_c \
  --output results/sft_triples.jsonl
```

### 3. Router Training

Fine-tune a routing classifier:

```bash
python src/router_sft_train.py \
  --data_path results/sft_triples.jsonl \
  --model_name Qwen/Qwen3-1.7B \
  --output_dir checkpoints/router \
  --epochs 3 --lora_r 16 --gpu_id 0
```

### 4. End-to-End Evaluation

Evaluate routing strategies (always-internal, always-deterministic, oracle, random, trained-router):

```bash
python src/eval_end_to_end.py \
  --data_dir results/plan_c \
  --router_checkpoint checkpoints/router \
  --output_dir results/eval
```

## Datasets

- **ScienceWorld**: Install via `pip install scienceworld`. 11 task types with routing signal (out of 30 total).
- **ALFWorld**: Follow setup at [alfworld repo](https://github.com/alfworld/alfworld). Cross-environment validation in `src/alfworld_counterfactual.py`.
- **MATH**: Uses problems from [MATH dataset](https://github.com/hendrycks/math). Extension analysis in `src/math_counterfactual.py`.

## License

This code is released for research purposes.
