# Generative Router SFT: Finetune Qwen3-1.7B as implicit generative router.
#
# Based on ablation-b's implicit_router_sft.py (the first non-degenerate router,
# internal recall=51.5%). Key difference from plan-002 classification head approach:
# routing signal is embedded in generated text ([SIMULATE]/raw action), not a
# separate classification output. Classification head collapsed to all-deterministic
# on 90:10 imbalanced data; generative approach avoids this via upsample + causal LM.
#
# Input:  implicit_sft.jsonl (chat-format, from format_implicit_data.py)
#         Fields: messages=[system,user,assistant], route_label, task_type, variation, step_idx
# Output: LoRA adapter in --output_dir/final_adapter/
#
# vs implicit_router_sft.py:
#   - Removes class_weights path entirely (upsample-only, proven better)
#   - Default batch_size=4 + grad_accum=8 (effective 32) to fit GPU memory
#   - Adds --max_steps for quick dry-run validation
#   - Adds --upsample_ratio to control minority oversampling target
#   - Computes internal recall/precision during eval (key metric for non-degeneracy)
#   - Explicitly sets model.config.pad_token_id

import argparse
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import re

MODEL_ID = "Qwen/Qwen3-1.7B"
SIMULATE_PREFIX = "[SIMULATE]"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GenerativeSFTDataset(Dataset):
    """Chat-format SFT dataset for causal LM training.

    Each record has messages: [system, user, assistant].
    Tokenizes full conversation; masks system+user tokens (label=-100),
    only supervises assistant response tokens.
    """

    def __init__(self, records: List[Dict], tokenizer, max_seq_len: int = 2048):
        self.records = records
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        messages = rec["messages"]

        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        prompt_messages = messages[:-1]
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

        full_ids = self.tokenizer(
            full_text, truncation=True, max_length=self.max_seq_len,
            return_tensors="pt"
        )["input_ids"].squeeze(0)

        prompt_ids = self.tokenizer(
            prompt_text, truncation=True, max_length=self.max_seq_len,
            return_tensors="pt"
        )["input_ids"].squeeze(0)

        labels = full_ids.clone()
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = -100

        return {
            "input_ids": full_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(full_ids),
        }


class PaddingCollator:
    """Dynamic padding collator for variable-length sequences."""

    def __init__(self, tokenizer, max_seq_len: int = 2048):
        self.pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.max_seq_len = max_seq_len

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        max_len = min(
            max(f["input_ids"].size(0) for f in features),
            self.max_seq_len,
        )

        input_ids = []
        labels = []
        attention_mask = []

        for f in features:
            seq_len = f["input_ids"].size(0)
            pad_len = max_len - seq_len

            if pad_len > 0:
                input_ids.append(torch.cat([
                    f["input_ids"],
                    torch.full((pad_len,), self.pad_id, dtype=torch.long),
                ]))
                labels.append(torch.cat([
                    f["labels"],
                    torch.full((pad_len,), -100, dtype=torch.long),
                ]))
                attention_mask.append(torch.cat([
                    f["attention_mask"],
                    torch.zeros(pad_len, dtype=torch.long),
                ]))
            else:
                input_ids.append(f["input_ids"][:max_len])
                labels.append(f["labels"][:max_len])
                attention_mask.append(f["attention_mask"][:max_len])

        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attention_mask),
        }


# ---------------------------------------------------------------------------
# Data loading with episode-level stratified split + upsampling
# ---------------------------------------------------------------------------

def load_data(
    data_path: str,
    eval_split: float = 0.15,
    seed: int = 42,
    upsample_ratio: float = 1.0,
) -> Tuple[List[Dict], List[Dict]]:
    """Load implicit SFT data with episode-level stratified split and upsampling.

    Args:
        data_path: Path to implicit_sft.jsonl or directory containing it.
        eval_split: Fraction of episodes held out for validation.
        seed: Random seed for reproducibility.
        upsample_ratio: Target ratio of minority:majority after upsampling.
                        1.0 = equal counts, 0.33 = 1:3 ratio.

    Returns:
        (train_records, val_records) with upsampling applied to train only.
    """
    from sklearn.model_selection import train_test_split as sklearn_split

    if os.path.isdir(data_path):
        data_path = os.path.join(data_path, "implicit_sft.jsonl")

    records = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        raise ValueError(f"No records loaded from {data_path}")

    rng = random.Random(seed)

    # Group by episode (task_type, variation)
    episodes: Dict[tuple, List[Dict]] = {}
    for r in records:
        key = (r.get("task_type", "unknown"), r.get("variation", 0))
        episodes.setdefault(key, []).append(r)

    ep_keys = list(episodes.keys())
    ep_labels = []
    for key in ep_keys:
        lbls = Counter(r["route_label"] for r in episodes[key])
        ep_labels.append(lbls.most_common(1)[0][0])

    train_keys, val_keys = sklearn_split(
        ep_keys, test_size=eval_split, random_state=seed, stratify=ep_labels
    )

    train_records = [r for k in train_keys for r in episodes[k]]
    val_records = [r for k in val_keys for r in episodes[k]]

    rng.shuffle(train_records)
    rng.shuffle(val_records)

    train_lbl = Counter(r["route_label"] for r in train_records)
    val_lbl = Counter(r["route_label"] for r in val_records)
    print(f"Episode-level stratified split: {len(train_keys)} train / {len(val_keys)} val episodes")
    print(f"  -> {len(train_records)} train steps, {len(val_records)} val steps")
    print(f"  Train route labels: {dict(train_lbl)}")
    print(f"  Val route labels:   {dict(val_lbl)}")

    # Upsample minority class in train set
    # "internal" is always the minority in our 90:10 data
    majority_count = max(train_lbl.values())
    minority_label = min(train_lbl, key=train_lbl.get)
    minority_count = train_lbl[minority_label]
    target_minority = int(majority_count * upsample_ratio)

    if target_minority > minority_count:
        minority_records = [r for r in train_records if r["route_label"] == minority_label]
        n_extra = target_minority - minority_count
        upsampled = rng.choices(minority_records, k=n_extra)
        train_records.extend(upsampled)
        rng.shuffle(train_records)
        new_lbl = Counter(r["route_label"] for r in train_records)
        print(f"  Upsampled '{minority_label}': {minority_count} -> {new_lbl[minority_label]} "
              f"(target ratio {upsample_ratio:.2f})")
        print(f"  Train after upsample: {dict(new_lbl)}")

    return train_records, val_records


# ---------------------------------------------------------------------------
# Eval metrics: internal recall is the key metric for non-degeneracy
# ---------------------------------------------------------------------------

def make_eval_fn(tokenizer):
    """Create eval function that checks if model predicts [SIMULATE] vs raw action."""
    simulate_token_ids = tokenizer.encode(SIMULATE_PREFIX, add_special_tokens=False)

    def compute_metrics(eval_pred):
        logits_arr, labels_arr = eval_pred
        logits = torch.from_numpy(logits_arr)
        labels = torch.from_numpy(labels_arr)

        # For each sample, find first supervised token and check prediction
        correct = 0
        total = 0
        internal_correct = 0
        internal_total = 0
        det_correct = 0
        det_total = 0

        for i in range(labels.shape[0]):
            mask = labels[i] != -100
            if not mask.any():
                continue
            first_sup = mask.nonzero(as_tuple=True)[0][0].item()

            # Ground truth: does the label start with [SIMULATE] tokens?
            gt_starts_simulate = True
            for j, tid in enumerate(simulate_token_ids):
                pos = first_sup + j
                if pos >= labels.shape[1] or labels[i, pos].item() != tid:
                    gt_starts_simulate = False
                    break

            # Prediction: does the model predict [SIMULATE] tokens?
            pred_starts_simulate = True
            for j, tid in enumerate(simulate_token_ids):
                # logits at position (first_sup + j - 1) predicts token at (first_sup + j)
                pred_pos = first_sup + j - 1
                if pred_pos < 0 or pred_pos >= logits.shape[1]:
                    pred_starts_simulate = False
                    break
                pred_token = logits[i, pred_pos].argmax().item()
                if pred_token != tid:
                    pred_starts_simulate = False
                    break

            total += 1
            if gt_starts_simulate == pred_starts_simulate:
                correct += 1

            if gt_starts_simulate:
                det_total += 1
                if pred_starts_simulate:
                    det_correct += 1
            else:
                internal_total += 1
                if not pred_starts_simulate:
                    internal_correct += 1

        acc = correct / max(total, 1)
        internal_recall = internal_correct / max(internal_total, 1)
        det_recall = det_correct / max(det_total, 1)

        return {
            "routing_acc": acc,
            "internal_recall": internal_recall,
            "det_recall": det_recall,
            "internal_n": internal_total,
            "det_n": det_total,
        }

    return compute_metrics


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    """Finetune Qwen3-1.7B with LoRA on implicit SFT data (generative router)."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_records, val_records = load_data(
        args.data_dir,
        eval_split=args.eval_split,
        seed=args.seed,
        upsample_ratio=args.upsample_ratio,
    )
    print(f"Train: {len(train_records)}, Val: {len(val_records)}")

    route_dist = Counter(r["route_label"] for r in train_records)
    print(f"Route distribution (train): {dict(route_dist)}")

    # WeightedRandomSampler for additional epoch-level balancing
    weighted_sampler = None
    if args.upsample_minority and len(route_dist) > 1:
        sample_weights = [1.0 / route_dist[r["route_label"]] for r in train_records]
        weighted_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_records),
            replacement=True,
        )
        print(f"WeightedRandomSampler enabled: {len(train_records)} draws/epoch")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": int(os.environ.get("LOCAL_RANK", args.gpu_id))},
        trust_remote_code=True,
    )
    # Avoid warning: model.config.pad_token_id must be set for generation
    model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = GenerativeSFTDataset(train_records, tokenizer, args.max_seq_len)
    val_dataset = GenerativeSFTDataset(val_records, tokenizer, args.max_seq_len)
    collator = PaddingCollator(tokenizer, args.max_seq_len)

    os.makedirs(args.output_dir, exist_ok=True)

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=use_bf16,
        fp16=(not use_bf16 and torch.cuda.is_available()),
        logging_steps=10,
        report_to="wandb" if not os.environ.get("WANDB_MODE") == "offline" else "none",
        run_name=f"gen-router-lr{args.lr}-bs{args.batch_size}x{args.gradient_accumulation_steps}",
        seed=args.seed,
        save_total_limit=2,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
    )

    os.environ.setdefault("WANDB_PROJECT", "simgate-generative-router")

    class SamplerTrainer(Trainer):
        """Trainer that optionally uses WeightedRandomSampler."""

        def __init__(self, weighted_sampler=None, collator_ref=None, **kwargs):
            super().__init__(**kwargs)
            self._weighted_sampler = weighted_sampler
            self._collator_ref = collator_ref

        def get_train_dataloader(self) -> DataLoader:
            if self._weighted_sampler is None:
                return super().get_train_dataloader()
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                sampler=self._weighted_sampler,
                collate_fn=self._collator_ref or self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

    trainer = SamplerTrainer(
        weighted_sampler=weighted_sampler,
        collator_ref=collator,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    print(f"\n{'='*60}")
    print(f"Starting training: {len(train_dataset)} train / {len(val_dataset)} val samples")
    print(f"Model: {args.model_name}")
    print(f"LoRA r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"lr={args.lr}, epochs={args.epochs}, effective_batch={args.batch_size * args.gradient_accumulation_steps}")
    print(f"Upsample ratio: {args.upsample_ratio}, WeightedSampler: {weighted_sampler is not None}")
    if args.max_steps > 0:
        print(f"max_steps={args.max_steps} (dry-run mode)")
    print(f"{'='*60}\n")

    trainer.train()

    adapter_dir = os.path.join(args.output_dir, "final_adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Adapter saved to {adapter_dir}")

    config_path = os.path.join(args.output_dir, "train_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    sentinel_path = os.path.join(args.output_dir, "TRAINING_COMPLETE")
    with open(sentinel_path, "w") as f:
        f.write(f"completed_at={datetime.now().isoformat()}\n")
        f.write(f"output_dir={args.output_dir}\n")
        f.write(f"train_samples={len(train_dataset)}\n")
        f.write(f"val_samples={len(val_dataset)}\n")
    print(f"Sentinel written: {sentinel_path}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def parse_output(text: str) -> Tuple[str, str]:
    """Parse model output into (action, route_decision).

    Starts with [SIMULATE] -> route = "deterministic", action = remainder.
    Otherwise -> route = "internal", action = full text.
    """
    text = text.strip()
    if text.startswith(SIMULATE_PREFIX):
        action = text[len(SIMULATE_PREFIX):].strip()
        return action, "deterministic"
    return text, "internal"


class GenerativeRouter:
    """Inference wrapper: loads finetuned adapter, generates actions with routing."""

    def __init__(self, adapter_path: str, base_model: str = MODEL_ID, gpu_id: int = 0):
        self.tokenizer = AutoTokenizer.from_pretrained(
            adapter_path, trust_remote_code=True
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map={"": gpu_id},
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.model.eval()
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from training.format_implicit_data import IMPLICIT_SYSTEM_PROMPT, format_action_history
        self._system_prompt = IMPLICIT_SYSTEM_PROMPT
        self._format_history = format_action_history

    def __call__(
        self,
        observation: str,
        valid_actions: List[str],
        history: List[dict],
    ) -> Tuple[str, str]:
        """RouterFn interface: (obs, valid_actions, history) -> (action, route)."""
        hist_strs = [h.get("action", str(h)) for h in history]
        user_content = (
            f"Observation: {observation}\n"
            f"Action history: {self._format_history(hist_strs)}"
        )

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=64, temperature=0.1, do_sample=True
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        raw_output = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        raw_output = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()
        action, route = parse_output(raw_output)
        if not action:
            action = valid_actions[0] if valid_actions else "look around"

        return action, route


# ---------------------------------------------------------------------------
# Evaluation entry point
# ---------------------------------------------------------------------------

def evaluate(args):
    """Run evaluation using ablation_eval_common."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from ablation_eval_common import run_ablation_eval, config_from_args, add_eval_args

    router = GenerativeRouter(
        adapter_path=args.adapter_path,
        base_model=args.model_name,
        gpu_id=args.gpu_id,
    )
    config = config_from_args(args, condition_name="generative_router")
    results = run_ablation_eval(config, router_fn=router)

    print(f"\n=== Generative Router Results ===")
    print(f"Mean score: {results['episode_metrics']['mean_score']:.2f}")
    print(f"Death rate: {results['episode_metrics']['death_rate']:.1%}")
    print(f"Routing accuracy: {results['routing_accuracy']['accuracy']:.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generative Router SFT: finetune LLM to output [SIMULATE]/raw action tokens"
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- train ---
    train_p = subparsers.add_parser("train")
    train_p.add_argument("--data_dir", type=str, required=True,
                         help="Path to implicit_sft.jsonl or directory containing it")
    train_p.add_argument("--model_name", type=str, default=MODEL_ID)
    train_p.add_argument("--output_dir", type=str, default="checkpoints/plan-002-generative")
    train_p.add_argument("--epochs", type=int, default=3)
    train_p.add_argument("--batch_size", type=int, default=4)
    train_p.add_argument("--lr", type=float, default=2e-5)
    train_p.add_argument("--lora_r", type=int, default=16)
    train_p.add_argument("--lora_alpha", type=int, default=32)
    train_p.add_argument("--lora_dropout", type=float, default=0.05)
    train_p.add_argument("--max_seq_len", type=int, default=2048)
    train_p.add_argument("--eval_split", type=float, default=0.15)
    train_p.add_argument("--gpu_id", type=int, default=1)
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument("--upsample_minority", action="store_true", default=True,
                         help="Use WeightedRandomSampler (default: enabled)")
    train_p.add_argument("--no_upsample_minority", dest="upsample_minority", action="store_false")
    train_p.add_argument("--upsample_ratio", type=float, default=1.0,
                         help="Target minority:majority ratio after upsampling (default: 1.0 = balanced)")
    train_p.add_argument("--gradient_accumulation_steps", type=int, default=8,
                         help="Gradient accumulation steps (effective batch = batch_size * this)")
    train_p.add_argument("--max_steps", type=int, default=0,
                         help="Max training steps (0 = use epochs, >0 for dry-run)")

    # --- eval ---
    eval_p = subparsers.add_parser("eval")
    eval_p.add_argument("--adapter_path", type=str, required=True)
    eval_p.add_argument("--model_name", type=str, default=MODEL_ID)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    try:
        from ablation_eval_common import add_eval_args
        add_eval_args(eval_p)
    except ImportError:
        eval_p.add_argument("--task_types", type=str, default="boil,melt,change-the-state-of-matter-of")
        eval_p.add_argument("--max_variations", type=int, default=5)
        eval_p.add_argument("--max_steps", type=int, default=30)
        eval_p.add_argument("--gpu_id", type=int, default=0)
        eval_p.add_argument("--output_dir", type=str, default="results/ablation/")
        eval_p.add_argument("--oracle_data", type=str,
                            default="results/sft_data_death_aware/sft_triples.jsonl")

    args = parser.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "eval":
        evaluate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
