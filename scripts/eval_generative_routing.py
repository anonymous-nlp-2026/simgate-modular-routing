"""Evaluate generative router on val split: routing accuracy metrics."""
import json, os, sys, random, time
from collections import Counter
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

SIMULATE_PREFIX = "[SIMULATE]"

def log(msg):
    print(msg, flush=True)

def load_and_split(data_dir, eval_split=0.15, seed=42):
    path = data_dir if data_dir.endswith(".jsonl") else os.path.join(data_dir, "implicit_sft.jsonl")
    with open(path) as f:
        records = [json.loads(l) for l in f if l.strip()]
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    n_val = int(len(records) * eval_split)
    val = [records[i] for i in indices[:n_val]]
    return val

def parse_output(text):
    """Extract route from generated text, handling Qwen3 <think> blocks."""
    text = text.strip()
    # Strip thinking block if present
    if "<think>" in text:
        think_end = text.find("</think>")
        if think_end != -1:
            text = text[think_end + len("</think>"):].strip()
        else:
            # Thinking not closed = truncated, treat remaining as the output
            text = text[text.find("<think>") + len("<think>"):].strip()
    if text.startswith(SIMULATE_PREFIX):
        return "deterministic", text
    return "internal", text

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--adapter_path", required=True)
    p.add_argument("--model_name", default="./models/Qwen/Qwen3-1.7B")
    p.add_argument("--data_dir", default="data/plan-004-implicit/")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--eval_split", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    log(f"Loading val split (seed={args.seed}, eval_split={args.eval_split})...")
    val_records = load_and_split(args.data_dir, args.eval_split, args.seed)
    log(f"Val samples: {len(val_records)}")
    gt_dist = Counter(r["route_label"] for r in val_records)
    log(f"Ground truth distribution: {dict(gt_dist)}")

    log(f"Loading model + adapter from {args.adapter_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16,
        device_map={"": args.gpu_id}, trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter_path)
    model.eval()
    model.config.pad_token_id = tokenizer.pad_token_id
    log("Model loaded.")

    # Inference
    preds = []
    gts = []
    pred_texts = []
    t0 = time.time()

    for i, rec in enumerate(val_records):
        messages = rec["messages"]
        gt_label = rec["route_label"]
        prompt_msgs = messages[:-1]

        prompt = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False)

        pred_route, clean_text = parse_output(gen_text)
        preds.append(pred_route)
        gts.append(gt_label)
        pred_texts.append(gen_text[:200])

        if i == 0:
            log(f"  Sample 0: gt={gt_label}, pred={pred_route}")
            log(f"  Raw output: {gen_text[:200]}")

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(val_records) - i - 1) / rate
            log(f"  [{i+1}/{len(val_records)}] {rate:.1f} samples/s, ETA {eta:.0f}s")

    elapsed = time.time() - t0
    log(f"Inference done: {len(val_records)} samples in {elapsed:.1f}s ({len(val_records)/elapsed:.1f} samples/s)")

    # Metrics
    labels = ["deterministic", "internal"]
    tp = {l: 0 for l in labels}
    fp = {l: 0 for l in labels}
    fn = {l: 0 for l in labels}
    correct = 0
    cm = {g: {p: 0 for p in labels} for g in labels}

    for gt, pred in zip(gts, preds):
        cm[gt][pred] += 1
        if gt == pred:
            correct += 1
            tp[gt] += 1
        else:
            fp[pred] += 1
            fn[gt] += 1

    total = len(gts)
    accuracy = correct / max(total, 1)

    precision = {}
    recall = {}
    f1 = {}
    for l in labels:
        precision[l] = tp[l] / max(tp[l] + fp[l], 1)
        recall[l] = tp[l] / max(tp[l] + fn[l], 1)
        if precision[l] + recall[l] > 0:
            f1[l] = 2 * precision[l] * recall[l] / (precision[l] + recall[l])
        else:
            f1[l] = 0.0

    f1_macro = np.mean(list(f1.values()))
    bal_acc = np.mean([recall[l] for l in labels])

    pred_dist = Counter(preds)
    is_degenerate = len(pred_dist) == 1

    log(f"\n{'='*60}")
    log(f"GENERATIVE ROUTER EVAL (plan-002-generative-v1)")
    log(f"{'='*60}")
    log(f"Total samples:       {total}")
    log(f"GT distribution:     {dict(gt_dist)}")
    log(f"Pred distribution:   {dict(pred_dist)}")
    log(f"Is degenerate:       {is_degenerate}")
    log(f"\neval_accuracy:       {accuracy:.4f}")
    log(f"balanced_accuracy:   {bal_acc:.4f}")
    log(f"f1_macro:            {f1_macro:.4f}")
    log(f"\ninternal_recall:     {recall['internal']:.4f}")
    log(f"internal_precision:  {precision['internal']:.4f}")
    log(f"internal_f1:         {f1['internal']:.4f}")
    log(f"\ndet_recall:          {recall['deterministic']:.4f}")
    log(f"det_precision:       {precision['deterministic']:.4f}")
    log(f"det_f1:              {f1['deterministic']:.4f}")
    log(f"\nConfusion Matrix (rows=GT, cols=Pred):")
    log(f"{'':>15} {'det_pred':>10} {'int_pred':>10}")
    for g in labels:
        log(f"{g:>15} {cm[g]['deterministic']:>10} {cm[g]['internal']:>10}")

    log(f"\n--- Sample predictions (first 10) ---")
    for i in range(min(10, total)):
        marker = "OK" if gts[i] == preds[i] else "WRONG"
        log(f"  [{marker}] gt={gts[i]:>15}, pred={preds[i]:>15} | {pred_texts[i][:80]}")

    results = {
        "eval_accuracy": round(accuracy, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "f1_macro": round(f1_macro, 4),
        "internal_recall": round(recall["internal"], 4),
        "internal_precision": round(precision["internal"], 4),
        "internal_f1": round(f1["internal"], 4),
        "det_recall": round(recall["deterministic"], 4),
        "det_precision": round(precision["deterministic"], 4),
        "det_f1": round(f1["deterministic"], 4),
        "is_degenerate": is_degenerate,
        "n_val": total,
        "gt_distribution": dict(gt_dist),
        "pred_distribution": dict(pred_dist),
        "confusion_matrix": cm,
    }
    out_path = os.path.join(os.path.dirname(args.adapter_path), "eval_routing_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to {out_path}")
    log(f"\nJSON: {json.dumps(results, indent=2)}")

if __name__ == "__main__":
    main()
