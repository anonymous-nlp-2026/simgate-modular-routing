"""Eval plan-008-v4 router checkpoints using same pipeline as seed-robustness-core."""
import json, os, sys, time, argparse
from collections import Counter

import torch
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from eval_routing_metrics import load_val_split, parse_output

DATA_DIR = "./data/plan-004-implicit"

def eval_one(adapter_path, base_model, seed, gpu_id, output_path):
    val_records = load_val_split(DATA_DIR, seed=seed)
    val_lbl = Counter(r["route_label"] for r in val_records)
    print(f"[Eval] adapter={adapter_path}, seed={seed}, gpu={gpu_id}")
    print(f"  Val: {len(val_records)} steps, dist={dict(val_lbl)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16,
        device_map={"": gpu_id}, trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    model.config.pad_token_id = tokenizer.pad_token_id

    y_true, y_pred = [], []
    t0 = time.time()

    for i, rec in enumerate(val_records):
        prompt_messages = rec["messages"][:-1]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=128, temperature=0.1, do_sample=True,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        y_pred.append(parse_output(raw))
        y_true.append(rec["route_label"])

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(val_records)}] ({time.time()-t0:.0f}s)", flush=True)

    del model, base
    torch.cuda.empty_cache()

    labels = ["internal", "deterministic"]
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)

    result = {
        "seed": seed,
        "balanced_acc": round(bal_acc, 4),
        "internal_recall": round(rec[0], 4),
        "internal_precision": round(prec[0], 4),
        "det_recall": round(rec[1], 4),
        "det_precision": round(prec[1], 4),
        "val_size": len(val_records),
    }
    print(f"  balanced_acc={result['balanced_acc']}, internal_recall={result['internal_recall']}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {output_path}")
    return result

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--configs", type=str, required=True, help="JSON array of {adapter, base_model, seed, output}")
    p.add_argument("--gpu_id", type=int, required=True)
    args = p.parse_args()

    configs = json.loads(args.configs)
    for cfg in configs:
        eval_one(cfg["adapter"], cfg["base_model"], cfg["seed"], args.gpu_id, cfg["output"])
    print("\nAll evals done.")
