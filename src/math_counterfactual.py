"""MATH counterfactual labeling pipeline.

For each MATH problem, runs two trajectories:
1. Pure Internal: LLM reasoning only, no code execution
2. Code Interpreter: LLM + Python code sandbox (ReAct-style)

Produces episode-level labels:
- det-preferred: code interpreter correct + internal wrong
- internal-preferred: internal correct + code interpreter wrong
- no-preference: both correct or both wrong

Input:  MATH dataset (Level 4-5), loaded from data/math-raw or HuggingFace
Output: data/math-counterfactual/episodes.jsonl
"""

import argparse
import json
import os
import re
import sys
import time
from typing import List, Dict, Any, Optional, Tuple
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from code_sandbox import execute_code
from math_answer_utils import extract_boxed_answer, answers_equal

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

INTERNAL_SYSTEM = (
    "Solve this math problem step by step. "
    "Show your reasoning clearly. "
    "Give your final answer as \\boxed{answer}."
)

CODE_INTERPRETER_SYSTEM = (
    "You MUST solve this problem by writing and executing Python code. "
    "Show your code in a ```python\n...\n``` block, execute it, and report the result. "
    "Do not use mental math or reasoning alone. "
    "Every answer must come from code execution output. "
    "Give your final answer as \\boxed{answer}."
)


def extract_code_blocks(text: str) -> List[str]:
    """Extract Python code blocks from LLM output."""
    blocks = re.findall(r'```python\s*\n(.*?)```', text, re.DOTALL)
    if not blocks:
        blocks = re.findall(r'```\s*\n(.*?)```', text, re.DOTALL)
    return blocks


# ---------------------------------------------------------------------------
# LLM wrapper
# ---------------------------------------------------------------------------

class MathLLM:
    """LLM wrapper for MATH inference. Uses vLLM OpenAI-compatible API."""

    def __init__(self, model_path: str, api_url: str = "http://localhost:8000/v1",
                 use_vllm: bool = True, gpu_id: int = 0):
        self.model_path = model_path
        self.api_url = api_url
        self.use_vllm = use_vllm
        self.gpu_id = gpu_id
        self._llm = None
        self._tokenizer = None
        self._model = None
        self.max_new_tokens = 2048

    def _load_vllm(self):
        if self._llm is not None:
            return
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        from vllm import LLM
        self._llm = LLM(
            model=self.model_path,
            trust_remote_code=True,
            gpu_memory_utilization=0.85,
            max_model_len=4096,
        )
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )

    def _load_hf(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=f"cuda:{self.gpu_id}",
            trust_remote_code=True,
        )

    def generate(self, messages: List[dict], temperature: float = 0.0) -> str:
        if self.api_url:
            return self._generate_api(messages, temperature)
        if self.use_vllm:
            return self._generate_vllm(messages, temperature)
        return self._generate_hf(messages, temperature)

    def _generate_api(self, messages: List[dict], temperature: float) -> str:
        """Generate via vLLM OpenAI-compatible API."""
        payload = {
            "model": self.model_path,
            "messages": messages,
            "temperature": max(temperature, 0.01),
            "max_tokens": self.max_new_tokens,
        }
        for attempt in range(3):
            try:
                resp = requests.post(
                    f"{self.api_url}/chat/completions",
                    json=payload,
                    timeout=120,
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()
            except requests.exceptions.HTTPError as e:
                if attempt == 2:
                    raise
                print(f"  [API retry {attempt+1}/3] {e}")
                time.sleep(2)

    def _generate_vllm(self, messages: List[dict], temperature: float) -> str:
        self._load_vllm()
        from vllm import SamplingParams
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        params = SamplingParams(
            temperature=max(temperature, 0.01),
            max_tokens=self.max_new_tokens,
        )
        outputs = self._llm.generate([prompt], params)
        return outputs[0].outputs[0].text.strip()

    def _generate_hf(self, messages: List[dict], temperature: float) -> str:
        self._load_hf()
        import torch
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=max(temperature, 0.01),
                do_sample=temperature > 0,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def count_tokens(self, text: str) -> int:
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path, trust_remote_code=True
                )
            except Exception:
                return len(text) // 4
        return len(self._tokenizer.encode(text))


# ---------------------------------------------------------------------------
# Trajectory runners
# ---------------------------------------------------------------------------

def run_internal(llm: MathLLM, problem: str) -> dict:
    """Run pure internal reasoning trajectory (no tools)."""
    messages = [
        {"role": "system", "content": INTERNAL_SYSTEM},
        {"role": "user", "content": problem},
    ]

    t0 = time.time()
    response = llm.generate(messages)
    elapsed = time.time() - t0

    answer = extract_boxed_answer(response)
    trajectory = messages + [{"role": "assistant", "content": response}]

    return {
        "trajectory": trajectory,
        "final_answer": answer,
        "raw_response": response,
        "time_s": round(elapsed, 2),
        "tokens": llm.count_tokens(response),
    }


def run_code_interpreter(llm: MathLLM, problem: str, max_rounds: int = 5) -> dict:
    """Run code interpreter trajectory with ReAct loop.

    Loop: LLM generates response -> extract code -> execute -> feed result back.
    Stops when no code blocks found or max_rounds reached.
    """
    messages = [
        {"role": "system", "content": CODE_INTERPRETER_SYSTEM},
        {"role": "user", "content": problem},
    ]

    trajectory = list(messages)
    code_executions = []
    total_tokens = 0
    t0 = time.time()

    for round_idx in range(max_rounds):
        response = llm.generate(messages)
        messages.append({"role": "assistant", "content": response})
        trajectory.append({"role": "assistant", "content": response})
        total_tokens += llm.count_tokens(response)

        code_blocks = extract_code_blocks(response)

        if not code_blocks:
            break

        code = code_blocks[-1]
        exec_result = execute_code(code)
        code_executions.append({
            "code": code,
            "output": exec_result["stdout"],
            "error": exec_result["stderr"] if not exec_result["success"] else "",
            "success": exec_result["success"],
        })

        if exec_result["success"]:
            feedback = f"Code execution result:\n```\n{exec_result['stdout'][:2000]}\n```"
        else:
            feedback = f"Code execution error:\n```\n{exec_result['stderr'][:1000]}\n```"

        messages.append({"role": "user", "content": feedback})
        trajectory.append({"role": "user", "content": feedback})

    elapsed = time.time() - t0

    # Extract answer from last assistant message containing \\boxed{}
    answer = None
    for msg in reversed(trajectory):
        if msg["role"] == "assistant":
            answer = extract_boxed_answer(msg["content"])
            if answer is not None:
                break

    return {
        "trajectory": trajectory,
        "code_executions": code_executions,
        "final_answer": answer,
        "n_rounds": len(code_executions),
        "time_s": round(elapsed, 2),
        "tokens": total_tokens,
    }


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_math_problems(data_path: str, n_problems: int = 0, offset: int = 0) -> List[dict]:
    """Load MATH Level 4-5 problems from saved dataset or HuggingFace."""
    if os.path.isdir(data_path):
        from datasets import load_from_disk
        ds = load_from_disk(data_path)
    else:
        from datasets import load_dataset
        ds = load_dataset("hendrycks/competition_math", split="test")
        ds = ds.filter(lambda x: x['level'] in ['Level 4', 'Level 5'])

    problems = []
    for i, item in enumerate(ds):
        problems.append({
            "idx": i,
            "problem": item["problem"],
            "solution": item["solution"],
            "level": item["level"],
            "type": item["type"],
            "ground_truth": extract_boxed_answer(item["solution"]),
        })

    if offset > 0:
        problems = problems[offset:]
    if n_problems > 0:
        problems = problems[:n_problems]

    return problems


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_problem(llm: MathLLM, prob: dict, max_code_rounds: int = 5) -> dict:
    """Run both trajectories on one MATH problem and generate label."""
    problem_text = prob["problem"]
    ground_truth = prob["ground_truth"]

    internal_result = run_internal(llm, problem_text)
    internal_correct = answers_equal(internal_result["final_answer"], ground_truth)

    det_result = run_code_interpreter(llm, problem_text, max_rounds=max_code_rounds)
    det_correct = answers_equal(det_result["final_answer"], ground_truth)

    if det_correct and not internal_correct:
        raw_label = "det-preferred"
    elif internal_correct and not det_correct:
        raw_label = "internal-preferred"
    else:
        raw_label = "no-preference"

    filtered = det_result["n_rounds"] == 0
    label = "no-preference" if filtered else raw_label

    return {
        "episode_id": f"math_{prob['level'].replace(' ', '').lower()}_{prob['idx']:04d}",
        "environment": "math",
        "task_type": prob["type"].lower(),
        "task_id": f"test/{prob['type'].lower()}/{prob['idx']:04d}",
        "problem": problem_text,
        "ground_truth": ground_truth,
        "internal": {
            "trajectory": internal_result["trajectory"],
            "final_answer": internal_result["final_answer"],
            "correct": internal_correct,
            "score": 1.0 if internal_correct else 0.0,
            "time_s": internal_result["time_s"],
            "tokens": internal_result["tokens"],
        },
        "deterministic": {
            "trajectory": det_result["trajectory"],
            "code_executions": det_result["code_executions"],
            "final_answer": det_result["final_answer"],
            "correct": det_correct,
            "score": 1.0 if det_correct else 0.0,
            "n_rounds": det_result["n_rounds"],
            "time_s": det_result["time_s"],
            "tokens": det_result["tokens"],
        },
        "label": label,
        "raw_label": raw_label,
        "filtered": filtered,
        "metadata": {
            "level": prob["level"],
            "type": prob["type"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description="MATH counterfactual labeling")
    parser.add_argument("--n_problems", type=int, default=0, help="0=all")
    parser.add_argument("--output", type=str, default="data/math-counterfactual/episodes.jsonl")
    parser.add_argument("--data_path", type=str, default="data/math-raw")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--api_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--use_vllm", action="store_true", default=True)
    parser.add_argument("--no_vllm", action="store_true", default=False)
    parser.add_argument("--max_code_rounds", type=int, default=5)
    parser.add_argument("--offset", type=int, default=0, help="Skip first N problems")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.no_vllm:
        args.use_vllm = False

    import random
    random.seed(args.seed)

    if args.model_path is None:
        from llm_agent import find_model_path
        args.model_path = find_model_path()

    print(f"=== MATH Counterfactual Labeling ===")
    print(f"Model: {args.model_path}")
    print(f"GPU: {args.gpu_id} | vLLM: {args.use_vllm}")

    problems = load_math_problems(args.data_path, args.n_problems, args.offset)
    print(f"Loaded {len(problems)} problems")

    llm = MathLLM(
        model_path=args.model_path,
        api_url=args.api_url,
        use_vllm=args.use_vllm,
        gpu_id=args.gpu_id,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    label_counts = {"det-preferred": 0, "internal-preferred": 0, "no-preference": 0, "error": 0}
    internal_correct = 0
    det_correct = 0
    total_internal_tokens = 0
    total_det_tokens = 0
    total_time = 0
    n_filtered = 0
    n_code_executed = 0
    label_counts_raw = {"det-preferred": 0, "internal-preferred": 0, "no-preference": 0, "error": 0}

    for i, prob in enumerate(problems):
        print(f"\n--- [{i+1}/{len(problems)}] {prob['type']} ({prob['level']}) ---")
        print(f"  Q: {prob['problem'][:100]}...")

        try:
            result = process_problem(llm, prob, max_code_rounds=args.max_code_rounds)
        except Exception as e:
            print(f"  ERROR: {e}")
            result = {
                "episode_id": f"math_{prob['level'].replace(' ', '').lower()}_{prob['idx']:04d}",
                "environment": "math",
                "error": str(e),
                "problem": prob["problem"],
                "ground_truth": prob["ground_truth"],
                "internal": {"correct": False, "final_answer": None, "tokens": 0, "time_s": 0},
                "deterministic": {"correct": False, "final_answer": None, "tokens": 0, "time_s": 0, "n_rounds": 0},
                "label": "no-preference",
                "raw_label": "error",
                "filtered": True,
                "metadata": {"level": prob["level"], "type": prob["type"]},
            }

        with open(args.output, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")

        label_counts[result["label"]] += 1
        label_counts_raw[result.get("raw_label", result["label"])] += 1
        if result.get("filtered", False):
            n_filtered += 1
        if result["deterministic"]["n_rounds"] > 0:
            n_code_executed += 1
        if result["internal"]["correct"]:
            internal_correct += 1
        if result["deterministic"]["correct"]:
            det_correct += 1
        total_internal_tokens += result["internal"]["tokens"]
        total_det_tokens += result["deterministic"]["tokens"]
        total_time += result["internal"]["time_s"] + result["deterministic"]["time_s"]

        int_mark = "Y" if result["internal"]["correct"] else "N"
        det_mark = "Y" if result["deterministic"]["correct"] else "N"
        print(f"  Internal: {int_mark} ({result['internal']['final_answer']})")
        print(f"  CodeInterp: {det_mark} ({result['deterministic']['final_answer']}) "
              f"[{result['deterministic']['n_rounds']} code rounds]")
        filt_tag = " [FILTERED]" if result.get("filtered", False) else ""
        print(f"  Label: {result['label']}{filt_tag} | GT: {prob['ground_truth']}")

    n = len(problems)
    print(f"\n{'='*60}")
    print(f"MATH Counterfactual Labeling Summary")
    print(f"  Problems: {n}")
    print(f"  Internal acc: {internal_correct}/{n} ({100*internal_correct/n:.1f}%)")
    print(f"  CodeInterp acc: {det_correct}/{n} ({100*det_correct/n:.1f}%)")
    print(f"  Code execution rate: {n_code_executed}/{n} ({100*n_code_executed/n:.1f}%)")
    print(f"  Filtered (0 code rounds): {n_filtered}/{n}")
    print(f"  Labels (raw): det={label_counts_raw['det-preferred']}, "
          f"int={label_counts_raw['internal-preferred']}, "
          f"nopref={label_counts_raw['no-preference']}")
    print(f"  Labels (filtered): det={label_counts['det-preferred']}, "
          f"int={label_counts['internal-preferred']}, "
          f"nopref={label_counts['no-preference']}")
    print(f"  Avg internal tokens: {total_internal_tokens/n:.0f}")
    print(f"  Avg code-interp tokens: {total_det_tokens/n:.0f}")
    print(f"  Total time: {total_time:.0f}s ({total_time/n:.1f}s/problem)")
    print(f"  Output: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
