#!/usr/bin/env python3
import argparse
import sys
import os
import glob
import json
import subprocess
import time
from datetime import datetime

# Add src folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from runner import Runner
from judge import Judge


def load_dataset(dataset_dir: str) -> list:
    tasks = []
    # Load all json files in datasets/validation or test
    files = glob.glob(os.path.join(dataset_dir, "*.json"))
    for file_path in files:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    tasks.extend(data)
                elif isinstance(data, dict):
                    tasks.append(data)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return tasks


def load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {}


def collect_local_provenance(model_path: str) -> dict:
    """Collect reproducibility facts from k7000 without exposing credentials."""
    command = (
        "set -e; "
        f"echo MODEL_SIZE=$(stat -c %s '{model_path}'); "
        f"echo MODEL_SHA256=$(sha256sum '{model_path}' | cut -d' ' -f1); "
        "echo LLAMA_VERSION=$(/home/opencode/llama.cpp/build/bin/llama-cli "
        "--version 2>&1 | head -1); "
        "echo NVIDIA_DRIVER=$(nvidia-smi --query-gpu=driver_version "
        "--format=csv,noheader | head -1); "
        "nvidia-smi --query-gpu=index,name,memory.total "
        "--format=csv,noheader | sed 's/^/GPU=/'"
    )
    result = subprocess.run(
        ["ssh", "opencode@192.168.1.171", command],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        return {"status": "unavailable", "error": result.stderr.strip()}

    facts = {"gpus": []}
    for line in result.stdout.splitlines():
        key, separator, value = line.partition("=")
        if not separator:
            continue
        if key == "GPU":
            facts["gpus"].append(value)
        else:
            facts[key.lower()] = value
    return facts


def build_run_manifest(
    args: argparse.Namespace,
    profile_name: str,
    model_path: str,
    device: str,
    first_result: dict,
    provenance: dict,
) -> dict:
    """Build a compact manifest from the final resolved benchmark settings."""
    return {
        "profile": profile_name,
        "model": args.model,
        "model_path": model_path if args.model == "local" else None,
        "device": device,
        "dataset": args.dataset,
        "context_length": args.context_length,
        "max_tokens": args.max_tokens,
        "tensor_split": args.ts_split,
        "run_judge": args.run_judge,
        "command_args": first_result.get("command_args"),
        "provenance": provenance,
    }


def main():
    parser = argparse.ArgumentParser(
        description="AutoBench: LLM Benchmark and Performance Evaluation on k7000"
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Profile from config.json: 'fast-local' or 'frontier-precision'",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model type: 'local' (llama.cpp) or API model name (e.g. cl/gemini-2.5-flash)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/opencode/llama.cpp/models/qwen2.5-0.5b-instruct-q8_0.gguf",
        help="Path to local GGUF model file on k7000",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="Vulkan0",
        help="Vulkan GPU device (e.g. Vulkan0, Vulkan1, or Vulkan0,Vulkan1)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="validation",
        choices=["validation", "test"],
        help="Dataset split to run",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512, help="Max tokens to generate"
    )
    parser.add_argument(
        "--context-length", type=int, default=1024, help="Context window size (-c)"
    )
    parser.add_argument(
        "--ts-split", type=str, default=None, help="Custom tensor split ratio (-ts)"
    )
    parser.add_argument(
        "--run-judge",
        action="store_true",
        help="Run frontier LLM judge for quality grading",
    )
    args = parser.parse_args()

    cfg = load_config()
    profile_name = args.profile or cfg.get("active_profile", "fast-local")
    profiles = cfg.get("profiles", {})
    prof = profiles.get(profile_name, {})

    # Resolve arguments from profile if not explicitly overridden
    if args.model is None:
        args.model = prof.get("type", prof.get("model_name", "local"))
    
    model_path = args.model_path
    if prof.get("model_path") and (args.model_path == "/home/opencode/llama.cpp/models/qwen2.5-0.5b-instruct-q8_0.gguf" or args.profile):
        model_path = prof.get("model_path")

    device = args.device
    if prof.get("device") and (args.device == "Vulkan0" or args.profile):
        device = prof.get("device")
    args.device = device

    if prof.get("model_name") and args.model == prof.get("type"):
        args.model = prof.get("model_name")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "datasets", args.dataset)

    print("=== AutoBench Start ===")
    print(f"DateTime:   {datetime.now().isoformat()}")
    print(f"Model:      {args.model}")
    print(f"Device:     {args.device}")
    print(f"Dataset:    {args.dataset} (loading from {dataset_dir})")
    print("=======================")

    tasks = load_dataset(dataset_dir)
    if not tasks:
        print(f"No tasks found in {dataset_dir}. Exiting.")
        sys.exit(1)

    print(f"Loaded {len(tasks)} benchmark tasks.")

    results = []
    first_run_result = {}
    total_passed = 0
    total_prompt_speed = 0.0
    total_gen_speed = 0.0
    total_judge_score = 0.0
    judge_count = 0

    provenance = (
        collect_local_provenance(model_path) if args.model == "local" else {}
    )
    execution_start = time.time()

    for idx, t in enumerate(tasks):
        item_id = t.get("id", f"case_{idx}")
        task_type = t.get("task", "unknown")
        prompt = t.get("prompt", "")
        constraints = t.get("constraints", {})

        print(f"\n[Task {idx + 1}/{len(tasks)}] ID: {item_id} | Type: {task_type}")
        print(f"Prompt: {prompt[:80]}...")

        # Run inference
        if args.model == "local":
            run_res = Runner.run_local_vulkan(
                prompt,
                max_tokens=args.max_tokens,
                device=device,
                model_path=model_path,
                timeout=180,
                context_length=args.context_length,
                ts_split=args.ts_split,
            )
        else:
            run_res = Runner.run_frontier_api(
                args.model, prompt, max_tokens=args.max_tokens
            )

        if not first_run_result:
            first_run_result = run_res

        if not run_res.get("success"):
            print(f"  ERR: Generation failed: {run_res.get('error')}")
            results.append({"task": t, "success": False, "error": run_res.get("error")})
            continue

        generated_text = run_res.get("response", "")
        print(
            f"  Generated ({run_res.get('tokens_approx'):.1f} tokens in {run_res.get('elapsed_seconds'):.2f}s)"
        )
        print(f"  Text: {generated_text[:120].strip()}...")

        # Grade output deterministically
        det_res = Judge.validate_deterministic(generated_text, constraints)
        passed_det = det_res["passed"]
        det_score = det_res["score"]
        issues = det_res["issues"]

        if passed_det:
            print("  Deterministic Check: PASSED")
            total_passed += 1
        else:
            print(f"  Deterministic Check: FAILED. Issues: {issues}")

        # Optional LLM Judge
        judge_score = None
        judge_reason = None
        if args.run_judge:
            print("  Querying Frontier LLM Judge...")
            lh_res = Judge.run_llm_judge(
                Runner, args.model, task_type, prompt, generated_text
            )
            if lh_res.get("success"):
                judge_score = lh_res.get("judge_score")
                judge_reason = lh_res.get("reason")
                total_judge_score += judge_score
                judge_count += 1
                print(f"    Judge Score: {judge_score}/5 | Reason: {judge_reason}")
            else:
                print(f"    Judge query failed: {lh_res.get('error')}")

        # Sum speeds for averages
        total_prompt_speed += run_res.get("prompt_speed_ts", 0.0)
        total_gen_speed += run_res.get("generation_speed_ts", 0.0)

        results.append(
            {
                "task": t,
                "success": True,
                "generated_text": generated_text,
                "performance": {
                    "elapsed_seconds": run_res.get("elapsed_seconds"),
                    "prompt_speed_ts": run_res.get("prompt_speed_ts"),
                    "generation_speed_ts": run_res.get("generation_speed_ts"),
                    "tokens_approx": run_res.get("tokens_approx"),
                },
                "eval": {
                    "deterministic_passed": passed_det,
                    "deterministic_score": det_score,
                    "issues": issues,
                    "llm_judge_score": judge_score,
                    "llm_judge_reason": judge_reason,
                },
            }
        )

    execution_end = time.time()
    total_elapsed = execution_end - execution_start

    # Calculate statistics
    n_successful = sum(1 for r in results if r.get("success"))
    avg_prompt_speed = total_prompt_speed / n_successful if n_successful > 0 else 0.0
    avg_gen_speed = total_gen_speed / n_successful if n_successful > 0 else 0.0
    avg_judge_score = total_judge_score / judge_count if judge_count > 0 else 0.0
    det_pass_rate = (total_passed / len(tasks)) * 100 if len(tasks) > 0 else 0.0

    # Compile final run details
    run_log = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
            "model_path": model_path if args.model == "local" else None,
            "device": device,
            "dataset": args.dataset,
            "context_length": args.context_length,
            "max_tokens": args.max_tokens,
            "tensor_split": args.ts_split,
            "total_tasks": len(tasks),
            "run_judge": args.run_judge,
        },
        "manifest": build_run_manifest(
            args,
            profile_name,
            model_path,
            device,
            first_run_result,
            provenance,
        ),
        "stats": {
            "total_elapsed_seconds": total_elapsed,
            "successful_runs": n_successful,
            "passed_deterministic": total_passed,
            "deterministic_pass_rate": det_pass_rate,
            "avg_prompt_speed_ts": avg_prompt_speed,
            "avg_generation_speed_ts": avg_gen_speed,
            "avg_judge_score": avg_judge_score,
        },
        "results": results,
    }

    # Save results to runs directory
    timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_filename = (
        f"run_{args.model.replace('/', '_')}_{args.device}_{timestamp_str}.json"
    )
    output_path = os.path.join(base_dir, "results", "runs", output_filename)

    with open(output_path, "w") as f:
        json.dump(run_log, f, indent=2)

    print("\n=======================")
    print("=== BENCHMARK REPORT ===")
    print("=======================")
    print(f"Save Path:   {output_path}")
    print(f"Elapsed:     {total_elapsed:.2f} seconds")
    print(f"Pass Rate:   {det_pass_rate:.1f}% ({total_passed}/{len(tasks)} passed)")
    print(f"Avg Prompt:  {avg_prompt_speed:.1f} t/s")
    print(f"Avg Gen:     {avg_gen_speed:.1f} t/s")
    if args.run_judge:
        print(f"Avg Quality: {avg_judge_score:.2f} / 5.0 (Judge consensus)")
    print("=======================")


if __name__ == "__main__":
    main()
