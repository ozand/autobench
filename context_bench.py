#!/usr/bin/env python3
"""
AutoBench Context Window Scaling & GPU Allocation Benchmark

Evaluates maximum effective context window size (512, 1024, 2048, 4096, 8192)
across single-GPU vs dual-GPU tensor splitting strategies (1,1 vs 2,1 offload)
for local GGUF models on k7000.
"""

import sys
import os
import argparse
import json
import time
import re
from datetime import datetime

# Add src folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from runner import Runner


def generate_haystack(
    target_tokens: int, needle: str, needle_position: float = 0.50
) -> str:
    """Generate a log haystack with a needle near the requested fractional position."""
    if not 0 <= needle_position <= 1:
        raise ValueError("needle_position must be between 0 and 1")
    log_templates = [
        "Jul 20 18:00:01 k7000 systemd[1]: Started Periodic Background Scan.",
        "Jul 20 18:00:02 k7000 kernel: [ 4512.1023] ata1.00: status: { DRDY }",
        "Jul 20 18:00:03 k7000 openclaw-gateway[3412]: [INFO] Client connection established from 192.168.1.35",
        "Jul 20 18:00:04 k7000 vulkan: [DEBUG] Device Vulkan0 memory heap 0 allocated 128MB.",
        "Jul 20 18:00:05 k7000 sshd[88412]: Accepted publickey for opencode from 100.67.171.58 port 44321",
        "Jul 20 18:00:06 k7000 smartd[1204]: Device /dev/sda: Temperature 34 C OK.",
    ]

    # Each log line is approx 35 tokens
    tokens_per_line = 35
    total_lines = max(2, target_tokens // tokens_per_line)
    needle_line = min(total_lines - 1, max(0, round((total_lines - 1) * needle_position)))

    lines = []
    for i in range(total_lines):
        if i == needle_line:
            lines.append(f"Jul 20 18:00:00 k7000 AUDIT_LOG: [CONFIDENTIAL] {needle}")
        else:
            template = log_templates[i % len(log_templates)]
            lines.append(f"{template} (seq={i+1000})")

    return "\n".join(lines)


def build_context_prompt(
    model_path: str,
    context_size: int,
    max_tokens: int,
    utilization: float,
    needle_val: str,
    needle_position: float = 0.50,
    tokenizer_timeout: int = 60,
    calibration_steps: int = 16,
) -> tuple[str, int, int, int]:
    """Build a model-tokenized prompt without consuming output headroom."""
    if not 0 < utilization <= 1:
        raise ValueError("utilization must be between 0 and 1")
    prompt_budget = min(int(context_size * utilization), context_size - max_tokens)
    if prompt_budget <= 0:
        raise ValueError("context size must exceed reserved output tokens")

    needle_text = f"SECRET_MAINTENANCE_CODE = '{needle_val}'"
    def make_prompt(estimated_tokens: int) -> str:
        haystack = generate_haystack(
            estimated_tokens, needle_text, needle_position=needle_position
        )
        return (
            "You are a system audit assistant. Analyze the log stream below and "
            "extract the secret code.\n"
            'Output JSON format: {"secret_code": "..."}\n\n'
            f"LOG STREAM:\n{haystack}\n\n"
            "Question: What is the exact value of SECRET_MAINTENANCE_CODE?\n"
            "--- END OF PROMPT ---"
        )

    low = 2
    high = max(2, prompt_budget * 2)
    best_prompt = None
    best_tokens = 0
    for _ in range(calibration_steps):
        estimated_tokens = (low + high) // 2
        prompt = make_prompt(estimated_tokens)
        actual_tokens = Runner.count_local_tokens(
            model_path, prompt, timeout=tokenizer_timeout
        )
        if actual_tokens <= prompt_budget:
            best_prompt = prompt
            best_tokens = actual_tokens
            low = estimated_tokens + 1
        else:
            high = estimated_tokens - 1
        if low > high:
            break

    if best_prompt is None:
        raise RuntimeError("Minimum prompt exceeds reserved context budget")
    needle_prefix = best_prompt.split(needle_text, maxsplit=1)[0]
    needle_token_offset = Runner.count_local_tokens(
        model_path, needle_prefix, timeout=tokenizer_timeout
    )
    return best_prompt, best_tokens, prompt_budget, needle_token_offset


def execution_stages(result: dict, generated_tokens: int = 0) -> dict:
    """Expose execution stages without conflating them with retrieval quality."""
    status = result.get("status", "EXECUTION_ERROR")
    success = bool(result.get("success"))
    return {
        "load_ok": True if success else False if status == "MODEL_LOAD_ERROR" else None,
        "context_allocated": (
            True if success else False if status in {"OOM", "CONTEXT_OVERFLOW"} else None
        ),
        "prefill_ok": success and result.get("prompt_speed_ts", 0.0) > 0,
        "decode_ok": success and generated_tokens > 0,
    }


def classify_retrieval_attempt(
    response_text: str,
    expected_answer: str,
    status: str,
    generated_tokens: int,
) -> dict:
    """Classify one bounded retrieval attempt without treating missing evidence as a miss."""
    if status != "SUCCESS" or generated_tokens <= 0:
        return {"outcome": "INCONCLUSIVE", "reason": "execution_or_generation_incomplete"}
    if not response_text.strip():
        return {"outcome": "INCONCLUSIVE", "reason": "empty_response"}
    normalized = response_text.strip()
    if normalized == expected_answer or expected_answer in normalized:
        return {"outcome": "VERIFIED", "reason": "exact_answer_present"}
    return {"outcome": "MISSED", "reason": "answer_absent"}


def summarize_context_runs(
    runs: list[dict],
    reliability_threshold: float,
    min_prompt_ts: float,
    min_gen_ts: float,
) -> dict:
    """Summarize technical capacity and explicit retrieval outcomes separately."""
    by_context: dict[int, list[dict]] = {}
    for run in runs:
        by_context.setdefault(run["context_size"], []).append(run)

    contexts = {}
    for context_size, attempts in sorted(by_context.items()):
        total = len(attempts)
        successful = [run for run in attempts if run["status"] == "SUCCESS"]
        outcomes = [
            run.get("retrieval_outcome")
            or ("VERIFIED" if run.get("status") == "SUCCESS" and run.get("retrieved") else
                "MISSED" if run.get("status") == "SUCCESS" else "INCONCLUSIVE")
            for run in attempts
        ]
        verified = [run for run, outcome in zip(attempts, outcomes) if outcome == "VERIFIED"]
        missed = [run for run, outcome in zip(attempts, outcomes) if outcome == "MISSED"]
        inconclusive = [run for run, outcome in zip(attempts, outcomes) if outcome == "INCONCLUSIVE"]
        allocation_rate = len(successful) / total
        retrieval_rate = len(verified) / total
        avg_prompt = (
            sum(run["prompt_ts"] for run in successful) / len(successful)
            if successful
            else 0.0
        )
        avg_gen = (
            sum(run["gen_ts"] for run in successful) / len(successful)
            if successful
            else 0.0
        )
        contexts[context_size] = {
            "attempts": total,
            "allocation_rate": allocation_rate,
            "retrieval_rate": retrieval_rate,
            "verified_attempts": len(verified),
            "missed_attempts": len(missed),
            "inconclusive_attempts": len(inconclusive),
            "retrieval_outcome": (
                "INCONCLUSIVE" if inconclusive else
                "VERIFIED" if len(verified) == total else
                "MISSED" if len(missed) == total else
                "MIXED"
            ),
            "avg_prompt_ts": avg_prompt,
            "avg_gen_ts": avg_gen,
            "allocatable": allocation_rate >= reliability_threshold,
            "reliable": retrieval_rate >= reliability_threshold,
            "operational": (
                retrieval_rate >= reliability_threshold
                and avg_prompt >= min_prompt_ts
                and avg_gen >= min_gen_ts
            ),
        }

    def maximum(flag: str) -> int:
        return max(
            (context for context, metrics in contexts.items() if metrics[flag]),
            default=0,
        )

    def non_monotonic(flag: str) -> bool:
        seen_failure = False
        for metrics in contexts.values():
            if not metrics[flag]:
                seen_failure = True
            elif seen_failure:
                return True
        return False

    return {
        "contexts": contexts,
        "maximum_allocatable_context": maximum("allocatable"),
        "maximum_reliable_retrieval_context": maximum("reliable"),
        "maximum_operational_context": maximum("operational"),
        "rerun_required": non_monotonic("allocatable")
        or non_monotonic("reliable"),
        "task_quality": {
            "measured": False,
            "source": "run_bench.py",
            "note": "Task-quality pass rate is intentionally separate.",
        },
    }


def run_context_test(
    model_name: str,
    model_path: str,
    context_size: int,
    device: str,
    ts_split: str,
    needle_val: str = "K7000-KEY-8842",
    split_mode: str | None = None,
    timeout: int = 180,
    utilization: float = 0.90,
    max_tokens: int = 128,
    needle_position: float = 0.50,
    calibration_steps: int = 16,
    cache_type_k: str | None = None,
    cache_type_v: str | None = None,
    no_kv_offload: bool = False,
) -> dict:
    """
    Runs a single Needle-In-A-Haystack context evaluation case.
    """
    prompt, actual_prompt_tokens, prompt_budget, needle_token_offset = build_context_prompt(
        model_path=model_path,
        context_size=context_size,
        max_tokens=max_tokens,
        utilization=utilization,
        needle_val=needle_val,
        needle_position=needle_position,
        tokenizer_timeout=min(timeout, 60),
        calibration_steps=calibration_steps,
    )

    res = Runner.run_local_vulkan(
        prompt=prompt,
        max_tokens=max_tokens,
        device=device,
        model_path=model_path,
        timeout=timeout,
        context_length=context_size,
        ts_split=ts_split,
        split_mode=split_mode,
        cache_type_k=cache_type_k,
        cache_type_v=cache_type_v,
        no_kv_offload=no_kv_offload,
    )

    if not res.get("success"):
        return {
            "context_size": context_size,
            "actual_prompt_tokens": actual_prompt_tokens,
            "prompt_budget_tokens": prompt_budget,
            "reserved_output_tokens": max_tokens,
            "needle_token_offset": needle_token_offset,
            "needle_position": needle_position,
            "context_utilization_pct": round(actual_prompt_tokens / context_size * 100, 2),
            "device": device,
            "ts_split": ts_split or "N/A",
            "status": res.get("status", "EXECUTION_ERROR"),
            "error": res.get("error", "Unknown error"),
            "return_code": res.get("return_code"),
            "prompt_ts": 0.0,
            "gen_ts": 0.0,
            "retrieved": False,
            "retrieval_outcome": "INCONCLUSIVE",
            "retrieval_reason": "execution_or_generation_incomplete",
            "elapsed_seconds": res.get("elapsed_seconds", 0.0),
            "execution": execution_stages(res),
            "retrieval": {"correct": False},
            "performance": {
                "model_load_seconds": None,
                "time_to_first_token_seconds": None,
                "prompt_speed_ts": 0.0,
                "generation_speed_ts": 0.0,
                "generated_tokens": 0,
                "elapsed_seconds": res.get("elapsed_seconds", 0.0),
            },
        }

    response_text = res.get("response", "")
    generated_tokens = Runner.count_local_tokens(
        model_path, response_text, timeout=min(timeout, 60)
    )
    retrieval_result = classify_retrieval_attempt(
        response_text=response_text,
        expected_answer=needle_val,
        status=res.get("status", "EXECUTION_ERROR"),
        generated_tokens=generated_tokens,
    )
    retrieved = retrieval_result["outcome"] == "VERIFIED"

    return {
        "context_size": context_size,
        "actual_prompt_tokens": actual_prompt_tokens,
        "prompt_budget_tokens": prompt_budget,
        "reserved_output_tokens": max_tokens,
        "generated_tokens": generated_tokens,
        "needle_token_offset": needle_token_offset,
        "needle_position": needle_position,
        "context_utilization_pct": round(actual_prompt_tokens / context_size * 100, 2),
        "device": device,
        "ts_split": ts_split or "N/A",
        "status": "SUCCESS",
        "error": None,
        "return_code": res.get("return_code", 0),
        "prompt_ts": res.get("prompt_speed_ts", 0.0),
        "gen_ts": res.get("generation_speed_ts", 0.0),
        "retrieved": retrieved,
        "retrieval_outcome": retrieval_result["outcome"],
        "retrieval_reason": retrieval_result["reason"],
        "elapsed_seconds": res.get("elapsed_seconds", 0.0),
        "response": response_text[:120],
        "execution": execution_stages(res, generated_tokens),
        "retrieval": {"correct": retrieved},
        "performance": {
            "model_load_seconds": None,
            "time_to_first_token_seconds": None,
            "prompt_speed_ts": res.get("prompt_speed_ts", 0.0),
            "generation_speed_ts": res.get("generation_speed_ts", 0.0),
            "generated_tokens": generated_tokens,
            "elapsed_seconds": res.get("elapsed_seconds", 0.0),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="AutoBench Context Window Benchmark")
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model names or paths to test",
    )
    parser.add_argument(
        "--context-sizes",
        type=str,
        default="512,1024,2048,4096,8192",
        help="Comma-separated context window sizes to test",
    )
    parser.add_argument(
        "--utilization",
        type=float,
        default=0.90,
        help="Target prompt utilization before reserved output tokens (default: 0.90)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Completion tokens reserved inside the configured context",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Attempts per model/config/context (default: 1)",
    )
    parser.add_argument(
        "--reliability-threshold",
        type=float,
        default=0.80,
        help="Required allocation/retrieval pass fraction (default: 0.80)",
    )
    parser.add_argument(
        "--min-prompt-ts",
        type=float,
        default=1.0,
        help="Minimum prompt throughput for operational context",
    )
    parser.add_argument(
        "--min-gen-ts",
        type=float,
        default=1.0,
        help="Minimum generation throughput for operational context",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if not 0 < args.reliability_threshold <= 1:
        parser.error("--reliability-threshold must be between 0 and 1")

    # Default model targets from evaluated models
    target_models = [
        {"name": "Qwen-3.5-0.8B", "path": "/home/opencode/llama.cpp/models/Qwen3.5-0.8B-Q4_K_M.gguf", "configs": [("Vulkan0", None, None), ("Vulkan0,Vulkan1", "1,1", "tensor")]},
        {"name": "Qwen-2.5-Coder-1.5B", "path": "/home/opencode/llama.cpp/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", "configs": [("Vulkan0", None, None), ("Vulkan0,Vulkan1", "1,1", "tensor"), ("Vulkan0,Vulkan1", "2,1", "tensor")]},
        {"name": "SmolLM2-1.7B", "path": "/home/opencode/llama.cpp/models/smollm2-1.7b-instruct-q4_k_m.gguf", "configs": [("Vulkan0", None, None), ("Vulkan0,Vulkan1", "1,1", "tensor")]},
        {"name": "Gemma-4-E2B", "path": "/home/opencode/llama.cpp/models/gemma-4-E2B-it-Q4_K_M.gguf", "configs": [("Vulkan0", None, None), ("Vulkan0,Vulkan1", "1,1", "tensor")]},
        {"name": "Llama-3.2-3B", "path": "/home/opencode/llama.cpp/models/llama-3.2-3b-instruct-q4_k_m.gguf", "configs": [("Vulkan0,Vulkan1", "1,1", "tensor")]},
        {"name": "Nemotron-3-Nano-4B", "path": "/home/opencode/llama.cpp/models/NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf", "configs": [("Vulkan0,Vulkan1", "1,1", "tensor")]},
    ]

    context_sizes = [int(x.strip()) for x in args.context_sizes.split(",")]

    all_results = {}

    print("==================================================================")
    print("AutoBench Context Scaling & Dual-GPU Allocation Benchmark")
    print(f"Context sizes: {context_sizes}")
    print("==================================================================")

    for m in target_models:
        m_name = m["name"]
        m_path = m["path"]
        print(f"\n>>> Evaluating Model: {m_name}")
        all_results[m_name] = []

        for device, ts_split, split_mode in m["configs"]:
            config_label = f"{device} (mode={split_mode or 'layer'}, ts={ts_split or 'N/A'})"
            print(f"  --> Configuration Strategy: {config_label}")

            for ctx in context_sizes:
                for repeat in range(1, args.repeats + 1):
                    print(
                        f"    - Context {ctx}, attempt {repeat}/{args.repeats}...",
                        end="",
                        flush=True,
                    )
                    res = run_context_test(
                        model_name=m_name,
                        model_path=m_path,
                        context_size=ctx,
                        device=device,
                        ts_split=ts_split,
                        split_mode=split_mode,
                        timeout=180,
                        utilization=args.utilization,
                        max_tokens=args.max_tokens,
                        needle_val=f"K7000-{ctx}-{repeat}-8842",
                    )

                    if res["status"] != "SUCCESS":
                        status_str = res["status"]
                    elif res["retrieved"]:
                        status_str = "PASSED"
                    else:
                        status_str = "MISSED_NEEDLE"
                    print(
                        f" [{status_str}] Input: {res['actual_prompt_tokens']}/{ctx} "
                        f"({res['context_utilization_pct']:.1f}%) | "
                        f"Prompt: {res['prompt_ts']:.1f} t/s | "
                        f"Gen: {res['gen_ts']:.1f} t/s "
                        f"({res['elapsed_seconds']:.1f}s)"
                    )

                    res["config_label"] = config_label
                    res["repeat"] = repeat
                    all_results[m_name].append(res)

    # Build Markdown Matrix Report
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(os.path.dirname(__file__), "results", "comparisons")
    os.makedirs(output_dir, exist_ok=True)
    report_file = os.path.join(output_dir, f"context_matrix_{timestamp}.md")

    lines = [
        "# AutoBench Context Window & Allocation Strategy Matrix",
        f"**Generated:** {datetime.now().isoformat()}",
        "",
        f"Prompt utilization target: {args.utilization:.0%}; reserved output: {args.max_tokens} tokens.",
        f"Attempts per context: {args.repeats}; reliability threshold: {args.reliability_threshold:.0%}.",
        "Cells report actual tokenizer-derived input tokens, not configured context size.",
        "",
        "| Model Name | GPU Config Strategy | Max Allocatable | Max Reliable Retrieval | Max Operational | Rerun? |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
    ]

    detail_lines = []
    for m_name, runs in all_results.items():
        config_labels = list(dict.fromkeys(run["config_label"] for run in runs))
        for cfg_label in config_labels:
            config_runs = [
                run for run in runs if run["config_label"] == cfg_label
            ]
            summary = summarize_context_runs(
                config_runs,
                reliability_threshold=args.reliability_threshold,
                min_prompt_ts=args.min_prompt_ts,
                min_gen_ts=args.min_gen_ts,
            )
            rerun = "yes" if summary["rerun_required"] else "no"
            lines.append(
                f"| `{m_name}` | {cfg_label} | "
                f"{summary['maximum_allocatable_context']} | "
                f"{summary['maximum_reliable_retrieval_context']} | "
                f"{summary['maximum_operational_context']} | {rerun} |"
            )

            detail_lines.extend(
                [
                    "",
                    f"#### `{m_name}` — {cfg_label}",
                    "",
                    "| Context | Attempts | Allocation | Retrieval | Prompt t/s | "
                    "Generation t/s |",
                    "| :---: | :---: | :---: | :---: | :---: | :---: |",
                ]
            )
            for context_size, metrics in summary["contexts"].items():
                detail_lines.append(
                    f"| {context_size} | {metrics['attempts']} | "
                    f"{metrics['allocation_rate']:.0%} | "
                    f"{metrics['retrieval_rate']:.0%} | "
                    f"{metrics['avg_prompt_ts']:.1f} | "
                    f"{metrics['avg_gen_ts']:.1f} |"
                )
            if summary["rerun_required"]:
                detail_lines.extend(
                    [
                        "",
                        "⚠️ Non-monotonic result detected; do not promote the "
                        "highest isolated pass without rerunning.",
                    ]
                )

    lines.extend(["", "### Per-context detail"])
    lines.extend(detail_lines)

    lines.extend(
        [
            "",
            "### Metric separation",
            "- **Allocatable**: inference completed technically; retrieval correctness is ignored.",
            "- **Reliable retrieval**: correct retrieval met the configured pass threshold.",
            "- **Operational**: reliable retrieval also met prompt and generation speed thresholds.",
            "- **Task quality**: measured separately by `run_bench.py`; it is not inferred here.",
            "",
            f"Context Matrix saved to: `{report_file}`",
        ]
    )

    report_content = "\n".join(lines)
    with open(report_file, "w") as f:
        f.write(report_content)

    print("\n==================================================================")
    print(report_content)
    print("==================================================================")


if __name__ == "__main__":
    main()
