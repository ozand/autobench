#!/usr/bin/env python3
"""Plan and smoke-test the reproducible k7000 GGUF benchmark matrix."""

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from context_bench import build_context_prompt, run_context_test
from run_bench import load_dataset
from src.judge import Judge
from src.runner import Runner

MODEL_DIR = "/home/opencode/llama.cpp/models"
SSH_TARGET = "opencode@192.168.1.171"
GPU_MEMORY_BYTES = 2 * 1024**3
SINGLE_GPU_FIT_BYTES = 1_900_000_000
FULL_POLICY = {
    "coarse_contexts": [512, 1024, 2048, 4096, 8192],
    "boundary_step": 256,
    "retrieval_positions": [0.10, 0.50, 0.90],
    "retrieval_repetitions": 5,
    "performance_warmups": 1,
    "performance_repetitions": 3,
    "matched_prompt_tokens": 512,
    "matched_output_tokens": 64,
    "cache_mode": "f16",
    "task_quality_dataset": "datasets/validation",
    "task_quality_metric": "deterministic_pass_rate",
}


def discover_remote_models(timeout: int = 30) -> list[dict]:
    """Return every real GGUF model on k7000, excluding tokenizer fixtures."""
    command = (
        f"find {MODEL_DIR} -maxdepth 1 -type f -iname '*.gguf' "
        "-printf '%f\\t%s\\n' | sort"
    )
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", SSH_TARGET, command],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "remote GGUF inventory failed")

    models = []
    for line in result.stdout.splitlines():
        name, separator, size_text = line.partition("\t")
        if not separator or name.lower().startswith("ggml-vocab-"):
            continue
        models.append(
            {
                "id": Path(name).stem,
                "name": name,
                "path": f"{MODEL_DIR}/{name}",
                "size_bytes": int(size_text),
            }
        )
    return models


def configurations_for_model(size_bytes: int) -> list[dict]:
    """Plan asymmetry checks for fitting models and load-only checks otherwise."""
    if size_bytes <= SINGLE_GPU_FIT_BYTES:
        return [
            {"device": "Vulkan0", "tensor_split": None, "mode": "full"},
            {"device": "Vulkan1", "tensor_split": None, "mode": "full"},
        ]
    return [
        {"device": "Vulkan0", "tensor_split": None, "mode": "load_only"},
        {"device": "Vulkan1", "tensor_split": None, "mode": "load_only"},
        {"device": "Vulkan0,Vulkan1", "tensor_split": "1,1", "mode": "full"},
    ]


def fine_boundary_contexts(last_pass: int, first_failure: int, step: int = 256) -> list[int]:
    """Return finer context probes strictly between coarse boundary endpoints."""
    if last_pass >= first_failure or step <= 0:
        return []
    return list(range(last_pass + step, first_failure, step))


def build_plan(models: list[dict], mode: str) -> dict:
    """Build a serializable benchmark contract without executing inference."""
    policy = dict(FULL_POLICY)
    if mode == "smoke":
        policy.update(
            {
                "coarse_contexts": [512],
                "retrieval_repetitions": 1,
                "performance_repetitions": 1,
            }
        )
    return {
        "schema_version": 1,
        "mode": mode,
        "authoritative": False,
        "execution_status": "planned_not_run",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": SSH_TARGET,
        "gpu_memory_bytes_per_device": GPU_MEMORY_BYTES,
        "single_gpu_fit_threshold_bytes": SINGLE_GPU_FIT_BYTES,
        "policy": policy,
        "models": [
            {**model, "configurations": configurations_for_model(model["size_bytes"])}
            for model in models
        ],
    }


def select_smoke_models(models: list[dict]) -> list[dict]:
    """Select one fitting and one dual-GPU candidate deterministically."""
    fitting = next((model for model in models if model["size_bytes"] <= SINGLE_GPU_FIT_BYTES), None)
    oversized = next((model for model in models if model["size_bytes"] > SINGLE_GPU_FIT_BYTES), None)
    return [model for model in (fitting, oversized) if model is not None]


def run_load_probe(model: dict, config: dict, timeout: int) -> dict:
    """Run a minimal bounded inference to record actual model-load viability."""
    result = Runner.run_local_vulkan(
        prompt="Reply OK.",
        max_tokens=1,
        device=config["device"],
        model_path=model["path"],
        timeout=timeout,
        context_length=128,
        ts_split=config["tensor_split"],
    )
    return {
        "stage": "load_probe",
        "status": result.get("status", "EXECUTION_ERROR"),
        "load_ok": bool(result.get("success")),
        "elapsed_seconds": result.get("elapsed_seconds", 0.0),
        "return_code": result.get("return_code"),
        "command_args": result.get("command_args"),
    }


def execute_suite(
    plan: dict,
    timeout: int,
    context_sizes: list[int],
    boundary_step: int,
    retrieval_repetitions: int,
    reliability_threshold: float,
    performance_context: int,
    prompt_tokens: int,
    output_tokens: int,
    warmups: int,
    performance_repetitions: int,
    dataset_dir: str,
    max_tasks: int,
) -> dict:
    """Run all validated stages for one explicit model/configuration."""
    model = plan["models"][0]
    config_template = model["configurations"][0]

    def stage_plan(mode: str) -> dict:
        child = build_plan(
            [
                {
                    "id": model["id"],
                    "name": model["name"],
                    "path": model["path"],
                    "size_bytes": model["size_bytes"],
                }
            ],
            mode,
        )
        child["models"][0]["configurations"] = [
            {
                "device": config_template["device"],
                "tensor_split": config_template["tensor_split"],
                "mode": "full",
            }
        ]
        return child

    boundary = execute_boundary(
        stage_plan("boundary"), timeout, context_sizes, boundary_step
    )["models"][0]["configurations"][0]["result"]
    retrieval_context = boundary["maximum_allocatable_context"]
    if retrieval_context <= 0:
        config_template["result"] = {
            "stage": "suite",
            "status": "BLOCKED_BY_BOUNDARY",
            "stages": {"boundary": boundary},
        }
    elif prompt_tokens + output_tokens > min(
        performance_context, retrieval_context
    ):
        config_template["result"] = {
            "stage": "suite",
            "status": "BLOCKED_BY_CONTEXT_BUDGET",
            "stages": {"boundary": boundary},
        }
    else:
        operational_context = min(performance_context, retrieval_context)
        retrieval = execute_retrieval(
            stage_plan("retrieval"),
            timeout,
            retrieval_context,
            retrieval_repetitions,
            reliability_threshold,
        )["models"][0]["configurations"][0]["result"]
        performance = execute_performance(
            stage_plan("performance"),
            timeout,
            operational_context,
            prompt_tokens,
            output_tokens,
            warmups,
            performance_repetitions,
        )["models"][0]["configurations"][0]["result"]
        quality = execute_quality(
            stage_plan("quality"),
            timeout,
            dataset_dir,
            max_tasks,
            operational_context,
            output_tokens,
        )["models"][0]["configurations"][0]["result"]
        config_template["result"] = {
            "stage": "suite",
            "status": (
                "SUCCESS"
                if all(
                    stage["status"] == "SUCCESS"
                    for stage in (boundary, retrieval, performance, quality)
                )
                else "PARTIAL_FAILURE"
            ),
            "maximum_allocatable_context": boundary[
                "maximum_allocatable_context"
            ],
            "allocatable_is_lower_bound": boundary["first_failed_context"] is None,
            "first_failed_context": boundary["first_failed_context"],
            "maximum_reliable_context": (
                retrieval_context if retrieval["retrieved"] else 0
            ),
            "retrieval_rate": retrieval["retrieval_rate"],
            "prompt_ts": performance["prompt_ts"],
            "gen_ts": performance["gen_ts"],
            "elapsed_seconds": sum(
                stage.get("elapsed_seconds", 0.0)
                for stage in (boundary, retrieval, performance, quality)
            ),
            "task_pass_rate": quality["task_pass_rate"],
            "stages": {
                "boundary": boundary,
                "retrieval": retrieval,
                "performance": performance,
                "quality": quality,
            },
        }

    plan["execution_status"] = "completed_suite"
    plan["authoritative"] = False
    plan["completed_at"] = datetime.now(timezone.utc).isoformat()
    return plan


def execute_performance(
    plan: dict,
    timeout: int,
    context_size: int,
    prompt_tokens: int,
    output_tokens: int,
    warmups: int,
    repetitions: int,
) -> dict:
    """Measure a matched tokenizer-calibrated workload after discarded warm-ups."""
    full_configs = [
        (model, config)
        for model in plan["models"]
        for config in model["configurations"]
        if config["mode"] == "full"
    ]
    if len(full_configs) != 1:
        raise ValueError("performance mode requires exactly one full configuration")

    model, config = full_configs[0]
    utilization = min(1.0, prompt_tokens / context_size)
    try:
        prompt, actual_tokens, prompt_budget, _ = build_context_prompt(
            model_path=model["path"],
            context_size=context_size,
            max_tokens=output_tokens,
            utilization=utilization,
            needle_val="PERFORMANCE-WORKLOAD-8842",
            needle_position=0.50,
            tokenizer_timeout=min(timeout, 60),
            calibration_steps=8,
        )
    except (RuntimeError, subprocess.SubprocessError) as exc:
        config["result"] = {
            "stage": "performance",
            "status": "TOKENIZER_ERROR",
            "error": str(exc),
            "context_size": context_size,
            "retrieved": False,
            "actual_prompt_tokens": 0,
            "prompt_budget_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "warmups_discarded": warmups,
            "measured_repetitions": repetitions,
            "successful_measurements": 0,
            "prompt_ts": 0.0,
            "gen_ts": 0.0,
            "elapsed_seconds": 0.0,
            "runs": [],
        }
        plan["execution_status"] = "completed_performance"
        plan["authoritative"] = False
        plan["completed_at"] = datetime.now(timezone.utc).isoformat()
        return plan

    runs = []
    total_runs = warmups + repetitions
    for index in range(total_runs):
        result = Runner.run_local_vulkan(
            prompt=prompt,
            max_tokens=output_tokens,
            device=config["device"],
            model_path=model["path"],
            timeout=timeout,
            context_length=context_size,
            ts_split=config["tensor_split"],
        )
        runs.append(
            {
                "phase": "warmup" if index < warmups else "measured",
                "index": index + 1,
                "status": result.get("status", "EXECUTION_ERROR"),
                "success": bool(result.get("success")),
                "prompt_ts": result.get("prompt_speed_ts", 0.0),
                "gen_ts": result.get("generation_speed_ts", 0.0),
                "elapsed_seconds": result.get("elapsed_seconds", 0.0),
                "return_code": result.get("return_code"),
                "command_args": result.get("command_args"),
            }
        )

    measured = [run for run in runs if run["phase"] == "measured"]
    successful = [run for run in measured if run["success"]]
    config["result"] = {
        "stage": "performance",
        "status": (
            "SUCCESS"
            if len(successful) == repetitions
            else "PARTIAL_FAILURE"
        ),
        "context_size": context_size,
        "retrieved": False,
        "actual_prompt_tokens": actual_tokens,
        "prompt_budget_tokens": prompt_budget,
        "output_tokens": output_tokens,
        "warmups_discarded": warmups,
        "measured_repetitions": repetitions,
        "successful_measurements": len(successful),
        "prompt_ts": (
            sum(run["prompt_ts"] for run in successful) / len(successful)
            if successful
            else 0.0
        ),
        "gen_ts": (
            sum(run["gen_ts"] for run in successful) / len(successful)
            if successful
            else 0.0
        ),
        "elapsed_seconds": (
            sum(run["elapsed_seconds"] for run in successful) / len(successful)
            if successful
            else 0.0
        ),
        "runs": runs,
    }
    plan["execution_status"] = "completed_performance"
    plan["authoritative"] = False
    plan["completed_at"] = datetime.now(timezone.utc).isoformat()
    plan["policy"].update(
        {
            "matched_prompt_tokens": prompt_tokens,
            "matched_output_tokens": output_tokens,
            "performance_warmups": warmups,
            "performance_repetitions": repetitions,
            "performance_context": context_size,
        }
    )
    return plan


def execute_quality(
    plan: dict,
    timeout: int,
    dataset_dir: str,
    max_tasks: int,
    context_size: int,
    max_tokens: int,
) -> dict:
    """Measure deterministic task quality independently for one configuration."""
    full_configs = [
        (model, config)
        for model in plan["models"]
        for config in model["configurations"]
        if config["mode"] == "full"
    ]
    if len(full_configs) != 1:
        raise ValueError("quality mode requires exactly one full configuration")

    tasks = sorted(load_dataset(dataset_dir), key=lambda task: task.get("id", ""))
    tasks = tasks[:max_tasks]
    if not tasks:
        raise ValueError("quality dataset contains no tasks")

    model, config = full_configs[0]
    task_results = []
    for task in tasks:
        run_result = Runner.run_local_vulkan(
            prompt=task.get("prompt", ""),
            max_tokens=max_tokens,
            device=config["device"],
            model_path=model["path"],
            timeout=timeout,
            context_length=context_size,
            ts_split=config["tensor_split"],
        )
        result = {
            "task_id": task.get("id"),
            "task_type": task.get("task"),
            "status": run_result.get("status", "EXECUTION_ERROR"),
            "success": bool(run_result.get("success")),
            "elapsed_seconds": run_result.get("elapsed_seconds", 0.0),
            "prompt_ts": run_result.get("prompt_speed_ts", 0.0),
            "gen_ts": run_result.get("generation_speed_ts", 0.0),
            "return_code": run_result.get("return_code"),
            "command_args": run_result.get("command_args"),
        }
        if run_result.get("success"):
            evaluation = Judge.validate_deterministic(
                run_result.get("response", ""), task.get("constraints", {})
            )
            result["deterministic_passed"] = evaluation["passed"]
            result["deterministic_score"] = evaluation["score"]
            result["issues"] = evaluation["issues"]
        else:
            result["deterministic_passed"] = False
            result["deterministic_score"] = 0.0
            result["issues"] = [run_result.get("error", result["status"])]
        task_results.append(result)

    passed = sum(result["deterministic_passed"] for result in task_results)
    config["result"] = {
        "stage": "quality",
        "status": (
            "SUCCESS"
            if all(result["success"] for result in task_results)
            else "PARTIAL_FAILURE"
        ),
        "context_size": context_size,
        "retrieved": False,
        "task_pass_rate": passed / len(task_results),
        "tasks_passed": passed,
        "tasks_total": len(task_results),
        "prompt_ts": 0.0,
        "gen_ts": 0.0,
        "elapsed_seconds": sum(
            result["elapsed_seconds"] for result in task_results
        ),
        "tasks": task_results,
    }
    plan["execution_status"] = "completed_quality"
    plan["authoritative"] = False
    plan["completed_at"] = datetime.now(timezone.utc).isoformat()
    plan["policy"].update(
        {
            "task_quality_dataset": dataset_dir,
            "task_quality_max_tasks": max_tasks,
            "task_quality_context": context_size,
            "task_quality_max_tokens": max_tokens,
        }
    )
    return plan


def execute_boundary(
    plan: dict,
    timeout: int,
    context_sizes: list[int],
    boundary_step: int,
) -> dict:
    """Measure an allocation boundary without conflating it with retrieval."""
    full_configs = [
        (model, config)
        for model in plan["models"]
        for config in model["configurations"]
        if config["mode"] == "full"
    ]
    if len(full_configs) != 1:
        raise ValueError("boundary mode requires exactly one full configuration")

    model, config = full_configs[0]
    probes = []

    def run_probe(context_size: int, phase: str) -> dict:
        try:
            result = run_context_test(
                model_name=model["name"],
                model_path=model["path"],
                context_size=context_size,
                device=config["device"],
                ts_split=config["tensor_split"],
                needle_val=f"BOUNDARY-{model['id']}-{context_size}-8842",
                timeout=timeout,
                utilization=0.50,
                max_tokens=16,
                needle_position=0.50,
                calibration_steps=8,
            )
        except (RuntimeError, subprocess.SubprocessError) as exc:
            result = {
                "status": "TOKENIZER_ERROR",
                "error": str(exc),
                "context_size": context_size,
                "retrieved": False,
                "prompt_ts": 0.0,
                "gen_ts": 0.0,
                "elapsed_seconds": 0.0,
            }
        result.update({"stage": "allocation_probe", "phase": phase})
        probes.append(result)
        return result

    last_pass = 0
    first_failure = 0
    terminal_capacity_failures = {"OOM", "CONTEXT_OVERFLOW"}
    inconclusive_status = None
    for context_size in sorted(set(context_sizes)):
        result = run_probe(context_size, "coarse")
        if result["status"] == "SUCCESS":
            last_pass = context_size
        elif result["status"] in terminal_capacity_failures:
            first_failure = context_size
            break
        else:
            inconclusive_status = result["status"]
            break

    if last_pass and first_failure:
        for context_size in fine_boundary_contexts(
            last_pass, first_failure, boundary_step
        ):
            result = run_probe(context_size, "fine")
            if result["status"] == "SUCCESS":
                last_pass = context_size
            elif result["status"] in terminal_capacity_failures:
                first_failure = context_size
                break
            else:
                inconclusive_status = result["status"]
                break

    config["result"] = {
        "stage": "boundary",
        "status": (
            "INCONCLUSIVE"
            if inconclusive_status
            else "SUCCESS"
            if probes
            else "NOT_RUN"
        ),
        "inconclusive_status": inconclusive_status,
        "context_size": last_pass,
        "retrieved": False,
        "maximum_allocatable_context": last_pass,
        "first_failed_context": first_failure or None,
        "prompt_ts": 0.0,
        "gen_ts": 0.0,
        "elapsed_seconds": sum(probe["elapsed_seconds"] for probe in probes),
        "probes": probes,
    }
    plan["execution_status"] = "completed_boundary"
    plan["authoritative"] = False
    plan["completed_at"] = datetime.now(timezone.utc).isoformat()
    plan["policy"].update(
        {
            "coarse_contexts": sorted(set(context_sizes)),
            "boundary_step": boundary_step,
        }
    )
    return plan


def execute_retrieval(
    plan: dict,
    timeout: int,
    context_size: int,
    repetitions: int,
    reliability_threshold: float,
) -> dict:
    """Execute repeated 10/50/90% retrieval for one full configuration."""
    full_configs = [
        (model, config)
        for model in plan["models"]
        for config in model["configurations"]
        if config["mode"] == "full"
    ]
    if len(full_configs) != 1:
        raise ValueError("retrieval mode requires exactly one full configuration")

    model, config = full_configs[0]
    attempts = []
    for position in FULL_POLICY["retrieval_positions"]:
        for repeat in range(1, repetitions + 1):
            try:
                result = run_context_test(
                    model_name=model["name"],
                    model_path=model["path"],
                    context_size=context_size,
                    device=config["device"],
                    ts_split=config["tensor_split"],
                    needle_val=(
                        f"RETRIEVAL-{model['id']}-{position:.2f}-{repeat}-8842"
                    ),
                    timeout=timeout,
                    utilization=0.90,
                    max_tokens=64,
                    needle_position=position,
                    calibration_steps=8,
                )
            except (RuntimeError, subprocess.SubprocessError) as exc:
                result = {
                    "status": "TOKENIZER_ERROR",
                    "error": str(exc),
                    "context_size": context_size,
                    "retrieved": False,
                    "prompt_ts": 0.0,
                    "gen_ts": 0.0,
                    "elapsed_seconds": 0.0,
                }
            result.update(
                {
                    "stage": "retrieval",
                    "requested_position": position,
                    "repeat": repeat,
                }
            )
            attempts.append(result)

    successful = [attempt for attempt in attempts if attempt["status"] == "SUCCESS"]
    retrieved = [attempt for attempt in attempts if attempt.get("retrieved")]
    retrieval_rate = len(retrieved) / len(attempts)
    config["result"] = {
        "stage": "repeated_retrieval",
        "status": "SUCCESS" if len(successful) == len(attempts) else "PARTIAL_FAILURE",
        "context_size": context_size,
        "retrieved": retrieval_rate >= reliability_threshold,
        "retrieval_rate": retrieval_rate,
        "attempts_completed": len(attempts),
        "attempts_expected": len(FULL_POLICY["retrieval_positions"]) * repetitions,
        "prompt_ts": (
            sum(attempt["prompt_ts"] for attempt in successful) / len(successful)
            if successful
            else 0.0
        ),
        "gen_ts": (
            sum(attempt["gen_ts"] for attempt in successful) / len(successful)
            if successful
            else 0.0
        ),
        "elapsed_seconds": sum(attempt["elapsed_seconds"] for attempt in attempts),
        "attempts": attempts,
    }
    plan["execution_status"] = "completed_retrieval"
    plan["authoritative"] = False
    plan["completed_at"] = datetime.now(timezone.utc).isoformat()
    plan["policy"].update(
        {
            "retrieval_repetitions": repetitions,
            "retrieval_context": context_size,
            "reliability_threshold": reliability_threshold,
        }
    )
    return plan


def execute_smoke(plan: dict, timeout: int = 180) -> dict:
    """Execute one bounded measured case for every planned smoke configuration."""
    if plan["mode"] != "smoke":
        raise ValueError("only smoke plans may be executed by this increment")

    for model in plan["models"]:
        for config in model["configurations"]:
            if config["mode"] == "load_only":
                result = run_load_probe(model, config, timeout)
            else:
                try:
                    result = run_context_test(
                        model_name=model["name"],
                        model_path=model["path"],
                        context_size=512,
                        device=config["device"],
                        ts_split=config["tensor_split"],
                        needle_val=f"SMOKE-{model['id']}-{config['device']}-8842",
                        timeout=timeout,
                        utilization=0.50,
                        max_tokens=32,
                        needle_position=0.50,
                        calibration_steps=8,
                    )
                except (RuntimeError, subprocess.SubprocessError) as exc:
                    result = {
                        "stage": "retrieval_performance",
                        "status": "TOKENIZER_ERROR",
                        "error": str(exc),
                        "context_size": 512,
                        "retrieved": False,
                        "prompt_ts": 0.0,
                        "gen_ts": 0.0,
                        "elapsed_seconds": 0.0,
                    }
                else:
                    result["stage"] = "retrieval_performance"
            config["result"] = result

    plan["execution_status"] = "completed_smoke"
    plan["completed_at"] = datetime.now(timezone.utc).isoformat()
    return plan


def render_matrix(plan: dict, manifest_name: str) -> str:
    """Render planned or measured rows without inventing missing evidence."""
    if plan["mode"] == "smoke":
        title = "# Smoke / Non-Authoritative Local GGUF Benchmark Matrix"
    elif plan["mode"] == "retrieval":
        title = "# Retrieval Slice / Non-Authoritative Local GGUF Matrix"
    elif plan["mode"] == "boundary":
        title = "# Context Boundary / Non-Authoritative Local GGUF Matrix"
    elif plan["mode"] == "quality":
        title = "# Task Quality / Non-Authoritative Local GGUF Matrix"
    elif plan["mode"] == "performance":
        title = "# Matched Performance / Non-Authoritative Local GGUF Matrix"
    elif plan["mode"] == "suite":
        title = "# End-to-End Suite / Non-Authoritative Local GGUF Matrix"
    else:
        title = "# Planned / Non-Authoritative Local GGUF Benchmark Matrix"
    lines = [
        title,
        "",
        f"Manifest: `{manifest_name}`",
        "",
        "| Model / configuration | Cache | Allocatable context | Reliable context | Retrieval rate | Prompt t/s | Generation t/s | Elapsed | Task pass rate | Manifest |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]
    for model in plan["models"]:
        for config in model["configurations"]:
            label = f"{model['name']} — {config['device']} ({config['mode']})"
            result = config.get("result")
            if not result:
                values = ["not_run"] * 7
            elif result["stage"] == "load_probe":
                allocation = "load_probe_pass" if result["load_ok"] else result["status"]
                values = [
                    allocation,
                    "not_run",
                    "not_run",
                    "not_run",
                    "not_run",
                    f"{result['elapsed_seconds']:.1f}s",
                    "not_run",
                ]
            elif result["stage"] == "suite":
                if result["status"].startswith("BLOCKED_BY_"):
                    values = [result["status"]] + ["not_run"] * 6
                else:
                    allocatable = str(result["maximum_allocatable_context"])
                    if result.get("allocatable_is_lower_bound"):
                        allocatable = f">={allocatable}"
                    values = [
                        allocatable,
                        str(result["maximum_reliable_context"]),
                        f"{result['retrieval_rate']:.0%}",
                        f"{result['prompt_ts']:.1f}",
                        f"{result['gen_ts']:.1f}",
                        f"{result['elapsed_seconds']:.1f}s",
                        f"{result['task_pass_rate']:.0%}",
                    ]
            elif result["stage"] == "performance":
                values = [
                    "not_run",
                    "not_run",
                    "not_run",
                    f"{result['prompt_ts']:.1f}",
                    f"{result['gen_ts']:.1f}",
                    f"{result['elapsed_seconds']:.1f}s",
                    "not_run",
                ]
            elif result["stage"] == "quality":
                values = [
                    "not_run",
                    "not_run",
                    "not_run",
                    "not_run",
                    "not_run",
                    f"{result['elapsed_seconds']:.1f}s",
                    f"{result['task_pass_rate']:.0%}",
                ]
            elif result["stage"] == "boundary":
                values = [
                    (
                        f">={result['maximum_allocatable_context']}"
                        if result["maximum_allocatable_context"]
                        and result.get("first_failed_context") is None
                        else str(result["maximum_allocatable_context"])
                    ),
                    "not_run",
                    "not_run",
                    "not_run",
                    "not_run",
                    f"{result['elapsed_seconds']:.1f}s",
                    "not_run",
                ]
            else:
                successful = result["status"] == "SUCCESS"
                measured = successful or result["stage"] == "repeated_retrieval"
                retrieval_rate = result.get(
                    "retrieval_rate", 1.0 if result["retrieved"] else 0.0
                )
                values = [
                    str(result["context_size"]) if measured else result["status"],
                    str(result["context_size"]) if result["retrieved"] else "0",
                    f"{retrieval_rate:.0%}",
                    f"{result['prompt_ts']:.1f}" if measured else "not_run",
                    f"{result['gen_ts']:.1f}" if measured else "not_run",
                    f"{result['elapsed_seconds']:.1f}s",
                    "not_run",
                ]
            lines.append(
                f"| {label} | {plan['policy']['cache_mode']} | "
                f"{' | '.join(values)} | `{manifest_name}` |"
            )
    lines.extend(
        [
            "",
            "These bounded measurements are diagnostic evidence, not the authoritative full matrix. `--full` remains planning-only.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_artifacts(plan: dict, output_dir: Path) -> tuple[Path, Path]:
    """Write a machine-readable plan manifest and human-readable matrix."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    prefix = plan["mode"]
    manifest_path = output_dir / f"{prefix}_plan_{stamp}.json"
    report_path = output_dir / f"{prefix}_matrix_{stamp}.md"
    manifest_path.write_text(json.dumps(plan, indent=2) + "\n")
    report_path.write_text(render_matrix(plan, manifest_path.name))
    return manifest_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan-only", action="store_true", help="Inventory and write the full execution plan (default)")
    mode.add_argument("--smoke", action="store_true", help="Execute a bounded two-model smoke benchmark")
    mode.add_argument("--retrieval", action="store_true", help="Run repeated retrieval for one model and device")
    mode.add_argument("--boundary", action="store_true", help="Discover allocatable context boundary for one configuration")
    mode.add_argument("--quality", action="store_true", help="Run deterministic task quality for one configuration")
    mode.add_argument("--performance", action="store_true", help="Run matched warm-up and measured performance")
    mode.add_argument("--suite", action="store_true", help="Run all validated stages for one configuration")
    mode.add_argument("--full", action="store_true", help="Write the full planned matrix contract; execution is not yet implemented")
    parser.add_argument("--models", help="Comma-separated exact GGUF basenames to include")
    parser.add_argument("--device", default="Vulkan0", help="Device for --retrieval")
    parser.add_argument("--ts-split", default=None, help="Tensor split for --retrieval")
    parser.add_argument("--context-size", type=int, default=512, help="Context for --retrieval")
    parser.add_argument(
        "--context-sizes",
        default="512,1024,2048,4096,8192",
        help="Ordered coarse contexts for --boundary",
    )
    parser.add_argument("--boundary-step", type=int, default=256)
    parser.add_argument("--repetitions", type=int, default=5, help="Repetitions per retrieval position")
    parser.add_argument("--max-tasks", type=int, default=2, help="Ordered validation tasks for --quality")
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--performance-repetitions", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128, help="Output tokens for --quality")
    parser.add_argument(
        "--dataset-dir",
        default=os.path.join(os.path.dirname(__file__), "datasets", "validation"),
        help="Dataset directory for --quality",
    )
    parser.add_argument("--reliability-threshold", type=float, default=0.80)
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Per-inference timeout for smoke execution (default: 180)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "results", "manifests"),
    )
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error("--timeout must be at least 1 second")

    selected_mode = (
        "full"
        if args.full
        else "retrieval"
        if args.retrieval
        else "boundary"
        if args.boundary
        else "quality"
        if args.quality
        else "performance"
        if args.performance
        else "suite"
        if args.suite
        else "smoke"
        if args.smoke
        else "plan"
    )
    models = discover_remote_models()
    if args.models:
        requested = {name.strip() for name in args.models.split(",") if name.strip()}
        models = [model for model in models if model["name"] in requested]
        missing = requested - {model["name"] for model in models}
        if missing:
            parser.error(f"models not found: {', '.join(sorted(missing))}")
    elif selected_mode == "smoke":
        models = select_smoke_models(models)
    if selected_mode == "smoke" and len(models) > 2:
        parser.error("--smoke accepts at most two models")
    if selected_mode in {"retrieval", "boundary", "quality", "performance", "suite"}:
        if len(models) != 1:
            parser.error(f"--{selected_mode} requires exactly one --models basename")
        if selected_mode in {"performance", "suite"}:
            if args.prompt_tokens < 1:
                parser.error("--prompt-tokens must be at least 1")
            if args.warmups < 0:
                parser.error("--warmups cannot be negative")
            if args.performance_repetitions < 1:
                parser.error("--performance-repetitions must be at least 1")
            if args.prompt_tokens + args.max_tokens > args.context_size:
                parser.error("prompt and output tokens must fit inside context")
        if selected_mode in {"quality", "suite"}:
            if args.max_tasks < 1:
                parser.error("--max-tasks must be at least 1")
            if args.max_tokens < 1:
                parser.error("--max-tokens must be at least 1")
        if selected_mode in {"boundary", "suite"}:
            try:
                context_sizes = [
                    int(value.strip())
                    for value in args.context_sizes.split(",")
                    if value.strip()
                ]
            except ValueError:
                parser.error("--context-sizes must contain integers")
            if not context_sizes or min(context_sizes) <= 16:
                parser.error("--context-sizes must exceed the 16-token reserve")
            if args.boundary_step < 1:
                parser.error("--boundary-step must be at least 1")
        if args.context_size <= 64:
            parser.error("--context-size must exceed the 64-token output reserve")
        if args.repetitions < 1:
            parser.error("--repetitions must be at least 1")
        if not 0 < args.reliability_threshold <= 1:
            parser.error("--reliability-threshold must be between 0 and 1")
        models[0]["configurations"] = [
            {
                "device": args.device,
                "tensor_split": args.ts_split,
                "mode": "full",
            }
        ]

    plan = build_plan(models, selected_mode)
    if selected_mode == "retrieval":
        plan["models"][0]["configurations"] = models[0]["configurations"]
        plan = execute_retrieval(
            plan,
            timeout=args.timeout,
            context_size=args.context_size,
            repetitions=args.repetitions,
            reliability_threshold=args.reliability_threshold,
        )
    elif selected_mode == "boundary":
        plan["models"][0]["configurations"] = models[0]["configurations"]
        plan = execute_boundary(
            plan,
            timeout=args.timeout,
            context_sizes=context_sizes,
            boundary_step=args.boundary_step,
        )
    elif selected_mode == "quality":
        plan["models"][0]["configurations"] = models[0]["configurations"]
        plan = execute_quality(
            plan,
            timeout=args.timeout,
            dataset_dir=args.dataset_dir,
            max_tasks=args.max_tasks,
            context_size=args.context_size,
            max_tokens=args.max_tokens,
        )
    elif selected_mode == "performance":
        plan["models"][0]["configurations"] = models[0]["configurations"]
        plan = execute_performance(
            plan,
            timeout=args.timeout,
            context_size=args.context_size,
            prompt_tokens=args.prompt_tokens,
            output_tokens=args.max_tokens,
            warmups=args.warmups,
            repetitions=args.performance_repetitions,
        )
    elif selected_mode == "suite":
        plan["models"][0]["configurations"] = models[0]["configurations"]
        plan = execute_suite(
            plan,
            timeout=args.timeout,
            context_sizes=context_sizes,
            boundary_step=args.boundary_step,
            retrieval_repetitions=args.repetitions,
            reliability_threshold=args.reliability_threshold,
            performance_context=args.context_size,
            prompt_tokens=args.prompt_tokens,
            output_tokens=args.max_tokens,
            warmups=args.warmups,
            performance_repetitions=args.performance_repetitions,
            dataset_dir=args.dataset_dir,
            max_tasks=args.max_tasks,
        )
    elif selected_mode == "smoke":
        plan = execute_smoke(plan, timeout=args.timeout)
    manifest_path, report_path = write_artifacts(plan, Path(args.output_dir))
    print(f"Inventory: {len(models)} model(s)")
    print(f"Mode: {selected_mode}; authoritative={plan['authoritative']}")
    print(f"Manifest: {manifest_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
