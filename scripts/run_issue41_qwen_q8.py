#!/usr/bin/env python3
"""Run the reviewed, issue-scoped Qwen Q8 follow-up for Issue #41."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from authoritative_bench import (
    FULL_POLICY,
    SINGLE_GPU_FIT_BYTES,
    discover_remote_models,
    execute_suite,
    render_matrix,
)
from src.protocol_receipt import ProtocolReceiptError, assert_all_models_have_valid_receipts
from src.statuses import sanitize_artifact

MODEL_NAME = "qwen2.5-0.5b-instruct-q8_0.gguf"
EXPECTED_SIZE_BYTES = 675710816
EXPECTED_JOB_COUNT = 3
EXPECTED_ALLOCATION_PROBES = 6
PRIMARY_TIMEOUT = 600
FALLBACK_TIMEOUT = 1200
DEFAULT_OUTPUT_DIR = Path("results/issue41-qwen-q8-followup")


def issue_configurations(model: dict) -> list[dict]:
    """Return the exact three reviewed suite configurations."""
    if model["name"] != MODEL_NAME:
        raise ValueError(f"unexpected model: {model['name']}")
    if model["size_bytes"] != EXPECTED_SIZE_BYTES:
        raise ValueError("target GGUF size does not match the reviewed artifact")
    return [
        {
            "device": "Vulkan0",
            "tensor_split": None,
            "split_mode": "none",
            "mode": "full",
            "cache_type_k": "f16",
            "cache_type_v": "f16",
            "no_kv_offload": False,
        },
        {
            "device": "Vulkan1",
            "tensor_split": None,
            "split_mode": "none",
            "mode": "full",
            "cache_type_k": "f16",
            "cache_type_v": "f16",
            "no_kv_offload": False,
        },
        {
            "device": "Vulkan0,Vulkan1",
            "tensor_split": "1,1",
            "split_mode": "layer",
            "mode": "full",
            "cache_type_k": "f16",
            "cache_type_v": "f16",
            "no_kv_offload": False,
        },
    ]


def reviewed_plan(model: dict, output_dir: Path, timeout: int) -> dict:
    """Build the sanitized plan contract without executing inference."""
    configurations = issue_configurations(model)
    return {
        "schema_version": 1,
        "issue": 41,
        "provenance": "Source: defect found while implementing #41",
        "mode": "issue41_qwen_q8",
        "authoritative": False,
        "execution_status": "planned_not_run",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "basename": model["name"],
            "size_bytes": model["size_bytes"],
            "quantization": "Q8_0",
            "backend": "vulkan",
        },
        "configurations": configurations,
        "policy": {
            "contexts_single_gpu": [1024],
            "contexts_dual_layer": [1024, 2048, 4096, 8192],
            "cache_type_k": "f16",
            "cache_type_v": "f16",
            "no_kv_offload": False,
            "prompt_tokens": 512,
            "output_tokens": 64,
            "boundary_output_tokens": 16,
            "retrieval_positions": FULL_POLICY["retrieval_positions"],
            "retrieval_repetitions": 5,
            "quality_tasks": 2,
            "performance_warmups": 1,
            "performance_repetitions": 3,
            "reliability_threshold": 0.8,
            "timeout_seconds": timeout,
            "fallback_timeout_seconds": FALLBACK_TIMEOUT,
            "fallback_condition": "Only after a primary 600-second run has no terminal result; record separately.",
        },
        "expected_job_count": EXPECTED_JOB_COUNT,
        "allocation_probe_count": EXPECTED_ALLOCATION_PROBES,
        "output_dir": output_dir.name,
        "stop_rule": "Stop at the first unexpected result and classify it before further inference.",
        "inference_authorized": False,
    }


def _suite_plan(model: dict, configuration: dict) -> dict:
    """Return the one-configuration plan accepted by execute_suite."""
    return {
        "schema_version": 1,
        "mode": "suite",
        "authoritative": False,
        "execution_status": "planned_not_run",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": "k7000",
        "gpu_memory_bytes_per_device": 2 * 1024**3,
        "single_gpu_fit_threshold_bytes": SINGLE_GPU_FIT_BYTES,
        "policy": dict(FULL_POLICY),
        "models": [
            {
                "id": model["id"],
                "name": model["name"],
                "path": model["path"],
                "size_bytes": model["size_bytes"],
                "configurations": [configuration],
            }
        ],
    }


def execute_reviewed_plan(model: dict, output_dir: Path, timeout: int) -> dict:
    """Execute the three suite jobs serially and stop on the first surprise."""
    configurations = issue_configurations(model)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = reviewed_plan(model, output_dir, timeout)
    summary["execution_status"] = "completed"
    summary["inference_authorized"] = True
    summary["jobs"] = []

    for index, configuration in enumerate(configurations, start=1):
        contexts = [1024] if configuration["device"] != "Vulkan0,Vulkan1" else [1024, 2048, 4096, 8192]
        result = execute_suite(
            _suite_plan(model, configuration),
            timeout=timeout,
            context_sizes=contexts,
            boundary_step=256,
            retrieval_repetitions=5,
            reliability_threshold=0.8,
            performance_context=1024,
            prompt_tokens=512,
            output_tokens=64,
            warmups=1,
            performance_repetitions=3,
            dataset_dir="datasets/validation",
            max_tasks=2,
        )
        job = {
            "job_index": index,
            "configuration": configuration,
            "contexts": contexts,
            "status": result["models"][0]["configurations"][0].get("result", {}).get("status", "EXECUTION_ERROR"),
            "result": result,
        }
        job_path = output_dir / f"job_{index:02d}.json"
        job_path.write_text(json.dumps(sanitize_artifact(job), indent=2) + "\n", encoding="utf-8")
        summary["jobs"].append({
            "job_index": index,
            "configuration": configuration,
            "contexts": contexts,
            "status": job["status"],
            "artifact": job_path.name,
        })
        if job["status"] != "SUCCESS":
            summary["execution_status"] = "stopped_after_first_unexpected_result"
            summary["stopped_after_job"] = index
            break

    summary["completed_job_count"] = len(summary["jobs"])
    summary["all_jobs_completed"] = len(summary["jobs"]) == EXPECTED_JOB_COUNT
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(sanitize_artifact(summary), indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Preview without inference")
    mode.add_argument("--execute", action="store_true", help="Execute the reviewed serial plan")
    parser.add_argument("--models", required=True, help="The exact GGUF basename")
    parser.add_argument("--receipt", required=True, help="The Issue #41 protocol receipt")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout", type=int, default=PRIMARY_TIMEOUT)
    args = parser.parse_args()
    if args.models != MODEL_NAME:
        parser.error("--models must name the exact Qwen Q8 GGUF")
    if args.timeout < 1:
        parser.error("--timeout must be positive")

    models = [model for model in discover_remote_models() if model["name"] == MODEL_NAME]
    if len(models) != 1:
        parser.error("exact target GGUF was not found exactly once")
    model = models[0]
    try:
        assert_all_models_have_valid_receipts(
            [model],
            receipt_paths={MODEL_NAME: args.receipt},
            governing_issue=41,
            check_files_exist=True,
        )
    except ProtocolReceiptError as exc:
        parser.error(str(exc))

    plan = reviewed_plan(model, args.output_dir, args.timeout)
    if args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        plan_path = args.output_dir / "dry_run_plan.json"
        report_path = args.output_dir / "dry_run_matrix.md"
        plan_path.write_text(json.dumps(sanitize_artifact(plan), indent=2) + "\n", encoding="utf-8")
        report_path.write_text(render_matrix({
            "mode": "issue41_qwen_q8",
            "policy": {"cache_mode": "f16"},
            "models": [{**model, "configurations": issue_configurations(model)}],
        }, plan_path.name), encoding="utf-8")
        print(f"Issue #41 Qwen Q8 dry-run: jobs={EXPECTED_JOB_COUNT}; allocation_probes={EXPECTED_ALLOCATION_PROBES}; inference=0")
        print(f"Plan: {plan_path}")
        print(f"Report: {report_path}")
        return

    summary = execute_reviewed_plan(model, args.output_dir, args.timeout)
    print(json.dumps({
        "issue": 41,
        "model": MODEL_NAME,
        "execution_status": summary["execution_status"],
        "completed_job_count": summary["completed_job_count"],
        "expected_job_count": EXPECTED_JOB_COUNT,
        "all_jobs_completed": summary["all_jobs_completed"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
