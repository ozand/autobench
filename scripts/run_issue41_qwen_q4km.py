#!/usr/bin/env python3
"""Run the reviewed, issue-scoped Qwen Q4_K_M follow-up for Issue #41."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from authoritative_bench import (
    FULL_POLICY,
    discover_remote_models,
    execute_suite,
    render_matrix,
)
from src.protocol_receipt import ProtocolReceiptError, load_and_validate_receipt
from src.statuses import sanitize_artifact

MODEL_NAME = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
EXPECTED_SIZE_BYTES = 491400032
EXPECTED_SHA256 = "74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db"
EXPECTED_JOB_COUNT = 3
PRIMARY_TIMEOUT = 600
FALLBACK_TIMEOUT = 1200
DEFAULT_OUTPUT_DIR = Path("results/runs/issue41-qwen-q4km-followup")
DEFAULT_RECEIPT = Path("results/receipts/qwen2.5-0.5b-instruct-q4_k_m.issue41.json")


def verify_artifact(model_path: Path) -> dict:
    """Verify the exact model artifact exists and matches size and SHA-256."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    size = model_path.stat().st_size
    if size != EXPECTED_SIZE_BYTES:
        raise ValueError(
            f"Model artifact size mismatch for {model_path.name}: "
            f"expected {EXPECTED_SIZE_BYTES}, got {size}"
        )
    hasher = hashlib.sha256()
    with model_path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    if digest.lower() != EXPECTED_SHA256.lower():
        raise ValueError(
            f"Model artifact SHA-256 mismatch for {model_path.name}: "
            f"expected {EXPECTED_SHA256}, got {digest}"
        )
    return {
        "basename": model_path.name,
        "size_bytes": size,
        "sha256": digest,
    }


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


def reviewed_plan(model: dict, output_dir: Path, timeout: int = PRIMARY_TIMEOUT) -> dict:
    """Build the sanitized plan contract without executing inference."""
    if timeout != PRIMARY_TIMEOUT:
        raise ValueError(f"Reviewed primary timeout is {PRIMARY_TIMEOUT}s; cannot override with {timeout}s")
    configurations = issue_configurations(model)
    return {
        "schema_version": 1,
        "issue": 41,
        "provenance": "Source: rerun follow-up under #41",
        "mode": "issue41_qwen_q4km",
        "authoritative": False,
        "execution_status": "planned_not_run",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "basename": model["name"],
            "size_bytes": model["size_bytes"],
            "quantization": "Q4_K_M",
            "backend": "vulkan",
        },
        "configurations": configurations,
        "policy": {
            "contexts_single_gpu": [1024],
            "contexts_dual_layer": [1024, 2048, 4096],
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
        "output_dir": str(output_dir),
        "stop_rule": "Stop at the first unexpected result and classify it before further inference.",
        "inference_authorized": False,
    }


def _suite_plan(model: dict, configuration: dict) -> dict:
    """Return the one-configuration plan accepted by execute_suite."""
    return {
        "schema_version": 1,
        "mode": "suite",
        "authoritative": False,
        "configurations": [configuration],
        "models": [model],
    }


def execute_job(
    model: dict,
    configuration: dict,
    output_dir: Path,
    index: int,
    total: int,
    timeout: int = PRIMARY_TIMEOUT,
) -> dict:
    """Execute one reviewed configuration with bound f16 KV."""
    if timeout != PRIMARY_TIMEOUT:
        raise ValueError(f"Reviewed primary timeout is {PRIMARY_TIMEOUT}s; cannot override with {timeout}s")
    is_dual = configuration["device"] == "Vulkan0,Vulkan1"
    contexts = [1024, 2048, 4096] if is_dual else [1024]
    suite_result = execute_suite(
        plan=_suite_plan(model, configuration),
        context_sizes=contexts,
        output_dir=output_dir,
        timeout_seconds=timeout,
        prompt_tokens=512,
        output_tokens=64,
        boundary_output_tokens=16,
        retrieval_repetitions=5,
        quality_tasks=2,
        performance_warmups=1,
        performance_repetitions=3,
        performance_context=1024,
        reliability_threshold=0.8,
        cache_type_k="f16",
        cache_type_v="f16",
        no_kv_offload=False,
    )
    result = {
        "job_index": index,
        "total_jobs": total,
        "device": configuration["device"],
        "split_mode": configuration["split_mode"],
        "tensor_split": configuration["tensor_split"],
        "status": suite_result.get("status"),
        "result": suite_result,
    }
    job_path = output_dir / f"job_{index:02d}.json"
    job_path.write_text(json.dumps(sanitize_artifact(result), indent=2), encoding="utf-8")
    return result


def execute(model: dict, output_dir: Path, timeout: int = PRIMARY_TIMEOUT) -> dict:
    """Execute the serial, stop-first-surprise 3-job suite for Qwen Q4_K_M."""
    if timeout != PRIMARY_TIMEOUT:
        raise ValueError(f"Reviewed primary timeout is {PRIMARY_TIMEOUT}s; cannot override with {timeout}s")
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_dict = reviewed_plan(model, output_dir, timeout=timeout)
    plan_dict["execution_status"] = "in_progress"
    plan_dict["inference_authorized"] = True

    configurations = issue_configurations(model)
    job_results = []
    terminal_status = "SUCCESS"

    for idx, config in enumerate(configurations, 1):
        job = execute_job(model, config, output_dir, idx, len(configurations), timeout=timeout)
        job_results.append(job)
        status = job.get("status")
        if status != "SUCCESS":
            terminal_status = status or "EXECUTION_ERROR"
            break

    summary = {
        "schema_version": 1,
        "issue": 41,
        "provenance": "Source: rerun follow-up under #41",
        "mode": "issue41_qwen_q4km",
        "authoritative": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "basename": model["name"],
            "size_bytes": model["size_bytes"],
            "sha256": EXPECTED_SHA256,
            "quantization": "Q4_K_M",
            "backend": "vulkan",
        },
        "configurations": configurations,
        "execution_status": terminal_status,
        "jobs_completed": len(job_results),
        "jobs_expected": len(configurations),
        "jobs": job_results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(sanitize_artifact(summary), indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print plan without running inference")
    parser.add_argument("--execute", action="store_true", help="Run the approved benchmark execution")
    parser.add_argument("--models", nargs="*", default=[MODEL_NAME], help="GGUF model basename")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Result directory")
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT, help="Dedicated receipt path")
    parser.add_argument("--timeout", type=int, default=PRIMARY_TIMEOUT, help="Primary timeout in seconds")
    args = parser.parse_args()

    if not args.dry_run and not args.execute:
        parser.error("Specify either --dry-run or --execute")
    if args.dry_run and args.execute:
        parser.error("Cannot combine --dry-run and --execute")
    if args.timeout != PRIMARY_TIMEOUT:
        parser.error(f"--timeout must be exactly {PRIMARY_TIMEOUT}")

    models = discover_remote_models()
    matched = [m for m in models if m["name"] in args.models]
    if not matched:
        print(f"ERROR: Model {args.models} not found", file=sys.stderr)
        return 1
    model = matched[0]

    # Validate receipt fail-closed
    receipt_result = load_and_validate_receipt(
        receipt_path=str(args.receipt),
        model_name=model["name"],
        backend="vulkan",
        governing_issue=41,
        check_files_exist=True,
        expected_job_count=EXPECTED_JOB_COUNT,
        expected_artifact={
            "basename": MODEL_NAME,
            "size_bytes": EXPECTED_SIZE_BYTES,
            "sha256": EXPECTED_SHA256,
        },
    )
    if receipt_result["status"] != "PROTOCOL_RECEIPT_VALID":
        print(f"ERROR: Receipt invalid: {receipt_result['errors']}", file=sys.stderr)
        return 2

    if args.dry_run:
        plan_dict = reviewed_plan(model, args.output_dir, timeout=args.timeout)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        plan_path = args.output_dir / "dry_run_plan.json"
        plan_path.write_text(json.dumps(plan_dict, indent=2), encoding="utf-8")
        print(
            f"Issue #41 Qwen Q4_K_M follow-up dry-run: "
            f"jobs={plan_dict['expected_job_count']}; "
            f"contexts={plan_dict['policy']['contexts_dual_layer']}; "
            f"inference=0"
        )
        print(f"Plan: {plan_path}")
        return 0

    if args.execute:
        # Check artifact before execution if available locally
        local_model_path = Path("models") / model["name"]
        if local_model_path.exists():
            verify_artifact(local_model_path)
        summary = execute(model, args.output_dir, timeout=args.timeout)
        print(f"Issue #41 execution completed: status={summary['execution_status']}, jobs={summary['jobs_completed']}/{summary['jobs_expected']}")
        return 0 if summary["execution_status"] == "SUCCESS" else 3

    return 0


if __name__ == "__main__":
    sys.exit(main())
