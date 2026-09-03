#!/usr/bin/env python3
"""Run the reviewed one-job same-context Qwen Q8 follow-up for Issue #41."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from authoritative_bench import FULL_POLICY, execute_suite
from src.protocol_receipt import ProtocolReceiptError, assert_all_models_have_valid_receipts
from src.remote import run_host_command
from src.statuses import sanitize_artifact

MODEL_NAME = "qwen2.5-0.5b-instruct-q8_0.gguf"
EXPECTED_SIZE_BYTES = 675710816
PRIMARY_TIMEOUT = 600
DECLARED_CONTEXT = 1024
DEFAULT_OUTPUT_DIR = Path("results/issue41-qwen-q8-same-context")


def configuration(model: dict) -> dict:
    if model["name"] != MODEL_NAME:
        raise ValueError("unexpected model basename")
    if model["size_bytes"] != EXPECTED_SIZE_BYTES:
        raise ValueError("target GGUF size does not match the reviewed artifact")
    return {
        "device": "Vulkan0,Vulkan1",
        "tensor_split": "1,1",
        "split_mode": "layer",
        "mode": "full",
        "cache_type_k": "f16",
        "cache_type_v": "f16",
        "no_kv_offload": False,
        "declared_context": DECLARED_CONTEXT,
    }


def verify_artifact(model: dict) -> None:
    """Verify the discovered target file against the reviewed checksum."""
    result = run_host_command(
        f"sha256sum /home/opencode/llama.cpp/models/{MODEL_NAME} | cut -d ' ' -f1",
        timeout=120,
    )
    expected = "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e"
    if result.returncode != 0 or result.stdout.strip() != expected:
        raise ValueError("target GGUF checksum does not match the reviewed artifact")


def plan(model: dict, output_dir: Path, timeout: int) -> dict:
    return {
        "schema_version": 1,
        "issue": 41,
        "provenance": "Source: defect found while implementing #41",
        "mode": "issue41_qwen_q8_same_context",
        "execution_status": "planned_not_run",
        "authoritative": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "basename": model["name"],
            "size_bytes": model["size_bytes"],
            "quantization": "Q8_0",
            "backend": "vulkan",
        },
        "configuration": configuration(model),
        "declared_context": DECLARED_CONTEXT,
        "workload": {
            "prompt_tokens": 512,
            "output_tokens": 64,
            "boundary_output_tokens": 16,
            "retrieval_positions": FULL_POLICY["retrieval_positions"],
            "retrieval_repetitions": 5,
            "quality_tasks": 2,
            "performance_warmups": 1,
            "performance_repetitions": 3,
            "timeout_seconds": timeout,
        },
        "expected_job_count": 1,
        "output_dir": output_dir.name,
        "publication_rule": "All publication metrics must use declared context 1024; no mixed-context row is eligible.",
        "inference_authorized": False,
    }


def execute(model: dict, output_dir: Path, timeout: int) -> dict:
    config = configuration(model)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = execute_suite(
        {
            "schema_version": 1,
            "mode": "suite",
            "authoritative": False,
            "execution_status": "planned_not_run",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "host": "k7000",
            "gpu_memory_bytes_per_device": 2 * 1024**3,
            "single_gpu_fit_threshold_bytes": 1900000000,
            "policy": dict(FULL_POLICY),
            "models": [{
                "id": Path(model["name"]).stem,
                "name": model["name"],
                "path": model["path"],
                "size_bytes": model["size_bytes"],
                "configurations": [config],
            }],
        },
        timeout=timeout,
        context_sizes=[DECLARED_CONTEXT],
        boundary_step=256,
        retrieval_repetitions=5,
        reliability_threshold=0.8,
        performance_context=DECLARED_CONTEXT,
        prompt_tokens=512,
        output_tokens=64,
        warmups=1,
        performance_repetitions=3,
        dataset_dir="datasets/validation",
        max_tasks=2,
    )
    suite_result = result["models"][0]["configurations"][0].get("result", {})
    summary = plan(model, output_dir, timeout)
    summary.update({
        "execution_status": "completed",
        "inference_authorized": True,
        "job_status": suite_result.get("status", "EXECUTION_ERROR"),
        "result": suite_result,
    })
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(sanitize_artifact(summary), indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--models", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout", type=int, default=PRIMARY_TIMEOUT)
    args = parser.parse_args()
    if args.models != MODEL_NAME:
        parser.error("--models must name the exact Qwen Q8 GGUF")
    expected_receipt = "results/receipts/qwen2.5-0.5b-instruct-q8_0.issue41.same-context.json"
    if args.receipt != expected_receipt:
        parser.error("--receipt must name the same-context Issue #41 receipt")
    if args.timeout != PRIMARY_TIMEOUT:
        parser.error("--timeout must remain the reviewed primary value of 600 seconds")

    models = [m for m in __import__("authoritative_bench").discover_remote_models() if m["name"] == MODEL_NAME]
    if len(models) != 1:
        parser.error("exact target GGUF was not found exactly once")
    try:
        assert_all_models_have_valid_receipts(
            models,
            receipt_paths={MODEL_NAME: args.receipt},
            governing_issue=41,
            check_files_exist=True,
        )
        verify_artifact(models[0])
    except (ProtocolReceiptError, ValueError) as exc:
        parser.error(str(exc))
    model = models[0]
    if args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        p = args.output_dir / "dry_run_plan.json"
        p.write_text(json.dumps(sanitize_artifact(plan(model, args.output_dir, args.timeout)), indent=2) + "\n", encoding="utf-8")
        print(f"Issue #41 Qwen Q8 same-context dry-run: jobs=1; context={DECLARED_CONTEXT}; inference=0")
        print(f"Plan: {p}")
        return
    summary = execute(model, args.output_dir, args.timeout)
    print(json.dumps({
        "issue": 41,
        "model": MODEL_NAME,
        "context": DECLARED_CONTEXT,
        "job_status": summary["job_status"],
        "publication_context_consistent": summary["declared_context"] == DECLARED_CONTEXT,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
