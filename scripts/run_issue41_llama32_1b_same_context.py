#!/usr/bin/env python3
"""Run the reviewed one-job same-context Llama 3.2 1B follow-up for Issue #41."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from authoritative_bench import FULL_POLICY, discover_remote_models, execute_suite
from src.protocol_receipt import load_and_validate_receipt
from src.remote import run_host_command
from src.statuses import sanitize_artifact

MODEL_NAME = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
EXPECTED_SIZE_BYTES = 807694368
EXPECTED_SHA256 = "3f5a22426976ab26cfe84dba63c1d08391717abb1af893e10f1b2968d862dcc1"
PRIMARY_TIMEOUT = 600
DECLARED_CONTEXT = 1024
EXPECTED_JOB_COUNT = 1
DEFAULT_RECEIPT = Path("results/receipts/Llama-3.2-1B-Instruct-Q4_K_M.issue41.same-context.json")
DEFAULT_OUTPUT_DIR = Path("results/issue41-llama32-1b-same-context")


def configuration(model: dict) -> dict:
    if model["name"] != MODEL_NAME:
        raise ValueError("unexpected model basename")
    if model["size_bytes"] != EXPECTED_SIZE_BYTES:
        raise ValueError("target GGUF size does not match reviewed artifact")
    return {"device": "Vulkan0,Vulkan1", "tensor_split": "1,1", "split_mode": "layer", "mode": "full", "cache_type_k": "f16", "cache_type_v": "f16", "no_kv_offload": False, "declared_context": DECLARED_CONTEXT}


def verify_artifact() -> None:
    result = run_host_command(f"sha256sum /home/opencode/llama.cpp/models/{MODEL_NAME} | cut -d ' ' -f1", timeout=120)
    if result.returncode != 0 or result.stdout.strip() != EXPECTED_SHA256:
        raise ValueError("target GGUF checksum does not match reviewed artifact")


def suite_plan(model: dict) -> dict:
    return {"schema_version": 1, "mode": "suite", "authoritative": False, "execution_status": "planned_not_run", "generated_at": datetime.now(timezone.utc).isoformat(), "host": "k7000", "gpu_memory_bytes_per_device": 2 * 1024**3, "single_gpu_fit_threshold_bytes": 1900000000, "policy": dict(FULL_POLICY), "models": [{"id": Path(model["name"]).stem, "name": model["name"], "path": model["path"], "size_bytes": model["size_bytes"], "configurations": [configuration(model)]}]}


def plan(model: dict, output_dir: Path, timeout: int) -> dict:
    if timeout != PRIMARY_TIMEOUT:
        raise ValueError("timeout must remain the reviewed primary value of 600 seconds")
    return {"schema_version": 1, "issue": 41, "provenance": "Source: governed same-context follow-up under #41", "mode": "issue41_llama32_1b_same_context", "execution_status": "planned_not_run", "generated_at": datetime.now(timezone.utc).isoformat(), "model": {"basename": model["name"], "size_bytes": model["size_bytes"], "quantization": "Q4_K_M", "backend": "vulkan"}, "configuration": configuration(model), "declared_context": DECLARED_CONTEXT, "workload": {"prompt_tokens": 512, "output_tokens": 64, "boundary_output_tokens": 16, "retrieval_positions": FULL_POLICY["retrieval_positions"], "retrieval_repetitions": 5, "quality_tasks": 2, "performance_warmups": 1, "performance_repetitions": 3, "timeout_seconds": timeout}, "expected_job_count": EXPECTED_JOB_COUNT, "output_dir": output_dir.name, "publication_rule": "All publication metrics must use declared context 1024; no mixed-context row is eligible.", "inference_authorized": False}


def execute(model: dict, output_dir: Path, timeout: int) -> dict:
    if timeout != PRIMARY_TIMEOUT:
        raise ValueError("timeout must remain the reviewed primary value of 600 seconds")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = execute_suite(suite_plan(model), timeout=timeout, context_sizes=[DECLARED_CONTEXT], boundary_step=256, retrieval_repetitions=5, reliability_threshold=0.8, performance_context=DECLARED_CONTEXT, prompt_tokens=512, output_tokens=64, warmups=1, performance_repetitions=3, dataset_dir="datasets/validation", max_tasks=2)
    data = result["models"][0]["configurations"][0].get("result", {})
    summary = plan(model, output_dir, timeout)
    summary.update({"execution_status": "completed", "inference_authorized": True, "job_status": data.get("status", "EXECUTION_ERROR"), "result": data})
    (output_dir / "summary.json").write_text(json.dumps(sanitize_artifact(summary), indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--models", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout", type=int, default=PRIMARY_TIMEOUT)
    args = parser.parse_args()
    if args.models != MODEL_NAME:
        parser.error("--models must name exact Llama GGUF")
    if args.receipt != DEFAULT_RECEIPT:
        parser.error("--receipt must name Issue #41 same-context receipt")
    if args.timeout != PRIMARY_TIMEOUT:
        parser.error("--timeout must remain 600")
    matches = [m for m in discover_remote_models() if m["name"] == MODEL_NAME]
    if len(matches) != 1:
        parser.error("exact target GGUF was not found exactly once")
    model = matches[0]
    try:
        validation = load_and_validate_receipt(args.receipt, model_name=MODEL_NAME, backend="vulkan", governing_issue=41, check_files_exist=True, expected_job_count=EXPECTED_JOB_COUNT, expected_artifact={"basename": MODEL_NAME, "size_bytes": EXPECTED_SIZE_BYTES, "sha256": EXPECTED_SHA256}, expected_configurations=[configuration(model)])
        if validation["status"] != "PROTOCOL_RECEIPT_VALID":
            parser.error("protocol receipt validation failed")
        verify_artifact()
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out = args.output_dir / "dry_run_plan.json"
        out.write_text(json.dumps(plan(model, args.output_dir, args.timeout), indent=2) + "\n", encoding="utf-8")
        print("Issue #41 Llama 3.2 1B same-context dry-run: jobs=1; context=1024; inference=0")
        print(f"Plan: {out}")
        return 0
    summary = execute(model, args.output_dir, args.timeout)
    print(json.dumps({"issue": 41, "model": MODEL_NAME, "context": DECLARED_CONTEXT, "job_status": summary["job_status"], "publication_context_consistent": summary["declared_context"] == DECLARED_CONTEXT}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
