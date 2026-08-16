#!/usr/bin/env python3
"""Run the validated AutoBench suite across a resumable GGUF inventory."""

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from authoritative_bench import (
    build_plan,
    discover_remote_models,
    execute_performance,
    execute_suite,
    run_load_probe,
    select_working_configuration,
    SINGLE_GPU_FIT_BYTES,
)
from src.statuses import sanitize_artifact


def job_id(model: dict, config: dict) -> str:
    """Return a stable filesystem-safe identifier for one model/configuration."""
    parts = [
        model["id"],
        config["device"],
        config.get("tensor_split") or "none",
        config["mode"],
    ]
    if config.get("split_mode"):
        parts.append(config["split_mode"])
    raw = "_".join(parts)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def build_jobs(models: list[dict], include_tensor: bool = False) -> list[dict]:
    """Expand the inventory plan into deterministic execution jobs."""
    plan = build_plan(models, "matrix" if include_tensor else "inventory")
    jobs = []
    for model in plan["models"]:
        for config in model["configurations"]:
            jobs.append(
                {
                    "id": job_id(model, config),
                    "model": {
                        key: model[key]
                        for key in ("id", "name", "path", "size_bytes")
                    },
                    "config": config,
                }
            )
    return jobs


def select_tensor_validation_models(models: list[dict], requested_names: list[str]) -> list[dict]:
    """Resolve exactly one named model on each side of the single-GPU threshold."""
    if len(requested_names) != 2 or len(set(requested_names)) != 2:
        raise ValueError("tensor validation requires exactly two distinct GGUF basenames")
    by_name = {model["name"]: model for model in models}
    missing = [name for name in requested_names if name not in by_name]
    if missing:
        raise ValueError(f"models not found: {', '.join(sorted(missing))}")
    selected = [by_name[name] for name in requested_names]
    small = [model for model in selected if model["size_bytes"] <= SINGLE_GPU_FIT_BYTES]
    large = [model for model in selected if model["size_bytes"] > SINGLE_GPU_FIT_BYTES]
    if len(small) != 1 or len(large) != 1:
        raise ValueError("tensor validation requires one GGUF <= 1.9 GB and one GGUF > 1.9 GB")
    return small + large


def build_tensor_validation_jobs(models: list[dict]) -> list[dict]:
    """Build the fixed six-job validation matrix without the full inventory suite."""
    if len(models) != 2:
        raise ValueError("tensor validation requires exactly two models")
    small, large = models
    if small["size_bytes"] > SINGLE_GPU_FIT_BYTES or large["size_bytes"] <= SINGLE_GPU_FIT_BYTES:
        raise ValueError("tensor validation models must be ordered small then large")
    configurations = {
        small["id"]: [
            {"device": "Vulkan0", "tensor_split": None, "mode": "performance"},
            {"device": "Vulkan1", "tensor_split": None, "mode": "performance"},
        ],
        large["id"]: [
            {"device": "Vulkan0", "tensor_split": None, "mode": "load_only"},
            {"device": "Vulkan1", "tensor_split": None, "mode": "load_only"},
            {"device": "Vulkan0,Vulkan1", "tensor_split": "1,1", "mode": "performance"},
            {
                "device": "Vulkan0,Vulkan1",
                "tensor_split": "1,1",
                "split_mode": "tensor",
                "mode": "performance",
                "capability_probe": "pending",
            },
        ],
    }
    jobs = []
    for model in models:
        for config in configurations[model["id"]]:
            jobs.append(
                {
                    "id": job_id(model, config),
                    "model": {key: model[key] for key in ("id", "name", "path", "size_bytes")},
                    "config": config,
                }
            )
    if len(jobs) != 6:
        raise ValueError("tensor validation must build exactly six jobs")
    return jobs


def policy_fingerprint(args: argparse.Namespace) -> dict:
    """Return the benchmark settings that determine job comparability."""
    return {
        "context_sizes": args.context_sizes,
        "boundary_step": args.boundary_step,
        "context_size": args.context_size,
        "repetitions": args.repetitions,
        "reliability_threshold": args.reliability_threshold,
        "prompt_tokens": args.prompt_tokens,
        "max_tokens": args.max_tokens,
        "warmups": args.warmups,
        "performance_repetitions": args.performance_repetitions,
        "max_tasks": args.max_tasks,
        "dataset_dir": Path(args.dataset_dir).name,
    }


def completed_job(path: Path, fingerprint: dict | None = None) -> bool:
    """Return true when a prior terminal manifest matches the requested policy."""
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    terminal = data.get("execution_status") in {
        "completed_load_probe",
        "completed_suite",
    }
    if not terminal:
        return False
    if fingerprint is None or data.get("execution_status") == "completed_load_probe":
        return True
    return data.get("policy_fingerprint") == fingerprint


def execute_tensor_validation_job(job: dict, args: argparse.Namespace) -> dict:
    """Run one bounded load-only or short performance validation job."""
    model = job["model"]
    config = job["config"]
    started_at = datetime.now(timezone.utc).isoformat()
    preflight = run_load_probe(model, config, args.timeout)
    if config["mode"] == "load_only" or not preflight["load_ok"]:
        return {
            "schema_version": 2,
            "job_id": job["id"],
            "execution_status": "completed_load_probe",
            "authoritative": False,
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "config": config,
            "workload": {"context_size": 128, "max_tokens": 1, "timeout": args.timeout},
            "result": preflight,
        }

    performance_config = dict(config)
    performance_config["mode"] = "full"
    plan = build_plan([model], "performance")
    plan["models"][0]["configurations"] = [performance_config]
    measured = execute_performance(
        plan,
        timeout=args.timeout,
        context_size=args.context_size,
        prompt_tokens=args.prompt_tokens,
        output_tokens=args.max_tokens,
        warmups=args.warmups,
        repetitions=args.performance_repetitions,
    )
    result = measured["models"][0]["configurations"][0]["result"]
    return {
        "schema_version": 2,
        "job_id": job["id"],
        "execution_status": "completed_performance",
        "authoritative": False,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "config": config,
        "workload": {
            "context_size": args.context_size,
            "prompt_tokens": args.prompt_tokens,
            "max_tokens": args.max_tokens,
            "warmups": args.warmups,
            "performance_repetitions": args.performance_repetitions,
            "timeout": args.timeout,
        },
        "result": result,
        "preflight": preflight,
    }


def execute_job(job: dict, args: argparse.Namespace) -> dict:
    """Execute one configuration with its own bounded preflight."""
    model = job["model"]
    config = job["config"]
    if config["mode"] in {"load_only", "unsupported"}:
        result = (
            {
                "stage": "load_probe",
                "probe_type": "tensor_split_capability",
                "status": config.get("capability_status", "UNSUPPORTED_BACKEND"),
                "load_ok": False,
                "elapsed_seconds": 0.0,
                "return_code": None,
                "command_args": [],
            }
            if config["mode"] == "unsupported"
            else run_load_probe(model, config, args.timeout)
        )
        return {
            "schema_version": 1,
            "policy_fingerprint": policy_fingerprint(args),
            "job_id": job["id"],
            "execution_status": "completed_load_probe",
            "authoritative": False,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "config": config,
            "result": result,
        }

    plan = build_plan([model], "suite")
    plan["models"][0]["configurations"] = [config]
    result = execute_suite(
        plan,
        timeout=args.timeout,
        context_sizes=args.context_sizes,
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
    result["job_id"] = job["id"]
    result["policy_fingerprint"] = policy_fingerprint(args)
    return result


def inventory_status(
    jobs: list[dict], output_dir: Path, fingerprint: dict | None = None
) -> dict:
    """Summarize cumulative job state from persisted local manifests."""
    status = {"completed": [], "stale": [], "failed": [], "pending": []}
    for job in jobs:
        path = output_dir / f"{job['id']}.json"
        if completed_job(path, fingerprint):
            status["completed"].append(job["id"])
        elif path.exists():
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                status["failed"].append(job["id"])
            else:
                if data.get("execution_status") in {
                    "completed_load_probe",
                    "completed_suite",
                }:
                    status["stale"].append(job["id"])
                else:
                    status["failed"].append(job["id"])
        else:
            status["pending"].append(job["id"])
    return status


def select_contiguous_jobs(jobs: list[dict], max_jobs: int = 0) -> list[dict]:
    """Select whole per-model batches without interleaving or splitting models."""
    if not max_jobs:
        return list(jobs)

    selected: list[dict] = []
    batch: list[dict] = []
    current_model = None
    for job in jobs:
        model = job.get("model", {})
        model_id = model.get("id") or model.get("name") or job["id"]
        if batch and model_id != current_model:
            if selected and len(selected) + len(batch) > max_jobs:
                break
            selected.extend(batch)
            batch = []
        current_model = model_id
        batch.append(job)
    if batch and (not selected or len(selected) + len(batch) <= max_jobs):
        selected.extend(batch)
    elif batch and not selected:
        # A model is the scheduling atom; do not split its configurations.
        selected.extend(batch)
    return selected


def run_tensor_validation(
    jobs: list[dict], output_dir: Path, args: argparse.Namespace
) -> dict:
    """Run the fixed matrix serially within an overall deadline."""
    if len(jobs) != args.expected_jobs:
        raise ValueError(f"expected {args.expected_jobs} jobs, planned {len(jobs)}")
    summary = {
        "schema_version": 2,
        "mode": "tensor_validation",
        "expected_job_count": args.expected_jobs,
        "completed": [],
        "failed": [],
        "pending": [job["id"] for job in jobs],
        "deadline_reached": False,
    }
    if args.dry_run:
        return summary

    output_dir.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.overall_timeout
    for job in jobs:
        if time.monotonic() >= deadline:
            summary["deadline_reached"] = True
            break
        path = output_dir / f"{job['id']}.json"
        try:
            result = execute_tensor_validation_job(job, args)
            path.write_text(json.dumps(sanitize_artifact(result), indent=2) + "\n")
            summary["completed"].append(job["id"])
        except (RuntimeError, ValueError) as exc:
            failure = {
                "schema_version": 2,
                "job_id": job["id"],
                "execution_status": "failed",
                "authoritative": False,
                "status": "EXECUTION_ERROR",
                "error": str(exc),
                "model": job["model"],
                "config": job["config"],
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            path.write_text(json.dumps(sanitize_artifact(failure), indent=2) + "\n")
            summary["failed"].append(job["id"])
        summary["pending"] = [
            candidate["id"]
            for candidate in jobs
            if candidate["id"] not in summary["completed"] + summary["failed"]
        ]
    return summary


def run_inventory(
    jobs: list[dict], output_dir: Path, args: argparse.Namespace
) -> dict:
    """Run pending jobs in model-contiguous batches and persist each result."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = policy_fingerprint(args)
    summary = {"completed": [], "skipped": [], "failed": [], "pending": []}
    runnable = []
    for job in jobs:
        path = output_dir / f"{job['id']}.json"
        if completed_job(path, fingerprint) and not args.force:
            summary["skipped"].append(job["id"])
        else:
            runnable.append(job)

    selected = select_contiguous_jobs(runnable, args.max_jobs)
    selected_ids = {job["id"] for job in selected}
    summary["pending"] = [job["id"] for job in runnable if job["id"] not in selected_ids]
    if args.dry_run:
        summary["pending"] = [job["id"] for job in runnable]
        return summary

    for job in selected:
        path = output_dir / f"{job['id']}.json"
        try:
            result = execute_job(job, args)
            path.write_text(json.dumps(sanitize_artifact(result), indent=2) + "\n")
            summary["completed"].append(job["id"])
        except (RuntimeError, ValueError) as exc:
            failure = {
                "schema_version": 1,
                "job_id": job["id"],
                "execution_status": "failed",
                "authoritative": False,
                "error": str(exc),
                "model": job["model"],
                "config": job["config"],
            }
            path.write_text(json.dumps(sanitize_artifact(failure), indent=2) + "\n")
            summary["failed"].append(job["id"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", help="Comma-separated exact GGUF basenames")
    parser.add_argument("--tensor-validation", action="store_true", help="Run the fixed bounded two-GGUF tensor matrix")
    parser.add_argument("--expected-jobs", type=int, default=6, help="Required job count for --tensor-validation")
    parser.add_argument("--overall-timeout", type=int, default=1800, help="Overall seconds allowed for --tensor-validation")
    parser.add_argument("--max-jobs", type=int, default=0, help="Run at most N pending jobs; 0 means all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true", help="Show cumulative persisted state without inference")
    parser.add_argument("--force", action="store_true", help="Rerun completed jobs")
    parser.add_argument("--context-sizes", default="512,1024,2048,4096,8192")
    parser.add_argument("--boundary-step", type=int, default=256)
    parser.add_argument("--context-size", type=int, default=1024)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--reliability-threshold", type=float, default=0.80)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--performance-repetitions", type=int, default=3)
    parser.add_argument("--max-tasks", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--dataset-dir",
        default=str(Path(__file__).parent / "datasets" / "validation"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent / "results" / "inventory"),
    )
    args = parser.parse_args()
    if args.max_jobs < 0 or args.timeout < 1 or args.overall_timeout < 1:
        parser.error("--max-jobs cannot be negative and timeouts must be positive")
    if args.expected_jobs < 1:
        parser.error("--expected-jobs must be positive")
    if args.tensor_validation and (args.status or args.max_jobs or args.force):
        parser.error("--tensor-validation cannot be combined with --status, --max-jobs, or --force")
    try:
        args.context_sizes = [
            int(value.strip()) for value in args.context_sizes.split(",") if value.strip()
        ]
    except ValueError:
        parser.error("--context-sizes must contain integers")

    models = discover_remote_models()
    requested_names = [value.strip() for value in (args.models or "").split(",") if value.strip()]
    if args.tensor_validation:
        try:
            models = select_tensor_validation_models(models, requested_names)
            jobs = build_tensor_validation_jobs(models)
        except ValueError as exc:
            parser.error(str(exc))
        if len(jobs) != args.expected_jobs:
            parser.error(f"expected {args.expected_jobs} jobs, planned {len(jobs)}")
        print("Tensor validation plan")
        print(f"Models selected: {len(models)}")
        print("Small models: 1")
        print("Large models: 1")
        print(f"Jobs planned: {len(jobs)}")
        print(f"Inference calls executed: {0 if args.dry_run else 'bounded'}")
        for job in jobs:
            model = job["model"]
            config = job["config"]
            size_class = "small" if model["size_bytes"] <= SINGLE_GPU_FIT_BYTES else "large"
            print(
                f"- {model['name']} | {model['size_bytes']} | {size_class} | "
                f"{config['device']} | {config['mode']} | "
                f"{config.get('tensor_split') or '-'} | {config.get('split_mode') or '-'}"
            )
        summary = run_tensor_validation(jobs, Path(args.output_dir), args)
        if args.dry_run:
            print("Job manifests written: 0")
            return
        summary_path = Path(args.output_dir) / "summary.json"
        summary_path.write_text(json.dumps(sanitize_artifact(summary), indent=2) + "\n")
        print(f"Summary: {summary_path}")
        return

    if args.models:
        requested = set(requested_names)
        models = [model for model in models if model["name"] in requested]
        missing = requested - {model["name"] for model in models}
        if missing:
            parser.error(f"models not found: {', '.join(sorted(missing))}")

    # Tensor mode was rejected by the bounded hardware gate and remains
    # diagnostic-only; the authoritative inventory uses the validated matrix.
    jobs = build_jobs(models, include_tensor=False)
    if args.status:
        summary = inventory_status(
            jobs, Path(args.output_dir), policy_fingerprint(args)
        )
        summary["skipped"] = []
    else:
        summary = run_inventory(jobs, Path(args.output_dir), args)
    if not args.status:
        selections = {}
        for model in models:
            model_jobs = [job for job in jobs if job["model"]["id"] == model["id"]]
            configs = []
            for job in model_jobs:
                path = Path(args.output_dir) / f"{job['id']}.json"
                if not path.exists():
                    continue
                try:
                    data = json.loads(path.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                config = dict(job["config"])
                if data.get("execution_status") == "completed_suite":
                    model_rows = data.get("models") or []
                    config_rows = model_rows[0].get("configurations", []) if model_rows else []
                    config["result"] = config_rows[0].get("result", {}) if config_rows else {}
                else:
                    config["result"] = data.get("result", {})
                configs.append(config)
            selections[model["id"]] = select_working_configuration(configs)
        summary["selections"] = selections

    summary_name = "status.json" if args.status else "summary.json"
    summary_path = Path(args.output_dir) / summary_name
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Inventory models: {len(models)}; jobs: {len(jobs)}")
    counts = {key: len(value) for key, value in summary.items()}
    status_parts = [f"completed: {counts['completed']}"]
    if "stale" in counts:
        status_parts.append(f"stale: {counts['stale']}")
    status_parts.extend(
        [
            f"skipped: {counts['skipped']}",
            f"failed: {counts['failed']}",
            f"pending: {counts['pending']}",
        ]
    )
    print("; ".join(status_parts).capitalize())
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
