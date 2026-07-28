#!/usr/bin/env python3
"""Run the validated AutoBench suite across a resumable GGUF inventory."""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from authoritative_bench import (
    build_plan,
    discover_remote_models,
    execute_suite,
    run_load_probe,
)


def job_id(model: dict, config: dict) -> str:
    """Return a stable filesystem-safe identifier for one model/configuration."""
    raw = "_".join(
        [
            model["id"],
            config["device"],
            config.get("tensor_split") or "none",
            config["mode"],
        ]
    )
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def build_jobs(models: list[dict]) -> list[dict]:
    """Expand the inventory plan into deterministic execution jobs."""
    plan = build_plan(models, "inventory")
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
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
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


def execute_job(job: dict, args: argparse.Namespace) -> dict:
    """Execute a load-only probe or the existing end-to-end suite."""
    model = job["model"]
    config = job["config"]
    if config["mode"] == "load_only":
        result = run_load_probe(model, config, args.timeout)
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


def run_inventory(
    jobs: list[dict], output_dir: Path, args: argparse.Namespace
) -> dict:
    """Run pending jobs in order and persist each result immediately."""
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

    selected = runnable[: args.max_jobs] if args.max_jobs else runnable
    summary["pending"] = [job["id"] for job in runnable[len(selected) :]]
    if args.dry_run:
        summary["pending"] = [job["id"] for job in runnable]
        return summary

    for job in selected:
        path = output_dir / f"{job['id']}.json"
        try:
            result = execute_job(job, args)
            path.write_text(json.dumps(result, indent=2) + "\n")
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
            path.write_text(json.dumps(failure, indent=2) + "\n")
            summary["failed"].append(job["id"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", help="Comma-separated exact GGUF basenames")
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
    if args.max_jobs < 0 or args.timeout < 1:
        parser.error("--max-jobs cannot be negative and --timeout must be positive")
    try:
        args.context_sizes = [
            int(value.strip()) for value in args.context_sizes.split(",") if value.strip()
        ]
    except ValueError:
        parser.error("--context-sizes must contain integers")

    models = discover_remote_models()
    if args.models:
        requested = {value.strip() for value in args.models.split(",") if value.strip()}
        models = [model for model in models if model["name"] in requested]
        missing = requested - {model["name"] for model in models}
        if missing:
            parser.error(f"models not found: {', '.join(sorted(missing))}")

    jobs = build_jobs(models)
    if args.status:
        summary = inventory_status(
            jobs, Path(args.output_dir), policy_fingerprint(args)
        )
        summary["skipped"] = []
    else:
        summary = run_inventory(jobs, Path(args.output_dir), args)
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
