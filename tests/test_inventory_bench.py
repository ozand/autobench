import argparse
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from inventory_bench import (
    build_jobs,
    completed_job,
    inventory_status,
    job_id,
    policy_fingerprint,
    run_inventory,
)


def args(**overrides):
    values = {
        "timeout": 5,
        "context_sizes": [512],
        "boundary_step": 128,
        "repetitions": 1,
        "reliability_threshold": 0.8,
        "context_size": 512,
        "prompt_tokens": 240,
        "max_tokens": 32,
        "warmups": 0,
        "performance_repetitions": 1,
        "dataset_dir": "/dataset",
        "max_tasks": 1,
        "force": False,
        "max_jobs": 0,
        "dry_run": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_build_jobs_expands_every_applicable_configuration():
    models = [
        {"id": "small", "name": "small.gguf", "path": "/small", "size_bytes": 10},
        {"id": "large", "name": "large.gguf", "path": "/large", "size_bytes": 3_000_000_000},
    ]

    jobs = build_jobs(models)

    assert len(jobs) == 5
    assert [job["config"]["device"] for job in jobs[:2]] == ["Vulkan0", "Vulkan1"]
    assert [job["config"]["mode"] for job in jobs[2:]] == [
        "load_only",
        "load_only",
        "full",
    ]
    assert len({job["id"] for job in jobs}) == 5


def test_job_id_is_stable_and_filesystem_safe():
    identifier = job_id(
        {"id": "Model 1/B"},
        {"device": "Vulkan0,Vulkan1", "tensor_split": "1,1", "mode": "full"},
    )

    assert identifier == "Model_1_B_Vulkan0_Vulkan1_1_1_full"


def test_complete_inventory_has_expected_job_count_for_mixed_sizes():
    models = [
        {"id": f"small-{index}", "name": f"small-{index}.gguf", "path": "/small", "size_bytes": 10}
        for index in range(10)
    ] + [
        {"id": f"large-{index}", "name": f"large-{index}.gguf", "path": "/large", "size_bytes": 3_000_000_000}
        for index in range(10)
    ]

    jobs = build_jobs(models)

    assert len(jobs) == 50


def test_policy_fingerprint_captures_comparable_settings():
    fingerprint = policy_fingerprint(args())

    assert fingerprint["context_sizes"] == [512]
    assert fingerprint["repetitions"] == 1
    assert fingerprint["performance_repetitions"] == 1
    assert fingerprint["dataset_dir"].replace("\\", "/").endswith("/dataset")


def test_completed_job_requires_terminal_manifest(tmp_path: Path):
    path = tmp_path / "job.json"
    assert completed_job(path) is False
    path.write_text("not json")
    assert completed_job(path) is False
    path.write_text(json.dumps({"execution_status": "failed"}))
    assert completed_job(path) is False
    path.write_text(
        json.dumps(
            {
                "execution_status": "completed_suite",
                "policy_fingerprint": {"repetitions": 5},
            }
        )
    )
    assert completed_job(path) is True
    assert completed_job(path, {"repetitions": 5}) is True
    assert completed_job(path, {"repetitions": 1}) is False


def test_inventory_status_reports_cumulative_persisted_state(tmp_path: Path):
    jobs = [
        {"id": "done", "model": {}, "config": {}},
        {"id": "failed", "model": {}, "config": {}},
        {"id": "pending", "model": {}, "config": {}},
    ]
    (tmp_path / "done.json").write_text(
        json.dumps({"execution_status": "completed_suite"})
    )
    (tmp_path / "failed.json").write_text(
        json.dumps({"execution_status": "failed", "error": "boom"})
    )

    status = inventory_status(jobs, tmp_path)

    assert status == {
        "completed": ["done"],
        "stale": [],
        "failed": ["failed"],
        "pending": ["pending"],
    }


def test_inventory_status_separates_stale_policy_from_failure(tmp_path: Path):
    jobs = [{"id": "old", "model": {}, "config": {}}]
    (tmp_path / "old.json").write_text(
        json.dumps(
            {
                "execution_status": "completed_suite",
                "policy_fingerprint": {"repetitions": 1},
            }
        )
    )

    status = inventory_status(jobs, tmp_path, {"repetitions": 5})

    assert status == {
        "completed": [],
        "stale": ["old"],
        "failed": [],
        "pending": [],
    }


def test_inventory_skips_completed_and_runs_only_bounded_pending_jobs(tmp_path: Path):
    jobs = [
        {"id": "done", "model": {}, "config": {}},
        {"id": "next", "model": {}, "config": {}},
        {"id": "later", "model": {}, "config": {}},
    ]
    (tmp_path / "done.json").write_text(
        json.dumps(
            {
                "execution_status": "completed_suite",
                "policy_fingerprint": policy_fingerprint(args()),
            }
        )
    )
    with patch(
        "inventory_bench.execute_job",
        return_value={"execution_status": "completed_suite"},
    ) as execute:
        summary = run_inventory(jobs, tmp_path, args(max_jobs=1))

    execute.assert_called_once()
    assert execute.call_args.args[0]["id"] == "next"
    assert summary == {
        "completed": ["next"],
        "skipped": ["done"],
        "failed": [],
        "pending": ["later"],
    }
    assert (tmp_path / "next.json").exists()


def test_inventory_reruns_completed_manifest_with_mismatched_policy(tmp_path: Path):
    jobs = [{"id": "old", "model": {}, "config": {}}]
    (tmp_path / "old.json").write_text(
        json.dumps(
            {
                "execution_status": "completed_suite",
                "policy_fingerprint": {"repetitions": 1},
            }
        )
    )
    with patch(
        "inventory_bench.execute_job",
        return_value={"execution_status": "completed_suite"},
    ) as execute:
        summary = run_inventory(jobs, tmp_path, args(repetitions=5))

    execute.assert_called_once()
    assert summary["completed"] == ["old"]
    assert summary["skipped"] == []


def test_inventory_dry_run_executes_nothing(tmp_path: Path):
    jobs = [{"id": "one", "model": {}, "config": {}}]
    with patch("inventory_bench.execute_job") as execute:
        summary = run_inventory(jobs, tmp_path, args(dry_run=True))

    execute.assert_not_called()
    assert summary["pending"] == ["one"]


def test_inventory_failure_is_persisted_and_next_job_continues(tmp_path: Path):
    jobs = [
        {"id": "bad", "model": {"name": "bad"}, "config": {"device": "Vulkan0"}},
        {"id": "good", "model": {"name": "good"}, "config": {"device": "Vulkan1"}},
    ]
    with patch(
        "inventory_bench.execute_job",
        side_effect=[RuntimeError("boom"), {"execution_status": "completed_suite"}],
    ):
        summary = run_inventory(jobs, tmp_path, args())

    assert summary["failed"] == ["bad"]
    assert summary["completed"] == ["good"]
    failure = json.loads((tmp_path / "bad.json").read_text())
    assert failure["execution_status"] == "failed"
    assert failure["error"] == "boom"
