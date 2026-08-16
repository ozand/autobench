import argparse
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from inventory_bench import (
    build_jobs,
    build_tensor_validation_jobs,
    completed_job,
    execute_job,
    execute_tensor_validation_job,
    inventory_status,
    job_id,
    policy_fingerprint,
    run_inventory,
    run_tensor_validation,
    select_contiguous_jobs,
    select_tensor_validation_models,
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
        "expected_jobs": 6,
        "overall_timeout": 1800,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def tensor_models():
    return [
        {"id": "small", "name": "small.gguf", "path": "/small", "size_bytes": 1_000_000_000},
        {"id": "large", "name": "large.gguf", "path": "/large", "size_bytes": 3_000_000_000},
    ]


def test_tensor_validation_selects_exact_named_size_classes():
    models = tensor_models() + [
        {"id": "other", "name": "other.gguf", "path": "/other", "size_bytes": 100},
    ]
    selected = select_tensor_validation_models(models, ["large.gguf", "small.gguf"])
    assert [model["name"] for model in selected] == ["small.gguf", "large.gguf"]


def test_tensor_validation_rejects_invalid_selections():
    models = tensor_models()
    for requested in (["small.gguf"], ["small.gguf", "small.gguf"], ["small.gguf", "missing.gguf"]):
        try:
            select_tensor_validation_models(models, requested)
        except ValueError:
            pass
        else:
            raise AssertionError(f"selection must fail: {requested}")
    same_class = models + [
        {"id": "small2", "name": "small2.gguf", "path": "/small2", "size_bytes": 2},
    ]
    try:
        select_tensor_validation_models(same_class, ["small.gguf", "small2.gguf"])
    except ValueError as exc:
        assert "one GGUF <= 1.9 GB" in str(exc)
    else:
        raise AssertionError("same-size-class selection must fail")


def test_tensor_validation_builds_exact_six_job_matrix():
    jobs = build_tensor_validation_jobs(tensor_models())
    assert len(jobs) == 6
    assert [job["config"]["mode"] for job in jobs] == [
        "performance", "performance", "load_only", "load_only", "performance", "performance"
    ]
    assert [job["config"].get("split_mode") for job in jobs] == [None, None, None, None, None, "tensor"]


def test_build_jobs_expands_every_applicable_configuration():
    models = [
        {"id": "small", "name": "small.gguf", "path": "/small", "size_bytes": 10},
        {"id": "large", "name": "large.gguf", "path": "/large", "size_bytes": 3_000_000_000},
    ]

    legacy_jobs = build_jobs(models)
    assert len(legacy_jobs) == 5

    jobs = build_jobs(models, include_tensor=True)

    assert len(jobs) == 7
    assert [job["config"]["device"] for job in jobs[:2]] == ["Vulkan0", "Vulkan1"]
    assert [job["config"]["mode"] for job in jobs[2:4]] == ["full", "load_only"]
    assert [job["config"].get("split_mode") for job in jobs] == [None, None, "tensor", None, None, None, "tensor"]
    assert len({job["id"] for job in jobs}) == 7


def test_inventory_authoritative_jobs_exclude_tensor_candidate():
    models = [
        {"id": "small", "name": "small.gguf", "path": "/small", "size_bytes": 10},
        {"id": "large", "name": "large.gguf", "path": "/large", "size_bytes": 3_000_000_000},
    ]
    jobs = build_jobs(models, include_tensor=False)
    assert len(jobs) == 5
    assert all(job["config"].get("split_mode") != "tensor" for job in jobs)


def test_execute_job_preserves_tensor_probe_configuration(tmp_path: Path):
    model = {"id": "large", "name": "large.gguf", "path": "/large", "size_bytes": 3_000_000_000}
    config = {
        "device": "Vulkan0,Vulkan1",
        "tensor_split": "1,1",
        "split_mode": "tensor",
        "mode": "full",
    }
    job = {"id": "large_dual_tensor", "model": model, "config": config}
    with patch(
        "inventory_bench.execute_suite",
        return_value={"execution_status": "completed_suite"},
    ) as execute:
        # The suite receives the exact tensor configuration; no implicit layer mode is substituted.
        result = execute_job(job, args())
    plan = execute.call_args.args[0]
    assert plan["models"][0]["configurations"][0]["split_mode"] == "tensor"
    assert plan["models"][0]["configurations"][0]["tensor_split"] == "1,1"
    assert result["execution_status"] == "completed_suite"


def test_select_contiguous_jobs_keeps_model_configurations_together():
    jobs = [
        {"id": "a0", "model": {"id": "a"}, "config": {}},
        {"id": "a1", "model": {"id": "a"}, "config": {}},
        {"id": "b0", "model": {"id": "b"}, "config": {}},
    ]
    assert [job["id"] for job in select_contiguous_jobs(jobs, max_jobs=1)] == ["a0", "a1"]
    assert [job["id"] for job in select_contiguous_jobs(jobs, max_jobs=2)] == ["a0", "a1"]


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

    jobs = build_jobs(models, include_tensor=True)

    assert len(jobs) == 70


def test_policy_fingerprint_captures_comparable_settings():
    fingerprint = policy_fingerprint(args())

    assert fingerprint["context_sizes"] == [512]
    assert fingerprint["repetitions"] == 1
    assert fingerprint["performance_repetitions"] == 1
    assert fingerprint["dataset_dir"].replace("\\", "/").endswith("/dataset")


def test_inventory_persists_sanitized_suite_artifact(tmp_path: Path):
    job = {
        "id": "job",
        "model": {"id": "model", "name": "model.gguf", "path": "/model", "size_bytes": 1},
        "config": {"device": "Vulkan0", "mode": "full"},
    }
    with patch(
        "inventory_bench.execute_job",
        return_value={
            "execution_status": "completed_suite",
            "models": [{"configurations": [{"result": {
                "stage": "suite",
                "status": "PREFLIGHT_EXECUTION_ERROR",
                "error": "SECRET_ERROR",
                "stages": {"preflight": {
                    "command_args": ["llama-cli", "-p", "SECRET_PROMPT"],
                    "stdout": "SECRET_STDOUT",
                }},
            }}]}],
        },
    ):
        summary = run_inventory([job], tmp_path, args())
    assert summary["completed"] == ["job"]
    serialized = (tmp_path / "job.json").read_text()
    assert "SECRET_PROMPT" not in serialized
    assert "SECRET_STDOUT" not in serialized
    assert "SECRET_ERROR" not in serialized


def test_tensor_validation_dry_run_writes_nothing(tmp_path: Path):
    jobs = build_tensor_validation_jobs(tensor_models())
    with patch("inventory_bench.execute_tensor_validation_job") as execute:
        summary = run_tensor_validation(jobs, tmp_path, args(dry_run=True))
    execute.assert_not_called()
    assert summary["expected_job_count"] == 6
    assert len(summary["pending"]) == 6
    assert list(tmp_path.iterdir()) == []


def test_tensor_validation_runs_short_performance_not_full_suite():
    job = build_tensor_validation_jobs(tensor_models())[0]
    preflight = {
        "stage": "load_probe", "status": "SUCCESS", "load_ok": True,
        "elapsed_seconds": 1.0, "return_code": 0, "command_args": [],
    }
    measured = {"models": [{"configurations": [{"result": {
        "stage": "performance", "status": "SUCCESS", "prompt_ts": 10.0, "gen_ts": 5.0
    }}]}]}
    with patch("inventory_bench.run_load_probe", return_value=preflight), patch(
        "inventory_bench.execute_performance", return_value=measured
    ) as performance, patch("inventory_bench.execute_suite") as suite:
        result = execute_tensor_validation_job(job, args())
    suite.assert_not_called()
    performance.assert_called_once()
    assert result["execution_status"] == "completed_performance"


def test_tensor_validation_stops_starting_jobs_after_overall_deadline(tmp_path: Path):
    jobs = build_tensor_validation_jobs(tensor_models())
    with patch("inventory_bench.time.monotonic", side_effect=[0.0, 2.0]), patch(
        "inventory_bench.execute_tensor_validation_job"
    ) as execute:
        summary = run_tensor_validation(jobs, tmp_path, args(overall_timeout=1))
    execute.assert_not_called()
    assert summary["deadline_reached"] is True
    assert len(summary["pending"]) == 6


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
