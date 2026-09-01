import argparse
import json
from pathlib import Path
from unittest.mock import patch

from scripts.run_issue41_qwen_q8 import (
    EXPECTED_ALLOCATION_PROBES,
    EXPECTED_JOB_COUNT,
    MODEL_NAME,
    EXPECTED_SIZE_BYTES,
    execute_reviewed_plan,
    issue_configurations,
    reviewed_plan,
)


def target_model():
    return {
        "id": "qwen2.5-0.5b-instruct-q8_0",
        "name": MODEL_NAME,
        "path": "/models/qwen2.5-0.5b-instruct-q8_0.gguf",
        "size_bytes": EXPECTED_SIZE_BYTES,
    }


def test_issue41_plan_has_exact_three_jobs_and_six_probes(tmp_path: Path):
    plan = reviewed_plan(target_model(), tmp_path, 600)

    assert plan["issue"] == 41
    assert plan["expected_job_count"] == EXPECTED_JOB_COUNT == 3
    assert plan["allocation_probe_count"] == EXPECTED_ALLOCATION_PROBES == 6
    assert [config["device"] for config in plan["configurations"]] == [
        "Vulkan0",
        "Vulkan1",
        "Vulkan0,Vulkan1",
    ]
    assert plan["configurations"][2]["split_mode"] == "layer"
    assert plan["configurations"][2]["tensor_split"] == "1,1"
    assert all(config["cache_type_k"] == "f16" for config in plan["configurations"])
    assert all(config["cache_type_v"] == "f16" for config in plan["configurations"])
    assert plan["inference_authorized"] is False


def test_execute_reviewed_plan_stops_after_first_unexpected_result(tmp_path: Path):
    model = target_model()
    unexpected = {
        "models": [{
            "configurations": [{
                "result": {"status": "BOUNDARY_SSH_TIMEOUT"}
            }]
        }]
    }
    with patch("scripts.run_issue41_qwen_q8.execute_suite", return_value=unexpected) as suite:
        summary = execute_reviewed_plan(model, tmp_path, 600)

    assert suite.call_count == 1
    assert summary["execution_status"] == "stopped_after_first_unexpected_result"
    assert summary["completed_job_count"] == 1
    assert summary["all_jobs_completed"] is False
    assert (tmp_path / "job_01.json").exists()
    persisted = json.loads((tmp_path / "summary.json").read_text())
    assert persisted["stopped_after_job"] == 1


def test_issue41_plan_rejects_wrong_artifact():
    wrong = target_model()
    wrong["size_bytes"] = EXPECTED_SIZE_BYTES - 1
    try:
        issue_configurations(wrong)
    except ValueError as exc:
        assert "size" in str(exc)
    else:
        raise AssertionError("wrong artifact size must fail closed")
