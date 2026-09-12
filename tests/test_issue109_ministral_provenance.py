import json
from pathlib import Path

import pytest
from unittest.mock import patch

from authoritative_bench import build_plan
from scripts.run_issue41_ministral_reauthorized import (
    EXPECTED_SIZE_BYTES,
    MODEL_NAME,
    execute,
)


def model():
    return {
        "id": Path(MODEL_NAME).stem,
        "name": MODEL_NAME,
        "path": f"/models/{MODEL_NAME}",
        "size_bytes": EXPECTED_SIZE_BYTES,
    }


def _stage_plan(stage, result):
    child = build_plan([model()], stage)
    child["models"][0]["configurations"] = [
        {"device": "Vulkan0,Vulkan1", "tensor_split": "1,1", "split_mode": "layer", "mode": "full"}
    ]
    child["models"][0]["configurations"][0]["result"] = result
    return child


def test_reauthorized_summary_preserves_positive_performance_provenance(tmp_path):
    boundary = _stage_plan("boundary", {
        "status": "SUCCESS", "maximum_allocatable_context": 1024,
        "first_failed_context": None, "elapsed_seconds": 1.0,
    })
    retrieval = _stage_plan("retrieval", {
        "status": "SUCCESS", "retrieved": True, "retrieval_rate": 1.0,
        "elapsed_seconds": 2.0,
    })
    quality = _stage_plan("quality", {
        "status": "SUCCESS", "task_pass_rate": 1.0, "elapsed_seconds": 3.0,
    })
    preflight = {
        "stage": "load_probe", "status": "SUCCESS", "load_ok": True,
        "elapsed_seconds": 0.5, "return_code": 0, "command_args": ["llama-cli"],
    }
    runner_result = {
        "success": True, "status": "SUCCESS", "metric_parse_status": "PARSED",
        "prompt_speed_ts": 2.6, "generation_speed_ts": 11.8,
        "elapsed_seconds": 1.0, "return_code": 0, "command_args": ["llama-cli"],
    }

    with patch("authoritative_bench.run_load_probe", return_value=preflight), patch(
        "authoritative_bench.execute_boundary", return_value=boundary
    ), patch(
        "authoritative_bench.execute_retrieval", return_value=retrieval
    ), patch(
        "authoritative_bench.execute_quality", return_value=quality
    ), patch(
        "authoritative_bench.build_context_prompt", return_value=("prompt", 10, 512, {})
    ), patch(
        "authoritative_bench.Runner.run_local_vulkan", return_value=runner_result
    ):
        summary = execute(model(), tmp_path, 600)

    serialized = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    performance = serialized["result"]["stages"]["performance"]
    assert summary["job_status"] == "SUCCESS"
    assert performance["status"] == "SUCCESS"
    assert performance["prompt_ts"] == pytest.approx(2.6)
    assert performance["gen_ts"] == pytest.approx(11.8)
    assert performance["successful_measurements"] == 3
    assert all(run["metric_parse_status"] == "PARSED" for run in performance["runs"])
    assert serialized["model"]["basename"] == MODEL_NAME
    assert serialized["configuration"]["declared_context"] == 1024
