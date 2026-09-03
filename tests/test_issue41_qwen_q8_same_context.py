from pathlib import Path
from unittest.mock import patch

from scripts.run_issue41_qwen_q8_same_context import (
    DECLARED_CONTEXT,
    EXPECTED_SIZE_BYTES,
    MODEL_NAME,
    configuration,
    execute,
    plan,
)


def model():
    return {
        "id": Path(MODEL_NAME).stem,
        "name": MODEL_NAME,
        "path": "/models/qwen2.5-0.5b-instruct-q8_0.gguf",
        "size_bytes": EXPECTED_SIZE_BYTES,
    }


def test_plan_is_one_job_and_one_declared_context(tmp_path):
    p = plan(model(), tmp_path, 600)
    assert p["expected_job_count"] == 1
    assert p["declared_context"] == DECLARED_CONTEXT
    assert p["configuration"]["split_mode"] == "layer"
    assert p["configuration"]["tensor_split"] == "1,1"
    assert p["configuration"]["cache_type_k"] == "f16"
    assert p["configuration"]["cache_type_v"] == "f16"


def test_execute_passes_only_declared_context(tmp_path):
    result = {"models": [{"configurations": [{"result": {"status": "SUCCESS"}}]}]}
    with patch("scripts.run_issue41_qwen_q8_same_context.execute_suite", return_value=result) as run:
        summary = execute(model(), tmp_path, 600)
    assert run.call_args.kwargs["context_sizes"] == [DECLARED_CONTEXT]
    assert run.call_args.kwargs["performance_context"] == DECLARED_CONTEXT
    assert summary["job_status"] == "SUCCESS"
    assert (tmp_path / "summary.json").exists()
