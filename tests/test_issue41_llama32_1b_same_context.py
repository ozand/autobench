from pathlib import Path
from unittest.mock import patch

from scripts.run_issue41_llama32_1b_same_context import (
    EXPECTED_SHA256,
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
        "path": f"/models/{MODEL_NAME}",
        "size_bytes": EXPECTED_SIZE_BYTES,
    }


def test_plan_is_one_job_same_context(tmp_path):
    result = plan(model(), tmp_path, 600)
    assert result["expected_job_count"] == 1
    assert result["declared_context"] == 1024
    assert result["configuration"]["device"] == "Vulkan0,Vulkan1"
    assert result["configuration"]["split_mode"] == "layer"
    assert result["configuration"]["tensor_split"] == "1,1"


def test_configuration_rejects_wrong_model_or_size():
    bad = model()
    bad["name"] = "other.gguf"
    try:
        configuration(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong model accepted")

    bad = model()
    bad["size_bytes"] += 1
    try:
        configuration(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong size accepted")


def test_plan_rejects_non_reviewed_timeout(tmp_path):
    try:
        plan(model(), tmp_path, 1200)
    except ValueError as exc:
        assert "600" in str(exc)
    else:
        raise AssertionError("non-reviewed timeout accepted")


def test_execute_uses_only_context_1024(tmp_path):
    result = {"models": [{"configurations": [{"result": {"status": "SUCCESS"}}]}]}
    with patch(
        "scripts.run_issue41_llama32_1b_same_context.execute_suite",
        return_value=result,
    ) as run:
        summary = execute(model(), tmp_path, 600)
    assert run.call_args.kwargs["context_sizes"] == [1024]
    assert run.call_args.kwargs["performance_context"] == 1024
    assert summary["job_status"] == "SUCCESS"


def test_binding():
    assert EXPECTED_SHA256 == (
        "3f5a22426976ab26cfe84dba63c1d08391717abb1af893e10f1b2968d862dcc1"
    )
