from pathlib import Path
from unittest.mock import patch

from scripts.run_issue41_deepseek_same_context import (
    DECLARED_CONTEXT,
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


def test_plan_is_one_job_and_same_context(tmp_path):
    result = plan(model(), tmp_path, 600)
    assert result["expected_job_count"] == 1
    assert result["declared_context"] == DECLARED_CONTEXT
    assert result["configuration"]["device"] == "Vulkan0,Vulkan1"
    assert result["configuration"]["split_mode"] == "layer"
    assert result["configuration"]["tensor_split"] == "1,1"
    assert result["configuration"]["cache_type_k"] == "f16"
    assert result["configuration"]["cache_type_v"] == "f16"


def test_configuration_rejects_wrong_model_or_size():
    bad = model()
    bad["name"] = "other.gguf"
    try:
        configuration(bad)
    except ValueError as exc:
        assert "basename" in str(exc)
    else:
        raise AssertionError("wrong model must fail closed")
    bad = model()
    bad["size_bytes"] += 1
    try:
        configuration(bad)
    except ValueError as exc:
        assert "size" in str(exc)
    else:
        raise AssertionError("wrong size must fail closed")


def test_plan_rejects_non_reviewed_timeout(tmp_path):
    try:
        plan(model(), tmp_path, 1200)
    except ValueError as exc:
        assert "600" in str(exc)
    else:
        raise AssertionError("plan must reject non-reviewed timeout")


def test_execute_passes_only_declared_context(tmp_path):
    result = {"models": [{"configurations": [{"result": {"status": "SUCCESS"}}]}]}
    with patch("scripts.run_issue41_deepseek_same_context.execute_suite", return_value=result) as run:
        summary = execute(model(), tmp_path, 600)
    assert run.call_args.kwargs["context_sizes"] == [DECLARED_CONTEXT]
    assert run.call_args.kwargs["performance_context"] == DECLARED_CONTEXT
    assert summary["job_status"] == "SUCCESS"
    assert (tmp_path / "summary.json").exists()


def test_expected_binding_is_explicit():
    assert EXPECTED_SHA256 == "f3bdf9cf31dee4b57ae4e455a1cb0d01b5c2c1b50d72d3112141c195506c2840"
