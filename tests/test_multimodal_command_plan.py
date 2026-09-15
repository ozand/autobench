from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from src.multimodal_command_plan import (
    COMMAND_BINARY,
    EXPECTED_FLAGS,
    MultimodalCommandPlanError,
    build_first_baseline_command_plan,
    render_first_baseline_dry_run,
)
from src.multimodal_runner import PreparedMultimodalInvocation


def _prepared(**config_overrides) -> PreparedMultimodalInvocation:
    config = {
        "device": "Vulkan0",
        "split_mode": "none",
        "split_ratio": None,
        "context_length": 1024,
        "cache_type_k": "f16",
        "cache_type_v": "f16",
        "max_tokens": 32,
    }
    config.update(config_overrides)
    pairing_id = "qwen2-vl-2b-instruct-q4km-q8proj"
    return PreparedMultimodalInvocation(
        image_reference=Path("C:/ephemeral/non-persisted.png"),
        model_artifact={
            "basename": "Qwen2-VL-2B-Instruct-Q4_K_M.gguf",
            "size_bytes": 986046944,
            "sha256": "5745685d2e607a82a0696c1118e56a2a1ae0901da450fd9cd4f161c6b62867d7",
            "pairing_id": pairing_id,
        },
        projector_artifact={
            "basename": "mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf",
            "size_bytes": 709883360,
            "sha256": "a0ad91f00a7a80dcf84d719a61b00ee2e07b71794f4ee2dfa81a254621a8c418",
            "pairing_id": pairing_id,
        },
        image_descriptor={"format": "png", "width": 28, "height": 28, "byte_class": "small", "validation_status": "VALID"},
        configuration=config,
    )


def test_first_baseline_plan_is_deterministic_sanitized_and_non_executing():
    prepared = _prepared()
    with patch("subprocess.Popen") as popen, patch("subprocess.run") as run, patch("pathlib.Path.open") as open_file:
        first = build_first_baseline_command_plan(prepared)
        second = build_first_baseline_command_plan(prepared)
        summary = render_first_baseline_dry_run(prepared)
    popen.assert_not_called()
    run.assert_not_called()
    open_file.assert_not_called()
    assert first == second
    assert first["binary"] == COMMAND_BINARY
    assert first["argument_flags"] == list(EXPECTED_FLAGS)
    assert first["planned_job_count"] == 1
    assert first["inference_invoked"] is False
    assert summary["plan"] == first
    assert "non-persisted.png" not in str(first)
    assert "path" not in str(first).lower()


@pytest.mark.parametrize(
    "overrides",
    [
        {"device": "CPU"},
        {"device": "Vulkan1"},
        {"device": "Vulkan0,Vulkan1", "split_mode": "layer", "split_ratio": "1,1"},
        {"context_length": 2048},
        {"cache_type_k": "q8_0", "cache_type_v": "q8_0"},
        {"max_tokens": 33},
        {"max_tokens": 0},
        {"max_tokens": True},
    ],
)
def test_command_plan_rejects_every_configuration_outside_first_baseline(overrides):
    with pytest.raises(MultimodalCommandPlanError):
        build_first_baseline_command_plan(_prepared(**overrides))


def test_command_plan_rejects_non_prepared_wrong_pair_or_unsafe_contract_data():
    with pytest.raises(MultimodalCommandPlanError):
        build_first_baseline_command_plan(object())
    with pytest.raises(MultimodalCommandPlanError):
        build_first_baseline_command_plan(replace(_prepared(), configuration=None))
    with pytest.raises(MultimodalCommandPlanError):
        build_first_baseline_command_plan(replace(_prepared(), configuration=[]))
    prepared = _prepared()
    prepared.model_artifact["basename"] = "arbitrary.gguf"
    prepared.model_artifact["sha256"] = "0" * 64
    prepared.projector_artifact["basename"] = "other.gguf"
    prepared.projector_artifact["sha256"] = "1" * 64
    prepared.model_artifact["pairing_id"] = prepared.projector_artifact["pairing_id"] = "arbitrary-pair"
    with pytest.raises(MultimodalCommandPlanError):
        build_first_baseline_command_plan(prepared)
    prepared = _prepared()
    prepared.model_artifact["path"] = "C:/secret/model.gguf"
    with pytest.raises(MultimodalCommandPlanError):
        build_first_baseline_command_plan(prepared)


def test_command_plan_accepts_sanitized_jpeg_descriptor_but_rejects_unsafe_shape():
    prepared = replace(_prepared(), image_descriptor={"format": "jpg", "width": 28, "height": 28, "byte_class": "small", "validation_status": "VALID"})
    assert build_first_baseline_command_plan(prepared)["image_descriptor"]["format"] == "jpg"
    unsafe = replace(prepared, image_descriptor={"format": "jpg", "width": 28, "height": 28, "byte_class": "small", "validation_status": "VALID", "path": "private"})
    with pytest.raises(MultimodalCommandPlanError):
        build_first_baseline_command_plan(unsafe)


def test_command_plan_receipt_rejects_boolean_job_count():
    receipt = build_first_baseline_command_plan(_prepared())
    receipt["planned_job_count"] = True
    from src.multimodal_receipt import validate_multimodal_receipt
    assert validate_multimodal_receipt(receipt)["status"] == "MULTIMODAL_RECEIPT_INVALID"


def test_command_plan_module_has_no_execution_or_image_runtime_imports():
    source = Path("src/multimodal_command_plan.py").read_text(encoding="utf-8")
    for forbidden in ("import subprocess", "Popen(", "subprocess.", "os.system", "requests", "run_host_command", "Path.open(", "PIL"):
        assert forbidden not in source
