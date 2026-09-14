from pathlib import Path

import pytest

from src.multimodal_command_plan import APPROVED_ARTIFACTS, APPROVED_IMAGE_DESCRIPTOR, build_first_baseline_command_plan
from src.multimodal_execution_driver import (
    OUTPUT_LIMIT_BYTES,
    MultimodalExecutionDriverError,
    ProcessObservation,
    build_private_smoke_argv,
    build_simulated_smoke_receipt,
    classify_mock_observation,
    ephemeral_smoke_prompt,
)
from src.multimodal_receipt import find_unsanitized_multimodal_content, validate_multimodal_receipt
from src.multimodal_runner import PreparedMultimodalInvocation


def _plan() -> dict:
    return build_first_baseline_command_plan(PreparedMultimodalInvocation(
        image_reference=Path("C:/non-persisted.png"),
        model_artifact=dict(APPROVED_ARTIFACTS["model_artifact"]),
        projector_artifact=dict(APPROVED_ARTIFACTS["projector_artifact"]),
        image_descriptor=dict(APPROVED_IMAGE_DESCRIPTOR),
        configuration={
            "device": "Vulkan0", "split_mode": "none", "split_ratio": None,
            "context_length": 1024, "cache_type_k": "f16", "cache_type_v": "f16", "max_tokens": 32,
        },
    ))


def test_private_argv_is_shell_free_and_exact_first_baseline(tmp_path):
    argv = build_private_smoke_argv(
        binary_path=Path("C:/llama-mtmd-cli"), model_path=Path("C:/Qwen2-VL-2B-Instruct-Q4_K_M.gguf"),
        projector_path=Path("C:/mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf"),
        temporary_image_path=tmp_path / "document.jpg", prompt_path=tmp_path / "task.txt", plan=_plan(),
    )
    assert isinstance(argv, list)
    assert argv[1:7] == ["-m", argv[2], "--mmproj", argv[4], "--image", argv[6]]
    assert argv[argv.index("-f") + 1].endswith("task.txt")
    assert argv[argv.index("-dev") + 1] == "Vulkan0"
    assert argv[argv.index("-c") + 1] == "1024"
    assert argv[argv.index("-n") + 1] == "32"


@pytest.mark.parametrize("observation, expected", [
    (ProcessObservation(0, b"output", b""), ("SUCCESS", "OCR_SMOKE_OUTPUT_OBSERVED")),
    (ProcessObservation(0, b"", b""), ("OCR_INCONCLUSIVE", "OCR_SMOKE_NO_OUTPUT")),
    (ProcessObservation(1, b"", b"Vulkan out of memory"), ("OOM", "RUNTIME_OOM")),
    (ProcessObservation(1, b"", b"context exceeded"), ("CONTEXT_OVERFLOW", "RUNTIME_CONTEXT")),
    (ProcessObservation(1, b"", b"Vulkan unsupported"), ("UNSUPPORTED_BACKEND", "RUNTIME_BACKEND")),
    (ProcessObservation(1, b"", b"other failure"), ("EXECUTION_ERROR", "RUNTIME_NONZERO")),
    (ProcessObservation(0, b"x" * (OUTPUT_LIMIT_BYTES + 1), b""), ("METRIC_PARSE_FAILED", "OUTPUT_OVERSIZE")),
    (ProcessObservation("bad", b"", b""), ("METRIC_PARSE_FAILED", "MALFORMED_STREAM")),
])
def test_classifier_is_bounded_and_never_returns_output(observation, expected):
    assert classify_mock_observation(observation) == expected


def test_simulated_receipt_is_strict_sanitized_and_never_claims_inference():
    receipt = build_simulated_smoke_receipt(_plan(), ProcessObservation(0, b"output", b""))
    assert receipt["simulated"] is True
    assert receipt["inference_invoked"] is False
    assert receipt["task_id"] == "ocr_smoke_v1"
    assert receipt["output_classification"] == "OCR_SMOKE_OUTPUT_OBSERVED"
    assert validate_multimodal_receipt(receipt)["status"] == "MULTIMODAL_RECEIPT_VALID"
    assert find_unsanitized_multimodal_content(receipt) == []


def test_prompt_is_ephemeral_after_normal_and_error_exit(tmp_path):
    path = tmp_path / "autobench-ocr-smoke-task.txt"
    with ephemeral_smoke_prompt(tmp_path) as prompt:
        assert prompt == path and prompt.is_file()
    assert not path.exists()
    with pytest.raises(RuntimeError):
        with ephemeral_smoke_prompt(tmp_path):
            raise RuntimeError("synthetic failure")
    assert not path.exists()


def test_invalid_plan_or_input_shape_fails_before_temporary_resources(tmp_path):
    invalid = _plan()
    invalid["configuration"]["device"] = "Vulkan1"
    with pytest.raises(MultimodalExecutionDriverError):
        build_private_smoke_argv(
            binary_path=Path("C:/llama-mtmd-cli"), model_path=Path("C:/Qwen2-VL-2B-Instruct-Q4_K_M.gguf"),
            projector_path=Path("C:/mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf"),
            temporary_image_path=tmp_path / "document.jpg", prompt_path=tmp_path / "task.txt", plan=invalid,
        )
    with pytest.raises(MultimodalExecutionDriverError):
        build_private_smoke_argv(
            binary_path=Path("C:/llama-mtmd-cli"), model_path=Path("C:/Qwen2-VL-2B-Instruct-Q4_K_M.gguf"),
            projector_path=Path("C:/mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf"),
            temporary_image_path=tmp_path / "document.png", prompt_path=tmp_path / "task.txt", plan=_plan(),
        )
    assert not list(tmp_path.iterdir())


def test_driver_has_no_process_image_copy_or_remote_runtime_primitive():
    source = Path("src/multimodal_execution_driver.py").read_text(encoding="utf-8")
    for forbidden in ("import subprocess", "subprocess.", "shutil", "copyfile", "run_host_command", "requests", "os.system", "Popen("):
        assert forbidden not in source
