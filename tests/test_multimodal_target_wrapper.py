import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.multimodal_command_plan import APPROVED_ARTIFACTS, APPROVED_IMAGE_DESCRIPTOR, build_first_baseline_command_plan
from src.multimodal_runner import PreparedMultimodalInvocation
from src.multimodal_target_harness import ProcessObservation
from src.multimodal_target_wrapper import MultimodalTargetWrapperError, run_one_target_smoke


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


def _jpeg() -> bytes:
    return b"\xff\xd8\xff\xc0\x00\x11\x08\x00\x1c\x00\x1c\x03" + b"\x00" * 10 + b"\xff\xd9"


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "public-document.jpg"
    source.write_bytes(_jpeg())
    return source


def _identity():
    return dict(APPROVED_ARTIFACTS["model_artifact"]), dict(APPROVED_ARTIFACTS["projector_artifact"])


def test_wrapper_calls_exactly_one_mock_process_and_returns_sanitized_receipt(tmp_path):
    source = _source(tmp_path)
    calls = []
    def runner(argv, timeout):
        calls.append((argv, timeout))
        return ProcessObservation(0, b"bounded-output", b"")
    with patch("src.multimodal_target_wrapper.validate_target_identity", return_value=_identity()):
        receipt = run_one_target_smoke(
            binary_path=tmp_path / "llama-mtmd-cli", model_path=tmp_path / "Qwen2-VL-2B-Instruct-Q4_K_M.gguf",
            projector_path=tmp_path / "mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf", source_image=source,
            plan=_plan(), temporary_root=tmp_path, process_runner=runner,
        )
    assert len(calls) == 1
    assert receipt["terminal_class"] == "SUCCESS"
    assert receipt["inference_invoked"] is True
    assert receipt["task_id"] == "ocr_smoke_v1"
    assert str(source) not in str(receipt)
    assert list(tmp_path.iterdir()) == [source]


@pytest.mark.parametrize("observation, expected", [
    (ProcessObservation(1, b"", b"out of memory"), ("OOM", False)),
    (ProcessObservation(1, b"", b"Vulkan unsupported"), ("UNSUPPORTED_BACKEND", False)),
    (ProcessObservation(0, b"", b""), ("OCR_INCONCLUSIVE", False)),
    (ProcessObservation(0, b"x" * 8193, b""), ("METRIC_PARSE_FAILED", False)),
])
def test_wrapper_classifies_one_mock_result_and_stops(tmp_path, observation, expected):
    source = _source(tmp_path)
    with patch("src.multimodal_target_wrapper.validate_target_identity", return_value=_identity()):
        receipt = run_one_target_smoke(
            binary_path=tmp_path / "llama-mtmd-cli", model_path=tmp_path / "Qwen2-VL-2B-Instruct-Q4_K_M.gguf",
            projector_path=tmp_path / "mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf", source_image=source,
            plan=_plan(), temporary_root=tmp_path, process_runner=lambda *_: observation,
        )
    assert receipt["terminal_class"] == expected[0]
    assert receipt["inference_invoked"] is expected[1]
    assert list(tmp_path.iterdir()) == [source]


def test_wrapper_plan_failure_happens_before_identity_or_runner(tmp_path):
    source = _source(tmp_path)
    bad_plan = _plan()
    bad_plan["configuration"]["device"] = "Vulkan1"
    with patch("src.multimodal_target_wrapper.validate_target_identity") as identity:
        with pytest.raises(MultimodalTargetWrapperError):
            run_one_target_smoke(
                binary_path=tmp_path / "llama-mtmd-cli", model_path=tmp_path / "Qwen2-VL-2B-Instruct-Q4_K_M.gguf",
                projector_path=tmp_path / "mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf", source_image=source,
                plan=bad_plan, temporary_root=tmp_path, process_runner=lambda *_: None,
            )
    identity.assert_not_called()
    assert list(tmp_path.iterdir()) == [source]


def test_wrapper_timeout_and_identity_failure_do_not_invoke_runner(tmp_path):
    source = _source(tmp_path)
    called = False
    def runner(*args):
        nonlocal called
        called = True
        raise TimeoutError
    with patch("src.multimodal_target_wrapper.validate_target_identity", return_value=_identity()):
        receipt = run_one_target_smoke(
            binary_path=tmp_path / "llama-mtmd-cli", model_path=tmp_path / "Qwen2-VL-2B-Instruct-Q4_K_M.gguf",
            projector_path=tmp_path / "mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf", source_image=source,
            plan=_plan(), temporary_root=tmp_path, process_runner=runner,
        )
    assert called is True and receipt["terminal_class"] == "INCONCLUSIVE"
    called = False
    with patch("src.multimodal_target_wrapper.validate_target_identity", side_effect=Exception("bad")):
        with pytest.raises(MultimodalTargetWrapperError):
            run_one_target_smoke(
                binary_path=tmp_path / "llama-mtmd-cli", model_path=tmp_path / "Qwen2-VL-2B-Instruct-Q4_K_M.gguf",
                projector_path=tmp_path / "mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf", source_image=source,
                plan=_plan(), temporary_root=tmp_path, process_runner=runner,
            )
    assert called is False


def test_wrapper_has_no_direct_process_or_transport_primitive():
    source = Path("src/multimodal_target_wrapper.py").read_text(encoding="utf-8")
    for forbidden in ("import subprocess", "subprocess.", "run_host_command", "requests", "os.system", "Popen("):
        assert forbidden not in source
