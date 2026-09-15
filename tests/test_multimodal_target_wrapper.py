import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.multimodal_command_plan import APPROVED_ARTIFACTS, APPROVED_IMAGE_DESCRIPTOR, build_first_baseline_command_plan
from src.multimodal_runner import PreparedMultimodalInvocation
from src.multimodal_target_harness import ProcessObservation
from src.multimodal_target_wrapper import MultimodalTargetWrapperError, run_one_target_smoke


def _plan(image_descriptor: dict | None = None) -> dict:
    return build_first_baseline_command_plan(PreparedMultimodalInvocation(
        image_reference=Path("C:/non-persisted.jpg"),
        model_artifact=dict(APPROVED_ARTIFACTS["model_artifact"]),
        projector_artifact=dict(APPROVED_ARTIFACTS["projector_artifact"]),
        image_descriptor=dict(image_descriptor or APPROVED_IMAGE_DESCRIPTOR),
        configuration={"device": "Vulkan0", "split_mode": "none", "split_ratio": None, "context_length": 1024, "cache_type_k": "f16", "cache_type_v": "f16", "max_tokens": 32},
    ))


def _jpeg() -> bytes:
    return b"\xff\xd8\xff\xc0\x00\x11\x08\x00\x1c\x00\x1c\x03" + b"\x00" * 10 + b"\xff\xd9"


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "public-document.jpg"
    source.write_bytes(_jpeg())
    return source


def _jpeg_descriptor() -> dict:
    return {"format": "jpg", "width": 28, "height": 28, "byte_class": "small", "validation_status": "VALID"}


def _identity():
    return dict(APPROVED_ARTIFACTS["model_artifact"]), dict(APPROVED_ARTIFACTS["projector_artifact"])


def test_wrapper_calls_exactly_one_mock_process_and_returns_sanitized_receipt(tmp_path):
    source, calls = _source(tmp_path), []
    def runner(argv, timeout):
        calls.append((argv, timeout)); return ProcessObservation(0, b"bounded-output", b"")
    with patch("src.multimodal_target_wrapper.validate_target_identity", return_value=_identity()):
        receipt = run_one_target_smoke(binary_path=tmp_path / "llama-mtmd-cli", model_path=tmp_path / "Qwen2-VL-2B-Instruct-Q4_K_M.gguf", projector_path=tmp_path / "mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf", source_image=source, plan=_plan(_jpeg_descriptor()), temporary_root=tmp_path, process_runner=runner)
    assert len(calls) == 1 and receipt["terminal_class"] == "SUCCESS" and receipt["inference_invoked"] is True
    assert str(source) not in str(receipt) and list(tmp_path.iterdir()) == [source]


@pytest.mark.parametrize("observation, expected", [
    (ProcessObservation(1, b"", b"out of memory"), ("OOM", False)),
    (ProcessObservation(1, b"", b"Vulkan unsupported"), ("UNSUPPORTED_BACKEND", False)),
    (ProcessObservation(0, b"", b""), ("OCR_INCONCLUSIVE", False)),
    (ProcessObservation(0, b"x" * 8193, b""), ("METRIC_PARSE_FAILED", False)),
])
def test_wrapper_classifies_one_mock_result_and_stops(tmp_path, observation, expected):
    source = _source(tmp_path)
    with patch("src.multimodal_target_wrapper.validate_target_identity", return_value=_identity()):
        receipt = run_one_target_smoke(binary_path=tmp_path / "llama-mtmd-cli", model_path=tmp_path / "Qwen2-VL-2B-Instruct-Q4_K_M.gguf", projector_path=tmp_path / "mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf", source_image=source, plan=_plan(_jpeg_descriptor()), temporary_root=tmp_path, process_runner=lambda *_: observation)
    assert receipt["terminal_class"] == expected[0] and receipt["inference_invoked"] is expected[1]


def test_wrapper_rejects_bad_baseline_before_identity_or_runner(tmp_path):
    source, bad_plan = _source(tmp_path), _plan(_jpeg_descriptor())
    bad_plan["configuration"]["device"] = "Vulkan1"
    with patch("src.multimodal_target_wrapper.validate_target_identity") as identity:
        with pytest.raises(MultimodalTargetWrapperError):
            run_one_target_smoke(binary_path=tmp_path / "llama-mtmd-cli", model_path=tmp_path / "Qwen2-VL-2B-Instruct-Q4_K_M.gguf", projector_path=tmp_path / "mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf", source_image=source, plan=bad_plan, temporary_root=tmp_path, process_runner=lambda *_: None)
    identity.assert_not_called()


def test_wrapper_timeout_and_descriptor_mismatch_do_not_expand_execution(tmp_path):
    source, called = _source(tmp_path), []
    def timeout_runner(*args):
        called.append(args); raise TimeoutError
    with patch("src.multimodal_target_wrapper.validate_target_identity", return_value=_identity()):
        receipt = run_one_target_smoke(binary_path=tmp_path / "llama-mtmd-cli", model_path=tmp_path / "Qwen2-VL-2B-Instruct-Q4_K_M.gguf", projector_path=tmp_path / "mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf", source_image=source, plan=_plan(_jpeg_descriptor()), temporary_root=tmp_path, process_runner=timeout_runner)
    assert len(called) == 1 and receipt["terminal_class"] == "INCONCLUSIVE"
    with patch("src.multimodal_target_wrapper.validate_target_identity", return_value=_identity()):
        with pytest.raises(MultimodalTargetWrapperError):
            run_one_target_smoke(binary_path=tmp_path / "llama-mtmd-cli", model_path=tmp_path / "Qwen2-VL-2B-Instruct-Q4_K_M.gguf", projector_path=tmp_path / "mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf", source_image=source, plan=_plan({**_jpeg_descriptor(), "width": 29}), temporary_root=tmp_path, process_runner=lambda *_: pytest.fail("runner called"))


def test_wrapper_has_no_direct_process_or_transport_primitive():
    source = Path("src/multimodal_target_wrapper.py").read_text(encoding="utf-8")
    for forbidden in ("import subprocess", "subprocess.", "run_host_command", "requests", "os.system", "Popen("):
        assert forbidden not in source
