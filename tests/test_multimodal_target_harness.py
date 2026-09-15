import hashlib
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.multimodal_command_plan import APPROVED_ARTIFACTS, APPROVED_IMAGE_DESCRIPTOR, build_first_baseline_command_plan
from src.multimodal_runner import PreparedMultimodalInvocation
from src.multimodal_target_harness import (
    MultimodalTargetHarnessError,
    ProcessObservation,
    build_mock_terminal_receipt,
    build_target_argv,
    temporary_document_and_prompt,
    validate_target_identity,
)


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


def _jpeg(width: int = 28, height: int = 28) -> bytes:
    # Minimal header sufficient for metadata-only SOF parsing; no pixel decoding occurs.
    return b"\xff\xd8" + b"\xff\xc0\x00\x11\x08" + height.to_bytes(2, "big") + width.to_bytes(2, "big") + b"\x03" + b"\x00" * 10 + b"\xff\xd9"


def _source_jpeg(tmp_path: Path) -> Path:
    source = tmp_path / "public-document.jpg"
    source.write_bytes(_jpeg())
    return source


def test_staging_one_jpeg_and_prompt_binds_metadata_and_cleans(tmp_path):
    source = _source_jpeg(tmp_path)
    with temporary_document_and_prompt(tmp_path, source) as (document, prompt):
        assert document.path.is_file() and prompt.is_file()
        assert document.descriptor == {
            "format": "jpg", "width": 28, "height": 28,
            "byte_class": "small", "validation_status": "VALID",
        }
        assert prompt.stat().st_size > 0
    assert list(tmp_path.iterdir()) == [source]


def test_staging_cleans_on_failure_and_rejects_invalid_jpeg_or_unsafe_root(tmp_path, tmp_path_factory):
    source = _source_jpeg(tmp_path)
    with pytest.raises(RuntimeError):
        with temporary_document_and_prompt(tmp_path, source):
            raise RuntimeError("mock failure")
    assert list(tmp_path.iterdir()) == [source]
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not-a-jpeg")
    with pytest.raises(MultimodalTargetHarnessError):
        with temporary_document_and_prompt(tmp_path, bad):
            pass
    outside = tmp_path_factory.mktemp("outside")
    nested = outside / "nested"
    nested.mkdir()
    ancestor_link = tmp_path / "ancestor-link"
    ancestor_link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(MultimodalTargetHarnessError):
        with temporary_document_and_prompt(ancestor_link / "nested", source):
            pass


def test_exact_identity_validation_streams_hash_and_fails_closed(tmp_path):
    binary = tmp_path / "llama-mtmd-cli"
    binary.write_bytes(b"binary")
    os.chmod(binary, 0o700)
    with patch("src.multimodal_target_harness._validate_file", side_effect=[dict(APPROVED_ARTIFACTS["model_artifact"]), dict(APPROVED_ARTIFACTS["projector_artifact"])]) as check:
        model, projector = validate_target_identity(binary, tmp_path / "model", tmp_path / "projector")
    assert check.call_count == 2
    assert model == APPROVED_ARTIFACTS["model_artifact"]
    assert projector == APPROVED_ARTIFACTS["projector_artifact"]
    with pytest.raises(MultimodalTargetHarnessError):
        validate_target_identity(tmp_path / "wrong", tmp_path / "model", tmp_path / "projector")


def test_target_argv_is_private_shell_free_and_exact(tmp_path):
    argv = build_target_argv(
        tmp_path / "llama-mtmd-cli", tmp_path / "Qwen2-VL-2B-Instruct-Q4_K_M.gguf",
        tmp_path / "mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf", tmp_path / "document.jpg",
        tmp_path / "task.txt", _plan(),
    )
    assert isinstance(argv, list)
    assert argv[1:7] == ["-m", argv[2], "--mmproj", argv[4], "--image", argv[6]]
    assert argv[argv.index("-dev") + 1] == "Vulkan0"
    assert argv[argv.index("-c") + 1] == "1024"
    assert argv[argv.index("-n") + 1] == "32"
    assert "-st" not in argv


@pytest.mark.parametrize("observation, terminal", [
    (ProcessObservation(0, b"raw-content-123", b""), "SUCCESS"),
    (ProcessObservation(1, b"", b"raw-memory-456 oom"), "OOM"),
    (ProcessObservation(0, b"x" * 8193, b""), "METRIC_PARSE_FAILED"),
])
def test_mock_terminal_receipt_is_bound_to_jpeg_and_sanitized(tmp_path, observation, terminal):
    source = _source_jpeg(tmp_path)
    with temporary_document_and_prompt(tmp_path, source) as (document, _):
        receipt = build_mock_terminal_receipt(_plan(), document, observation)
    assert receipt["terminal_class"] == terminal
    assert receipt["image_descriptor"]["format"] == "jpg"
    assert receipt["simulated"] is True
    assert receipt["inference_invoked"] is False
    raw_stdout = observation.stdout.decode("utf-8", errors="ignore")
    raw_stderr = observation.stderr.decode("utf-8", errors="ignore")
    if raw_stdout:
        assert raw_stdout not in str(receipt)
    if raw_stderr:
        assert raw_stderr not in str(receipt)


def test_harness_has_no_direct_process_remote_or_image_decoder_primitive():
    source = Path("src/multimodal_target_harness.py").read_text(encoding="utf-8")
    for forbidden in ("import subprocess", "subprocess.", "run_host_command", "requests", "os.system", "Popen(", "PIL"):
        assert forbidden not in source
