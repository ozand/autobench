import hashlib
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from src.multimodal_preflight import (
    ArtifactExpectation,
    MultimodalPreflightError,
    build_preflight_receipt,
    validate_artifact,
    validate_image_descriptor,
)
from src.multimodal_receipt import validate_multimodal_receipt


def artifact(tmp_path: Path, name: str, content: bytes) -> tuple[Path, ArtifactExpectation]:
    path = tmp_path / name
    path.write_bytes(content)
    return path, ArtifactExpectation(name, len(content), hashlib.sha256(content).hexdigest(), "qwen2-vl-2b")


def descriptor(**overrides):
    value = {"format": "png", "width": 640, "height": 480, "size_bytes": 4096}
    value.update(overrides)
    return value


def test_preflight_valid_pair_is_sanitized_and_never_invokes_inference(tmp_path):
    model, model_expected = artifact(tmp_path, "Qwen2-VL-2B-Instruct-Q4_K_M.gguf", b"model")
    projector, projector_expected = artifact(tmp_path, "mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf", b"projector")
    with patch("subprocess.run") as run:
        receipt = build_preflight_receipt(
            model_path=model,
            projector_path=projector,
            model_expected=model_expected,
            projector_expected=projector_expected,
            image_descriptor=descriptor(),
        )
    run.assert_not_called()
    assert receipt["preflight_status"] == "VALID"
    assert receipt["inference_invoked"] is False
    assert validate_multimodal_receipt(receipt)["status"] == "MULTIMODAL_RECEIPT_VALID"
    assert "path" not in str(receipt).lower()


@pytest.mark.parametrize("bad_expected", [
    ArtifactExpectation("wrong.gguf", 5, hashlib.sha256(b"model").hexdigest(), "qwen2-vl-2b"),
    ArtifactExpectation("Qwen2-VL-2B-Instruct-Q4_K_M.gguf", 6, hashlib.sha256(b"model").hexdigest(), "qwen2-vl-2b"),
    ArtifactExpectation("Qwen2-VL-2B-Instruct-Q4_K_M.gguf", 5, "0" * 64, "qwen2-vl-2b"),
])
def test_preflight_rejects_wrong_model_pair(tmp_path, bad_expected):
    model, _ = artifact(tmp_path, "Qwen2-VL-2B-Instruct-Q4_K_M.gguf", b"model")
    with pytest.raises(MultimodalPreflightError):
        validate_artifact(model, bad_expected)


def test_preflight_rejects_missing_projector(tmp_path):
    model, model_expected = artifact(tmp_path, "Qwen2-VL-2B-Instruct-Q4_K_M.gguf", b"model")
    projector_expected = ArtifactExpectation("mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf", 9, hashlib.sha256(b"projector").hexdigest(), "qwen2-vl-2b")
    with pytest.raises(MultimodalPreflightError):
        build_preflight_receipt(
            model_path=model,
            projector_path=tmp_path / projector_expected.basename,
            model_expected=model_expected,
            projector_expected=projector_expected,
            image_descriptor=descriptor(),
        )


def test_preflight_rejects_unpaired_artifacts(tmp_path):
    model, model_expected = artifact(tmp_path, "Qwen2-VL-2B-Instruct-Q4_K_M.gguf", b"model")
    projector, projector_expected = artifact(tmp_path, "mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf", b"projector")
    projector_expected = ArtifactExpectation(
        projector_expected.basename, projector_expected.size_bytes,
        projector_expected.sha256, "different-pair",
    )
    with pytest.raises(MultimodalPreflightError):
        build_preflight_receipt(
            model_path=model, projector_path=projector,
            model_expected=model_expected, projector_expected=projector_expected,
            image_descriptor=descriptor(),
        )


@pytest.mark.parametrize("invalid", [
    {},
    {"format": "gif", "width": 10, "height": 10, "size_bytes": 10},
    {"format": "png", "width": 0, "height": 10, "size_bytes": 10},
    {"format": "png", "width": 2049, "height": 10, "size_bytes": 10},
    {"format": "png", "width": 10, "height": 10, "size_bytes": 10 * 1024 * 1024 + 1},
    {"format": "png", "width": 10, "height": 10, "size_bytes": 10, "path": "forbidden"},
])
def test_image_descriptor_rejects_unsafe_or_ambiguous_metadata(invalid):
    with pytest.raises(MultimodalPreflightError):
        validate_image_descriptor(invalid)


def test_receipt_rejects_ambiguous_descriptor_and_path_basename(tmp_path):
    model, model_expected = artifact(tmp_path, "Qwen2-VL-2B-Instruct-Q4_K_M.gguf", b"model")
    projector, projector_expected = artifact(tmp_path, "mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf", b"projector")
    receipt = build_preflight_receipt(
        model_path=model, projector_path=projector,
        model_expected=model_expected, projector_expected=projector_expected,
        image_descriptor=descriptor(),
    )
    receipt["image_descriptor"]["source_url"] = "https://example.invalid/image"
    assert validate_multimodal_receipt(receipt)["status"] == "MULTIMODAL_RECEIPT_INVALID"
    receipt = build_preflight_receipt(
        model_path=model, projector_path=projector,
        model_expected=model_expected, projector_expected=projector_expected,
        image_descriptor=descriptor(),
    )
    receipt["model_artifact"]["basename"] = "../secret.gguf"
    assert validate_multimodal_receipt(receipt)["status"] == "MULTIMODAL_RECEIPT_INVALID"


def test_cli_help_is_importable_from_repository_root():
    result = subprocess.run(
        [sys.executable, "scripts/multimodal_ocr_preflight.py", "--help"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0
    assert "--model" in result.stdout


def test_receipt_rejects_raw_payload_and_path_values(tmp_path):
    model, model_expected = artifact(tmp_path, "Qwen2-VL-2B-Instruct-Q4_K_M.gguf", b"model")
    projector, projector_expected = artifact(tmp_path, "mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf", b"projector")
    receipt = build_preflight_receipt(
        model_path=model, projector_path=projector,
        model_expected=model_expected, projector_expected=projector_expected,
        image_descriptor=descriptor(),
    )
    receipt["image_descriptor"]["base64"] = "data:image/png;base64,abc"
    assert validate_multimodal_receipt(receipt)["status"] == "MULTIMODAL_RECEIPT_INVALID"
