"""Zero-inference validation for a future multimodal OCR invocation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.multimodal_receipt import sanitize_multimodal_receipt, validate_multimodal_receipt

SUPPORTED_IMAGE_FORMATS = {"png", "jpg", "jpeg"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_DIMENSION = 2048


class MultimodalPreflightError(ValueError):
    """Raised when a zero-inference multimodal preflight contract is violated."""


@dataclass(frozen=True)
class ArtifactExpectation:
    basename: str
    size_bytes: int
    sha256: str
    pairing_id: str


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def validate_artifact(path: Path, expected: ArtifactExpectation) -> dict[str, Any]:
    """Verify one explicit artifact without loading it into a model runtime."""
    if path.name != expected.basename:
        raise MultimodalPreflightError("artifact basename does not match contract")
    if not path.is_file():
        raise MultimodalPreflightError("artifact is missing")
    if path.stat().st_size != expected.size_bytes:
        raise MultimodalPreflightError("artifact size does not match contract")
    if _sha256(path) != expected.sha256:
        raise MultimodalPreflightError("artifact checksum does not match contract")
    return {
        "basename": expected.basename,
        "size_bytes": expected.size_bytes,
        "sha256": expected.sha256,
        "pairing_id": expected.pairing_id,
    }


def validate_image_descriptor(descriptor: Any) -> dict[str, Any]:
    """Validate metadata only; no image file is opened or decoded."""
    if not isinstance(descriptor, dict):
        raise MultimodalPreflightError("image descriptor must be an object")
    if any(key.lower() in {"path", "image", "base64", "source_url"} for key in descriptor):
        raise MultimodalPreflightError("IMAGE_INPUT_REJECTED: unsafe descriptor field")
    required = {"format", "width", "height", "size_bytes"}
    if set(descriptor) != required:
        raise MultimodalPreflightError("image descriptor fields are ambiguous")
    image_format = descriptor["format"]
    width = descriptor["width"]
    height = descriptor["height"]
    size_bytes = descriptor["size_bytes"]
    if not isinstance(image_format, str) or image_format.lower() not in SUPPORTED_IMAGE_FORMATS:
        raise MultimodalPreflightError("IMAGE_INPUT_REJECTED: unsupported format")
    if not all(isinstance(value, int) and value > 0 for value in (width, height, size_bytes)):
        raise MultimodalPreflightError("IMAGE_INPUT_REJECTED: invalid metadata")
    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        raise MultimodalPreflightError("IMAGE_INPUT_REJECTED: dimensions exceed contract")
    if size_bytes > MAX_IMAGE_BYTES:
        raise MultimodalPreflightError("IMAGE_INPUT_REJECTED: byte class exceeds contract")
    return {
        "format": image_format.lower(),
        "width": width,
        "height": height,
        "byte_class": "small" if size_bytes <= 1024 * 1024 else "medium",
        "validation_status": "VALID",
    }


def build_preflight_receipt(
    *,
    model_path: Path,
    projector_path: Path,
    model_expected: ArtifactExpectation,
    projector_expected: ArtifactExpectation,
    image_descriptor: Any,
) -> dict[str, Any]:
    """Build a sanitized receipt without invoking a model, image decoder, or subprocess."""
    if not model_expected.pairing_id or model_expected.pairing_id != projector_expected.pairing_id:
        raise MultimodalPreflightError("model/projector pairing does not match contract")
    model_artifact = validate_artifact(model_path, model_expected)
    projector_artifact = validate_artifact(projector_path, projector_expected)
    image = validate_image_descriptor(image_descriptor)
    receipt = {
        "schema_version": 1,
        "receipt_type": "MULTIMODAL_OCR_PREFLIGHT",
        "model_artifact": model_artifact,
        "projector_artifact": projector_artifact,
        "image_descriptor": image,
        "preflight_status": "VALID",
        "inference_invoked": False,
    }
    validation = validate_multimodal_receipt(receipt)
    if validation["status"] != "MULTIMODAL_RECEIPT_VALID":
        raise MultimodalPreflightError("generated receipt failed validation")
    return sanitize_multimodal_receipt(receipt)
