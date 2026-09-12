"""Fail-closed sanitization and validation for multimodal OCR preflight receipts."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

MULTIMODAL_RECEIPT_SCHEMA_VERSION = 1
MULTIMODAL_RECEIPT_TYPE = "MULTIMODAL_OCR_PREFLIGHT"
SUPPORTED_IMAGE_FORMATS = {"png", "jpg", "jpeg"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_DIMENSION = 2048
_IMAGE_DESCRIPTOR_KEYS = {"format", "width", "height", "byte_class", "validation_status"}

_FORBIDDEN_KEYS = {
    "path", "image", "image_bytes", "base64", "prompt", "response",
    "stdout", "stderr", "raw_output", "credential", "token", "secret",
    "password", "command", "command_args",
}
_FORBIDDEN_VALUE = re.compile(
    r"(?i)(?:/home/|[a-z]:[\\/](?:users|home|srv|tmp|var)[\\/]|"
    r"opencode@|100[.]67[.]\d+[.]\d+|data:image/|base64,)"
)


def sanitize_multimodal_receipt(value: Any) -> Any:
    """Remove prohibited payload keys recursively before serialization."""
    if isinstance(value, dict):
        return {
            str(key): sanitize_multimodal_receipt(item)
            for key, item in value.items()
            if str(key).lower() not in _FORBIDDEN_KEYS
        }
    if isinstance(value, list):
        return [sanitize_multimodal_receipt(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_multimodal_receipt(item) for item in value]
    return deepcopy(value)


def find_unsanitized_multimodal_content(value: Any, path: str = "") -> list[str]:
    """Return deterministic reasons why a receipt cannot be persisted."""
    errors: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            item_path = f"{path}.{key_text}" if path else key_text
            if key_text.lower() in _FORBIDDEN_KEYS:
                errors.append(f"forbidden key at {item_path}")
            errors.extend(find_unsanitized_multimodal_content(item, item_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            errors.extend(find_unsanitized_multimodal_content(item, f"{path}[{index}]"))
    elif isinstance(value, str) and _FORBIDDEN_VALUE.search(value):
        errors.append(f"forbidden value at {path}")
    return errors


def validate_multimodal_receipt(receipt: Any) -> dict[str, Any]:
    """Validate the independent, sanitized multimodal preflight receipt shape."""
    errors: list[str] = []
    if not isinstance(receipt, dict):
        return {"status": "MULTIMODAL_RECEIPT_INVALID", "errors": ["receipt must be an object"]}
    errors.extend(find_unsanitized_multimodal_content(receipt))
    if receipt.get("schema_version") != MULTIMODAL_RECEIPT_SCHEMA_VERSION:
        errors.append("unsupported schema_version")
    if receipt.get("receipt_type") != MULTIMODAL_RECEIPT_TYPE:
        errors.append("unexpected receipt_type")
    for artifact_key in ("model_artifact", "projector_artifact"):
        artifact = receipt.get(artifact_key)
        if not isinstance(artifact, dict):
            errors.append(f"missing {artifact_key}")
            continue
        basename = artifact.get("basename")
        if (
            not isinstance(basename, str)
            or not basename
            or basename != basename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
            or basename in {".", ".."}
        ):
            errors.append(f"invalid {artifact_key}.basename")
        if not isinstance(artifact.get("size_bytes"), int) or artifact["size_bytes"] < 1:
            errors.append(f"invalid {artifact_key}.size_bytes")
        digest = artifact.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            errors.append(f"invalid {artifact_key}.sha256")
        if not isinstance(artifact.get("pairing_id"), str) or not artifact["pairing_id"]:
            errors.append(f"invalid {artifact_key}.pairing_id")
    model_pairing = (receipt.get("model_artifact") or {}).get("pairing_id")
    projector_pairing = (receipt.get("projector_artifact") or {}).get("pairing_id")
    if model_pairing != projector_pairing:
        errors.append("model/projector pairing does not match")
    descriptor = receipt.get("image_descriptor")
    if not isinstance(descriptor, dict):
        errors.append("missing image_descriptor")
    elif set(descriptor) != _IMAGE_DESCRIPTOR_KEYS:
        errors.append("image_descriptor fields are ambiguous")
    else:
        image_format = descriptor.get("format")
        width = descriptor.get("width")
        height = descriptor.get("height")
        byte_class = descriptor.get("byte_class")
        if image_format not in SUPPORTED_IMAGE_FORMATS:
            errors.append("invalid image_descriptor.format")
        if not isinstance(width, int) or not 0 < width <= MAX_IMAGE_DIMENSION:
            errors.append("invalid image_descriptor.width")
        if not isinstance(height, int) or not 0 < height <= MAX_IMAGE_DIMENSION:
            errors.append("invalid image_descriptor.height")
        if byte_class not in {"small", "medium"}:
            errors.append("invalid image_descriptor.byte_class")
        if descriptor.get("validation_status") != "VALID":
            errors.append("invalid image_descriptor.validation_status")
    if receipt.get("preflight_status") not in {"VALID", "REJECTED"}:
        errors.append("invalid preflight_status")
    if receipt.get("inference_invoked") is not False:
        errors.append("inference_invoked must be false")
    return {
        "status": "MULTIMODAL_RECEIPT_VALID" if not errors else "MULTIMODAL_RECEIPT_INVALID",
        "errors": errors,
        "receipt": sanitize_multimodal_receipt(receipt) if not errors else None,
    }
