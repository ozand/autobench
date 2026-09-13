"""Fail-closed, non-executing multimodal OCR invocation contract.

This module deliberately does not open or decode images, load GGUF artifacts into a
runtime, invoke a model, or construct a shell command.  It validates the
non-persisted local-reference boundary that must succeed before a future,
separately authorized OCR runner may invoke anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from src.multimodal_preflight import (
    ArtifactExpectation,
    MultimodalPreflightError,
    validate_artifact,
    validate_image_descriptor,
)
from src.multimodal_receipt import (
    MULTIMODAL_EXECUTION_RECEIPT_TYPE,
    MULTIMODAL_INVOCATION_RECEIPT_TYPE,
    sanitize_multimodal_receipt,
    validate_multimodal_receipt,
)

OCR_TERMINAL_CLASSES = frozenset(
    {
        "IMAGE_INPUT_REJECTED",
        "PROJECTOR_MISMATCH",
        "UNSUPPORTED_BACKEND",
        "OOM",
        "CONTEXT_OVERFLOW",
        "METRIC_PARSE_FAILED",
        "OCR_MISSED",
        "OCR_INCONCLUSIVE",
        "SUCCESS",
    }
)

_NON_INVOCATION_TERMINAL_CLASSES = frozenset(
    {"IMAGE_INPUT_REJECTED", "PROJECTOR_MISMATCH"}
)
_SUPPORTED_SUFFIXES = {".png": "png", ".jpg": "jpg", ".jpeg": "jpeg"}
_CONFIGURATION_KEYS = {
    "device",
    "split_mode",
    "split_ratio",
    "context_length",
    "cache_type_k",
    "cache_type_v",
    "max_tokens",
}


class MultimodalContractError(ValueError):
    """Raised when an OCR invocation contract is unsafe or ambiguous."""


class ImageReferenceRejected(MultimodalContractError):
    """Raised before invocation when a local image reference violates the contract."""


@dataclass(frozen=True)
class PreparedMultimodalInvocation:
    """In-memory-only invocation material; never serialize ``image_reference``."""

    image_reference: Path
    model_artifact: dict[str, Any]
    projector_artifact: dict[str, Any]
    image_descriptor: dict[str, Any]
    configuration: dict[str, Any]


def _image_path(reference: Path | str) -> Path:
    """Accept a local absolute path only; reject URLs, data payloads and traversal."""
    if not isinstance(reference, (Path, str)):
        raise ImageReferenceRejected("IMAGE_INPUT_REJECTED: image reference is not a path")
    text = str(reference)
    if (
        not text
        or "://" in text
        or text.lower().startswith("data:")
        or text.startswith(("\\\\", "//"))
    ):
        raise ImageReferenceRejected("IMAGE_INPUT_REJECTED: image reference is not local")
    path = Path(text)
    if not path.is_absolute() or ".." in path.parts:
        raise ImageReferenceRejected("IMAGE_INPUT_REJECTED: image reference is unsafe")
    return path


def validate_local_image_reference(
    reference: Path | str,
    descriptor: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Stat, but never open/decode, a local image reference against supplied metadata.

    Dimensions remain caller-supplied metadata in this implementation.  A later
    execution runner must validate decoded dimensions in a non-persisted runtime
    boundary before model invocation; this module intentionally does not do so.
    """
    path = _image_path(reference)
    if not path.is_file() or path.is_symlink():
        raise ImageReferenceRejected("IMAGE_INPUT_REJECTED: image reference is missing or unsafe")
    suffix_format = _SUPPORTED_SUFFIXES.get(path.suffix.lower())
    if suffix_format is None:
        raise ImageReferenceRejected("IMAGE_INPUT_REJECTED: unsupported image format")
    if not isinstance(descriptor, Mapping):
        raise ImageReferenceRejected("IMAGE_INPUT_REJECTED: invalid image metadata")
    try:
        safe_descriptor = validate_image_descriptor(dict(descriptor))
    except (MultimodalPreflightError, TypeError, ValueError) as exc:
        raise ImageReferenceRejected("IMAGE_INPUT_REJECTED: invalid image metadata") from exc
    if safe_descriptor["format"] != suffix_format:
        raise ImageReferenceRejected("IMAGE_INPUT_REJECTED: metadata format mismatch")
    if path.stat().st_size != descriptor["size_bytes"]:
        raise ImageReferenceRejected("IMAGE_INPUT_REJECTED: metadata size mismatch")
    return path, safe_descriptor


def validate_configuration(configuration: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the bounded, payload-free configuration retained in a receipt."""
    if not isinstance(configuration, Mapping) or set(configuration) != _CONFIGURATION_KEYS:
        raise MultimodalContractError("configuration fields are ambiguous")
    value = dict(configuration)
    if not isinstance(value["device"], str) or value["device"] not in {"CPU", "Vulkan0", "Vulkan1", "Vulkan0,Vulkan1"}:
        raise MultimodalContractError("unsupported device")
    if not isinstance(value["split_mode"], str) or value["split_mode"] not in {"none", "layer"}:
        raise MultimodalContractError("unsupported split mode")
    if value["split_mode"] == "none" and value["split_ratio"] is not None:
        raise MultimodalContractError("single-device configuration cannot specify a split ratio")
    if value["split_mode"] == "layer" and (
        not isinstance(value["split_ratio"], str) or value["split_ratio"] not in {"1,1", "1,2"}
    ):
        raise MultimodalContractError("unsupported layer split ratio")
    if not isinstance(value["context_length"], int) or value["context_length"] not in {1024, 2048}:
        raise MultimodalContractError("unsupported context length")
    if (
        not isinstance(value["cache_type_k"], str)
        or not isinstance(value["cache_type_v"], str)
        or value["cache_type_k"] not in {"f16", "q8_0"}
        or value["cache_type_v"] not in {"f16", "q8_0"}
    ):
        raise MultimodalContractError("unsupported KV cache type")
    if value["cache_type_k"] != value["cache_type_v"]:
        raise MultimodalContractError("mixed KV cache types are not allowed")
    if type(value["max_tokens"]) is not int or not 1 <= value["max_tokens"] <= 128:
        raise MultimodalContractError("invalid generation cap")
    return value


def _receipt_artifacts(
    model_path: Path,
    projector_path: Path,
    model_expected: ArtifactExpectation,
    projector_expected: ArtifactExpectation,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not model_expected.pairing_id or model_expected.pairing_id != projector_expected.pairing_id:
        raise MultimodalContractError("PROJECTOR_MISMATCH: pairing identifier mismatch")
    try:
        model = validate_artifact(model_path, model_expected)
    except MultimodalPreflightError as exc:
        raise MultimodalContractError("artifact contract mismatch") from exc
    try:
        projector = validate_artifact(projector_path, projector_expected)
    except MultimodalPreflightError as exc:
        raise MultimodalContractError("PROJECTOR_MISMATCH: projector artifact contract mismatch") from exc
    return model, projector


def build_pre_invocation_rejection_receipt(
    *,
    model_artifact: Mapping[str, Any],
    projector_artifact: Mapping[str, Any],
    terminal_class: str,
) -> dict[str, Any]:
    """Create sanitized evidence for a failure that occurred before invocation."""
    if terminal_class not in _NON_INVOCATION_TERMINAL_CLASSES:
        raise MultimodalContractError("pre-invocation receipt has an invalid terminal class")
    model = dict(model_artifact)
    projector = dict(projector_artifact)
    if terminal_class == "PROJECTOR_MISMATCH":
        # Preserve the distinct, sanitized pairing identifiers as the evidence of
        # mismatch; equalizing them would make the terminal class unverifiable.
        if model.get("pairing_id") == projector.get("pairing_id"):
            raise MultimodalContractError("PROJECTOR_MISMATCH requires distinct pairing identifiers")
    elif model.get("pairing_id") != projector.get("pairing_id"):
        raise MultimodalContractError("non-mismatch rejection requires a paired artifact contract")
    receipt = {
        "schema_version": 1,
        "receipt_type": MULTIMODAL_EXECUTION_RECEIPT_TYPE,
        "model_artifact": model,
        "projector_artifact": projector,
        "image_descriptor": {"validation_status": "REJECTED"},
        "configuration": None,
        "command_family": "multimodal_ocr_runner",
        "terminal_class": terminal_class,
        "invocation_attempted": False,
        "inference_invoked": False,
    }
    validation = validate_multimodal_receipt(receipt)
    if validation["status"] != "MULTIMODAL_RECEIPT_VALID":
        raise MultimodalContractError("generated rejection receipt failed validation")
    return sanitize_multimodal_receipt(receipt)


def prepare_multimodal_invocation(
    *,
    model_path: Path,
    projector_path: Path,
    model_expected: ArtifactExpectation,
    projector_expected: ArtifactExpectation,
    image_reference: Path | str,
    image_descriptor: Mapping[str, Any],
    configuration: Mapping[str, Any],
) -> PreparedMultimodalInvocation:
    """Validate all boundaries before a future runner is allowed to invoke anything."""
    model, projector = _receipt_artifacts(
        model_path, projector_path, model_expected, projector_expected
    )
    image_path, safe_descriptor = validate_local_image_reference(image_reference, image_descriptor)
    return PreparedMultimodalInvocation(
        image_reference=image_path,
        model_artifact=model,
        projector_artifact=projector,
        image_descriptor=safe_descriptor,
        configuration=validate_configuration(configuration),
    )


def build_prepared_invocation_receipt(
    prepared: PreparedMultimodalInvocation,
) -> dict[str, Any]:
    """Serialize only the safe contract; never serialize ``prepared.image_reference``."""
    receipt = {
        "schema_version": 1,
        "receipt_type": MULTIMODAL_INVOCATION_RECEIPT_TYPE,
        "model_artifact": prepared.model_artifact,
        "projector_artifact": prepared.projector_artifact,
        "image_descriptor": prepared.image_descriptor,
        "configuration": prepared.configuration,
        "command_family": "multimodal_ocr_runner",
        "inference_invoked": False,
    }
    validation = validate_multimodal_receipt(receipt)
    if validation["status"] != "MULTIMODAL_RECEIPT_VALID":
        raise MultimodalContractError("generated prepared receipt failed validation")
    return sanitize_multimodal_receipt(receipt)


def terminal_class_for_event(event: str) -> str:
    """Map only known sanitized event identifiers to ADR-003 terminal classes."""
    mapping = {
        "image_rejected": "IMAGE_INPUT_REJECTED",
        "projector_mismatch": "PROJECTOR_MISMATCH",
        "unsupported_backend": "UNSUPPORTED_BACKEND",
        "oom": "OOM",
        "context_overflow": "CONTEXT_OVERFLOW",
        "metric_parse_failed": "METRIC_PARSE_FAILED",
        "ocr_missed": "OCR_MISSED",
        "ocr_inconclusive": "OCR_INCONCLUSIVE",
        "success": "SUCCESS",
    }
    if not isinstance(event, str):
        raise MultimodalContractError("unknown multimodal event")
    try:
        return mapping[event]
    except KeyError as exc:
        raise MultimodalContractError("unknown multimodal event") from exc
