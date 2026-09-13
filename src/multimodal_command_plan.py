"""Deterministic, non-executing plan renderer for the first bounded OCR command.

It represents flags only.  Artifact and image paths remain in-memory and are
never returned or serialized.  This module deliberately has no process,
model-loading, image, network, or remote execution primitive.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from src.multimodal_receipt import (
    MULTIMODAL_COMMAND_PLAN_RECEIPT_TYPE,
    sanitize_multimodal_receipt,
    validate_multimodal_receipt,
)
from src.multimodal_runner import MultimodalContractError, PreparedMultimodalInvocation

COMMAND_BINARY = "llama-mtmd-cli"
COMMAND_FAMILY = "multimodal_ocr_runner"
EXPECTED_FLAGS = ("-m", "--mmproj", "--image", "-ngl", "-dev", "-c", "-ctk", "-ctv", "-n")
APPROVED_PAIRING_ID = "qwen2-vl-2b-instruct-q4km-q8proj"
APPROVED_IMAGE_DESCRIPTOR = {
    "format": "png",
    "width": 28,
    "height": 28,
    "byte_class": "small",
    "validation_status": "VALID",
}
APPROVED_ARTIFACTS = {
    "model_artifact": {
        "basename": "Qwen2-VL-2B-Instruct-Q4_K_M.gguf",
        "size_bytes": 986046944,
        "sha256": "5745685d2e607a82a0696c1118e56a2a1ae0901da450fd9cd4f161c6b62867d7",
        "pairing_id": APPROVED_PAIRING_ID,
    },
    "projector_artifact": {
        "basename": "mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf",
        "size_bytes": 709883360,
        "sha256": "a0ad91f00a7a80dcf84d719a61b00ee2e07b71794f4ee2dfa81a254621a8c418",
        "pairing_id": APPROVED_PAIRING_ID,
    },
}


class MultimodalCommandPlanError(ValueError):
    """Raised when a planned OCR command is not the single approved shape."""


def _validate_first_baseline(prepared: PreparedMultimodalInvocation) -> None:
    if not isinstance(prepared, PreparedMultimodalInvocation):
        raise MultimodalCommandPlanError("command plan requires a prepared invocation")
    if (
        prepared.model_artifact != APPROVED_ARTIFACTS["model_artifact"]
        or prepared.projector_artifact != APPROVED_ARTIFACTS["projector_artifact"]
    ):
        raise MultimodalCommandPlanError("command plan artifact pair is not approved")
    config = prepared.configuration
    expected = {
        "device": "Vulkan0",
        "split_mode": "none",
        "split_ratio": None,
        "context_length": 1024,
        "cache_type_k": "f16",
        "cache_type_v": "f16",
    }
    if not isinstance(config, Mapping) or set(config) != {
        "device", "split_mode", "split_ratio", "context_length", "cache_type_k",
        "cache_type_v", "max_tokens",
    }:
        raise MultimodalCommandPlanError("command plan configuration is malformed")
    if any(config.get(key) != value for key, value in expected.items()):
        raise MultimodalCommandPlanError("command plan configuration is outside first baseline")
    if type(config.get("max_tokens")) is not int or not 1 <= config["max_tokens"] <= 32:
        raise MultimodalCommandPlanError("command plan generation cap is outside first baseline")


def validate_first_baseline_command_plan(plan: Any) -> dict[str, Any]:
    """Fail closed unless a serialized plan is the one approved first baseline."""
    validation = validate_multimodal_receipt(plan)
    if (
        validation["status"] != "MULTIMODAL_RECEIPT_VALID"
        or not isinstance(plan, Mapping)
        or plan.get("receipt_type") != MULTIMODAL_COMMAND_PLAN_RECEIPT_TYPE
    ):
        raise MultimodalCommandPlanError("command plan receipt is invalid")
    if (
        plan.get("model_artifact") != APPROVED_ARTIFACTS["model_artifact"]
        or plan.get("projector_artifact") != APPROVED_ARTIFACTS["projector_artifact"]
        or plan.get("image_descriptor") != APPROVED_IMAGE_DESCRIPTOR
    ):
        raise MultimodalCommandPlanError("command plan binding is not approved")
    prepared = PreparedMultimodalInvocation(
        image_reference=Path("C:/non-persisted-image-reference.png"),
        model_artifact=plan["model_artifact"],
        projector_artifact=plan["projector_artifact"],
        image_descriptor=plan["image_descriptor"],
        configuration=plan["configuration"],
    )
    _validate_first_baseline(prepared)
    return validation["receipt"]


def _plan_receipt(prepared: PreparedMultimodalInvocation) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "receipt_type": MULTIMODAL_COMMAND_PLAN_RECEIPT_TYPE,
        "model_artifact": prepared.model_artifact,
        "projector_artifact": prepared.projector_artifact,
        "image_descriptor": prepared.image_descriptor,
        "configuration": prepared.configuration,
        "command_family": COMMAND_FAMILY,
        "binary": COMMAND_BINARY,
        "argument_flags": list(EXPECTED_FLAGS),
        "planned_job_count": 1,
        "dry_run": True,
        "inference_invoked": False,
    }


def build_first_baseline_command_plan(prepared: PreparedMultimodalInvocation) -> dict[str, Any]:
    """Return one sanitized command-plan receipt without exposing raw argv values."""
    _validate_first_baseline(prepared)
    if prepared.image_descriptor != APPROVED_IMAGE_DESCRIPTOR:
        raise MultimodalCommandPlanError("command plan image descriptor is not approved")
    receipt = _plan_receipt(prepared)
    try:
        validated = validate_first_baseline_command_plan(receipt)
    except MultimodalCommandPlanError as exc:
        raise MultimodalCommandPlanError("generated command plan failed validation") from exc
    return sanitize_multimodal_receipt(validated)


def render_first_baseline_dry_run(prepared: PreparedMultimodalInvocation) -> dict[str, Any]:
    """Render a one-job zero-inference summary; do not construct executable argv."""
    plan = build_first_baseline_command_plan(prepared)
    return {
        "dry_run": True,
        "planned_job_count": 1,
        "command_family": COMMAND_FAMILY,
        "binary": COMMAND_BINARY,
        "argument_flags": list(EXPECTED_FLAGS),
        "inference_invoked": False,
        "plan": plan,
    }
