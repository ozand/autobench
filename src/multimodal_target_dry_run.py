"""Target-safe zero-inference validation of the approved first OCR plan.

This adapter validates file identity and an already prepared command-plan receipt.
It never creates an image, opens/decodes an image, loads model data, builds raw
argv values, or starts a process. The caller chooses how to deploy it; its
receipt contains no target identifier or path.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from src.multimodal_command_plan import (
    APPROVED_ARTIFACTS,
    COMMAND_BINARY,
    COMMAND_FAMILY,
    EXPECTED_FLAGS,
    MultimodalCommandPlanError,
    validate_first_baseline_command_plan,
)
from src.multimodal_preflight import ArtifactExpectation, MultimodalPreflightError, validate_artifact
from src.multimodal_receipt import (
    MULTIMODAL_COMMAND_PLAN_RECEIPT_TYPE,
    MULTIMODAL_TARGET_DRY_RUN_RECEIPT_TYPE,
    sanitize_multimodal_receipt,
    validate_multimodal_receipt,
)


class MultimodalTargetDryRunError(ValueError):
    """Raised when target-side zero-inference plan validation fails closed."""


def validate_target_binary(binary_path: Path) -> None:
    """Verify the selected executable by metadata only; never run or load it."""
    path = Path(binary_path)
    if path.name != COMMAND_BINARY or not path.is_file() or not os.access(path, os.X_OK):
        raise MultimodalTargetDryRunError("target binary selection is invalid")


def _expectation(kind: str) -> ArtifactExpectation:
    artifact = APPROVED_ARTIFACTS[kind]
    return ArtifactExpectation(**artifact)


def _validate_plan(plan: Any) -> dict[str, Any]:
    try:
        validated = validate_first_baseline_command_plan(plan)
    except MultimodalCommandPlanError as exc:
        raise MultimodalTargetDryRunError("target dry-run requires the approved first-baseline command plan") from exc
    return validated


def validate_target_first_baseline_plan(
    *,
    model_path: Path,
    projector_path: Path,
    binary_path: Path,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind exact existing target artifacts to one zero-inference command plan."""
    safe_plan = _validate_plan(plan)
    validate_target_binary(binary_path)
    try:
        model = validate_artifact(Path(model_path), _expectation("model_artifact"))
        projector = validate_artifact(Path(projector_path), _expectation("projector_artifact"))
    except Exception as exc:
        # The adapter exposes no backend/file payload or implementation detail.
        raise MultimodalTargetDryRunError("target artifact identity validation failed") from exc
    receipt = {
        "schema_version": 1,
        "receipt_type": MULTIMODAL_TARGET_DRY_RUN_RECEIPT_TYPE,
        "model_artifact": model,
        "projector_artifact": projector,
        "image_descriptor": safe_plan["image_descriptor"],
        "configuration": safe_plan["configuration"],
        "command_family": COMMAND_FAMILY,
        "binary": COMMAND_BINARY,
        "argument_flags": list(EXPECTED_FLAGS),
        "planned_job_count": 1,
        "dry_run": True,
        "inference_invoked": False,
        "validation_scope": "ARTIFACT_IDENTITY_AND_PLAN",
    }
    validation = validate_multimodal_receipt(receipt)
    if validation["status"] != "MULTIMODAL_RECEIPT_VALID":
        raise MultimodalTargetDryRunError("generated target dry-run receipt failed validation")
    return sanitize_multimodal_receipt(receipt)
