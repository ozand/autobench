"""One-job, zero-inference dry-run rendering for the isolated OCR contract.

The module has no execution primitive: it does not create processes, open an
image, load artifacts, or communicate with a remote host.  It turns an already
prepared in-memory contract into a sanitized plan summary only.
"""

from __future__ import annotations

from typing import Any, Mapping

from src.multimodal_receipt import (
    MULTIMODAL_INVOCATION_RECEIPT_TYPE,
    sanitize_multimodal_receipt,
    validate_multimodal_receipt,
)
from src.multimodal_runner import (
    MultimodalContractError,
    PreparedMultimodalInvocation,
    build_prepared_invocation_receipt,
)

DRY_RUN_COMMAND_FAMILY = "multimodal_ocr_runner"


class MultimodalDryRunError(ValueError):
    """Raised when a dry-run job is malformed, unsafe, or ambiguous."""


def _validate_invocation_receipt(receipt: Any) -> dict[str, Any]:
    """Accept exactly the safe serialized form of one prepared invocation."""
    validation = validate_multimodal_receipt(receipt)
    if (
        validation["status"] != "MULTIMODAL_RECEIPT_VALID"
        or not isinstance(receipt, dict)
        or receipt.get("receipt_type") != MULTIMODAL_INVOCATION_RECEIPT_TYPE
    ):
        raise MultimodalDryRunError("dry-run job is not a valid invocation contract")
    return validation["receipt"]


def build_one_job_dry_run(
    prepared: PreparedMultimodalInvocation,
) -> dict[str, Any]:
    """Render exactly one safe plan from an already validated in-memory contract."""
    if not isinstance(prepared, PreparedMultimodalInvocation):
        raise MultimodalDryRunError("dry-run requires a prepared multimodal invocation")
    try:
        receipt = build_prepared_invocation_receipt(prepared)
    except MultimodalContractError as exc:
        raise MultimodalDryRunError("prepared contract cannot be rendered") from exc
    return build_one_serialized_job_dry_run(receipt)


def build_one_serialized_job_dry_run(job: Mapping[str, Any]) -> dict[str, Any]:
    """Render one serialized safe contract supplied to the CLI; never execute it."""
    if not isinstance(job, Mapping):
        raise MultimodalDryRunError("dry-run job must be an object")
    receipt = _validate_invocation_receipt(dict(job))
    summary = {
        "dry_run": True,
        "planned_job_count": 1,
        "command_family": DRY_RUN_COMMAND_FAMILY,
        "inference_invoked": False,
        "job": receipt,
    }
    # Deliberately re-sanitize at the final serialization boundary.
    return sanitize_multimodal_receipt(summary)
