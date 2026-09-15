"""One-run target OCR wrapper contract.

The wrapper composes the approved identity, staging, argv, stream-bound and receipt
contracts. Its only invocation boundary is an injected callable; tests use mocks.
A later #138 command supplies the reviewed real process function exactly once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.multimodal_execution_driver import ProcessObservation, classify_mock_observation
from src.multimodal_receipt import (
    MULTIMODAL_SMOKE_EXECUTION_RECEIPT_TYPE,
    sanitize_multimodal_receipt,
    validate_multimodal_receipt,
)
from src.multimodal_target_harness import (
    TASK_ID,
    TASK_VERSION,
    TIMEOUT_SECONDS,
    MultimodalTargetHarnessError,
    StagedDocument,
    build_target_argv,
    temporary_document_and_prompt,
    validate_target_identity,
)


class MultimodalTargetWrapperError(ValueError):
    """Raised when the one-run target wrapper rejects its contract."""


ProcessRunner = Callable[[Sequence[str], int], ProcessObservation]


def _receipt(
    *,
    plan: Mapping[str, Any],
    model_artifact: Mapping[str, Any],
    projector_artifact: Mapping[str, Any],
    document: StagedDocument,
    terminal_class: str,
    output_classification: str,
    invocation_attempted: bool,
    inference_invoked: bool,
) -> dict[str, Any]:
    receipt = {
        "schema_version": 1,
        "receipt_type": MULTIMODAL_SMOKE_EXECUTION_RECEIPT_TYPE,
        "model_artifact": dict(model_artifact),
        "projector_artifact": dict(projector_artifact),
        "image_descriptor": document.descriptor,
        "configuration": plan["configuration"],
        "command_family": "multimodal_ocr_runner",
        "terminal_class": terminal_class,
        "task_id": TASK_ID,
        "task_version": TASK_VERSION,
        "output_classification": output_classification,
        "invocation_attempted": invocation_attempted,
        "inference_invoked": inference_invoked,
    }
    validation = validate_multimodal_receipt(receipt)
    if validation["status"] != "MULTIMODAL_RECEIPT_VALID":
        raise MultimodalTargetWrapperError("terminal receipt is invalid")
    return sanitize_multimodal_receipt(receipt)


def run_one_target_smoke(
    *,
    binary_path: Path,
    model_path: Path,
    projector_path: Path,
    source_image: Path,
    plan: Mapping[str, Any],
    temporary_root: Path,
    process_runner: ProcessRunner,
) -> dict[str, Any]:
    """Prepare and invoke exactly one injected shell-free process, then stop."""
    if not callable(process_runner):
        raise MultimodalTargetWrapperError("process runner is invalid")
    try:
        from src.multimodal_command_plan import validate_first_baseline_command_plan
        safe_plan = validate_first_baseline_command_plan(plan)
        model_artifact, projector_artifact = validate_target_identity(
            binary_path, model_path, projector_path
        )
        with temporary_document_and_prompt(temporary_root, source_image) as (document, prompt):
            if safe_plan["image_descriptor"] != document.descriptor:
                raise MultimodalTargetWrapperError("approved plan does not bind staged JPEG descriptor")
            argv = build_target_argv(binary_path, model_path, projector_path, document.path, prompt, safe_plan)
            invocation_attempted = False
            try:
                invocation_attempted = True
                observation = process_runner(argv, TIMEOUT_SECONDS)
            except TimeoutError:
                return _receipt(
                    plan=safe_plan, model_artifact=model_artifact, projector_artifact=projector_artifact,
                    document=document, terminal_class="INCONCLUSIVE", output_classification="RUNTIME_TIMEOUT",
                    invocation_attempted=invocation_attempted, inference_invoked=False,
                )
            except Exception:
                return _receipt(
                    plan=safe_plan, model_artifact=model_artifact, projector_artifact=projector_artifact,
                    document=document, terminal_class="EXECUTION_ERROR", output_classification="RUNTIME_DRIVER_ERROR",
                    invocation_attempted=invocation_attempted, inference_invoked=False,
                )
            terminal_class, output_classification = classify_mock_observation(observation)
            return _receipt(
                plan=safe_plan, model_artifact=model_artifact, projector_artifact=projector_artifact,
                document=document, terminal_class=terminal_class, output_classification=output_classification,
                invocation_attempted=invocation_attempted, inference_invoked=terminal_class == "SUCCESS",
            )
    except Exception as exc:
        raise MultimodalTargetWrapperError("target wrapper precondition failed") from exc
