"""Non-executing preparation for the approved first OCR GPU smoke.

This module intentionally has no process, model, image-decoder, SSH, or deployment
primitive. It prepares private list argv, temporary prompt delivery, and a bounded
classifier for mocked tests. The later execution Issue owns the actual invocation.
"""

from __future__ import annotations

import os
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from src.multimodal_command_plan import (
    APPROVED_ARTIFACTS,
    COMMAND_BINARY,
    validate_first_baseline_command_plan,
)
from src.multimodal_receipt import (
    MULTIMODAL_SMOKE_SIMULATION_RECEIPT_TYPE,
    sanitize_multimodal_receipt,
    validate_multimodal_receipt,
)

TASK_ID = "ocr_smoke_v1"
TASK_VERSION = 1
OUTPUT_LIMIT_BYTES = 8192
_SMOKE_PROMPT = "Read the text in the image."


class MultimodalExecutionDriverError(ValueError):
    """Raised when non-executing smoke preparation is invalid."""


@dataclass(frozen=True)
class ProcessObservation:
    """Synthetic process observation for unit-tested classification only."""

    returncode: int
    stdout: bytes
    stderr: bytes


def _approved_plan(plan: Any) -> dict[str, Any]:
    try:
        return validate_first_baseline_command_plan(plan)
    except Exception as exc:
        raise MultimodalExecutionDriverError("execution plan is not approved") from exc


@contextmanager
def ephemeral_smoke_prompt(temporary_root: Path) -> Iterator[Path]:
    """Write the fixed non-sensitive prompt to one 0600 temporary file."""
    root = Path(temporary_root)
    if not root.is_dir() or root.is_symlink():
        raise MultimodalExecutionDriverError("prompt temporary root is unsafe")
    path = root.resolve(strict=True) / "autobench-ocr-smoke-task.txt"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    identity: tuple[int, int] | None = None
    try:
        try:
            descriptor = os.open(path, flags, 0o600)
        except FileExistsError as exc:
            raise MultimodalExecutionDriverError("prompt destination is ambiguous") from exc
        try:
            initial = os.fstat(descriptor)
            identity = (initial.st_dev, initial.st_ino)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                descriptor = None
                handle.write(_SMOKE_PROMPT)
        except Exception:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            raise
        yield path
    finally:
        try:
            current = path.lstat()
            if stat.S_ISREG(current.st_mode) and (
                identity is None or (current.st_dev, current.st_ino) == identity
            ):
                path.unlink()
        except FileNotFoundError:
            pass


def build_private_smoke_argv(
    *,
    binary_path: Path,
    model_path: Path,
    projector_path: Path,
    temporary_image_path: Path,
    prompt_path: Path,
    plan: Mapping[str, Any],
) -> list[str]:
    """Build private shell-free argv. It does not validate or open the image."""
    config = _approved_plan(plan)["configuration"]
    if Path(binary_path).name != COMMAND_BINARY:
        raise MultimodalExecutionDriverError("execution binary is not approved")
    if Path(model_path).name != APPROVED_ARTIFACTS["model_artifact"]["basename"]:
        raise MultimodalExecutionDriverError("execution model is not approved")
    if Path(projector_path).name != APPROVED_ARTIFACTS["projector_artifact"]["basename"]:
        raise MultimodalExecutionDriverError("execution projector is not approved")
    if Path(temporary_image_path).suffix.lower() != ".jpg" or Path(prompt_path).suffix.lower() != ".txt":
        raise MultimodalExecutionDriverError("execution temporary input shape is not approved")
    return [
        str(binary_path), "-m", str(model_path), "--mmproj", str(projector_path),
        "--image", str(temporary_image_path), "-f", str(prompt_path), "-ngl", "99",
        "-dev", config["device"], "-c", str(config["context_length"]),
        "-ctk", config["cache_type_k"], "-ctv", config["cache_type_v"],
        "-n", str(config["max_tokens"]), "-no-cnv", "--no-display-prompt", "--simple-io",
    ]


def classify_mock_observation(observation: ProcessObservation) -> tuple[str, str]:
    """Return only bounded status tags; never return stream content."""
    if (
        not isinstance(observation, ProcessObservation)
        or type(observation.returncode) is not int
        or not isinstance(observation.stdout, bytes)
        or not isinstance(observation.stderr, bytes)
    ):
        return "METRIC_PARSE_FAILED", "MALFORMED_STREAM"
    if len(observation.stdout) + len(observation.stderr) > OUTPUT_LIMIT_BYTES:
        return "METRIC_PARSE_FAILED", "OUTPUT_OVERSIZE"
    combined = (observation.stdout + b"\n" + observation.stderr).lower()
    if observation.returncode != 0:
        if b"out of memory" in combined or b"oom" in combined:
            return "OOM", "RUNTIME_OOM"
        if b"context" in combined and b"exceed" in combined:
            return "CONTEXT_OVERFLOW", "RUNTIME_CONTEXT"
        if b"vulkan" in combined and (b"unsupported" in combined or b"not supported" in combined):
            return "UNSUPPORTED_BACKEND", "RUNTIME_BACKEND"
        return "EXECUTION_ERROR", "RUNTIME_NONZERO"
    if observation.stdout.strip():
        return "SUCCESS", "OCR_SMOKE_OUTPUT_OBSERVED"
    return "OCR_INCONCLUSIVE", "OCR_SMOKE_NO_OUTPUT"


def build_simulated_smoke_receipt(plan: Mapping[str, Any], observation: ProcessObservation) -> dict[str, Any]:
    """Test-only sanitized classification evidence; it never claims real inference."""
    safe_plan = _approved_plan(plan)
    terminal_class, output_classification = classify_mock_observation(observation)
    receipt = {
        "schema_version": 1,
        "receipt_type": MULTIMODAL_SMOKE_SIMULATION_RECEIPT_TYPE,
        "model_artifact": APPROVED_ARTIFACTS["model_artifact"],
        "projector_artifact": APPROVED_ARTIFACTS["projector_artifact"],
        "image_descriptor": safe_plan["image_descriptor"],
        "configuration": safe_plan["configuration"],
        "command_family": "multimodal_ocr_runner",
        "terminal_class": terminal_class,
        "task_id": TASK_ID,
        "task_version": TASK_VERSION,
        "output_classification": output_classification,
        "simulated": True,
        "inference_invoked": False,
    }
    validation = validate_multimodal_receipt(receipt)
    if validation["status"] != "MULTIMODAL_RECEIPT_VALID":
        raise MultimodalExecutionDriverError("simulated smoke receipt is invalid")
    return sanitize_multimodal_receipt(receipt)
