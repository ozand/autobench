"""Mock-only target harness contract for the single approved OCR smoke.

It never starts a process. The later #138 execution wrapper must call this harness
for target identity and temporary JPEG/prompt preparation before its one process.
The staged JPEG descriptor is metadata-only and is distinct from the historical
synthetic PNG dry-run descriptor.
"""

from __future__ import annotations

import hashlib
import os
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator, Mapping

from src.multimodal_command_plan import APPROVED_ARTIFACTS, COMMAND_BINARY, validate_first_baseline_command_plan
from src.multimodal_execution_driver import ProcessObservation, classify_mock_observation
from src.multimodal_receipt import (
    MULTIMODAL_SMOKE_SIMULATION_RECEIPT_TYPE,
    sanitize_multimodal_receipt,
    validate_multimodal_receipt,
)

TIMEOUT_SECONDS = 120
TASK_ID = "ocr_smoke_v1"
TASK_VERSION = 1
_MAX_IMAGE_BYTES = 10 * 1024 * 1024
_HASH_BLOCK_BYTES = 1024 * 1024
_SMOKE_PROMPT = "Read the text in the image."


class MultimodalTargetHarnessError(ValueError):
    """Raised when the mock-only harness contract is unsafe or ambiguous."""


@dataclass(frozen=True)
class StagedDocument:
    path: Path
    descriptor: dict[str, Any]


def _trusted_temporary_root(root: Path) -> Path:
    candidate = Path(root)
    if not candidate.is_dir() or candidate.is_symlink():
        raise MultimodalTargetHarnessError("temporary root is invalid")
    try:
        lexical = os.path.normcase(os.path.normpath(str(candidate.absolute())))
        resolved = candidate.resolve(strict=True)
        canonical = os.path.normcase(os.path.normpath(str(resolved)))
    except OSError as exc:
        raise MultimodalTargetHarnessError("temporary root is invalid") from exc
    if lexical != canonical:
        raise MultimodalTargetHarnessError("temporary root resolves through an unsafe ancestor")
    return resolved


def _stream_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_HASH_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_file(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    candidate = Path(path)
    if candidate.name != expected["basename"] or not candidate.is_file() or candidate.is_symlink():
        raise MultimodalTargetHarnessError("target artifact identity is invalid")
    before = candidate.stat()
    if before.st_size != expected["size_bytes"] or _stream_sha256(candidate) != expected["sha256"]:
        raise MultimodalTargetHarnessError("target artifact identity is invalid")
    after = candidate.stat()
    if (before.st_dev, before.st_ino, before.st_size) != (after.st_dev, after.st_ino, after.st_size):
        raise MultimodalTargetHarnessError("target artifact changed during validation")
    return dict(expected)


def validate_target_identity(binary_path: Path, model_path: Path, projector_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify executable metadata and exact GGUF identities without runtime loading."""
    binary = Path(binary_path)
    if binary.name != COMMAND_BINARY or not binary.is_file() or binary.is_symlink() or not os.access(binary, os.X_OK):
        raise MultimodalTargetHarnessError("target binary identity is invalid")
    return (
        _validate_file(Path(model_path), APPROVED_ARTIFACTS["model_artifact"]),
        _validate_file(Path(projector_path), APPROVED_ARTIFACTS["projector_artifact"]),
    )


def _jpeg_descriptor(path: Path) -> dict[str, Any]:
    """Read JPEG headers only; never decode or inspect document pixels/content."""
    with path.open("rb") as handle:
        if handle.read(2) != b"\xff\xd8":
            raise MultimodalTargetHarnessError("designated JPEG header is invalid")
        while True:
            marker_prefix = handle.read(1)
            if not marker_prefix:
                break
            if marker_prefix != b"\xff":
                continue
            marker = handle.read(1)
            while marker == b"\xff":
                marker = handle.read(1)
            if not marker:
                break
            code = marker[0]
            if code in {0xD8, 0xD9}:
                continue
            length_bytes = handle.read(2)
            if len(length_bytes) != 2:
                break
            length = int.from_bytes(length_bytes, "big")
            if length < 2:
                break
            body = handle.read(length - 2)
            if len(body) != length - 2:
                break
            if code in {*range(0xC0, 0xC4), *range(0xC5, 0xC8), *range(0xC9, 0xCC), *range(0xCD, 0xD0)}:
                if len(body) < 5:
                    break
                height = int.from_bytes(body[1:3], "big")
                width = int.from_bytes(body[3:5], "big")
                if not 1 <= width <= 2048 or not 1 <= height <= 2048:
                    raise MultimodalTargetHarnessError("designated JPEG dimensions exceed contract")
                return {
                    "format": "jpg", "width": width, "height": height,
                    "byte_class": "small" if path.stat().st_size <= 1024 * 1024 else "medium",
                    "validation_status": "VALID",
                }
    raise MultimodalTargetHarnessError("designated JPEG dimensions are unavailable")


@contextmanager
def temporary_document_and_prompt(temporary_root: Path, source_image: Path) -> Iterator[tuple[StagedDocument, Path]]:
    """Stage one bounded regular JPEG + one 0600 prompt, deleting both on exit."""
    source = Path(source_image)
    root = _trusted_temporary_root(temporary_root)
    if (
        not source.is_file() or source.is_symlink() or source.suffix.lower() != ".jpg"
        or source.stat().st_size < 1 or source.stat().st_size > _MAX_IMAGE_BYTES
    ):
        raise MultimodalTargetHarnessError("designated image is invalid")
    source_descriptor = _jpeg_descriptor(source)
    source_stat = source.stat()
    with TemporaryDirectory(dir=root) as directory:
        run_dir = Path(directory)
        image = run_dir / "document.jpg"
        prompt = run_dir / "task.txt"
        image_fd = os.open(image, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
        try:
            with source.open("rb") as reader, os.fdopen(image_fd, "wb") as writer:
                image_fd = None
                for block in iter(lambda: reader.read(_HASH_BLOCK_BYTES), b""):
                    writer.write(block)
        finally:
            if image_fd is not None:
                os.close(image_fd)
        staged_stat = image.stat()
        if (
            image.is_symlink()
            or staged_stat.st_size != source_stat.st_size
            or _stream_sha256(image) != _stream_sha256(source)
        ):
            raise MultimodalTargetHarnessError("temporary image staging failed")
        prompt_fd = os.open(prompt, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
        try:
            with os.fdopen(prompt_fd, "w", encoding="utf-8") as handle:
                prompt_fd = None
                handle.write(_SMOKE_PROMPT)
        finally:
            if prompt_fd is not None:
                os.close(prompt_fd)
        yield StagedDocument(path=image, descriptor=source_descriptor), prompt


def build_target_argv(binary_path: Path, model_path: Path, projector_path: Path, image_path: Path, prompt_path: Path, plan: Mapping[str, Any]) -> list[str]:
    """Private shell-free shape renderer; caller must first call identity validation."""
    try:
        config = validate_first_baseline_command_plan(plan)["configuration"]
    except Exception as exc:
        raise MultimodalTargetHarnessError("target plan is not approved") from exc
    return [
        str(binary_path), "-m", str(model_path), "--mmproj", str(projector_path),
        "--image", str(image_path), "-f", str(prompt_path), "-ngl", "99",
        "-dev", config["device"], "-c", str(config["context_length"]),
        "-ctk", config["cache_type_k"], "-ctv", config["cache_type_v"],
        "-n", str(config["max_tokens"]), "-st", "-no-cnv", "--no-display-prompt", "--simple-io",
    ]


def build_mock_terminal_receipt(plan: Mapping[str, Any], document: StagedDocument, observation: ProcessObservation) -> dict[str, Any]:
    """Return simulation-only evidence bound to staged JPEG metadata, never inference."""
    try:
        safe_plan = validate_first_baseline_command_plan(plan)
    except Exception as exc:
        raise MultimodalTargetHarnessError("target plan is not approved") from exc
    terminal_class, output_classification = classify_mock_observation(observation)
    receipt = {
        "schema_version": 1,
        "receipt_type": MULTIMODAL_SMOKE_SIMULATION_RECEIPT_TYPE,
        "model_artifact": APPROVED_ARTIFACTS["model_artifact"],
        "projector_artifact": APPROVED_ARTIFACTS["projector_artifact"],
        "image_descriptor": document.descriptor,
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
        raise MultimodalTargetHarnessError("mock terminal receipt is invalid")
    return sanitize_multimodal_receipt(receipt)
