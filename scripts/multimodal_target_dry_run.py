#!/usr/bin/env python3
"""Validate one approved OCR command plan against target artifact identity only."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.multimodal_target_dry_run import (
    MultimodalTargetDryRunError,
    validate_target_first_baseline_plan,
)


def _plan(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("plan must be a JSON object") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("plan must be a JSON object")
    return parsed


def _safe_receipt_directory(directory: Path) -> Path:
    """Reject a directory chain whose lexical route resolves through a link/reparse point."""
    candidate = Path(directory)
    if not candidate.is_dir() or candidate.is_symlink():
        raise MultimodalTargetDryRunError("receipt directory is unsafe")
    try:
        lexical = os.path.normcase(os.path.normpath(str(candidate.absolute())))
        resolved = candidate.resolve(strict=True)
        canonical = os.path.normcase(os.path.normpath(str(resolved)))
    except OSError as exc:
        raise MultimodalTargetDryRunError("receipt directory is unsafe") from exc
    if lexical != canonical:
        raise MultimodalTargetDryRunError("receipt directory resolves through an unsafe ancestor")
    return resolved


def _write_new_receipt(output: Path, receipt: dict[str, Any], protected: tuple[Path, ...]) -> None:
    """Write one new receipt without following links or mutating protected inputs."""
    path = Path(output)
    directory = _safe_receipt_directory(path.parent)
    try:
        resolved_output = directory / path.name
        if any(resolved_output == protected_path.resolve(strict=False) for protected_path in protected):
            raise MultimodalTargetDryRunError("receipt output collides with protected input")
    except OSError as exc:
        raise MultimodalTargetDryRunError("receipt output path is unsafe") from exc
    if path.exists() or path.is_symlink():
        raise MultimodalTargetDryRunError("receipt output must be a new regular file")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise MultimodalTargetDryRunError("receipt output cannot be safely created") from exc
    descriptor_owned = True
    try:
        handle = os.fdopen(descriptor, "w", encoding="utf-8")
        descriptor_owned = False
        with handle:
            file_stat = os.fstat(handle.fileno())
            if not stat.S_ISREG(file_stat.st_mode):
                raise MultimodalTargetDryRunError("receipt output is not regular")
            handle.write(json.dumps(receipt, sort_keys=True) + "\n")
    except Exception:
        if descriptor_owned:
            try:
                os.close(descriptor)
            except OSError:
                pass
        try:
            if path.exists() and not path.is_symlink():
                path.unlink()
        except OSError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--projector", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--plan", type=_plan, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = validate_target_first_baseline_plan(
            model_path=args.model,
            projector_path=args.projector,
            binary_path=args.binary,
            plan=args.plan,
        )
    except MultimodalTargetDryRunError as exc:
        parser.error(str(exc))
    try:
        _write_new_receipt(args.output, receipt, (args.model, args.projector, args.binary))
    except MultimodalTargetDryRunError as exc:
        parser.error(str(exc))
    print(json.dumps({
        "validation_scope": receipt["validation_scope"],
        "planned_job_count": receipt["planned_job_count"],
        "inference_invoked": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
