#!/usr/bin/env python3
"""Validate a paired multimodal OCR preflight contract without model or image inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.multimodal_preflight import ArtifactExpectation, MultimodalPreflightError, build_preflight_receipt


def _expectation(value: str) -> ArtifactExpectation:
    try:
        basename, size, digest, pairing_id = value.split(":", 3)
        return ArtifactExpectation(
            basename=basename,
            size_bytes=int(size),
            sha256=digest,
            pairing_id=pairing_id,
        )
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("artifact expectation must be basename:size:sha256:pairing_id") from exc


def _descriptor(value: str) -> dict:
    try:
        descriptor = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("image descriptor must be JSON") from exc
    if not isinstance(descriptor, dict):
        raise argparse.ArgumentTypeError("image descriptor must be an object")
    return descriptor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--projector", type=Path, required=True)
    parser.add_argument("--model-expected", type=_expectation, required=True)
    parser.add_argument("--projector-expected", type=_expectation, required=True)
    parser.add_argument("--image-descriptor", type=_descriptor, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = build_preflight_receipt(
            model_path=args.model,
            projector_path=args.projector,
            model_expected=args.model_expected,
            projector_expected=args.projector_expected,
            image_descriptor=args.image_descriptor,
        )
    except MultimodalPreflightError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"preflight_status": receipt["preflight_status"], "inference_invoked": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
