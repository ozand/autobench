#!/usr/bin/env python3
"""Render exactly one zero-inference multimodal OCR dry-run job from safe JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.multimodal_dry_run import MultimodalDryRunError, build_one_serialized_job_dry_run


def _job(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("job must be a JSON object") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("job must be a JSON object")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", required=True)
    parser.add_argument("--job", action="append", type=_job, required=True)
    args = parser.parse_args()
    if len(args.job) != 1:
        parser.error("exactly one --job is required")
    try:
        summary = build_one_serialized_job_dry_run(args.job[0])
    except MultimodalDryRunError as exc:
        parser.error(str(exc))
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
