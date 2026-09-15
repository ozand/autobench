#!/usr/bin/env python3
"""Run exactly one approved target-local OCR smoke through the reviewed launcher.

This CLI is deployed as code under Issue #147. Issue #138 alone authorizes a real
invocation after its fresh execution review. The receipt output is sanitized by
the launcher; raw process output is never printed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.multimodal_command_plan import APPROVED_ARTIFACTS, build_first_baseline_command_plan
from src.multimodal_runner import PreparedMultimodalInvocation
from src.multimodal_target_harness import _jpeg_descriptor
from src.multimodal_target_launcher import MultimodalTargetLauncherError, launch_one_target_smoke
from src.multimodal_target_wrapper import MultimodalTargetWrapperError


def _plan(image_descriptor: dict) -> dict:
    return build_first_baseline_command_plan(PreparedMultimodalInvocation(
        image_reference=Path("/non-persisted-image-reference.jpg"),
        model_artifact=dict(APPROVED_ARTIFACTS["model_artifact"]),
        projector_artifact=dict(APPROVED_ARTIFACTS["projector_artifact"]),
        image_descriptor=dict(image_descriptor),
        configuration={
            "device": "Vulkan0", "split_mode": "none", "split_ratio": None,
            "context_length": 1024, "cache_type_k": "f16", "cache_type_v": "f16", "max_tokens": 32,
        },
    ))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--projector", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--temporary-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostic-output", type=Path, default=None)
    args = parser.parse_args()
    try:
        receipt = launch_one_target_smoke(
            binary_path=args.binary,
            model_path=args.model,
            projector_path=args.projector,
            target_image_path=args.image,
            plan=_plan(_jpeg_descriptor(args.image)),
            temporary_root=args.temporary_root,
            output=args.output,
            diagnostic_output=args.diagnostic_output,
        )
    except (MultimodalTargetLauncherError, MultimodalTargetWrapperError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "terminal_class": receipt["terminal_class"],
        "inference_invoked": receipt["inference_invoked"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
