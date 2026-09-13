import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.multimodal_ocr_dry_run import main
from src.multimodal_dry_run import (
    MultimodalDryRunError,
    build_one_job_dry_run,
    build_one_serialized_job_dry_run,
)
from src.multimodal_runner import PreparedMultimodalInvocation


def _prepared() -> PreparedMultimodalInvocation:
    """An already validated in-memory contract; no artifacts or image are created."""
    pairing_id = "qwen2-vl-2b-instruct-q4km-q8proj"
    return PreparedMultimodalInvocation(
        image_reference=Path("C:/non-persisted-reference.png"),
        model_artifact={
            "basename": "Qwen2-VL-2B-Instruct-Q4_K_M.gguf",
            "size_bytes": 986046944,
            "sha256": "5745685d2e607a82a0696c1118e56a2a1ae0901da450fd9cd4f161c6b62867d7",
            "pairing_id": pairing_id,
        },
        projector_artifact={
            "basename": "mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf",
            "size_bytes": 709883360,
            "sha256": "a0ad91f00a7a80dcf84d719a61b00ee2e07b71794f4ee2dfa81a254621a8c418",
            "pairing_id": pairing_id,
        },
        image_descriptor={"format": "png", "width": 28, "height": 28, "byte_class": "small", "validation_status": "VALID"},
        configuration={
            "device": "Vulkan0",
            "split_mode": "none",
            "split_ratio": None,
            "context_length": 1024,
            "cache_type_k": "f16",
            "cache_type_v": "f16",
            "max_tokens": 32,
        },
    )


def test_one_job_dry_run_renders_sanitized_plan_without_subprocess_or_image_open():
    prepared = _prepared()
    with patch("subprocess.Popen") as popen, patch("subprocess.run") as run, patch("pathlib.Path.open") as open_file:
        summary = build_one_job_dry_run(prepared)
    popen.assert_not_called()
    run.assert_not_called()
    open_file.assert_not_called()
    assert summary["dry_run"] is True
    assert summary["planned_job_count"] == 1
    assert summary["inference_invoked"] is False
    assert summary["command_family"] == "multimodal_ocr_runner"
    assert "synthetic.png" not in str(summary)
    assert "path" not in str(summary).lower()


def test_serialized_one_job_dry_run_accepts_only_safe_invocation_contract():
    summary = build_one_job_dry_run(_prepared())
    assert build_one_serialized_job_dry_run(summary["job"]) == summary
    with pytest.raises(MultimodalDryRunError):
        build_one_serialized_job_dry_run({})
    with pytest.raises(MultimodalDryRunError):
        build_one_serialized_job_dry_run([])
    unsafe = dict(summary["job"])
    unsafe["prompt"] = "forbidden"
    with pytest.raises(MultimodalDryRunError):
        build_one_serialized_job_dry_run(unsafe)


def test_cli_requires_exactly_one_dry_run_job_and_never_reaches_subprocess(capsys):
    job = build_one_job_dry_run(_prepared())["job"]
    with patch.object(sys, "argv", ["multimodal_ocr_dry_run.py", "--dry-run", "--job", json.dumps(job)]), patch("subprocess.Popen") as popen, patch("subprocess.run") as run:
        assert main() == 0
    popen.assert_not_called()
    run.assert_not_called()
    rendered = json.loads(capsys.readouterr().out)
    assert rendered["planned_job_count"] == 1
    with patch.object(sys, "argv", ["multimodal_ocr_dry_run.py", "--dry-run"]):
        with pytest.raises(SystemExit):
            main()
    with patch.object(sys, "argv", ["multimodal_ocr_dry_run.py", "--dry-run", "--job", json.dumps(job), "--job", json.dumps(job)]):
        with pytest.raises(SystemExit):
            main()


def test_dry_run_module_has_no_execution_or_image_runtime_imports():
    source = Path("src/multimodal_dry_run.py").read_text(encoding="utf-8")
    for forbidden in ("import subprocess", "Popen(", "subprocess.", "Path.open(", "PIL", "requests"):
        assert forbidden not in source
