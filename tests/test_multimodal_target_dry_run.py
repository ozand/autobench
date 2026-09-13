import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.multimodal_target_dry_run import _write_new_receipt, main
from src.multimodal_command_plan import APPROVED_ARTIFACTS, build_first_baseline_command_plan
from src.multimodal_target_dry_run import (
    MultimodalTargetDryRunError,
    validate_target_first_baseline_plan,
)
from src.multimodal_runner import PreparedMultimodalInvocation


def _plan() -> dict:
    return build_first_baseline_command_plan(
        PreparedMultimodalInvocation(
            image_reference=Path("C:/ephemeral/non-persisted.png"),
            model_artifact=dict(APPROVED_ARTIFACTS["model_artifact"]),
            projector_artifact=dict(APPROVED_ARTIFACTS["projector_artifact"]),
            image_descriptor={"format": "png", "width": 28, "height": 28, "byte_class": "small", "validation_status": "VALID"},
            configuration={
                "device": "Vulkan0", "split_mode": "none", "split_ratio": None,
                "context_length": 1024, "cache_type_k": "f16", "cache_type_v": "f16",
                "max_tokens": 32,
            },
        )
    )


def test_target_dry_run_binds_only_identity_binary_and_plan_without_execution_or_image_open():
    plan = _plan()
    with patch(
        "src.multimodal_target_dry_run.validate_artifact",
        side_effect=[dict(APPROVED_ARTIFACTS["model_artifact"]), dict(APPROVED_ARTIFACTS["projector_artifact"])],
    ) as artifact_check, patch("src.multimodal_target_dry_run.validate_target_binary") as binary_check, patch("subprocess.Popen") as popen, patch("subprocess.run") as run, patch("pathlib.Path.open") as open_file:
        receipt = validate_target_first_baseline_plan(
            model_path=Path("C:/not-loaded-model.gguf"),
            projector_path=Path("C:/not-loaded-projector.gguf"),
            binary_path=Path("C:/llama-mtmd-cli"),
            plan=plan,
        )
    assert artifact_check.call_count == 2
    binary_check.assert_called_once()
    popen.assert_not_called()
    run.assert_not_called()
    open_file.assert_not_called()
    assert receipt["receipt_type"] == "MULTIMODAL_OCR_TARGET_DRY_RUN"
    assert receipt["validation_scope"] == "ARTIFACT_IDENTITY_AND_PLAN"
    assert receipt["planned_job_count"] == 1
    assert receipt["inference_invoked"] is False
    assert "not-loaded" not in str(receipt)
    assert "path" not in str(receipt).lower()


@pytest.mark.parametrize("mutator", [
    lambda plan: plan["configuration"].update({"device": "Vulkan1"}),
    lambda plan: plan["configuration"].update({
        "device": "Vulkan0,Vulkan1", "split_mode": "layer", "split_ratio": "1,1",
        "context_length": 2048, "cache_type_k": "q8_0", "cache_type_v": "q8_0", "max_tokens": 128,
    }),
    lambda plan: plan.update({
        "image_descriptor": {"format": "jpg", "width": 2048, "height": 2048, "byte_class": "medium", "validation_status": "VALID"}
    }),
])
def test_target_dry_run_rejects_plan_outside_exact_first_baseline_before_artifact_reads(mutator):
    bad_plan = _plan()
    mutator(bad_plan)
    with patch("src.multimodal_target_dry_run.validate_artifact") as artifact_check:
        with pytest.raises(MultimodalTargetDryRunError):
            validate_target_first_baseline_plan(
                model_path=Path("C:/model.gguf"), projector_path=Path("C:/projector.gguf"), binary_path=Path("C:/llama-mtmd-cli"), plan=bad_plan
            )
    artifact_check.assert_not_called()


def test_target_dry_run_sanitizes_artifact_identity_exception():
    with patch("src.multimodal_target_dry_run.validate_target_binary"), patch("src.multimodal_target_dry_run.validate_artifact", side_effect=Exception("must not leak")):
        with pytest.raises(MultimodalTargetDryRunError):
            validate_target_first_baseline_plan(
                model_path=Path("C:/model.gguf"), projector_path=Path("C:/projector.gguf"), binary_path=Path("C:/llama-mtmd-cli"), plan=_plan()
            )


@pytest.mark.parametrize("binary", [Path("C:/wrong-binary"), Path("C:/llama-mtmd-cli")])
def test_binary_validation_rejects_missing_or_wrong_binary(binary):
    with patch("src.multimodal_target_dry_run.Path.is_file", return_value=False), patch("src.multimodal_target_dry_run.os.access", return_value=False):
        from src.multimodal_target_dry_run import validate_target_binary
        with pytest.raises(MultimodalTargetDryRunError):
            validate_target_binary(binary)


def test_target_dry_run_cli_writes_only_sanitized_receipt(tmp_path, capsys):
    output = tmp_path / "receipt.json"
    plan = _plan()
    with patch.object(
        sys,
        "argv",
        [
            "multimodal_target_dry_run.py", "--dry-run", "--model", "C:/model.gguf",
            "--projector", "C:/projector.gguf", "--binary", "C:/llama-mtmd-cli", "--plan", json.dumps(plan), "--output", str(output),
        ],
    ), patch(
        "scripts.multimodal_target_dry_run.validate_target_first_baseline_plan",
        return_value={
            "validation_scope": "ARTIFACT_IDENTITY_AND_PLAN",
            "planned_job_count": 1,
            "inference_invoked": False,
        },
    ) as validate:
        assert main() == 0
    validate.assert_called_once()
    assert json.loads(output.read_text()) == {
        "validation_scope": "ARTIFACT_IDENTITY_AND_PLAN",
        "planned_job_count": 1,
        "inference_invoked": False,
    }
    assert json.loads(capsys.readouterr().out)["inference_invoked"] is False


def test_target_dry_run_receipt_writer_rejects_collision_existing_and_symlink_outputs(tmp_path):
    protected = tmp_path / "Qwen2-VL-2B-Instruct-Q4_K_M.gguf"
    protected.write_bytes(b"unchanged")
    receipt = {"inference_invoked": False}
    with pytest.raises(MultimodalTargetDryRunError):
        _write_new_receipt(protected, receipt, (protected,))
    assert protected.read_bytes() == b"unchanged"
    existing = tmp_path / "existing.json"
    existing.write_text("unchanged", encoding="utf-8")
    with pytest.raises(MultimodalTargetDryRunError):
        _write_new_receipt(existing, receipt, (protected,))
    assert existing.read_text(encoding="utf-8") == "unchanged"
    target = tmp_path / "outside.json"
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(MultimodalTargetDryRunError):
        _write_new_receipt(link, receipt, (protected,))
    assert not target.exists()
    outside_directory = tmp_path / "outside-directory"
    outside_directory.mkdir()
    ancestor_link = tmp_path / "ancestor-link"
    ancestor_link.symlink_to(outside_directory, target_is_directory=True)
    nested = outside_directory / "nested"
    nested.mkdir()
    redirected = ancestor_link / "nested" / "receipt.json"
    with pytest.raises(MultimodalTargetDryRunError):
        _write_new_receipt(redirected, receipt, (protected,))
    assert not (nested / "receipt.json").exists()


def test_target_dry_run_receipt_writer_creates_new_regular_file(tmp_path):
    output = tmp_path / "new.json"
    _write_new_receipt(output, {"inference_invoked": False}, ())
    assert json.loads(output.read_text(encoding="utf-8")) == {"inference_invoked": False}


def test_target_dry_run_receipt_writer_closes_descriptor_and_removes_output_on_fdopen_failure(tmp_path):
    output = tmp_path / "fdopen-failure.json"
    real_close = __import__("os").close
    with patch("scripts.multimodal_target_dry_run.os.fdopen", side_effect=OSError("synthetic failure")), patch("scripts.multimodal_target_dry_run.os.close", side_effect=real_close) as close:
        with pytest.raises(OSError):
            _write_new_receipt(output, {"inference_invoked": False}, ())
    close.assert_called_once()
    assert not output.exists()


def test_target_dry_run_module_has_no_process_image_or_remote_runtime_imports():
    source = Path("src/multimodal_target_dry_run.py").read_text(encoding="utf-8")
    for forbidden in ("import subprocess", "Popen(", "subprocess.", "run_host_command", "requests", "Path.open(", "PIL", "llama-"):
        assert forbidden not in source
