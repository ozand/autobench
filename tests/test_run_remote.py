"""Tests for the safe local-to-remote workflow helper."""

from pathlib import Path
import sys

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from run_remote import (  # noqa: E402
    MAX_STAGED_IMAGE_BYTES,
    WorkflowError,
    _validated_staging_source,
    _remote_staging_root_check,
    _validated_remote_staging_root,
    ensure_clean_local_repository,
    stage_designated_jpeg,
    run_staged_ocr_smoke,
    _cleanup_staged_jpeg,
    OCR_RUNTIME_ROOTS,
    OCR_STAGING_ROOT,
    OCR_WORK_ROOT,
    OCR_OUTPUT_ROOT,
    provision_ocr_runtime_roots,
    ensure_origin_is_expected,
    command_requires_model_route,
    normalize_remote_command,
    parse_arguments,
    validate_required_model_route,
)


class GitResult:
    """Minimal completed-process substitute used by Git status tests."""

    def __init__(self, stdout: str) -> None:
        self.stdout = stdout


def test_remote_python_uses_virtualenv_and_quotes_arguments() -> None:
    command = normalize_remote_command(
        ["python", "inventory_bench.py", "--models", "model name.gguf"],
        "/srv/auto bench",
    )

    assert "cd '/srv/auto bench'" in command
    assert "export AUTOBENCH_EXECUTION_MODE=local" in command
    assert ".venv/bin/python inventory_bench.py" in command
    assert "'model name.gguf'" in command


def test_remote_pytest_uses_virtualenv_module() -> None:
    command = normalize_remote_command(["pytest", "-q"], "/srv/autobench")

    assert ".venv/bin/python -m pytest -q" in command


def test_remote_command_is_required() -> None:
    with pytest.raises(WorkflowError, match="No remote command supplied"):
        normalize_remote_command([], "/srv/autobench")


def test_dirty_repository_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "run_remote.run",
        lambda *args, **kwargs: GitResult(" M src/runner.py\n?? notes.txt\n"),
    )

    with pytest.raises(WorkflowError, match="not clean"):
        ensure_clean_local_repository(Path("."))


def test_unexpected_origin_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "run_remote.run",
        lambda *args, **kwargs: GitResult("https://example.invalid/wrong.git\n"),
    )

    with pytest.raises(WorkflowError, match="Unexpected origin URL"):
        ensure_origin_is_expected(Path("."), "https://example.invalid/right.git")


def _jpeg() -> bytes:
    return b"\xff\xd8\xff\xc0\x00\x11\x08\x00\x1c\x00\x1c\x03" + b"\x00" * 10 + b"\xff\xd9"


def test_staging_rejects_non_jpeg_oversize_or_unsafe_remote_root(tmp_path: Path) -> None:
    image = tmp_path / "input.jpg"
    image.write_bytes(_jpeg())
    assert _validated_staging_source(image) == image
    image.write_bytes(b"x" * (MAX_STAGED_IMAGE_BYTES + 1))
    with pytest.raises(WorkflowError):
        _validated_staging_source(image)
    with pytest.raises(WorkflowError):
        _validated_remote_staging_root("/tmp/root; injection", "/srv/autobench")
    with pytest.raises(WorkflowError):
        _validated_remote_staging_root("/srv/autobench/private", "/srv/autobench")
    with pytest.raises(WorkflowError):
        _validated_remote_staging_root("/srv/autobench/../tmp", "/srv/autobench")


def test_remote_root_check_requires_preexisting_canonical_private_root() -> None:
    script = _remote_staging_root_check(OCR_STAGING_ROOT, "/srv/autobench")
    assert "realpath -e" in script and "stat -c %a" in script and "mkdir" not in script
    assert "test ! -e" not in script


def test_staging_uses_injected_transport_and_cleans_target_after_failure(tmp_path: Path) -> None:
    image = tmp_path / "input.jpg"; image.write_bytes(_jpeg())
    remote_calls, copy_calls = [], []
    def remote(target, script):
        remote_calls.append((target, script))
        return f"{OCR_STAGING_ROOT}/.autobench-ocr-abc123.jpg" if "mktemp" in script else ""
    def fail_copy(command):
        copy_calls.append(command)
        raise OSError("transport")
    with pytest.raises(WorkflowError, match="staging failed"):
        stage_designated_jpeg("target", image, OCR_STAGING_ROOT, "/srv/autobench", remote_runner=remote, copy_runner=fail_copy)
    assert len(copy_calls) == 1 and len(remote_calls) == 4
    assert "input.jpg" not in str(remote_calls)
    assert remote_calls[0][1] == _remote_staging_root_check(OCR_STAGING_ROOT, "/srv/autobench")
    assert "mktemp" in remote_calls[1][1]
    assert "realpath -e" in remote_calls[2][1]
    assert "rm -f" in remote_calls[-1][1]


def test_staging_returns_private_created_location_without_printing_or_persisting_it(tmp_path: Path) -> None:
    image = tmp_path / "input.jpg"; image.write_bytes(_jpeg())
    calls = []
    def remote(target, script):
        calls.append(script)
        return f"{OCR_STAGING_ROOT}/.autobench-ocr-abc123.jpg" if "mktemp" in script else ""
    location = stage_designated_jpeg("target", image, OCR_STAGING_ROOT, "/srv/autobench", remote_runner=remote, copy_runner=lambda *_: None)
    assert location == f"{OCR_STAGING_ROOT}/.autobench-ocr-abc123.jpg"
    assert "chmod 600" in calls[-1]


def test_staging_rejects_malformed_private_destination_before_copy(tmp_path: Path) -> None:
    image = tmp_path / "input.jpg"; image.write_bytes(_jpeg())
    def remote(target, script):
        return "/tmp/staging/.autobench-ocr-x/../../checkout/file" if "mktemp" in script else ""
    with pytest.raises(WorkflowError, match="staging failed"):
        stage_designated_jpeg("target", image, OCR_STAGING_ROOT, "/srv/autobench", remote_runner=remote, copy_runner=lambda *_: pytest.fail("copy called"))


def test_staging_rejects_collision_before_copy_and_does_not_cleanup_unowned_destination(tmp_path: Path) -> None:
    image = tmp_path / "input.jpg"; image.write_bytes(_jpeg())
    calls = []
    def remote(target, script):
        calls.append(script)
        if "mktemp" in script:
            raise OSError("collision")
        return ""
    with pytest.raises(WorkflowError, match="staging failed"):
        stage_designated_jpeg("target", image, OCR_STAGING_ROOT, "/srv/autobench", remote_runner=remote, copy_runner=lambda *_: pytest.fail("copy called"))
    assert len(calls) == 2 and all("rm -f" not in call for call in calls)


def test_provisioning_creates_only_fixed_private_roots_without_leaking_paths() -> None:
    calls = []
    provision_ocr_runtime_roots("target", "/srv/autobench", remote_runner=lambda host, script: calls.append((host, script)))
    assert len(calls) == 1
    script = calls[0][1]
    for root in OCR_RUNTIME_ROOTS:
        assert f"mkdir -m 700 -- {root}" in script
        assert f"test ! -L {root}" in script
    assert "/srv/autobench" in script and "printf" not in script


def test_provisioning_refuses_checkout_overlap_before_remote_call(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr("run_remote.OCR_RUNTIME_ROOTS", ("/srv/autobench/unsafe",))
    with pytest.raises(WorkflowError, match="outside the checkout"):
        provision_ocr_runtime_roots("target", "/srv/autobench", remote_runner=lambda *args: calls.append(args))
    assert calls == []


def _ocr_roots() -> dict:
    return {"staging_root": OCR_STAGING_ROOT, "temporary_root": OCR_WORK_ROOT, "output_root": OCR_OUTPUT_ROOT}


def test_staged_ocr_handoff_requires_fixed_runtime_roots(tmp_path: Path) -> None:
    image = tmp_path / "input.jpg"; image.write_bytes(_jpeg())
    with pytest.raises(WorkflowError, match="not approved"):
        run_staged_ocr_smoke(
            "target", image, "/tmp/not-approved", "/srv/autobench",
            binary="/opt/bin/llama-mtmd-cli", model="/opt/models/model.gguf", projector="/opt/models/projector.gguf",
            temporary_root=OCR_WORK_ROOT, output_root=OCR_OUTPUT_ROOT,
            remote_runner=lambda *_: pytest.fail("remote called"),
        )


def test_staged_ocr_handoff_invokes_one_hidden_command_and_cleans(tmp_path: Path) -> None:
    image = tmp_path / "input.jpg"; image.write_bytes(_jpeg())
    calls, cleanups = [], []
    staged = f"{OCR_STAGING_ROOT}/.autobench-ocr-abc123.jpg"
    run_staged_ocr_smoke(
        "target", image, OCR_STAGING_ROOT, "/srv/autobench",
        binary="/opt/bin/llama-mtmd-cli", model="/opt/models/model.gguf", projector="/opt/models/projector.gguf",
        temporary_root=OCR_WORK_ROOT, output_root=OCR_OUTPUT_ROOT,
        stage_runner=lambda *_: staged,
        remote_runner=lambda host, script: calls.append((host, script)),
        cleanup_runner=lambda host, path, root: cleanups.append((host, path, root)),
    )
    assert len(calls) == 5 and cleanups == [("target", staged, OCR_STAGING_ROOT)]
    command = calls[-1][1]
    assert "run_ocr_target_smoke.py" in command and f"--image {staged}" in command
    assert f"--diagnostic-output {OCR_OUTPUT_ROOT}/diagnostic.json" in command
    assert f"--output {OCR_OUTPUT_ROOT}/receipt.json" in command


def test_staged_ocr_handoff_rejects_unowned_stage_return_before_launch(tmp_path: Path) -> None:
    image = tmp_path / "input.jpg"; image.write_bytes(_jpeg())
    calls = []
    with pytest.raises(WorkflowError, match="handoff is unsafe"):
        run_staged_ocr_smoke(
            "target", image, OCR_STAGING_ROOT, "/srv/autobench",
            binary="/opt/bin/llama-mtmd-cli", model="/opt/models/model.gguf", projector="/opt/models/projector.gguf",
            temporary_root=OCR_WORK_ROOT, output_root=OCR_OUTPUT_ROOT,
            stage_runner=lambda *_: "/tmp/other/input.jpg", remote_runner=lambda *args: calls.append(args),
        )
    assert not any("run_ocr_target_smoke.py" in call[1] for call in calls)


def test_staged_ocr_handoff_does_not_launch_after_stage_or_preflight_failure(tmp_path: Path) -> None:
    image = tmp_path / "input.jpg"; image.write_bytes(_jpeg())
    calls = []
    with pytest.raises(WorkflowError, match="stage failed"):
        run_staged_ocr_smoke(
            "target", image, OCR_STAGING_ROOT, "/srv/autobench",
            binary="/opt/bin/llama-mtmd-cli", model="/opt/models/model.gguf", projector="/opt/models/projector.gguf",
            temporary_root=OCR_WORK_ROOT, output_root=OCR_OUTPUT_ROOT,
            stage_runner=lambda *_: (_ for _ in ()).throw(WorkflowError("stage failed")), remote_runner=lambda *args: calls.append(args),
        )
    assert len(calls) == 4 and not any("run_ocr_target_smoke.py" in call[1] for call in calls)
    calls.clear()
    def fail_preflight(host, script):
        calls.append(script)
        raise WorkflowError("output collision")
    with pytest.raises(WorkflowError, match="output collision"):
        run_staged_ocr_smoke(
            "target", image, OCR_STAGING_ROOT, "/srv/autobench",
            binary="/opt/bin/llama-mtmd-cli", model="/opt/models/model.gguf", projector="/opt/models/projector.gguf",
            temporary_root=OCR_WORK_ROOT, output_root=OCR_OUTPUT_ROOT,
            stage_runner=lambda *_: pytest.fail("stage called"), remote_runner=fail_preflight,
        )
    assert len(calls) == 1 and "run_ocr_target_smoke.py" not in calls[0]


def test_staged_ocr_handoff_propagates_cleanup_failure(tmp_path: Path) -> None:
    image = tmp_path / "input.jpg"; image.write_bytes(_jpeg())
    staged = f"{OCR_STAGING_ROOT}/.autobench-ocr-abc123.jpg"
    with pytest.raises(WorkflowError, match="cleanup failed"):
        run_staged_ocr_smoke(
            "target", image, OCR_STAGING_ROOT, "/srv/autobench",
            binary="/opt/bin/llama-mtmd-cli", model="/opt/models/model.gguf", projector="/opt/models/projector.gguf",
            temporary_root=OCR_WORK_ROOT, output_root=OCR_OUTPUT_ROOT,
            stage_runner=lambda *_: staged, remote_runner=lambda *_: None,
            cleanup_runner=lambda *_: (_ for _ in ()).throw(WorkflowError("cleanup failed")),
        )


def test_staged_ocr_handoff_cleans_after_invocation_failure(tmp_path: Path) -> None:
    image = tmp_path / "input.jpg"; image.write_bytes(_jpeg())
    staged, calls, cleanups = f"{OCR_STAGING_ROOT}/.autobench-ocr-abc123.jpg", [], []
    def remote(host, script):
        calls.append(script)
        if "run_ocr_target_smoke.py" in script:
            raise WorkflowError("failed")
    with pytest.raises(WorkflowError, match="failed"):
        run_staged_ocr_smoke(
            "target", image, OCR_STAGING_ROOT, "/srv/autobench",
            binary="/opt/bin/llama-mtmd-cli", model="/opt/models/model.gguf", projector="/opt/models/projector.gguf",
            temporary_root=OCR_WORK_ROOT, output_root=OCR_OUTPUT_ROOT,
            stage_runner=lambda *_: staged, remote_runner=remote,
            cleanup_runner=lambda host, path, root: cleanups.append((host, path, root)),
        )
    assert sum("run_ocr_target_smoke.py" in call for call in calls) == 1
    assert cleanups == [("target", staged, OCR_STAGING_ROOT)]


def test_staged_ocr_handoff_requires_exact_cli_shape_and_order(tmp_path: Path) -> None:
    image = tmp_path / "input.jpg"; image.write_bytes(_jpeg())
    events, staged = [], f"{OCR_STAGING_ROOT}/.autobench-ocr-abc123.jpg"
    def remote(host, script):
        events.append(("launch" if "run_ocr_target_smoke.py" in script else "preflight", script))
    run_staged_ocr_smoke(
        "target", image, OCR_STAGING_ROOT, "/srv/autobench",
        binary="/opt/bin/llama-mtmd-cli", model="/opt/models/model.gguf", projector="/opt/models/projector.gguf",
        temporary_root=OCR_WORK_ROOT, output_root=OCR_OUTPUT_ROOT,
        stage_runner=lambda *_: events.append(("stage", "")) or staged,
        remote_runner=remote, cleanup_runner=lambda *_: events.append(("cleanup", "")),
    )
    command = [text for event, text in events if event == "launch"]
    assert len(command) == 1 and " -st " not in command[0]
    assert [event for event, _ in events] == ["preflight", "preflight", "preflight", "preflight", "stage", "launch", "cleanup"]


def test_staged_ocr_handoff_rejects_all_nonapproved_root_combinations(tmp_path: Path) -> None:
    image = tmp_path / "input.jpg"; image.write_bytes(_jpeg())
    common = dict(host="target", source=image, staging_root=OCR_STAGING_ROOT, remote_dir="/srv/autobench", binary="/opt/bin/llama-mtmd-cli", model="/opt/models/model.gguf", projector="/opt/models/projector.gguf")
    for temporary_root, output_root in ((f"{OCR_STAGING_ROOT}/work", OCR_OUTPUT_ROOT), (OCR_WORK_ROOT, "/tmp"), (OCR_WORK_ROOT, f"{OCR_WORK_ROOT}/results"), (OCR_STAGING_ROOT, OCR_OUTPUT_ROOT), (OCR_WORK_ROOT, OCR_WORK_ROOT)):
        with pytest.raises(WorkflowError, match="not approved"):
            run_staged_ocr_smoke(**common, temporary_root=temporary_root, output_root=output_root)


def test_staged_ocr_cli_requires_fixed_inputs_and_rejects_arbitrary_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["run_remote.py", "--run-staged-ocr-smoke", "--stage-image", "input.jpg", "--stage-root", OCR_STAGING_ROOT])
    with pytest.raises(SystemExit):
        parse_arguments()
    monkeypatch.setattr("sys.argv", ["run_remote.py", "--run-staged-ocr-smoke", "--stage-image", "input.jpg", "--stage-root", OCR_STAGING_ROOT, "--ocr-binary", "/opt/bin/llama-mtmd-cli", "--ocr-model", "/opt/models/model.gguf", "--ocr-projector", "/opt/models/projector.gguf", "--ocr-temporary-root", "/tmp/work", "--ocr-output-root", "/tmp/output", "--", "echo", "unsafe"])
    with pytest.raises(SystemExit):
        parse_arguments()


def test_staged_cleanup_refuses_unowned_path(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(WorkflowError, match="cleanup is unsafe"):
        _cleanup_staged_jpeg("target", "/tmp/other/file.jpg", "/tmp/staging")


def test_staging_cli_rejects_result_sync_or_remote_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["run_remote.py", "--deploy-only", "--stage-image", "input.jpg", "--stage-root", "/tmp/staging", "--sync-results"])
    with pytest.raises(SystemExit):
        parse_arguments()
    monkeypatch.setattr("sys.argv", ["run_remote.py", "--deploy-only", "--stage-image", "input.jpg", "--stage-root", "/tmp/staging", "--", "pytest"])
    with pytest.raises(SystemExit):
        parse_arguments()


def test_workload_commands_require_route_validation() -> None:
    assert command_requires_model_route(["authoritative_bench.py", "--suite"])
    assert command_requires_model_route(["inventory_bench.py", "--models", "model.gguf"])
    assert command_requires_model_route(["inventory_bench.py", "--dry-run"])
    assert command_requires_model_route(["authoritative_bench.py", "--plan-only"])
    assert not command_requires_model_route(["inventory_bench.py", "--status"])
    assert not command_requires_model_route(["inventory_bench.py", "--help"])
    assert not command_requires_model_route(["authoritative_bench.py", "-h"])
    assert not command_requires_model_route(["python", "-m", "pytest"])


def test_required_route_validation_accepts_verified_luna(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_remote.py",
            "--require-model-route",
            "--required-model-route",
            "litellm-edge/cl/gpt-5.6-luna",
            "--configured-model-route",
            "litellm-edge/cl/gpt-5.6-luna",
            "--resolved-provider",
            "litellm-edge",
            "--resolved-model",
            "cl/gpt-5.6-luna",
            "--identity-check",
            "verified",
            "--route-evidence-output",
            str(tmp_path / "route.json"),
            "--",
            "echo",
            "ok",
        ],
    )
    args = parse_arguments()

    evidence = validate_required_model_route(args)

    assert evidence["status"] == "MODEL_ROUTE_VALID"
    assert '"status":"MODEL_ROUTE_VALID"' in (tmp_path / "route.json").read_text()


def test_required_route_validation_rejects_silent_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_remote.py",
            "--require-model-route",
            "--required-model-route",
            "litellm-edge/cl/gpt-5.6-luna",
            "--configured-model-route",
            "litellm-edge/cl/gpt-5.6-luna",
            "--resolved-provider",
            "litellm-edge",
            "--resolved-model",
            "an/gemini-3.7-flash-high",
            "--identity-check",
            "verified",
            "--",
            "echo",
            "never",
        ],
    )
    args = parse_arguments()

    with pytest.raises(WorkflowError, match="MODEL_ROUTE_IDENTITY_MISMATCH"):
        validate_required_model_route(args)
