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
    script = _remote_staging_root_check("/tmp/staging", "/srv/autobench")
    assert "realpath -e" in script and "stat -c %a" in script and "mkdir" not in script
    assert "test ! -e" not in script


def test_staging_uses_injected_transport_and_cleans_target_after_failure(tmp_path: Path) -> None:
    image = tmp_path / "input.jpg"; image.write_bytes(_jpeg())
    remote_calls, copy_calls = [], []
    def remote(target, script):
        remote_calls.append((target, script))
        return "/tmp/staging/.autobench-ocr-abc123.jpg" if "mktemp" in script else ""
    def fail_copy(command):
        copy_calls.append(command)
        raise OSError("transport")
    with pytest.raises(WorkflowError, match="staging failed"):
        stage_designated_jpeg("target", image, "/tmp/staging", "/srv/autobench", remote_runner=remote, copy_runner=fail_copy)
    assert len(copy_calls) == 1 and len(remote_calls) == 4
    assert "input.jpg" not in str(remote_calls)
    assert remote_calls[0][1] == _remote_staging_root_check("/tmp/staging", "/srv/autobench")
    assert "mktemp" in remote_calls[1][1]
    assert "realpath -e" in remote_calls[2][1]
    assert "rm -f" in remote_calls[-1][1]


def test_staging_returns_private_created_location_without_printing_or_persisting_it(tmp_path: Path) -> None:
    image = tmp_path / "input.jpg"; image.write_bytes(_jpeg())
    calls = []
    def remote(target, script):
        calls.append(script)
        return "/tmp/staging/.autobench-ocr-abc123.jpg" if "mktemp" in script else ""
    location = stage_designated_jpeg("target", image, "/tmp/staging", "/srv/autobench", remote_runner=remote, copy_runner=lambda *_: None)
    assert location == "/tmp/staging/.autobench-ocr-abc123.jpg"
    assert "chmod 600" in calls[-1]


def test_staging_rejects_malformed_private_destination_before_copy(tmp_path: Path) -> None:
    image = tmp_path / "input.jpg"; image.write_bytes(_jpeg())
    def remote(target, script):
        return "/tmp/staging/.autobench-ocr-x/../../checkout/file" if "mktemp" in script else ""
    with pytest.raises(WorkflowError, match="staging failed"):
        stage_designated_jpeg("target", image, "/tmp/staging", "/srv/autobench", remote_runner=remote, copy_runner=lambda *_: pytest.fail("copy called"))


def test_staging_rejects_collision_before_copy_and_does_not_cleanup_unowned_destination(tmp_path: Path) -> None:
    image = tmp_path / "input.jpg"; image.write_bytes(_jpeg())
    calls = []
    def remote(target, script):
        calls.append(script)
        if "mktemp" in script:
            raise OSError("collision")
        return ""
    with pytest.raises(WorkflowError, match="staging failed"):
        stage_designated_jpeg("target", image, "/tmp/staging", "/srv/autobench", remote_runner=remote, copy_runner=lambda *_: pytest.fail("copy called"))
    assert len(calls) == 2 and all("rm -f" not in call for call in calls)


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
