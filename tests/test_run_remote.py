"""Tests for the safe local-to-remote workflow helper."""

from pathlib import Path
import sys

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from run_remote import (  # noqa: E402
    WorkflowError,
    ensure_clean_local_repository,
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
