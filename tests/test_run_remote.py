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
    normalize_remote_command,
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
