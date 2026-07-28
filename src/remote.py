"""Shared SSH configuration for AutoBench host operations."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence

DEFAULT_SSH_TARGET = "opencode@100.67.171.58"
SSH_TARGET = os.environ.get("AUTOBENCH_SSH_TARGET", DEFAULT_SSH_TARGET)
SSH_OPTIONS = [
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=10",
    "-o",
    "StrictHostKeyChecking=accept-new",
]


def running_on_target_host() -> bool:
    """Return whether benchmark commands should execute directly on this host."""
    value = os.environ.get("AUTOBENCH_EXECUTION_MODE", "").strip().lower()
    return value == "local"


def host_command(command: str) -> list[str]:
    """Return a direct shell command locally or an SSH command remotely."""
    if running_on_target_host():
        return ["bash", "-lc", command]
    return ["ssh", *SSH_OPTIONS, SSH_TARGET, command]


def run_host_command(
    command: str,
    *,
    timeout: int | None = None,
    input_text: str | None = None,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a command on the execution host using the configured transport."""
    return subprocess.run(
        host_command(command),
        input=input_text,
        capture_output=capture_output,
        text=True,
        timeout=timeout,
    )
