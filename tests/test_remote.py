"""Tests for local-versus-SSH execution host selection."""

from src.remote import DEFAULT_SSH_TARGET, host_command


def test_host_command_uses_local_shell_on_execution_host(monkeypatch) -> None:
    monkeypatch.setenv("AUTOBENCH_EXECUTION_MODE", "local")

    assert host_command("printf ok") == ["bash", "-lc", "printf ok"]


def test_host_command_uses_configured_ssh_target(monkeypatch) -> None:
    monkeypatch.delenv("AUTOBENCH_EXECUTION_MODE", raising=False)
    monkeypatch.setattr("src.remote.SSH_TARGET", DEFAULT_SSH_TARGET)

    command = host_command("printf ok")

    assert command[0] == "ssh"
    assert DEFAULT_SSH_TARGET in command
    assert command[-1] == "printf ok"
