"""Regression tests for local llama.cpp execution result classification."""

import os
import subprocess
import sys
from unittest.mock import patch

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from runner import Runner


def completed(return_code: int, stdout: str = "", stderr: str = ""):
    """Return a subprocess result suitable for mocking the SSH invocation."""
    return subprocess.CompletedProcess(
        args=["ssh"],
        returncode=return_code,
        stdout=stdout,
        stderr=stderr,
    )


def run_with_result(result):
    """Execute the runner with a mocked subprocess result."""
    with patch("runner.subprocess.run", return_value=result):
        return Runner.run_local_vulkan("test prompt", timeout=1)


def test_count_local_tokens_uses_remote_model_tokenizer():
    tokenizer_output = "Total number of tokens: 42\n"
    with patch(
        "runner.subprocess.run",
        return_value=completed(0, stdout=tokenizer_output),
    ) as mocked_run:
        count = Runner.count_local_tokens("/models/test.gguf", "hello world")

    assert count == 42
    assert mocked_run.call_args.kwargs["input"] == "hello world"
    assert "llama-tokenize" in mocked_run.call_args.args[0][2]


def test_count_local_tokens_rejects_missing_count():
    with patch("runner.subprocess.run", return_value=completed(0, stdout="tokens")):
        try:
            Runner.count_local_tokens("/models/test.gguf", "hello world")
        except RuntimeError as exc:
            assert "did not return a token count" in str(exc)
        else:
            raise AssertionError("missing token count must fail")


def test_remote_timeout_is_failure():
    result = run_with_result(completed(124, stdout="partial output"))

    assert result["success"] is False
    assert result["status"] == "REMOTE_TIMEOUT"
    assert result["return_code"] == 124
    assert result["stdout"] == "partial output"
    assert result["command_args"][0] == "timeout"


def test_local_ssh_timeout_is_failure():
    with patch(
        "runner.subprocess.run",
        side_effect=subprocess.TimeoutExpired(
            cmd=["ssh"],
            timeout=11,
            output="partial stdout",
            stderr="partial stderr",
        ),
    ):
        result = Runner.run_local_vulkan("test prompt", timeout=1)

    assert result["success"] is False
    assert result["status"] == "SSH_TIMEOUT"
    assert result["return_code"] is None
    assert result["elapsed_seconds"] >= 0


def test_out_of_memory_is_classified():
    result = run_with_result(
        completed(1, stderr="vk::Device::allocateMemory: ErrorOutOfDeviceMemory")
    )

    assert result["status"] == "OOM"


def test_context_overflow_is_classified():
    result = run_with_result(
        completed(1, stdout="request exceeds the available context size")
    )

    assert result["status"] == "CONTEXT_OVERFLOW"


def test_unsupported_split_backend_is_classified():
    result = run_with_result(
        completed(1, stderr="device Vulkan0 does not support split buffers")
    )

    assert result["status"] == "UNSUPPORTED_BACKEND"


def test_model_load_failure_is_classified():
    result = run_with_result(completed(1, stderr="failed to load model"))

    assert result["status"] == "MODEL_LOAD_ERROR"


def test_ssh_failure_is_classified():
    result = run_with_result(completed(255, stderr="Connection refused"))

    assert result["status"] == "SSH_ERROR"


def test_truncated_prompt_echo_is_excluded_from_response():
    stdout = """\
> long prompt start
SECRET = 'needle'
Q ... (truncated)
actual answer
[ Prompt: 12.5 t/s | Generation: 22.0 t/s ]
Exiting...
"""
    result = run_with_result(completed(0, stdout=stdout))

    assert result["response"] == "actual answer"
    assert "needle" not in result["response"]


def test_success_includes_stable_status_and_diagnostics():
    stdout = """\
> test prompt
answer
[ Prompt: 12.5 t/s | Generation: 22.0 t/s ]
Exiting...
"""
    result = run_with_result(completed(0, stdout=stdout))

    assert result["success"] is True
    assert result["status"] == "SUCCESS"
    assert result["return_code"] == 0
    assert result["response"] == "answer"
    assert result["prompt_speed_ts"] == 12.5
    assert result["generation_speed_ts"] == 22.0
    assert result["command_args"][0] == "timeout"
