"""Stable benchmark status helpers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


_SENSITIVE_ARTIFACT_KEYS = {
    "prompt",
    "response",
    "generated_text",
    "stdout",
    "stderr",
    "raw_output",
    "task",
    "tasks",
    "issues",
    "llm_judge_reason",
}


def _sanitize_command_args(arguments: Any) -> list[str] | None:
    """Keep safe executable flags while removing prompt and private path values."""
    if not isinstance(arguments, list):
        return None
    sanitized: list[str] = []
    skip_value = False
    previous_flag: str | None = None
    for argument in arguments:
        value = str(argument)
        if skip_value:
            skip_value = False
            if previous_flag == "-m":
                sanitized.append(_basename(value) or "[REDACTED_MODEL]")
            continue
        if value in {"-p", "--prompt"}:
            skip_value = True
            previous_flag = value
            continue
        if value == "-m":
            sanitized.append(value)
            previous_flag = value
            skip_value = True
            continue
        sanitized.append(_basename(value) if value.startswith("/home/") else value)
        previous_flag = None
    return sanitized


def sanitize_artifact(value: Any) -> Any:
    """Return a JSON-safe artifact without raw prompts or runtime payloads."""
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if key in _SENSITIVE_ARTIFACT_KEYS:
                continue
            if key == "command_args":
                sanitized[key] = _sanitize_command_args(item)
            elif key == "error":
                sanitized[key] = item if _safe_error(item) else "details redacted"
            elif key in {"path", "model_path"}:
                sanitized[f"{key}_basename"] = _basename(item)
            elif key in {"dataset_dir", "output_dir"}:
                sanitized[key] = _basename(item)
            elif key == "host":
                sanitized[key] = "k7000"
            else:
                sanitized[key] = sanitize_artifact(item)
        return sanitized
    if isinstance(value, list):
        return [sanitize_artifact(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_artifact(item) for item in value]
    return deepcopy(value)


def _safe_error(value: Any) -> bool:
    """Keep short non-payload errors for operational diagnosis."""
    text = str(value)
    return len(text) <= 160 and not any(
        marker in text.lower()
        for marker in ("prompt", "response", "stdout", "stderr", "secret")
    )


def _basename(value: Any) -> str | None:
    """Return only a portable basename for persisted local or remote paths."""
    if value is None:
        return None
    text = str(value).replace("\\", "/").rstrip("/")
    return text.rsplit("/", 1)[-1]


PREFLIGHT_CAUSE_STATUSES = {
    "OOM": "PREFLIGHT_OOM",
    "REMOTE_TIMEOUT": "PREFLIGHT_TIMEOUT",
    "SSH_TIMEOUT": "PREFLIGHT_SSH_TIMEOUT",
    "UNSUPPORTED_BACKEND": "PREFLIGHT_UNSUPPORTED_BACKEND",
    "EXECUTION_ERROR": "PREFLIGHT_EXECUTION_ERROR",
    "MODEL_LOAD_ERROR": "PREFLIGHT_MODEL_LOAD_ERROR",
    "SSH_ERROR": "PREFLIGHT_SSH_ERROR",
    "CONTEXT_OVERFLOW": "PREFLIGHT_EXECUTION_ERROR",
}


def preflight_cause(probe: dict) -> str:
    """Map a load-probe result to a stable preflight cause status."""
    raw_status = str(probe.get("status", "EXECUTION_ERROR"))
    if raw_status in PREFLIGHT_CAUSE_STATUSES:
        return PREFLIGHT_CAUSE_STATUSES[raw_status]
    if raw_status.startswith("PREFLIGHT_"):
        return raw_status
    return "PREFLIGHT_EXECUTION_ERROR"


BOUNDARY_CAUSE_STATUSES = {
    "OOM": "BOUNDARY_OOM",
    "CONTEXT_OVERFLOW": "BOUNDARY_CONTEXT_OVERFLOW",
    "REMOTE_TIMEOUT": "BOUNDARY_TIMEOUT",
    "SSH_TIMEOUT": "BOUNDARY_SSH_TIMEOUT",
    "TOKENIZER_ERROR": "BOUNDARY_TOKENIZER_ERROR",
    "UNSUPPORTED_BACKEND": "BOUNDARY_UNSUPPORTED_BACKEND",
    "EXECUTION_ERROR": "BOUNDARY_EXECUTION_ERROR",
}


def boundary_cause(result: dict) -> str:
    """Map a boundary result to a stable cause without losing its raw status."""
    raw_status = str(result.get("status", "EXECUTION_ERROR"))
    if raw_status in BOUNDARY_CAUSE_STATUSES:
        return BOUNDARY_CAUSE_STATUSES[raw_status]
    if raw_status.startswith("BOUNDARY_"):
        return raw_status
    return "BOUNDARY_EXECUTION_ERROR"


def boundary_summary(result: dict) -> dict:
    """Return compact diagnostic metadata for suite-level blocked results."""
    probes = result.get("probes") or []
    failing_probe = next(
        (probe for probe in probes if probe.get("status") != "SUCCESS"), None
    )
    source = failing_probe or result
    return {
        "cause_status": boundary_cause(source),
        "source_status": source.get("status", "EXECUTION_ERROR"),
        "context_size": source.get("context_size"),
        "error": source.get("error"),
        "return_code": source.get("return_code"),
        "inconclusive_status": result.get("inconclusive_status"),
    }
