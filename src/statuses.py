"""Stable benchmark status helpers."""

from __future__ import annotations


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
