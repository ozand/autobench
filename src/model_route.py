"""Sanitized provider-route validation for supervised benchmark workloads."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

CANONICAL_LUNA_ROUTE = "litellm-edge/cl/gpt-5.6-luna"
VALIDATION_SCHEMA_VERSION = 1

_ROUTE_PARTS = {
    "provider": "provider",
    "model": "model",
}
_SAFE_ROUTE_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-/")


def _safe_route(value: Any, field: str) -> str:
    """Return a route component or reject values that could contain payloads."""
    if not isinstance(value, str) or not value or any(char not in _SAFE_ROUTE_CHARS for char in value):
        raise ValueError(f"invalid {field}")
    return value


def split_route(route: Any) -> tuple[str, str]:
    """Split a provider-qualified route without exposing credentials or payloads."""
    value = _safe_route(route, "route")
    provider, separator, model = value.partition("/")
    if not separator or not provider or not model or "/" in provider:
        raise ValueError("route must be provider/model")
    return provider, model


def _base_evidence(required_route: str, configured_route: Any) -> dict[str, Any]:
    """Build evidence containing only stable route identity fields."""
    required_provider, required_model = split_route(required_route)
    configured_provider = configured_model = None
    try:
        configured_provider, configured_model = split_route(configured_route)
    except ValueError:
        pass
    safe_configured_route = None
    if configured_provider is not None and configured_model is not None:
        safe_configured_route = f"{configured_provider}/{configured_model}"
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "required_route": required_route,
        "required_provider": required_provider,
        "required_model": required_model,
        "configured_route": safe_configured_route,
        "configured_provider": configured_provider,
        "configured_model": configured_model,
        "resolved_provider": None,
        "resolved_model": None,
        "identity_check": "unverified",
        "status": "MODEL_ROUTE_INVALID",
        "validation_timestamp": datetime.now(timezone.utc).isoformat(),
    }


def validate_model_route(
    configured_route: Any,
    *,
    required_route: str = CANONICAL_LUNA_ROUTE,
    resolved_provider: Any = None,
    resolved_model: Any = None,
    identity_check: str = "unverified",
) -> dict[str, Any]:
    """Validate configured and observed provider/model identity.

    The caller supplies only sanitized identity metadata from its provider
    resolver/completion layer. This function never accepts or persists tokens,
    headers, raw provider payloads, prompts, or responses.
    """
    required_provider, required_model = split_route(required_route)
    evidence = _base_evidence(required_route, configured_route)
    evidence["required_model"] = required_model
    evidence["identity_check"] = identity_check

    try:
        configured_provider, configured_model = split_route(configured_route)
    except ValueError:
        return evidence

    if configured_provider != required_provider or configured_model != required_model:
        evidence["status"] = "MODEL_ROUTE_INVALID"
        return evidence

    if identity_check in {"auth_failed", "rejected"}:
        evidence["status"] = "MODEL_ROUTE_AUTH_FAILED"
        return evidence
    if identity_check != "verified":
        evidence["status"] = "MODEL_ROUTE_IDENTITY_UNVERIFIED"
        return evidence

    try:
        observed_provider = _safe_route(resolved_provider, "resolved_provider")
        observed_model = _safe_route(resolved_model, "resolved_model")
    except ValueError:
        evidence["status"] = "MODEL_ROUTE_IDENTITY_UNVERIFIED"
        return evidence

    evidence["resolved_provider"] = observed_provider
    evidence["resolved_model"] = observed_model
    if observed_provider != required_provider or observed_model != required_model:
        evidence["status"] = "MODEL_ROUTE_IDENTITY_MISMATCH"
        return evidence

    evidence["status"] = "MODEL_ROUTE_VALID"
    return evidence


def validate_route_evidence(evidence: dict[str, Any], *, required_route: str) -> dict[str, Any]:
    """Revalidate a previously generated sanitized evidence document."""
    if not isinstance(evidence, dict):
        raise ValueError("route evidence must be an object")
    return validate_model_route(
        evidence.get("configured_route"),
        required_route=required_route,
        resolved_provider=evidence.get("resolved_provider"),
        resolved_model=evidence.get("resolved_model"),
        identity_check=evidence.get("identity_check", "unverified"),
    )


def route_is_valid(evidence: dict[str, Any]) -> bool:
    """Return whether evidence authorizes a required model workload."""
    return evidence.get("status") == "MODEL_ROUTE_VALID"
