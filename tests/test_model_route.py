from src.model_route import (
    CANONICAL_LUNA_ROUTE,
    route_is_valid,
    split_route,
    validate_model_route,
)


def test_canonical_luna_route_is_verified_only_with_matching_identity():
    evidence = validate_model_route(
        CANONICAL_LUNA_ROUTE,
        resolved_provider="litellm-edge",
        resolved_model="cl/gpt-5.6-luna",
        identity_check="verified",
    )

    assert evidence["status"] == "MODEL_ROUTE_VALID"
    assert route_is_valid(evidence)
    assert evidence["configured_route"] == CANONICAL_LUNA_ROUTE


def test_direct_openai_route_is_rejected_for_luna():
    evidence = validate_model_route(
        "openai/gpt-5.6-luna",
        required_route=CANONICAL_LUNA_ROUTE,
        resolved_provider="openai",
        resolved_model="gpt-5.6-luna",
        identity_check="verified",
    )

    assert evidence["status"] == "MODEL_ROUTE_INVALID"
    assert not route_is_valid(evidence)


def test_provider_rejection_cannot_fallback_to_another_model():
    evidence = validate_model_route(
        CANONICAL_LUNA_ROUTE,
        resolved_provider="litellm-edge",
        resolved_model="cl/gpt-5.6-luna",
        identity_check="auth_failed",
    )

    assert evidence["status"] == "MODEL_ROUTE_AUTH_FAILED"
    assert not route_is_valid(evidence)


def test_missing_or_ambiguous_identity_is_blocked():
    for kwargs in (
        {},
        {"resolved_provider": "litellm-edge", "resolved_model": "cl/gpt-5.6-luna"},
        {"resolved_provider": "litellm-edge", "resolved_model": "an/gemini-3.7-flash-high"},
    ):
        evidence = validate_model_route(CANONICAL_LUNA_ROUTE, **kwargs)
        assert evidence["status"] == "MODEL_ROUTE_IDENTITY_UNVERIFIED"
        assert not route_is_valid(evidence)


def test_route_parser_rejects_payload_like_values():
    try:
        split_route("litellm-edge/gpt-5.6-luna?token=secret")
    except ValueError:
        pass
    else:
        raise AssertionError("route payload must be rejected")
