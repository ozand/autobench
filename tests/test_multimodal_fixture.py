from pathlib import Path
from unittest.mock import patch

import pytest

from src.multimodal_dry_run import build_one_job_dry_run
from src.multimodal_fixture import (
    EphemeralFixtureError,
    deterministic_png_bytes,
    ephemeral_png_fixture,
    validate_ephemeral_png_fixture,
)
from src.multimodal_preflight import ArtifactExpectation
from src.multimodal_runner import prepare_multimodal_invocation


def _configuration() -> dict:
    return {
        "device": "Vulkan0",
        "split_mode": "none",
        "split_ratio": None,
        "context_length": 1024,
        "cache_type_k": "f16",
        "cache_type_v": "f16",
        "max_tokens": 32,
    }


def _expectations() -> tuple[ArtifactExpectation, ArtifactExpectation]:
    pairing_id = "qwen2-vl-2b-instruct-q4km-q8proj"
    return (
        ArtifactExpectation(
            "Qwen2-VL-2B-Instruct-Q4_K_M.gguf", 986046944,
            "5745685d2e607a82a0696c1118e56a2a1ae0901da450fd9cd4f161c6b62867d7", pairing_id,
        ),
        ArtifactExpectation(
            "mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf", 709883360,
            "a0ad91f00a7a80dcf84d719a61b00ee2e07b71794f4ee2dfa81a254621a8c418", pairing_id,
        ),
    )


def _safe_artifact(expectation: ArtifactExpectation) -> dict:
    return {
        "basename": expectation.basename,
        "size_bytes": expectation.size_bytes,
        "sha256": expectation.sha256,
        "pairing_id": expectation.pairing_id,
    }


def test_fixture_is_deterministic_bounded_and_removed_after_success(tmp_path):
    expected_path = tmp_path / "autobench-ephemeral-fixture.png"
    first = deterministic_png_bytes(width=28, height=28)
    assert first == deterministic_png_bytes(width=28, height=28)
    assert not expected_path.exists()
    with ephemeral_png_fixture(tmp_path, width=28, height=28) as fixture:
        assert fixture.path == expected_path
        assert fixture.path.is_file()
        assert fixture.descriptor == {
            "format": "png", "width": 28, "height": 28,
            "byte_class": "small", "validation_status": "VALID",
        }
        assert fixture.path.read_bytes() == first
    assert not expected_path.exists()


def test_fixture_is_removed_when_consumer_validation_or_setup_fails(tmp_path):
    expected_path = tmp_path / "autobench-ephemeral-fixture.png"
    with pytest.raises(RuntimeError):
        with ephemeral_png_fixture(tmp_path) as fixture:
            assert fixture.path.exists()
            raise RuntimeError("test consumer failure")
    assert not expected_path.exists()
    with patch("src.multimodal_fixture.validate_ephemeral_png_fixture", side_effect=EphemeralFixtureError("validation failure")):
        with pytest.raises(EphemeralFixtureError):
            with ephemeral_png_fixture(tmp_path):
                pass
    assert not expected_path.exists()
    with patch("src.multimodal_fixture.os.fdopen", side_effect=OSError("synthetic setup failure")):
        with pytest.raises(OSError):
            with ephemeral_png_fixture(tmp_path):
                pass
    assert not expected_path.exists()
    with patch("src.multimodal_fixture.os.fstat", side_effect=OSError("synthetic stat failure")):
        with pytest.raises(OSError):
            with ephemeral_png_fixture(tmp_path):
                pass
    assert not expected_path.exists()


def test_fixture_rejects_invalid_dimensions_unsafe_root_and_ambiguous_destination(tmp_path):
    with pytest.raises(EphemeralFixtureError):
        deterministic_png_bytes(width=0, height=28)
    with pytest.raises(EphemeralFixtureError):
        ephemeral_png_fixture(tmp_path, width=2049).__enter__()
    destination = tmp_path / "autobench-ephemeral-fixture.png"
    destination.write_bytes(b"not-a-fixture")
    with pytest.raises(EphemeralFixtureError):
        with ephemeral_png_fixture(tmp_path):
            pass


def test_container_validation_rejects_malformed_outside_corrupt_and_metadata_mismatch(tmp_path, tmp_path_factory):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not-a-png")
    descriptor = {"format": "png", "width": 28, "height": 28, "size_bytes": bad.stat().st_size}
    with pytest.raises(EphemeralFixtureError):
        validate_ephemeral_png_fixture(bad, descriptor, temporary_root=tmp_path)
    outside_root = tmp_path_factory.mktemp("outside")
    outside = outside_root / "outside.png"
    payload = deterministic_png_bytes(width=28, height=28)
    outside.write_bytes(payload)
    descriptor["size_bytes"] = len(payload)
    with pytest.raises(EphemeralFixtureError):
        validate_ephemeral_png_fixture(outside, descriptor, temporary_root=tmp_path)
    good = tmp_path / "good.png"
    good.write_bytes(payload)
    with pytest.raises(EphemeralFixtureError):
        validate_ephemeral_png_fixture(
            good, {"format": "png", "width": 29, "height": 28, "size_bytes": len(payload)},
            temporary_root=tmp_path,
        )
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(payload[:-1])
    with pytest.raises(EphemeralFixtureError):
        validate_ephemeral_png_fixture(
            corrupt, {"format": "png", "width": 28, "height": 28, "size_bytes": corrupt.stat().st_size},
            temporary_root=tmp_path,
        )


def test_fixture_rejects_symlinked_parent_and_never_follows_existing_destination(tmp_path, tmp_path_factory):
    outside_root = tmp_path_factory.mktemp("outside-parent")
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(outside_root, target_is_directory=True)
    with pytest.raises(EphemeralFixtureError):
        with ephemeral_png_fixture(linked_parent):
            pass
    destination = tmp_path / "autobench-ephemeral-fixture.png"
    destination.symlink_to(outside_root / "outside.png")
    with pytest.raises(EphemeralFixtureError):
        with ephemeral_png_fixture(tmp_path):
            pass


def test_ephemeral_fixture_feeds_prepared_and_dry_run_without_receipt_reference(tmp_path):
    model_expected, projector_expected = _expectations()
    with ephemeral_png_fixture(tmp_path) as fixture:
        with patch(
            "src.multimodal_runner.validate_artifact",
            side_effect=[_safe_artifact(model_expected), _safe_artifact(projector_expected)],
        ):
            prepared = prepare_multimodal_invocation(
                model_path=Path("C:/not-loaded-model.gguf"),
                projector_path=Path("C:/not-loaded-projector.gguf"),
                model_expected=model_expected,
                projector_expected=projector_expected,
                image_reference=fixture.path,
                image_descriptor={
                    "format": "png", "width": 28, "height": 28,
                    "size_bytes": fixture.path.stat().st_size,
                },
                configuration=_configuration(),
            )
        summary = build_one_job_dry_run(prepared)
        assert summary["dry_run"] is True
        assert summary["inference_invoked"] is False
        assert str(fixture.path) not in str(summary)
        assert fixture.path.name not in str(summary)
    assert not (tmp_path / "autobench-ephemeral-fixture.png").exists()


def test_fixture_module_has_no_model_process_or_remote_execution_path():
    source = Path("src/multimodal_fixture.py").read_text(encoding="utf-8")
    for forbidden in ("import subprocess", "Popen(", "subprocess.", "requests", "run_host_command", "llama-"):
        assert forbidden not in source
