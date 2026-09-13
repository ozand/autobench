import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from src.multimodal_preflight import ArtifactExpectation, MultimodalPreflightError, validate_image_descriptor
from src.multimodal_receipt import validate_multimodal_receipt
from src.multimodal_runner import (
    ImageReferenceRejected,
    MultimodalContractError,
    build_pre_invocation_rejection_receipt,
    build_prepared_invocation_receipt,
    prepare_multimodal_invocation,
    terminal_class_for_event,
    validate_configuration,
    validate_local_image_reference,
)


def _artifact(tmp_path: Path, name: str, contents: bytes) -> tuple[Path, ArtifactExpectation]:
    path = tmp_path / name
    path.write_bytes(contents)
    return path, ArtifactExpectation(
        basename=name,
        size_bytes=len(contents),
        sha256=hashlib.sha256(contents).hexdigest(),
        pairing_id="qwen2-vl-2b-instruct-q4km-q8proj",
    )


def _image(tmp_path: Path, name: str = "fixture.png", contents: bytes = b"synthetic") -> tuple[Path, dict]:
    path = tmp_path / name
    path.write_bytes(contents)
    return path, {"format": "png", "width": 28, "height": 28, "size_bytes": len(contents)}


def _configuration(**overrides) -> dict:
    value = {
        "device": "Vulkan0",
        "split_mode": "none",
        "split_ratio": None,
        "context_length": 1024,
        "cache_type_k": "f16",
        "cache_type_v": "f16",
        "max_tokens": 32,
    }
    value.update(overrides)
    return value


def _prepared(tmp_path: Path):
    model, expected_model = _artifact(tmp_path, "Qwen2-VL-2B-Instruct-Q4_K_M.gguf", b"model")
    projector, expected_projector = _artifact(tmp_path, "mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf", b"projector")
    image, descriptor = _image(tmp_path)
    return prepare_multimodal_invocation(
        model_path=model,
        projector_path=projector,
        model_expected=expected_model,
        projector_expected=expected_projector,
        image_reference=image,
        image_descriptor=descriptor,
        configuration=_configuration(),
    )


def test_prepare_validates_synthetic_local_reference_without_opening_image_or_invoking(tmp_path):
    model, expected_model = _artifact(tmp_path, "Qwen2-VL-2B-Instruct-Q4_K_M.gguf", b"model")
    projector, expected_projector = _artifact(tmp_path, "mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf", b"projector")
    image, descriptor = _image(tmp_path)
    model_artifact = {"basename": expected_model.basename, "size_bytes": expected_model.size_bytes, "sha256": expected_model.sha256, "pairing_id": expected_model.pairing_id}
    projector_artifact = {"basename": expected_projector.basename, "size_bytes": expected_projector.size_bytes, "sha256": expected_projector.sha256, "pairing_id": expected_projector.pairing_id}
    with patch("src.multimodal_runner.validate_artifact", side_effect=[model_artifact, projector_artifact]), patch("src.multimodal_runner.Path.open") as open_file, patch("subprocess.run") as run:
        prepared = prepare_multimodal_invocation(
            model_path=model, projector_path=projector,
            model_expected=expected_model, projector_expected=expected_projector,
            image_reference=image, image_descriptor=descriptor,
            configuration=_configuration(),
        )
    open_file.assert_not_called()
    run.assert_not_called()
    assert prepared.image_reference.name == "fixture.png"
    assert prepared.image_descriptor["validation_status"] == "VALID"
    receipt = build_prepared_invocation_receipt(prepared)
    assert receipt["inference_invoked"] is False
    assert "fixture.png" not in str(receipt)
    assert validate_multimodal_receipt(receipt)["status"] == "MULTIMODAL_RECEIPT_VALID"


@pytest.mark.parametrize(
    "reference, descriptor",
    [
        ("https://example.invalid/image.png", {"format": "png", "width": 28, "height": 28, "size_bytes": 1}),
        ("data:image/png;base64,abcd", {"format": "png", "width": 28, "height": 28, "size_bytes": 1}),
        ("relative.png", {"format": "png", "width": 28, "height": 28, "size_bytes": 1}),
        ("//server/share/image.png", {"format": "png", "width": 28, "height": 28, "size_bytes": 1}),
    ],
)
def test_local_reference_rejects_nonlocal_unsafe_or_missing_reference(tmp_path, reference, descriptor):
    with pytest.raises(ImageReferenceRejected):
        validate_local_image_reference(reference, descriptor)


def test_local_reference_rejects_missing_unsupported_symlink_and_metadata_mismatch(tmp_path):
    missing = tmp_path / "missing.png"
    with pytest.raises(ImageReferenceRejected):
        validate_local_image_reference(missing, {"format": "png", "width": 28, "height": 28, "size_bytes": 1})
    unsupported, descriptor = _image(tmp_path, "fixture.gif")
    with pytest.raises(ImageReferenceRejected):
        validate_local_image_reference(unsupported, descriptor)
    image, descriptor = _image(tmp_path, "actual.png")
    symlink = tmp_path / "link.png"
    symlink.symlink_to(image)
    with pytest.raises(ImageReferenceRejected):
        validate_local_image_reference(symlink, descriptor)
    with pytest.raises(ImageReferenceRejected):
        validate_local_image_reference(image, {**descriptor, "format": "jpg"})
    with pytest.raises(ImageReferenceRejected):
        validate_local_image_reference(image, {**descriptor, "size_bytes": descriptor["size_bytes"] + 1})


@pytest.mark.parametrize(
    "configuration",
    [
        {"device": "Vulkan0"},
        _configuration(split_mode="tensor", split_ratio="1,1"),
        _configuration(split_mode="layer", split_ratio="2,1"),
        _configuration(context_length=4096),
        _configuration(cache_type_k="f16", cache_type_v="q8_0"),
        _configuration(max_tokens=129),
        _configuration(max_tokens=True),
        _configuration(device=[]),
        _configuration(cache_type_k=[]),
    ],
)
def test_configuration_is_bounded_and_fail_closed(configuration):
    with pytest.raises(MultimodalContractError):
        validate_configuration(configuration)


def test_projector_pairing_mismatch_never_reaches_image_reference_validation(tmp_path):
    model, expected_model = _artifact(tmp_path, "Qwen2-VL-2B-Instruct-Q4_K_M.gguf", b"model")
    projector, expected_projector = _artifact(tmp_path, "mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf", b"projector")
    image, descriptor = _image(tmp_path)
    expected_projector = ArtifactExpectation(
        expected_projector.basename, expected_projector.size_bytes,
        expected_projector.sha256, "wrong-pair",
    )
    with patch("src.multimodal_runner.validate_local_image_reference") as validate_image:
        with pytest.raises(MultimodalContractError, match="PROJECTOR_MISMATCH"):
            prepare_multimodal_invocation(
                model_path=model, projector_path=projector,
                model_expected=expected_model, projector_expected=expected_projector,
                image_reference=image, image_descriptor=descriptor,
                configuration=_configuration(),
            )
    validate_image.assert_not_called()


def test_rejection_receipt_has_no_image_reference_or_execution_claim(tmp_path):
    prepared = _prepared(tmp_path)
    receipt = build_pre_invocation_rejection_receipt(
        model_artifact=prepared.model_artifact,
        projector_artifact=prepared.projector_artifact,
        terminal_class="IMAGE_INPUT_REJECTED",
    )
    assert receipt["inference_invoked"] is False
    assert receipt["image_descriptor"] == {"validation_status": "REJECTED"}
    assert "fixture.png" not in str(receipt)
    assert validate_multimodal_receipt(receipt)["status"] == "MULTIMODAL_RECEIPT_VALID"


def test_projector_mismatch_receipt_preserves_distinct_pairing_evidence(tmp_path):
    prepared = _prepared(tmp_path)
    mismatched_projector = {**prepared.projector_artifact, "pairing_id": "other-pair"}
    receipt = build_pre_invocation_rejection_receipt(
        model_artifact=prepared.model_artifact,
        projector_artifact=mismatched_projector,
        terminal_class="PROJECTOR_MISMATCH",
    )
    assert receipt["model_artifact"]["pairing_id"] != receipt["projector_artifact"]["pairing_id"]
    assert validate_multimodal_receipt(receipt)["status"] == "MULTIMODAL_RECEIPT_VALID"
    equal_pairing = build_pre_invocation_rejection_receipt(
        model_artifact=prepared.model_artifact,
        projector_artifact=prepared.projector_artifact,
        terminal_class="IMAGE_INPUT_REJECTED",
    )
    equal_pairing["terminal_class"] = "PROJECTOR_MISMATCH"
    assert validate_multimodal_receipt(equal_pairing)["status"] == "MULTIMODAL_RECEIPT_INVALID"


@pytest.mark.parametrize(
    "event, expected",
    [
        ("image_rejected", "IMAGE_INPUT_REJECTED"),
        ("projector_mismatch", "PROJECTOR_MISMATCH"),
        ("unsupported_backend", "UNSUPPORTED_BACKEND"),
        ("oom", "OOM"),
        ("context_overflow", "CONTEXT_OVERFLOW"),
        ("metric_parse_failed", "METRIC_PARSE_FAILED"),
        ("ocr_missed", "OCR_MISSED"),
        ("ocr_inconclusive", "OCR_INCONCLUSIVE"),
        ("success", "SUCCESS"),
    ],
)
def test_all_adr003_terminal_classes_have_deterministic_mapping(event, expected):
    assert terminal_class_for_event(event) == expected


def test_unknown_event_and_unsafe_execution_claim_are_rejected(tmp_path):
    with pytest.raises(MultimodalContractError):
        terminal_class_for_event("unexpected")
    with pytest.raises(MultimodalContractError):
        terminal_class_for_event([])
    prepared = _prepared(tmp_path)
    unsafe_execution_receipt = {
        "schema_version": 1,
        "receipt_type": "MULTIMODAL_OCR_EXECUTION",
        "model_artifact": prepared.model_artifact,
        "projector_artifact": prepared.projector_artifact,
        "image_descriptor": prepared.image_descriptor,
        "configuration": prepared.configuration,
        "command_family": "multimodal_ocr_runner",
        "terminal_class": "SUCCESS",
        "invocation_attempted": False,
        "inference_invoked": False,
    }
    assert validate_multimodal_receipt(unsafe_execution_receipt)["status"] == "MULTIMODAL_RECEIPT_INVALID"


def test_receipt_rejects_prohibited_content_even_with_valid_contract(tmp_path):
    receipt = build_prepared_invocation_receipt(_prepared(tmp_path))
    receipt["prompt"] = "forbidden"
    assert validate_multimodal_receipt(receipt)["status"] == "MULTIMODAL_RECEIPT_INVALID"


def test_receipt_validator_fail_closes_malformed_types_extra_artifact_fields_and_invalid_config(tmp_path):
    prepared = _prepared(tmp_path)
    receipt = build_prepared_invocation_receipt(prepared)
    receipt["configuration"]["context_length"] = 999999
    assert validate_multimodal_receipt(receipt)["status"] == "MULTIMODAL_RECEIPT_INVALID"
    receipt = build_prepared_invocation_receipt(prepared)
    receipt["model_artifact"]["extra"] = "forbidden"
    assert validate_multimodal_receipt(receipt)["status"] == "MULTIMODAL_RECEIPT_INVALID"
    receipt = build_prepared_invocation_receipt(prepared)
    receipt["model_artifact"]["pairing_id"] = "/etc/passwd"
    receipt["projector_artifact"]["pairing_id"] = "/etc/passwd"
    assert validate_multimodal_receipt(receipt)["status"] == "MULTIMODAL_RECEIPT_INVALID"
    malformed = {"receipt_type": []}
    assert validate_multimodal_receipt(malformed)["status"] == "MULTIMODAL_RECEIPT_INVALID"
    malformed = {"receipt_type": "MULTIMODAL_OCR_EXECUTION", "terminal_class": []}
    assert validate_multimodal_receipt(malformed)["status"] == "MULTIMODAL_RECEIPT_INVALID"
    receipt = build_prepared_invocation_receipt(prepared)
    receipt["schema_version"] = True
    assert validate_multimodal_receipt(receipt)["status"] == "MULTIMODAL_RECEIPT_INVALID"


@pytest.mark.parametrize("field, value", [
    ("format", []),
    ("byte_class", []),
])
def test_receipt_validator_rejects_unhashable_descriptor_values(tmp_path, field, value):
    receipt = build_prepared_invocation_receipt(_prepared(tmp_path))
    receipt["image_descriptor"][field] = value
    assert validate_multimodal_receipt(receipt)["status"] == "MULTIMODAL_RECEIPT_INVALID"


@pytest.mark.parametrize("field, value", [
    ("device", []),
    ("cache_type_k", []),
    ("split_mode", []),
])
def test_receipt_validator_rejects_unhashable_configuration_values(tmp_path, field, value):
    receipt = build_prepared_invocation_receipt(_prepared(tmp_path))
    receipt["configuration"][field] = value
    assert validate_multimodal_receipt(receipt)["status"] == "MULTIMODAL_RECEIPT_INVALID"


def test_execution_receipt_rejects_uninvoked_ocr_outcomes(tmp_path):
    prepared = _prepared(tmp_path)
    for terminal_class in ("OCR_MISSED", "OCR_INCONCLUSIVE"):
        receipt = {
            "schema_version": 1,
            "receipt_type": "MULTIMODAL_OCR_EXECUTION",
            "model_artifact": prepared.model_artifact,
            "projector_artifact": prepared.projector_artifact,
            "image_descriptor": prepared.image_descriptor,
            "configuration": prepared.configuration,
            "command_family": "multimodal_ocr_runner",
            "terminal_class": terminal_class,
            "invocation_attempted": True,
            "inference_invoked": False,
        }
        assert validate_multimodal_receipt(receipt)["status"] == "MULTIMODAL_RECEIPT_INVALID"


def test_image_descriptor_rejects_bool_and_non_string_keys():
    with pytest.raises(MultimodalPreflightError):
        validate_image_descriptor({"format": "png", "width": True, "height": 28, "size_bytes": 1})
    with pytest.raises(MultimodalPreflightError):
        validate_image_descriptor({1: "png", "width": 28, "height": 28, "size_bytes": 1})
