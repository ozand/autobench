"""Fail-closed sanitization and validation for multimodal OCR preflight receipts."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

MULTIMODAL_RECEIPT_SCHEMA_VERSION = 1
MULTIMODAL_RECEIPT_TYPE = "MULTIMODAL_OCR_PREFLIGHT"
MULTIMODAL_INVOCATION_RECEIPT_TYPE = "MULTIMODAL_OCR_INVOCATION_CONTRACT"
MULTIMODAL_COMMAND_PLAN_RECEIPT_TYPE = "MULTIMODAL_OCR_COMMAND_PLAN"
MULTIMODAL_TARGET_DRY_RUN_RECEIPT_TYPE = "MULTIMODAL_OCR_TARGET_DRY_RUN"
MULTIMODAL_EXECUTION_RECEIPT_TYPE = "MULTIMODAL_OCR_EXECUTION"
MULTIMODAL_SMOKE_EXECUTION_RECEIPT_TYPE = "MULTIMODAL_OCR_SMOKE_EXECUTION"
MULTIMODAL_SMOKE_SIMULATION_RECEIPT_TYPE = "MULTIMODAL_OCR_SMOKE_SIMULATION"
SUPPORTED_IMAGE_FORMATS = {"png", "jpg", "jpeg"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_DIMENSION = 2048
_IMAGE_DESCRIPTOR_KEYS = {"format", "width", "height", "byte_class", "validation_status"}
_CONFIGURATION_KEYS = {
    "device", "split_mode", "split_ratio", "context_length", "cache_type_k",
    "cache_type_v", "max_tokens",
}
_EXECUTION_TERMINAL_CLASSES = {
    "IMAGE_INPUT_REJECTED", "PROJECTOR_MISMATCH", "UNSUPPORTED_BACKEND", "OOM",
    "CONTEXT_OVERFLOW", "METRIC_PARSE_FAILED", "OCR_MISSED", "OCR_INCONCLUSIVE", "SUCCESS",
}
_ALLOWED_DEVICES = {"CPU", "Vulkan0", "Vulkan1", "Vulkan0,Vulkan1"}
_ALLOWED_SPLIT_MODES = {"none", "layer"}
_ALLOWED_KV_TYPES = {"f16", "q8_0"}
_PAIRING_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")

_FORBIDDEN_KEYS = {
    "path", "image", "image_bytes", "base64", "filename", "original_filename",
    "url", "source_url", "prompt", "response", "ocr_text", "generated_text",
    "stdout", "stderr", "raw_output", "credential", "token", "secret", "password",
    "command", "command_args", "exception",
}
_FORBIDDEN_VALUE = re.compile(
    r"(?i)(?:/home/|[a-z]:[\\/](?:users|home|srv|tmp|var)[\\/]|"
    r"opencode@|100[.]67[.]\d+[.]\d+|data:image/|base64,)"
)


def sanitize_multimodal_receipt(value: Any) -> Any:
    """Remove prohibited payload keys recursively before serialization."""
    if isinstance(value, dict):
        return {
            str(key): sanitize_multimodal_receipt(item)
            for key, item in value.items()
            if str(key).lower() not in _FORBIDDEN_KEYS
        }
    if isinstance(value, list):
        return [sanitize_multimodal_receipt(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_multimodal_receipt(item) for item in value]
    return deepcopy(value)


def find_unsanitized_multimodal_content(value: Any, path: str = "") -> list[str]:
    """Return deterministic reasons why a receipt cannot be persisted."""
    errors: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            item_path = f"{path}.{key_text}" if path else key_text
            if key_text.lower() in _FORBIDDEN_KEYS:
                errors.append(f"forbidden key at {item_path}")
            errors.extend(find_unsanitized_multimodal_content(item, item_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            errors.extend(find_unsanitized_multimodal_content(item, f"{path}[{index}]"))
    elif isinstance(value, str) and _FORBIDDEN_VALUE.search(value):
        errors.append(f"forbidden value at {path}")
    return errors


def _validate_artifacts(
    receipt: dict[str, Any], errors: list[str], *, allow_pairing_mismatch: bool = False
) -> None:
    pairings: dict[str, Any] = {}
    for artifact_key in ("model_artifact", "projector_artifact"):
        artifact = receipt.get(artifact_key)
        if not isinstance(artifact, dict):
            errors.append(f"missing {artifact_key}")
            continue
        if set(artifact) != {"basename", "size_bytes", "sha256", "pairing_id"}:
            errors.append(f"ambiguous {artifact_key} fields")
        basename = artifact.get("basename")
        if (
            not isinstance(basename, str)
            or not basename
            or basename != basename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
            or basename in {".", ".."}
        ):
            errors.append(f"invalid {artifact_key}.basename")
        if type(artifact.get("size_bytes")) is not int or artifact["size_bytes"] < 1:
            errors.append(f"invalid {artifact_key}.size_bytes")
        digest = artifact.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            errors.append(f"invalid {artifact_key}.sha256")
        pairing = artifact.get("pairing_id")
        if not isinstance(pairing, str) or not _PAIRING_ID.fullmatch(pairing):
            errors.append(f"invalid {artifact_key}.pairing_id")
        pairings[artifact_key] = pairing
    model_pairing = pairings.get("model_artifact")
    projector_pairing = pairings.get("projector_artifact")
    if not model_pairing or not projector_pairing:
        errors.append("model/projector pairing does not match")
    elif allow_pairing_mismatch and model_pairing == projector_pairing:
        errors.append("PROJECTOR_MISMATCH requires distinct pairing identifiers")
    elif not allow_pairing_mismatch and model_pairing != projector_pairing:
        errors.append("model/projector pairing does not match")


def _validate_descriptor(descriptor: Any, errors: list[str], *, rejected: bool) -> None:
    if rejected:
        if descriptor != {"validation_status": "REJECTED"}:
            errors.append("invalid rejected image_descriptor")
        return
    if not isinstance(descriptor, dict):
        errors.append("missing image_descriptor")
    elif set(descriptor) != _IMAGE_DESCRIPTOR_KEYS:
        errors.append("image_descriptor fields are ambiguous")
    else:
        image_format = descriptor.get("format")
        width = descriptor.get("width")
        height = descriptor.get("height")
        byte_class = descriptor.get("byte_class")
        if not isinstance(image_format, str) or image_format not in SUPPORTED_IMAGE_FORMATS:
            errors.append("invalid image_descriptor.format")
        if type(width) is not int or not 0 < width <= MAX_IMAGE_DIMENSION:
            errors.append("invalid image_descriptor.width")
        if type(height) is not int or not 0 < height <= MAX_IMAGE_DIMENSION:
            errors.append("invalid image_descriptor.height")
        if not isinstance(byte_class, str) or byte_class not in {"small", "medium"}:
            errors.append("invalid image_descriptor.byte_class")
        if descriptor.get("validation_status") != "VALID":
            errors.append("invalid image_descriptor.validation_status")


def _validate_configuration(configuration: Any, errors: list[str], *, required: bool) -> None:
    if configuration is None and not required:
        return
    if not isinstance(configuration, dict) or set(configuration) != _CONFIGURATION_KEYS:
        errors.append("invalid configuration")
        return
    device = configuration["device"]
    split_mode = configuration["split_mode"]
    split_ratio = configuration["split_ratio"]
    context_length = configuration["context_length"]
    cache_type_k = configuration["cache_type_k"]
    cache_type_v = configuration["cache_type_v"]
    max_tokens = configuration["max_tokens"]
    if not isinstance(device, str) or device not in _ALLOWED_DEVICES:
        errors.append("invalid configuration.device")
    if not isinstance(split_mode, str) or split_mode not in _ALLOWED_SPLIT_MODES:
        errors.append("invalid configuration.split_mode")
    elif split_mode == "none" and split_ratio is not None:
        errors.append("invalid configuration.split_ratio")
    elif split_mode == "layer" and (
        not isinstance(split_ratio, str) or split_ratio not in {"1,1", "1,2"}
    ):
        errors.append("invalid configuration.split_ratio")
    if type(context_length) is not int or context_length not in {1024, 2048}:
        errors.append("invalid configuration.context_length")
    if (
        not isinstance(cache_type_k, str)
        or not isinstance(cache_type_v, str)
        or cache_type_k not in _ALLOWED_KV_TYPES
        or cache_type_v not in _ALLOWED_KV_TYPES
    ):
        errors.append("invalid configuration.cache_type")
    elif cache_type_k != cache_type_v:
        errors.append("invalid configuration.cache_type")
    if type(max_tokens) is not int or not 1 <= max_tokens <= 128:
        errors.append("invalid configuration.max_tokens")


def validate_multimodal_receipt(receipt: Any) -> dict[str, Any]:
    """Validate fail-closed preflight, invocation, and execution receipt shapes."""
    errors: list[str] = []
    if not isinstance(receipt, dict):
        return {"status": "MULTIMODAL_RECEIPT_INVALID", "errors": ["receipt must be an object"]}
    errors.extend(find_unsanitized_multimodal_content(receipt))
    if type(receipt.get("schema_version")) is not int or receipt.get("schema_version") != MULTIMODAL_RECEIPT_SCHEMA_VERSION:
        errors.append("unsupported schema_version")
    receipt_type = receipt.get("receipt_type")
    if not isinstance(receipt_type, str):
        errors.append("unexpected receipt_type")
        receipt_type = ""
    allowed_keys = {
        MULTIMODAL_RECEIPT_TYPE: {
            "schema_version", "receipt_type", "model_artifact", "projector_artifact",
            "image_descriptor", "preflight_status", "inference_invoked",
        },
        MULTIMODAL_INVOCATION_RECEIPT_TYPE: {
            "schema_version", "receipt_type", "model_artifact", "projector_artifact",
            "image_descriptor", "configuration", "command_family", "inference_invoked",
        },
        MULTIMODAL_COMMAND_PLAN_RECEIPT_TYPE: {
            "schema_version", "receipt_type", "model_artifact", "projector_artifact",
            "image_descriptor", "configuration", "command_family", "binary",
            "argument_flags", "planned_job_count", "dry_run", "inference_invoked",
        },
        MULTIMODAL_TARGET_DRY_RUN_RECEIPT_TYPE: {
            "schema_version", "receipt_type", "model_artifact", "projector_artifact",
            "image_descriptor", "configuration", "command_family", "binary",
            "argument_flags", "planned_job_count", "dry_run", "inference_invoked",
            "validation_scope",
        },
        MULTIMODAL_SMOKE_EXECUTION_RECEIPT_TYPE: {
            "schema_version", "receipt_type", "model_artifact", "projector_artifact",
            "image_descriptor", "configuration", "command_family", "terminal_class",
            "task_id", "task_version", "output_classification", "invocation_attempted",
            "inference_invoked",
        },
        MULTIMODAL_SMOKE_SIMULATION_RECEIPT_TYPE: {
            "schema_version", "receipt_type", "model_artifact", "projector_artifact",
            "image_descriptor", "configuration", "command_family", "terminal_class",
            "task_id", "task_version", "output_classification", "simulated", "inference_invoked",
        },
        MULTIMODAL_EXECUTION_RECEIPT_TYPE: {
            "schema_version", "receipt_type", "model_artifact", "projector_artifact",
            "image_descriptor", "configuration", "command_family", "terminal_class",
            "invocation_attempted", "inference_invoked",
        },
    }
    if receipt_type in allowed_keys and set(receipt) != allowed_keys[receipt_type]:
        errors.append("receipt fields are ambiguous")
    _validate_artifacts(
        receipt,
        errors,
        allow_pairing_mismatch=(
            receipt_type == MULTIMODAL_EXECUTION_RECEIPT_TYPE
            and receipt.get("terminal_class") == "PROJECTOR_MISMATCH"
        ),
    )
    if receipt_type == MULTIMODAL_RECEIPT_TYPE:
        _validate_descriptor(receipt.get("image_descriptor"), errors, rejected=False)
        if receipt.get("preflight_status") not in {"VALID", "REJECTED"}:
            errors.append("invalid preflight_status")
        if receipt.get("inference_invoked") is not False:
            errors.append("inference_invoked must be false")
    elif receipt_type == MULTIMODAL_INVOCATION_RECEIPT_TYPE:
        _validate_descriptor(receipt.get("image_descriptor"), errors, rejected=False)
        _validate_configuration(receipt.get("configuration"), errors, required=True)
        if receipt.get("command_family") != "multimodal_ocr_runner":
            errors.append("invalid command_family")
        if receipt.get("inference_invoked") is not False:
            errors.append("inference_invoked must be false")
    elif receipt_type == MULTIMODAL_COMMAND_PLAN_RECEIPT_TYPE:
        _validate_descriptor(receipt.get("image_descriptor"), errors, rejected=False)
        _validate_configuration(receipt.get("configuration"), errors, required=True)
        if receipt.get("command_family") != "multimodal_ocr_runner":
            errors.append("invalid command_family")
        if receipt.get("binary") != "llama-mtmd-cli":
            errors.append("invalid binary")
        expected_flags = ["-m", "--mmproj", "--image", "-ngl", "-dev", "-c", "-ctk", "-ctv", "-n"]
        if receipt.get("argument_flags") != expected_flags:
            errors.append("invalid argument_flags")
        if type(receipt.get("planned_job_count")) is not int or receipt.get("planned_job_count") != 1 or receipt.get("dry_run") is not True:
            errors.append("invalid dry-run job count")
        if receipt.get("inference_invoked") is not False:
            errors.append("inference_invoked must be false")
    elif receipt_type == MULTIMODAL_TARGET_DRY_RUN_RECEIPT_TYPE:
        _validate_descriptor(receipt.get("image_descriptor"), errors, rejected=False)
        _validate_configuration(receipt.get("configuration"), errors, required=True)
        if receipt.get("command_family") != "multimodal_ocr_runner":
            errors.append("invalid command_family")
        if receipt.get("binary") != "llama-mtmd-cli":
            errors.append("invalid binary")
        expected_flags = ["-m", "--mmproj", "--image", "-ngl", "-dev", "-c", "-ctk", "-ctv", "-n"]
        if receipt.get("argument_flags") != expected_flags:
            errors.append("invalid argument_flags")
        if type(receipt.get("planned_job_count")) is not int or receipt.get("planned_job_count") != 1 or receipt.get("dry_run") is not True:
            errors.append("invalid dry-run job count")
        if receipt.get("inference_invoked") is not False:
            errors.append("inference_invoked must be false")
        if receipt.get("validation_scope") != "ARTIFACT_IDENTITY_AND_PLAN":
            errors.append("invalid validation_scope")
    elif receipt_type in {MULTIMODAL_SMOKE_EXECUTION_RECEIPT_TYPE, MULTIMODAL_SMOKE_SIMULATION_RECEIPT_TYPE}:
        terminal_class = receipt.get("terminal_class")
        classifications = {
            "SUCCESS": "OCR_SMOKE_OUTPUT_OBSERVED",
            "OCR_INCONCLUSIVE": "OCR_SMOKE_NO_OUTPUT",
            "METRIC_PARSE_FAILED": {"MALFORMED_STREAM", "OUTPUT_OVERSIZE"},
            "OOM": "RUNTIME_OOM",
            "CONTEXT_OVERFLOW": "RUNTIME_CONTEXT",
            "UNSUPPORTED_BACKEND": "RUNTIME_BACKEND",
            "EXECUTION_ERROR": {"RUNTIME_NONZERO", "RUNTIME_DRIVER_ERROR"},
            "INCONCLUSIVE": "RUNTIME_TIMEOUT",
        }
        expected_classification = classifications.get(terminal_class)
        if (
            (isinstance(expected_classification, set) and receipt.get("output_classification") not in expected_classification)
            or (not isinstance(expected_classification, set) and expected_classification != receipt.get("output_classification"))
        ):
            errors.append("invalid smoke terminal/output classification")
        _validate_descriptor(receipt.get("image_descriptor"), errors, rejected=False)
        _validate_configuration(receipt.get("configuration"), errors, required=True)
        if receipt.get("command_family") != "multimodal_ocr_runner":
            errors.append("invalid command_family")
        if receipt.get("task_id") != "ocr_smoke_v1" or type(receipt.get("task_version")) is not int or receipt.get("task_version") != 1:
            errors.append("invalid smoke task")
        if receipt_type == MULTIMODAL_SMOKE_SIMULATION_RECEIPT_TYPE:
            if receipt.get("simulated") is not True or receipt.get("inference_invoked") is not False:
                errors.append("invalid smoke simulation state")
        elif type(receipt.get("invocation_attempted")) is not bool or type(receipt.get("inference_invoked")) is not bool:
            errors.append("invalid smoke execution state")
        elif terminal_class == "SUCCESS" and (
            receipt.get("invocation_attempted") is not True or receipt.get("inference_invoked") is not True
        ):
            errors.append("smoke success requires invoked inference")
        elif receipt.get("invocation_attempted") is False and receipt.get("inference_invoked") is not False:
            errors.append("unattempted smoke execution cannot invoke inference")
    elif receipt_type == MULTIMODAL_EXECUTION_RECEIPT_TYPE:
        terminal_class = receipt.get("terminal_class")
        is_pre_invocation_class = (
            isinstance(terminal_class, str)
            and terminal_class in {"IMAGE_INPUT_REJECTED", "PROJECTOR_MISMATCH"}
        )
        if not isinstance(terminal_class, str) or terminal_class not in _EXECUTION_TERMINAL_CLASSES:
            errors.append("invalid terminal_class")
        _validate_descriptor(
            receipt.get("image_descriptor"), errors,
            rejected=is_pre_invocation_class,
        )
        _validate_configuration(
            receipt.get("configuration"), errors,
            required=not is_pre_invocation_class,
        )
        if receipt.get("command_family") != "multimodal_ocr_runner":
            errors.append("invalid command_family")
        invocation_attempted = receipt.get("invocation_attempted")
        inference_invoked = receipt.get("inference_invoked")
        if is_pre_invocation_class:
            if invocation_attempted is not False or inference_invoked is not False:
                errors.append("pre-invocation terminal class must not invoke inference")
        else:
            if invocation_attempted is not True:
                errors.append("execution terminal class requires an invocation attempt")
            if not isinstance(inference_invoked, bool):
                errors.append("inference_invoked must be boolean")
            if (
                isinstance(terminal_class, str)
                and terminal_class in {"SUCCESS", "OCR_MISSED", "OCR_INCONCLUSIVE"}
                and inference_invoked is not True
            ):
                errors.append("OCR outcome requires invoked inference")
    else:
        errors.append("unexpected receipt_type")
    return {
        "status": "MULTIMODAL_RECEIPT_VALID" if not errors else "MULTIMODAL_RECEIPT_INVALID",
        "errors": errors,
        "receipt": sanitize_multimodal_receipt(receipt) if not errors else None,
    }
