"""Sanitized protocol receipt validation for model-specific AutoBench workloads."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

RECEIPT_SCHEMA_VERSION = 1

# Status codes
PROTOCOL_RECEIPT_VALID = "PROTOCOL_RECEIPT_VALID"
PROTOCOL_RECEIPT_INVALID = "PROTOCOL_RECEIPT_INVALID"
PROTOCOL_RECEIPT_MISSING = "PROTOCOL_RECEIPT_MISSING"
PROTOCOL_RECEIPT_STALE = "PROTOCOL_RECEIPT_STALE"
PROTOCOL_RECEIPT_UNSANITIZED = "PROTOCOL_RECEIPT_UNSANITIZED"
PROTOCOL_RECEIPT_MODEL_MISMATCH = "PROTOCOL_RECEIPT_MODEL_MISMATCH"
PROTOCOL_RECEIPT_UNVERIFIED = "PROTOCOL_RECEIPT_UNVERIFIED"
PROTOCOL_RECEIPT_CORRUPT = "PROTOCOL_RECEIPT_CORRUPT"

FORBIDDEN_PATTERNS = {
    "private_path": re.compile(
        r"(?i)(?:/home/|[A-Za-z]:[\\/](?:Users|home|srv|tmp|var)[\\/])"
    ),
    "host_identifier": re.compile(
        r"(?i)(?:opencode@|192[.]168[.]\d+[.]\d+|100[.]67[.]\d+[.]\d+)"
    ),
    "sensitive_payload_key": re.compile(
        r"(?i)\b(?:stdout|stderr|raw_output|password|api[_-]?key|credential|token|secret)\b"
    ),
}

_SENSITIVE_KEY_NAMES = {
    "prompt",
    "response",
    "generated_text",
    "stdout",
    "stderr",
    "raw_output",
    "password",
    "api_key",
    "token",
    "secret",
    "credential",
}

DEFAULT_RECEIPT_DIRS = ("results/receipts", "receipts")


class ProtocolReceiptError(ValueError):
    """Raised when a model lacks a valid verified protocol receipt."""


def sanitize_receipt_data(value: Any) -> Any:
    """Recursively sanitize a receipt data structure, removing sensitive keys."""
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for k, v in value.items():
            if str(k).lower() in _SENSITIVE_KEY_NAMES:
                continue
            sanitized[k] = sanitize_receipt_data(v)
        return sanitized
    if isinstance(value, list):
        return [sanitize_receipt_data(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_receipt_data(item) for item in value]
    return deepcopy(value)


def check_for_unsanitized_content(value: Any, path: str = "") -> list[str]:
    """Check if any string value or key contains forbidden sensitive patterns."""
    errors: list[str] = []
    if isinstance(value, dict):
        for k, v in value.items():
            k_str = str(k)
            if k_str.lower() in _SENSITIVE_KEY_NAMES:
                errors.append(f"contains sensitive key '{k_str}' at {path}")
            for pat_name, pattern in FORBIDDEN_PATTERNS.items():
                if pattern.search(k_str):
                    errors.append(f"key '{k_str}' contains forbidden {pat_name} at {path}")
            errors.extend(check_for_unsanitized_content(v, f"{path}.{k}" if path else k_str))
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            errors.extend(check_for_unsanitized_content(item, f"{path}[{idx}]"))
    elif isinstance(value, str):
        for pat_name, pattern in FORBIDDEN_PATTERNS.items():
            if pattern.search(value):
                errors.append(f"value at {path} contains forbidden {pat_name}")
    return errors


def validate_protocol_receipt(
    receipt: Any,
    *,
    model_name: str | None = None,
    backend: str | None = None,
    check_files_exist: bool = False,
    base_dir: Path | str | None = None,
    governing_issue: int | str | None = None,
) -> dict[str, Any]:
    """Validate a protocol receipt object against the schema and model requirements.

    Returns a dict with:
        "status": status code string
        "model_name": target model name
        "errors": list of error messages (empty if valid)
        "receipt": sanitized receipt object or None
    """
    base_path = Path(base_dir).resolve() if base_dir else Path.cwd().resolve()
    result: dict[str, Any] = {
        "status": PROTOCOL_RECEIPT_INVALID,
        "model_name": model_name,
        "errors": [],
        "receipt": None,
    }

    if not isinstance(receipt, dict):
        result["errors"].append("receipt must be a JSON object")
        return result

    # Check for forbidden/unsanitized patterns first
    unsanitized_errors = check_for_unsanitized_content(receipt)
    if unsanitized_errors:
        result["status"] = PROTOCOL_RECEIPT_UNSANITIZED
        result["errors"].extend(unsanitized_errors)
        return result

    # Check schema version
    schema_version = receipt.get("schema_version")
    if schema_version != RECEIPT_SCHEMA_VERSION:
        result["errors"].append(
            f"unsupported schema_version: expected {RECEIPT_SCHEMA_VERSION}, got {schema_version}"
        )

    # Check required top-level fields
    receipt_model_name = receipt.get("model_name")
    if not isinstance(receipt_model_name, str) or not receipt_model_name:
        result["errors"].append("missing or invalid 'model_name'")
    else:
        result["model_name"] = receipt_model_name

    receipt_model_id = receipt.get("model_id")
    if not isinstance(receipt_model_id, str) or not receipt_model_id:
        result["errors"].append("missing or invalid 'model_id'")

    receipt_backend = receipt.get("backend")
    if not isinstance(receipt_backend, str) or not receipt_backend:
        result["errors"].append("missing or invalid 'backend'")

    receipt_governing_issue = receipt.get("governing_issue")
    if receipt_governing_issue is None:
        result["errors"].append("missing 'governing_issue'")

    # Validate model_name match if requested
    if model_name is not None and receipt_model_name is not None:
        if receipt_model_name.lower() != model_name.lower():
            result["status"] = PROTOCOL_RECEIPT_MODEL_MISMATCH
            result["errors"].append(
                f"model mismatch: expected '{model_name}', receipt is for '{receipt_model_name}'"
            )
            return result

    # Validate backend match if requested
    if backend is not None and receipt_backend is not None:
        if receipt_backend.lower() != backend.lower():
            result["status"] = PROTOCOL_RECEIPT_INVALID
            result["errors"].append(
                f"backend mismatch: expected '{backend}', receipt is for '{receipt_backend}'"
            )
            return result

    if governing_issue is not None and receipt_governing_issue != governing_issue:
        result["status"] = PROTOCOL_RECEIPT_INVALID
        result["errors"].append(
            f"governing issue mismatch: expected '{governing_issue}', receipt is for '{receipt_governing_issue}'"
        )
        return result

    # Validate stages
    stages = receipt.get("stages")
    if not isinstance(stages, dict):
        result["errors"].append("missing or invalid 'stages' object")
    else:
        # Stage 1: Research
        s1 = stages.get("stage_1_research")
        if not isinstance(s1, dict):
            result["errors"].append("missing 'stage_1_research'")
        else:
            if s1.get("status") != "VERIFIED":
                result["errors"].append(
                    f"stage_1_research status must be 'VERIFIED', got {s1.get('status')!r}"
                )
            raw_note = s1.get("raw_note")
            if not isinstance(raw_note, str) or not raw_note:
                result["errors"].append("stage_1_research missing 'raw_note'")
            elif check_files_exist:
                raw_path = (base_path / raw_note).resolve()
                if not raw_path.exists():
                    result["errors"].append(f"stage_1 raw_note not found: {raw_note}")
            if not isinstance(s1.get("retrieval_date"), str) or not s1.get("retrieval_date"):
                result["errors"].append("stage_1_research missing 'retrieval_date'")
            source_urls = s1.get("source_urls")
            if not isinstance(source_urls, list) or not source_urls:
                result["errors"].append("stage_1_research missing 'source_urls' list")
            else:
                for url in source_urls:
                    if not isinstance(url, str) or not urlparse(url).netloc:
                        result["errors"].append(f"stage_1 invalid source_url: {url}")

        # Stage 2: KB Validation
        s2 = stages.get("stage_2_kb_validation")
        if not isinstance(s2, dict):
            result["errors"].append("missing 'stage_2_kb_validation'")
        else:
            if s2.get("status") != "VERIFIED":
                result["errors"].append(
                    f"stage_2_kb_validation status must be 'VERIFIED', got {s2.get('status')!r}"
                )
            wiki_note = s2.get("wiki_note")
            if not isinstance(wiki_note, str) or not wiki_note:
                result["errors"].append("stage_2_kb_validation missing 'wiki_note'")
            elif check_files_exist:
                wiki_path = (base_path / wiki_note).resolve()
                if not wiki_path.exists():
                    result["errors"].append(f"stage_2 wiki_note not found: {wiki_note}")
            if s2.get("pre_run_check_passed") is not True:
                result["errors"].append("stage_2 'pre_run_check_passed' must be True")
            if s2.get("qmd_indexed") is not True:
                result["errors"].append("stage_2 'qmd_indexed' must be True")

        # Stage 3: Plan Review
        s3 = stages.get("stage_3_plan_review")
        if not isinstance(s3, dict):
            result["errors"].append("missing 'stage_3_plan_review'")
        else:
            if s3.get("status") not in {"APPROVED", "REVIEWED"}:
                result["errors"].append(
                    f"stage_3_plan_review status must be 'APPROVED' or 'REVIEWED', got {s3.get('status')!r}"
                )
            if s3.get("plan_reviewed") is not True:
                result["errors"].append("stage_3 'plan_reviewed' must be True")
            if not isinstance(s3.get("review_evidence"), str) or not s3.get("review_evidence"):
                result["errors"].append("stage_3 missing 'review_evidence'")
            expected_jobs = s3.get("expected_job_count")
            if not isinstance(expected_jobs, int) or expected_jobs < 1:
                result["errors"].append("stage_3 'expected_job_count' must be a positive integer")

    # Overall receipt status check
    receipt_status = receipt.get("status")
    if receipt_status != PROTOCOL_RECEIPT_VALID:
        result["errors"].append(
            f"receipt status must be '{PROTOCOL_RECEIPT_VALID}', got {receipt_status!r}"
        )

    if result["errors"]:
        result["status"] = PROTOCOL_RECEIPT_INVALID
    else:
        result["status"] = PROTOCOL_RECEIPT_VALID
        result["receipt"] = sanitize_receipt_data(receipt)

    return result


def find_receipt_for_model(
    model_name: str,
    search_dirs: Sequence[Path | str] | None = None,
    base_dir: Path | str | None = None,
    governing_issue: int | str | None = None,
) -> Path | None:
    """Find one unambiguous receipt, optionally constrained to an issue."""
    base_path = Path(base_dir).resolve() if base_dir else Path.cwd().resolve()
    dirs = [Path(d) for d in (search_dirs or DEFAULT_RECEIPT_DIRS)]
    clean_name = Path(model_name).name
    stem = Path(model_name).stem

    for d in dirs:
        dir_path = (base_path / d) if not d.is_absolute() else d
        if not dir_path.exists():
            continue
        candidates = set(dir_path.glob(f"{clean_name}*.json"))
        candidates.update(dir_path.glob(f"{stem}*.json"))
        matches = []
        for candidate in sorted(candidates):
            try:
                content = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(content, dict):
                continue
            if str(content.get("model_name", "")).lower() != clean_name.lower():
                continue
            if governing_issue is not None and content.get("governing_issue") != governing_issue:
                continue
            matches.append(candidate)
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            return None
    return None


def load_and_validate_receipt(
    receipt_source: Path | str | dict,
    *,
    model_name: str | None = None,
    backend: str | None = None,
    check_files_exist: bool = False,
    base_dir: Path | str | None = None,
    governing_issue: int | str | None = None,
) -> dict[str, Any]:
    """Load a receipt from a file or dict and validate it."""
    if isinstance(receipt_source, dict):
        return validate_protocol_receipt(
            receipt_source,
            model_name=model_name,
            backend=backend,
            check_files_exist=check_files_exist,
            base_dir=base_dir,
            governing_issue=governing_issue,
        )

    path = Path(receipt_source)
    if not path.exists():
        return {
            "status": PROTOCOL_RECEIPT_MISSING,
            "model_name": model_name,
            "errors": [f"receipt file not found: {path}"],
            "receipt": None,
        }

    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": PROTOCOL_RECEIPT_CORRUPT,
            "model_name": model_name,
            "errors": [f"failed to read or parse receipt JSON {path}: {exc}"],
            "receipt": None,
        }

    return validate_protocol_receipt(
        content,
        model_name=model_name,
        backend=backend,
        check_files_exist=check_files_exist,
        base_dir=base_dir,
        governing_issue=governing_issue,
    )


def verify_models_have_receipts(
    models: Sequence[dict | str],
    *,
    receipt_dir: Path | str | None = None,
    receipt_paths: dict[str, Path | str] | None = None,
    backend: str = "vulkan",
    check_files_exist: bool = False,
    base_dir: Path | str | None = None,
    governing_issue: int | str | None = None,
) -> dict[str, dict[str, Any]]:
    """Verify receipts for all models in the sequence.

    Returns a dict mapping model_name -> validation_result.
    """
    search_dirs = [receipt_dir] if receipt_dir else list(DEFAULT_RECEIPT_DIRS)
    results: dict[str, dict[str, Any]] = {}

    for model in models:
        m_name = model["name"] if isinstance(model, dict) else str(model)
        explicit_path = (receipt_paths or {}).get(m_name)
        if explicit_path:
            target_path = Path(explicit_path)
        else:
            target_path = find_receipt_for_model(
                m_name,
                search_dirs=search_dirs,
                base_dir=base_dir,
                governing_issue=governing_issue,
            )

        if target_path is None:
            results[m_name] = {
                "status": PROTOCOL_RECEIPT_MISSING,
                "model_name": m_name,
                "errors": [f"no protocol receipt found for model '{m_name}' in {search_dirs}"],
                "receipt": None,
            }
        else:
            results[m_name] = load_and_validate_receipt(
                target_path,
                model_name=m_name,
                backend=backend,
                check_files_exist=check_files_exist,
                base_dir=base_dir,
                governing_issue=governing_issue,
            )

    return results


def assert_all_models_have_valid_receipts(
    models: Sequence[dict | str],
    *,
    receipt_dir: Path | str | None = None,
    receipt_paths: dict[str, Path | str] | None = None,
    backend: str = "vulkan",
    check_files_exist: bool = False,
    base_dir: Path | str | None = None,
    governing_issue: int | str | None = None,
) -> dict[str, dict[str, Any]]:
    """Assert all models have valid protocol receipts, or raise ProtocolReceiptError."""
    results = verify_models_have_receipts(
        models,
        receipt_dir=receipt_dir,
        receipt_paths=receipt_paths,
        backend=backend,
        check_files_exist=check_files_exist,
        base_dir=base_dir,
        governing_issue=governing_issue,
    )
    invalid_or_missing = [
        f"{name} ({res['status']}: {', '.join(res['errors'])})"
        for name, res in results.items()
        if res["status"] != PROTOCOL_RECEIPT_VALID
    ]
    if invalid_or_missing:
        raise ProtocolReceiptError(
            f"model protocol receipt verification failed: {'; '.join(invalid_or_missing)}"
        )
    return results


def build_protocol_receipt(
    *,
    model_name: str,
    model_id: str,
    quantization: str,
    backend: str,
    governing_issue: int | str,
    raw_note: str,
    retrieval_date: str,
    source_urls: list[str],
    wiki_note: str,
    review_evidence: str,
    planned_configurations: list[dict],
    expected_job_count: int,
    validation_timestamp: str | None = None,
) -> dict[str, Any]:
    """Helper to construct a valid sanitized protocol receipt document."""
    timestamp = validation_timestamp or datetime.now(timezone.utc).isoformat()
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "model_name": model_name,
        "model_id": model_id,
        "quantization": quantization,
        "backend": backend,
        "governing_issue": governing_issue,
        "stages": {
            "stage_1_research": {
                "status": "VERIFIED",
                "raw_note": raw_note,
                "retrieval_date": retrieval_date,
                "source_urls": source_urls,
            },
            "stage_2_kb_validation": {
                "status": "VERIFIED",
                "wiki_note": wiki_note,
                "pre_run_check_passed": True,
                "qmd_indexed": True,
            },
            "stage_3_plan_review": {
                "status": "APPROVED",
                "plan_reviewed": True,
                "review_evidence": review_evidence,
                "planned_configurations": planned_configurations,
                "expected_job_count": expected_job_count,
            },
        },
        "validation_timestamp": timestamp,
        "status": PROTOCOL_RECEIPT_VALID,
    }
    return sanitize_receipt_data(receipt)
