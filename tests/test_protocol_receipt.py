import json
import pytest
from pathlib import Path
from src.protocol_receipt import (
    PROTOCOL_RECEIPT_CORRUPT,
    PROTOCOL_RECEIPT_INVALID,
    PROTOCOL_RECEIPT_MISSING,
    PROTOCOL_RECEIPT_MODEL_MISMATCH,
    PROTOCOL_RECEIPT_UNSANITIZED,
    PROTOCOL_RECEIPT_VALID,
    ProtocolReceiptError,
    assert_all_models_have_valid_receipts,
    build_protocol_receipt,
    find_receipt_for_model,
    load_and_validate_receipt,
    validate_protocol_receipt,
    verify_models_have_receipts,
)


def sample_receipt_dict(**overrides):
    base = {
        "schema_version": 1,
        "model_name": "Qwen3.5-0.8B-Q4_K_M.gguf",
        "model_id": "Qwen3.5-0.8B-Q4_K_M",
        "quantization": "Q4_K_M",
        "backend": "vulkan",
        "governing_issue": 22,
        "stages": {
            "stage_1_research": {
                "status": "VERIFIED",
                "raw_note": "kb/raw/qwen3.5-0.8b-model-card.md",
                "retrieval_date": "2026-08-19",
                "source_urls": [
                    "https://huggingface.co/Qwen/Qwen3.5-0.8B",
                    "https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md",
                ],
            },
            "stage_2_kb_validation": {
                "status": "VERIFIED",
                "wiki_note": "kb/wiki/QWEN35-001-qwen35-08b-vulkan-diagnostic.md",
                "pre_run_check_passed": True,
                "qmd_indexed": True,
            },
            "stage_3_plan_review": {
                "status": "APPROVED",
                "plan_reviewed": True,
                "review_evidence": "reviewed_in_issue_22",
                "planned_configurations": [
                    {"device": "Vulkan0", "mode": "full"},
                    {"device": "Vulkan1", "mode": "full"},
                ],
                "expected_job_count": 2,
            },
        },
        "validation_timestamp": "2026-08-19T10:00:00Z",
        "status": "PROTOCOL_RECEIPT_VALID",
    }
    base.update(overrides)
    return base


def test_valid_protocol_receipt():
    data = sample_receipt_dict()
    res = validate_protocol_receipt(data, model_name="Qwen3.5-0.8B-Q4_K_M.gguf", backend="vulkan")
    assert res["status"] == PROTOCOL_RECEIPT_VALID
    assert not res["errors"]
    assert res["receipt"]["model_name"] == "Qwen3.5-0.8B-Q4_K_M.gguf"


def test_missing_required_fields_fails_closed():
    res = validate_protocol_receipt({})
    assert res["status"] == PROTOCOL_RECEIPT_INVALID
    assert len(res["errors"]) >= 4

    res = validate_protocol_receipt(None)
    assert res["status"] == PROTOCOL_RECEIPT_INVALID


def test_model_mismatch_fails_closed():
    data = sample_receipt_dict()
    res = validate_protocol_receipt(data, model_name="different-model.gguf")
    assert res["status"] == PROTOCOL_RECEIPT_MODEL_MISMATCH
    assert "model mismatch" in res["errors"][0]


def test_backend_mismatch_fails_closed():
    data = sample_receipt_dict()
    res = validate_protocol_receipt(data, backend="cuda")
    assert res["status"] == PROTOCOL_RECEIPT_INVALID
    assert "backend mismatch" in res["errors"][0]


def test_unsanitized_private_paths_fail_closed():
    data = sample_receipt_dict()
    data["stages"]["stage_1_research"]["raw_note"] = "/home/opencode/kb/raw/note.md"
    res = validate_protocol_receipt(data)
    assert res["status"] == PROTOCOL_RECEIPT_UNSANITIZED
    assert any("private_path" in err for err in res["errors"])

    data2 = sample_receipt_dict()
    data2["stages"]["stage_1_research"]["raw_note"] = "C:\\Users\\alice\\note.md"
    res2 = validate_protocol_receipt(data2)
    assert res2["status"] == PROTOCOL_RECEIPT_UNSANITIZED


def test_unsanitized_sensitive_keys_fail_closed():
    data = sample_receipt_dict()
    data["stdout"] = "raw logs"
    res = validate_protocol_receipt(data)
    assert res["status"] == PROTOCOL_RECEIPT_UNSANITIZED
    assert any("sensitive key" in err for err in res["errors"])

    data2 = sample_receipt_dict()
    data2["stages"]["stage_3_plan_review"]["prompt"] = "secret prompt"
    res2 = validate_protocol_receipt(data2)
    assert res2["status"] == PROTOCOL_RECEIPT_UNSANITIZED


def test_unsanitized_host_identifiers_fail_closed():
    data = sample_receipt_dict()
    data["stages"]["stage_3_plan_review"]["review_evidence"] = "tested on opencode@100.67.171.58"
    res = validate_protocol_receipt(data)
    assert res["status"] == PROTOCOL_RECEIPT_UNSANITIZED
    assert any("host_identifier" in err for err in res["errors"])


def test_unverified_stages_fail_closed():
    # Stage 1 unverified
    d1 = sample_receipt_dict()
    d1["stages"]["stage_1_research"]["status"] = "PENDING"
    res1 = validate_protocol_receipt(d1)
    assert res1["status"] == PROTOCOL_RECEIPT_INVALID
    assert any("stage_1_research status" in err for err in res1["errors"])

    # Stage 2 unverified
    d2 = sample_receipt_dict()
    d2["stages"]["stage_2_kb_validation"]["pre_run_check_passed"] = False
    res2 = validate_protocol_receipt(d2)
    assert res2["status"] == PROTOCOL_RECEIPT_INVALID
    assert any("pre_run_check_passed" in err for err in res2["errors"])

    # Stage 3 unreviewed
    d3 = sample_receipt_dict()
    d3["stages"]["stage_3_plan_review"]["status"] = "UNREVIEWED"
    d3["stages"]["stage_3_plan_review"]["plan_reviewed"] = False
    res3 = validate_protocol_receipt(d3)
    assert res3["status"] == PROTOCOL_RECEIPT_INVALID
    assert any("stage_3_plan_review status" in err for err in res3["errors"])


def test_contract_constraints_reject_mismatched_plan():
    data = sample_receipt_dict(governing_issue=41)
    result = validate_protocol_receipt(data, governing_issue=41, expected_job_count=1)
    assert result["status"] == PROTOCOL_RECEIPT_INVALID
    assert "expected job count mismatch" in result["errors"][0]


def test_issue_constraint_rejects_unrelated_receipt():
    data = sample_receipt_dict(governing_issue=1)
    result = validate_protocol_receipt(data, governing_issue=41)
    assert result["status"] == PROTOCOL_RECEIPT_INVALID
    assert "governing issue mismatch" in result["errors"][0]


def test_find_and_load_receipt(tmp_path):
    receipt_dir = tmp_path / "receipts"
    receipt_dir.mkdir()
    receipt_file = receipt_dir / "Qwen3.5-0.8B-Q4_K_M.gguf.json"
    data = sample_receipt_dict()
    receipt_file.write_text(json.dumps(data), encoding="utf-8")

    found = find_receipt_for_model("Qwen3.5-0.8B-Q4_K_M.gguf", search_dirs=[receipt_dir])
    assert found == receipt_file

    loaded = load_and_validate_receipt(found, model_name="Qwen3.5-0.8B-Q4_K_M.gguf")
    assert loaded["status"] == PROTOCOL_RECEIPT_VALID

    missing = load_and_validate_receipt(tmp_path / "nonexistent.json", model_name="foo")
    assert missing["status"] == PROTOCOL_RECEIPT_MISSING

    corrupt_file = tmp_path / "corrupt.json"
    corrupt_file.write_text("{not valid json", encoding="utf-8")
    corrupt = load_and_validate_receipt(corrupt_file, model_name="foo")
    assert corrupt["status"] == PROTOCOL_RECEIPT_CORRUPT


def test_find_receipt_rejects_ambiguous_matching_candidates(tmp_path):
    receipt_dir = tmp_path / "receipts"
    receipt_dir.mkdir()
    data = sample_receipt_dict(model_name="model.gguf", model_id="model")
    for suffix in ("", ".issue41"):
        (receipt_dir / f"model.gguf{suffix}.json").write_text(json.dumps(data), encoding="utf-8")
    assert find_receipt_for_model("model.gguf", search_dirs=[receipt_dir]) is None


def test_find_receipt_can_select_issue_constrained_candidate(tmp_path):
    receipt_dir = tmp_path / "receipts"
    receipt_dir.mkdir()
    legacy = sample_receipt_dict(model_name="model.gguf", model_id="model", governing_issue=1)
    issue = sample_receipt_dict(model_name="model.gguf", model_id="model", governing_issue=41)
    (receipt_dir / "model.gguf.json").write_text(json.dumps(legacy), encoding="utf-8")
    (receipt_dir / "model.gguf.issue41.json").write_text(json.dumps(issue), encoding="utf-8")
    found = find_receipt_for_model("model.gguf", search_dirs=[receipt_dir], governing_issue=41)
    assert found.name == "model.gguf.issue41.json"


def test_verify_models_have_receipts_batch(tmp_path):
    receipt_dir = tmp_path / "receipts"
    receipt_dir.mkdir()
    (receipt_dir / "model_a.gguf.json").write_text(
        json.dumps(sample_receipt_dict(model_name="model_a.gguf", model_id="model_a")),
        encoding="utf-8",
    )

    models = [
        {"id": "model_a", "name": "model_a.gguf"},
        {"id": "model_b", "name": "model_b.gguf"},
    ]
    results = verify_models_have_receipts(models, receipt_dir=receipt_dir)
    assert results["model_a.gguf"]["status"] == PROTOCOL_RECEIPT_VALID
    assert results["model_b.gguf"]["status"] == PROTOCOL_RECEIPT_MISSING

    with pytest.raises(ProtocolReceiptError) as exc_info:
        assert_all_models_have_valid_receipts(models, receipt_dir=receipt_dir)
    assert "model_b.gguf" in str(exc_info.value)


def test_build_protocol_receipt_helper():
    receipt = build_protocol_receipt(
        model_name="test-model.gguf",
        model_id="test-model",
        quantization="Q4_K_M",
        backend="vulkan",
        governing_issue=22,
        raw_note="kb/raw/test.md",
        retrieval_date="2026-08-19",
        source_urls=["https://example.com/model"],
        wiki_note="kb/wiki/TEST-001.md",
        review_evidence="issue_22_plan",
        planned_configurations=[{"device": "Vulkan0", "mode": "full"}],
        expected_job_count=1,
    )
    res = validate_protocol_receipt(receipt, model_name="test-model.gguf")
    assert res["status"] == PROTOCOL_RECEIPT_VALID


def test_check_files_exist_validation(tmp_path):
    data = sample_receipt_dict()
    # Notes don't exist in tmp_path
    res = validate_protocol_receipt(data, check_files_exist=True, base_dir=tmp_path)
    assert res["status"] == PROTOCOL_RECEIPT_INVALID
    assert any("raw_note not found" in err for err in res["errors"])
    assert any("wiki_note not found" in err for err in res["errors"])

    # Create notes
    (tmp_path / "kb" / "raw").mkdir(parents=True)
    (tmp_path / "kb" / "wiki").mkdir(parents=True)
    (tmp_path / data["stages"]["stage_1_research"]["raw_note"]).write_text("raw note")
    (tmp_path / data["stages"]["stage_2_kb_validation"]["wiki_note"]).write_text("wiki note")

    res2 = validate_protocol_receipt(data, check_files_exist=True, base_dir=tmp_path)
    assert res2["status"] == PROTOCOL_RECEIPT_VALID
