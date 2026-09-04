import json
import pytest
from pathlib import Path
from scripts.run_issue41_qwen_q4km import (
    MODEL_NAME,
    EXPECTED_SIZE_BYTES,
    EXPECTED_SHA256,
    EXPECTED_JOB_COUNT,
    PRIMARY_TIMEOUT,
    issue_configurations,
    reviewed_plan,
    verify_artifact,
)


def test_issue_configurations():
    model = {"name": MODEL_NAME, "size_bytes": EXPECTED_SIZE_BYTES}
    configs = issue_configurations(model)
    assert len(configs) == 3
    assert configs[0]["device"] == "Vulkan0"
    assert configs[0]["split_mode"] == "none"
    assert configs[1]["device"] == "Vulkan1"
    assert configs[1]["split_mode"] == "none"
    assert configs[2]["device"] == "Vulkan0,Vulkan1"
    assert configs[2]["split_mode"] == "layer"
    assert configs[2]["tensor_split"] == "1,1"
    for c in configs:
        assert c["cache_type_k"] == "f16"
        assert c["cache_type_v"] == "f16"
        assert c["no_kv_offload"] is False


def test_reviewed_plan():
    model = {"name": MODEL_NAME, "size_bytes": EXPECTED_SIZE_BYTES}
    out_dir = Path("results/test")
    plan = reviewed_plan(model, out_dir)
    assert plan["schema_version"] == 1
    assert plan["issue"] == 41
    assert plan["expected_job_count"] == 3
    assert plan["authoritative"] is False
    assert plan["inference_authorized"] is False
    assert plan["policy"]["timeout_seconds"] == 600
    assert plan["policy"]["contexts_dual_layer"] == [1024, 2048, 4096]


def test_timeout_override_rejected():
    model = {"name": MODEL_NAME, "size_bytes": EXPECTED_SIZE_BYTES}
    out_dir = Path("results/test")
    with pytest.raises(ValueError, match="cannot override"):
        reviewed_plan(model, out_dir, timeout=1200)


def test_verify_artifact(tmp_path):
    f = tmp_path / MODEL_NAME
    f.write_bytes(b"bad content")
    with pytest.raises(ValueError, match="size mismatch"):
        verify_artifact(f)
