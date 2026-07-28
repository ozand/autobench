import sys
import os

# Add AutoBench root and src folders to path
base_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, "src"))
from judge import Judge
from run_bench import build_run_manifest


def test_validate_deterministic_max_length():
    constraints = {"max_length": 10}

    # Matches length <= 10
    res = Judge.validate_deterministic("12345", constraints)
    assert res["passed"] is True
    assert res["score"] == 1.0

    # Exceeds length
    res = Judge.validate_deterministic("12345678901", constraints)
    assert res["passed"] is False
    assert len(res["issues"]) == 1
    assert "Response length" in res["issues"][0]


def test_validate_deterministic_banned_phrases():
    constraints = {"banned_phrases": ["class"]}

    # No banned phrase
    res = Judge.validate_deterministic("def my_func(): pass", constraints)
    assert res["passed"] is True

    # Banned phrase exists
    res = Judge.validate_deterministic("class MyClass: pass", constraints)
    assert res["passed"] is False
    assert "Banned phrase detected" in res["issues"][0]


def test_validate_deterministic_required_regex():
    constraints = {"required_regex": ["def is_prime", "returns True"]}

    # Missing both
    res = Judge.validate_deterministic("hello world", constraints)
    assert res["passed"] is False
    assert len(res["issues"]) == 2

    # Matches both (case insensitive)
    res = Judge.validate_deterministic("Def Is_Prime(): returns true", constraints)
    assert res["passed"] is True
    assert len(res["issues"]) == 0


def test_validate_deterministic_json_schema():
    constraints = {
        "json_schema": {
            "type": "object",
            "properties": {
                "critical": {"type": "boolean"},
                "component": {"type": "string"},
                "exit_code": {"type": "integer"},
            },
            "required": ["critical", "component"],
        }
    }

    # Valid json matching schema
    res = Judge.validate_deterministic(
        '{"critical": true, "component": "CPU"}', constraints
    )
    assert res["passed"] is True

    # Valid json matching schema with markdown wrap
    res = Judge.validate_deterministic(
        '```json\n{"critical": true, "component": "CPU"}\n```', constraints
    )
    assert res["passed"] is True

    res = Judge.validate_deterministic(
        '```json\n{"critical": false, "component": "RAM", "exit_code": 0}\n```',
        constraints,
    )
    assert res["passed"] is True

    # Missing required property
    res = Judge.validate_deterministic('{"critical": true}', constraints)
    assert res["passed"] is False
    assert "Missing required parameter 'component'" in res["issues"][0]

    # Wrong type
    res = Judge.validate_deterministic(
        '{"critical": "yes", "component": "CPU"}', constraints
    )
    assert res["passed"] is False
    assert "expects boolean" in res["issues"][0]


def test_autobench_config_profiles():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    config_path = os.path.join(base_dir, "config.json")
    assert os.path.exists(config_path)

    import json
    with open(config_path, "r") as f:
        cfg = json.load(f)

    assert "active_profile" in cfg
    assert "profiles" in cfg
    assert "fast-local" in cfg["profiles"]
    assert "frontier-precision" in cfg["profiles"]
    assert cfg["profiles"]["fast-local"]["type"] == "local"
    assert cfg["profiles"]["frontier-precision"]["type"] == "frontier"

def test_build_run_manifest_uses_resolved_values():
    class Args:
        model = "local"
        dataset = "validation"
        context_length = 2048
        max_tokens = 128
        ts_split = "1,1"
        run_judge = False

    manifest = build_run_manifest(
        Args(),
        profile_name="dual-test",
        model_path="/resolved/model.gguf",
        device="Vulkan0,Vulkan1",
        first_result={"command_args": ["timeout", "180s", "llama-cli"]},
        provenance={"model_sha256": "abc123"},
    )

    assert manifest["profile"] == "dual-test"
    assert manifest["model_path"] == "/resolved/model.gguf"
    assert manifest["device"] == "Vulkan0,Vulkan1"
    assert manifest["context_length"] == 2048
    assert manifest["tensor_split"] == "1,1"
    assert manifest["command_args"][0] == "timeout"
    assert manifest["provenance"]["model_sha256"] == "abc123"


def test_strip_thinking():
    # 1. XML style
    text_xml = "<think>some internal thought process</think>Final Output"
    assert Judge.strip_thinking(text_xml) == "Final Output"

    # 2. Bracket style
    text_bracket = "[Start thinking]\nanalyse this\n[End thinking]Actual answer"
    assert Judge.strip_thinking(text_bracket) == "Actual answer"

    # 3. Raw thinking process style
    text_raw = "Thinking Process:\n1. analyze\n2. output\n\nCorrect Answer"
    assert Judge.strip_thinking(text_raw) == "Correct Answer"

    # 4. Truncated XML style
    text_truncated = "<think>incomplete thought process"
    assert Judge.strip_thinking(text_truncated) == ""

