import os
import sys
from unittest.mock import patch

# Add src and root autobench folder to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from context_bench import (
    classify_retrieval_attempt,
    build_context_prompt,
    execution_stages,
    generate_haystack,
    run_context_test,
    summarize_context_runs,
)


def test_generate_haystack():
    needle = "SECRET_CODE = 'K7000-KEY-1234'"
    haystack = generate_haystack(target_tokens=500, needle=needle)

    assert needle in haystack
    assert len(haystack.splitlines()) >= 10
    assert "k7000" in haystack


def test_generate_haystack_places_needles_at_ordered_positions():
    needle = "SECRET_CODE = 'POSITION'"
    offsets = []
    for position in (0.10, 0.50, 0.90):
        lines = generate_haystack(1000, needle, position).splitlines()
        offsets.append(next(i for i, line in enumerate(lines) if needle in line))

    assert offsets[0] < offsets[1] < offsets[2]


def test_build_context_prompt_reserves_output_headroom():
    def count_tokens(_model_path, text, timeout=60):
        if "SECRET_MAINTENANCE_CODE =" not in text:
            return 190
        return len(text.split())

    with patch(
        "context_bench.Runner.count_local_tokens",
        side_effect=count_tokens,
    ):
        prompt, actual_tokens, budget, needle_offset = build_context_prompt(
            model_path="/tmp/test.gguf",
            context_size=512,
            max_tokens=128,
            utilization=0.90,
            needle_val="TEST-KEY",
        )

    assert "TEST-KEY" in prompt
    assert budget == 384
    assert actual_tokens <= budget
    assert needle_offset == 190
    assert actual_tokens + 128 <= 512


def test_build_context_prompt_honors_calibration_step_limit():
    with patch(
        "context_bench.Runner.count_local_tokens",
        side_effect=lambda _model, text, timeout=60: len(text.split()),
    ) as tokenizer:
        build_context_prompt(
            model_path="/tmp/test.gguf",
            context_size=512,
            max_tokens=64,
            utilization=0.90,
            needle_val="TEST-KEY",
            calibration_steps=3,
        )

    assert tokenizer.call_count <= 4


def test_build_context_prompt_validates_utilization():
    try:
        build_context_prompt(
            model_path="/tmp/test.gguf",
            context_size=512,
            max_tokens=128,
            utilization=1.1,
            needle_val="TEST-KEY",
        )
    except ValueError as exc:
        assert "utilization" in str(exc)
    else:
        raise AssertionError("invalid utilization must fail")


def test_retrieval_classification_distinguishes_verified_missed_and_inconclusive():
    assert classify_retrieval_attempt("TEST-KEY", "TEST-KEY", "SUCCESS", 3) == {
        "outcome": "VERIFIED", "reason": "exact_answer_present"
    }
    assert classify_retrieval_attempt("wrong", "TEST-KEY", "SUCCESS", 3) == {
        "outcome": "MISSED", "reason": "answer_absent"
    }
    assert classify_retrieval_attempt("", "TEST-KEY", "SUCCESS", 0) == {
        "outcome": "INCONCLUSIVE", "reason": "execution_or_generation_incomplete"
    }
    assert classify_retrieval_attempt("wrong", "TEST-KEY", "TIMEOUT", 3) == {
        "outcome": "INCONCLUSIVE", "reason": "execution_or_generation_incomplete"
    }


def test_execution_stages_keep_retrieval_separate():
    stages = execution_stages(
        {"success": True, "status": "SUCCESS", "prompt_speed_ts": 10.0},
        generated_tokens=12,
    )

    assert stages == {
        "load_ok": True,
        "context_allocated": True,
        "prefill_ok": True,
        "decode_ok": True,
    }


def test_summarize_context_runs_separates_context_limits():
    runs = [
        {
            "context_size": 512,
            "status": "SUCCESS",
            "retrieved": True,
            "prompt_ts": 10.0,
            "gen_ts": 20.0,
        },
        {
            "context_size": 512,
            "status": "SUCCESS",
            "retrieved": True,
            "prompt_ts": 9.0,
            "gen_ts": 19.0,
        },
        {
            "context_size": 1024,
            "status": "SUCCESS",
            "retrieved": False,
            "prompt_ts": 8.0,
            "gen_ts": 18.0,
        },
        {
            "context_size": 1024,
            "status": "SUCCESS",
            "retrieved": True,
            "prompt_ts": 8.0,
            "gen_ts": 18.0,
        },
    ]

    summary = summarize_context_runs(
        runs,
        reliability_threshold=0.80,
        min_prompt_ts=5.0,
        min_gen_ts=10.0,
    )

    assert summary["maximum_allocatable_context"] == 1024
    assert summary["maximum_reliable_retrieval_context"] == 512
    assert summary["maximum_operational_context"] == 512
    assert summary["task_quality"]["measured"] is False
    assert summary["rerun_required"] is False


def test_summarize_context_runs_flags_non_monotonic_result():
    runs = [
        {
            "context_size": 512,
            "status": "SUCCESS",
            "retrieved": True,
            "prompt_ts": 10.0,
            "gen_ts": 20.0,
        },
        {
            "context_size": 1024,
            "status": "SUCCESS",
            "retrieved": False,
            "prompt_ts": 10.0,
            "gen_ts": 20.0,
        },
        {
            "context_size": 2048,
            "status": "SUCCESS",
            "retrieved": True,
            "prompt_ts": 10.0,
            "gen_ts": 20.0,
        },
    ]

    summary = summarize_context_runs(
        runs,
        reliability_threshold=1.0,
        min_prompt_ts=1.0,
        min_gen_ts=1.0,
    )

    assert summary["rerun_required"] is True


def test_context_test_preserves_runner_failure_status():
    with patch(
        "context_bench.build_context_prompt",
        return_value=("prompt", 380, 384, 190),
    ), patch(
        "context_bench.Runner.run_local_vulkan",
        return_value={
            "success": False,
            "status": "UNSUPPORTED_BACKEND",
            "error": "split buffers unsupported",
            "return_code": 1,
            "elapsed_seconds": 0.25,
        },
    ):
        result = run_context_test(
            model_name="test",
            model_path="/tmp/test.gguf",
            context_size=512,
            device="Vulkan0,Vulkan1",
            ts_split="1,1",
        )

    assert result["status"] == "UNSUPPORTED_BACKEND"
    assert result["return_code"] == 1
    assert result["actual_prompt_tokens"] == 380
    assert result["reserved_output_tokens"] == 128
    assert result["needle_token_offset"] == 190
    assert result["execution"]["context_allocated"] is None
    assert result["retrieval"]["correct"] is False
    assert result["retrieved"] is False


def test_context_test_records_exact_generated_tokens():
    with patch(
        "context_bench.build_context_prompt",
        return_value=("prompt", 380, 384, 190),
    ), patch(
        "context_bench.Runner.run_local_vulkan",
        return_value={
            "success": True,
            "status": "SUCCESS",
            "response": '{"secret_code": "K7000-KEY-8842"}',
            "return_code": 0,
            "prompt_speed_ts": 10.0,
            "generation_speed_ts": 20.0,
            "elapsed_seconds": 1.5,
        },
    ), patch("context_bench.Runner.count_local_tokens", return_value=12) as tokenizer:
        result = run_context_test(
            model_name="test",
            model_path="/tmp/test.gguf",
            context_size=512,
            device="Vulkan0",
            ts_split=None,
            needle_position=0.90,
        )

    assert result["generated_tokens"] == 12
    assert all(call.kwargs["timeout"] == 60 for call in tokenizer.call_args_list)
    assert result["needle_position"] == 0.90
    assert result["context_utilization_pct"] == 74.22
    assert result["execution"]["decode_ok"] is True
    assert result["retrieval"]["correct"] is True
    assert result["performance"]["generation_speed_ts"] == 20.0
    assert result["retrieved"] is True
