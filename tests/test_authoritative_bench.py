import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from authoritative_bench import (
    FULL_POLICY,
    build_plan,
    configurations_for_model,
    discover_remote_models,
    execute_boundary,
    execute_performance,
    execute_quality,
    execute_retrieval,
    execute_smoke,
    execute_suite,
    fine_boundary_contexts,
    render_matrix,
    run_load_probe,
    select_smoke_models,
    write_artifacts,
)


def test_inventory_excludes_vocab_and_keeps_quantization_variants():
    completed = Mock(
        returncode=0,
        stdout=(
            "ggml-vocab-qwen2.gguf\t100\n"
            "model-Q4_K_M.gguf\t1000\n"
            "model-Q8_0.gguf\t2000\n"
        ),
        stderr="",
    )
    with patch("authoritative_bench.subprocess.run", return_value=completed):
        models = discover_remote_models()

    assert [model["name"] for model in models] == [
        "model-Q4_K_M.gguf",
        "model-Q8_0.gguf",
    ]


def test_configuration_policy_covers_device_asymmetry_and_load_only():
    fitting = configurations_for_model(1_000_000_000)
    oversized = configurations_for_model(3_000_000_000)

    assert [config["device"] for config in fitting] == ["Vulkan0", "Vulkan1"]
    assert {config["mode"] for config in fitting} == {"full"}
    assert oversized[0]["mode"] == "load_only"
    assert oversized[1]["device"] == "Vulkan1"
    assert oversized[2] == {
        "device": "Vulkan0,Vulkan1",
        "tensor_split": "1,1",
        "mode": "full",
    }


def test_full_plan_pins_required_repetitions_and_positions():
    plan = build_plan(
        [{"id": "model", "name": "model.gguf", "path": "/models/model.gguf", "size_bytes": 1}],
        "full",
    )

    assert plan["authoritative"] is False
    assert plan["execution_status"] == "planned_not_run"
    assert plan["policy"]["retrieval_positions"] == [0.10, 0.50, 0.90]
    assert plan["policy"]["retrieval_repetitions"] == 5
    assert plan["policy"]["performance_warmups"] == 1
    assert plan["policy"]["performance_repetitions"] == 3
    assert plan["policy"]["matched_prompt_tokens"] == 512
    assert plan["policy"]["matched_output_tokens"] == 64
    assert plan["policy"] == FULL_POLICY


def test_smoke_is_bounded_and_never_authoritative():
    models = [
        {"id": "small", "name": "small.gguf", "path": "/small", "size_bytes": 10},
        {"id": "large", "name": "large.gguf", "path": "/large", "size_bytes": 3_000_000_000},
        {"id": "other", "name": "other.gguf", "path": "/other", "size_bytes": 20},
    ]

    selected = select_smoke_models(models)
    plan = build_plan(selected, "smoke")

    assert [model["name"] for model in selected] == ["small.gguf", "large.gguf"]
    assert plan["authoritative"] is False
    assert plan["policy"]["coarse_contexts"] == [512]
    assert plan["policy"]["retrieval_repetitions"] == 1


def test_fine_boundary_contexts_add_non_power_of_two_steps():
    assert fine_boundary_contexts(1024, 2048, 256) == [1280, 1536, 1792]


def test_execute_suite_combines_all_stage_results():
    plan = build_plan(
        [{"id": "model", "name": "model.gguf", "path": "/model", "size_bytes": 1}],
        "suite",
    )
    plan["models"][0]["configurations"] = [
        {"device": "Vulkan0", "tensor_split": None, "mode": "full"}
    ]
    boundary = {
        "status": "SUCCESS",
        "maximum_allocatable_context": 1024,
        "first_failed_context": 1280,
        "elapsed_seconds": 1.0,
    }
    retrieval = {
        "status": "SUCCESS",
        "retrieved": True,
        "retrieval_rate": 0.80,
        "elapsed_seconds": 2.0,
    }
    performance = {
        "status": "SUCCESS",
        "prompt_ts": 12.0,
        "gen_ts": 22.0,
        "elapsed_seconds": 3.0,
    }
    quality = {
        "status": "SUCCESS",
        "task_pass_rate": 0.50,
        "elapsed_seconds": 4.0,
    }

    def result_plan(stage_result):
        child = build_plan(
            [{"id": "model", "name": "model.gguf", "path": "/model", "size_bytes": 1}],
            "child",
        )
        child["models"][0]["configurations"][0]["result"] = stage_result
        return child

    with patch("authoritative_bench.execute_boundary", return_value=result_plan(boundary)), patch(
        "authoritative_bench.execute_retrieval", return_value=result_plan(retrieval)
    ), patch(
        "authoritative_bench.execute_performance", return_value=result_plan(performance)
    ), patch("authoritative_bench.execute_quality", return_value=result_plan(quality)):
        executed = execute_suite(
            plan,
            timeout=15,
            context_sizes=[512, 1024, 2048],
            boundary_step=256,
            retrieval_repetitions=5,
            reliability_threshold=0.80,
            performance_context=1024,
            prompt_tokens=512,
            output_tokens=64,
            warmups=1,
            performance_repetitions=3,
            dataset_dir="/dataset",
            max_tasks=2,
        )

    result = executed["models"][0]["configurations"][0]["result"]
    assert result["status"] == "SUCCESS"
    assert result["maximum_allocatable_context"] == 1024
    assert result["allocatable_is_lower_bound"] is False
    assert result["maximum_reliable_context"] == 1024
    assert result["retrieval_rate"] == 0.80
    assert result["prompt_ts"] == 12.0
    assert result["task_pass_rate"] == 0.50
    assert result["elapsed_seconds"] == 10.0
    assert executed["execution_status"] == "completed_suite"
    assert executed["authoritative"] is False


def test_execute_suite_stops_if_boundary_has_no_allocatable_context():
    plan = build_plan(
        [{"id": "model", "name": "model.gguf", "path": "/model", "size_bytes": 1}],
        "suite",
    )
    plan["models"][0]["configurations"] = [
        {"device": "Vulkan0", "tensor_split": None, "mode": "full"}
    ]
    child = build_plan(
        [{"id": "model", "name": "model.gguf", "path": "/model", "size_bytes": 1}],
        "boundary",
    )
    child["models"][0]["configurations"][0]["result"] = {
        "status": "INCONCLUSIVE",
        "maximum_allocatable_context": 0,
        "first_failed_context": None,
    }
    with patch("authoritative_bench.execute_boundary", return_value=child), patch(
        "authoritative_bench.execute_retrieval"
    ) as retrieval:
        executed = execute_suite(
            plan, 5, [512], 256, 1, 0.8, 512, 240, 32, 0, 1, "/dataset", 1
        )

    result = executed["models"][0]["configurations"][0]["result"]
    assert result["status"] == "BLOCKED_BY_BOUNDARY"
    retrieval.assert_not_called()


def test_execute_suite_blocks_if_workload_exceeds_measured_capacity():
    plan = build_plan(
        [{"id": "model", "name": "model.gguf", "path": "/model", "size_bytes": 1}],
        "suite",
    )
    plan["models"][0]["configurations"] = [
        {"device": "Vulkan0", "tensor_split": None, "mode": "full"}
    ]
    child = build_plan(
        [{"id": "model", "name": "model.gguf", "path": "/model", "size_bytes": 1}],
        "boundary",
    )
    child["models"][0]["configurations"][0]["result"] = {
        "status": "SUCCESS",
        "maximum_allocatable_context": 512,
        "first_failed_context": 768,
    }
    with patch("authoritative_bench.execute_boundary", return_value=child), patch(
        "authoritative_bench.execute_retrieval"
    ) as retrieval:
        executed = execute_suite(
            plan, 5, [512, 768], 128, 1, 0.8, 1024, 500, 64, 0, 1, "/dataset", 1
        )

    result = executed["models"][0]["configurations"][0]["result"]
    assert result["status"] == "BLOCKED_BY_CONTEXT_BUDGET"
    retrieval.assert_not_called()


def test_execute_performance_discards_warmup_and_averages_measured_runs():
    plan = build_plan(
        [{"id": "model", "name": "model.gguf", "path": "/model", "size_bytes": 1}],
        "performance",
    )
    plan["models"][0]["configurations"] = [
        {"device": "Vulkan0", "tensor_split": None, "mode": "full"}
    ]
    runs = [
        {"success": True, "status": "SUCCESS", "prompt_speed_ts": 100.0, "generation_speed_ts": 100.0, "elapsed_seconds": 99.0, "return_code": 0, "command_args": ["warmup"]},
        {"success": True, "status": "SUCCESS", "prompt_speed_ts": 10.0, "generation_speed_ts": 20.0, "elapsed_seconds": 2.0, "return_code": 0, "command_args": ["one"]},
        {"success": True, "status": "SUCCESS", "prompt_speed_ts": 12.0, "generation_speed_ts": 22.0, "elapsed_seconds": 4.0, "return_code": 0, "command_args": ["two"]},
        {"success": True, "status": "SUCCESS", "prompt_speed_ts": 14.0, "generation_speed_ts": 24.0, "elapsed_seconds": 6.0, "return_code": 0, "command_args": ["three"]},
    ]
    with patch(
        "authoritative_bench.build_context_prompt",
        return_value=("matched prompt", 500, 512, 250),
    ), patch(
        "authoritative_bench.Runner.run_local_vulkan", side_effect=runs
    ) as runner:
        executed = execute_performance(
            plan,
            timeout=15,
            context_size=1024,
            prompt_tokens=512,
            output_tokens=64,
            warmups=1,
            repetitions=3,
        )

    result = executed["models"][0]["configurations"][0]["result"]
    assert runner.call_count == 4
    assert result["warmups_discarded"] == 1
    assert result["measured_repetitions"] == 3
    assert result["prompt_ts"] == 12.0
    assert result["gen_ts"] == 22.0
    assert result["elapsed_seconds"] == 4.0
    assert result["actual_prompt_tokens"] == 500
    assert executed["execution_status"] == "completed_performance"
    assert executed["authoritative"] is False


def test_execute_performance_preserves_tokenizer_failure():
    plan = build_plan(
        [{"id": "model", "name": "model.gguf", "path": "/model", "size_bytes": 1}],
        "performance",
    )
    plan["models"][0]["configurations"] = [
        {"device": "Vulkan0", "tensor_split": None, "mode": "full"}
    ]
    with patch(
        "authoritative_bench.build_context_prompt",
        side_effect=RuntimeError("tokenizer unavailable"),
    ):
        executed = execute_performance(
            plan,
            timeout=5,
            context_size=1024,
            prompt_tokens=512,
            output_tokens=64,
            warmups=1,
            repetitions=3,
        )

    result = executed["models"][0]["configurations"][0]["result"]
    assert result["status"] == "TOKENIZER_ERROR"
    assert result["successful_measurements"] == 0
    assert result["runs"] == []
    assert executed["authoritative"] is False


def test_execute_performance_preserves_failed_measurement():
    plan = build_plan(
        [{"id": "model", "name": "model.gguf", "path": "/model", "size_bytes": 1}],
        "performance",
    )
    plan["models"][0]["configurations"] = [
        {"device": "Vulkan0", "tensor_split": None, "mode": "full"}
    ]
    with patch(
        "authoritative_bench.build_context_prompt",
        return_value=("matched prompt", 500, 512, 250),
    ), patch(
        "authoritative_bench.Runner.run_local_vulkan",
        side_effect=[
            {"success": True, "status": "SUCCESS", "prompt_speed_ts": 10.0, "generation_speed_ts": 20.0, "elapsed_seconds": 2.0, "return_code": 0, "command_args": []},
            {"success": False, "status": "REMOTE_TIMEOUT", "elapsed_seconds": 5.0, "return_code": 124, "command_args": []},
        ],
    ):
        executed = execute_performance(
            plan,
            timeout=5,
            context_size=1024,
            prompt_tokens=512,
            output_tokens=64,
            warmups=0,
            repetitions=2,
        )

    result = executed["models"][0]["configurations"][0]["result"]
    assert result["status"] == "PARTIAL_FAILURE"
    assert result["successful_measurements"] == 1
    assert result["runs"][1]["status"] == "REMOTE_TIMEOUT"


def test_execute_quality_uses_fixed_order_and_deterministic_pass_rate():
    plan = build_plan(
        [{"id": "model", "name": "model.gguf", "path": "/model", "size_bytes": 1}],
        "quality",
    )
    plan["models"][0]["configurations"] = [
        {"device": "Vulkan0", "tensor_split": None, "mode": "full"}
    ]
    tasks = [
        {"id": "b", "task": "analysis", "prompt": "B", "constraints": {}},
        {"id": "a", "task": "writing", "prompt": "A", "constraints": {}},
        {"id": "c", "task": "tools", "prompt": "C", "constraints": {}},
    ]
    runs = [
        {
            "success": True,
            "status": "SUCCESS",
            "response": "good",
            "elapsed_seconds": 1.0,
            "prompt_speed_ts": 2.0,
            "generation_speed_ts": 3.0,
            "return_code": 0,
            "command_args": ["llama-cli"],
        },
        {
            "success": True,
            "status": "SUCCESS",
            "response": "bad",
            "elapsed_seconds": 2.0,
            "prompt_speed_ts": 2.0,
            "generation_speed_ts": 3.0,
            "return_code": 0,
            "command_args": ["llama-cli"],
        },
    ]
    with patch("authoritative_bench.load_dataset", return_value=tasks), patch(
        "authoritative_bench.Runner.run_local_vulkan", side_effect=runs
    ) as runner, patch(
        "authoritative_bench.Judge.validate_deterministic",
        side_effect=[
            {"passed": True, "score": 1.0, "issues": []},
            {"passed": False, "score": 0.0, "issues": ["failed"]},
        ],
    ):
        executed = execute_quality(
            plan,
            timeout=15,
            dataset_dir="/dataset",
            max_tasks=2,
            context_size=1024,
            max_tokens=64,
        )

    result = executed["models"][0]["configurations"][0]["result"]
    assert [task["task_id"] for task in result["tasks"]] == ["a", "b"]
    assert [call.kwargs["prompt"] for call in runner.call_args_list] == ["A", "B"]
    assert result["task_pass_rate"] == 0.5
    assert result["tasks_passed"] == 1
    assert result["tasks_total"] == 2
    assert executed["execution_status"] == "completed_quality"
    assert executed["authoritative"] is False


def test_execute_quality_counts_execution_failure_as_failed_task():
    plan = build_plan(
        [{"id": "model", "name": "model.gguf", "path": "/model", "size_bytes": 1}],
        "quality",
    )
    plan["models"][0]["configurations"] = [
        {"device": "Vulkan0", "tensor_split": None, "mode": "full"}
    ]
    with patch(
        "authoritative_bench.load_dataset",
        return_value=[{"id": "a", "task": "analysis", "prompt": "A", "constraints": {}}],
    ), patch(
        "authoritative_bench.Runner.run_local_vulkan",
        return_value={
            "success": False,
            "status": "REMOTE_TIMEOUT",
            "error": "timed out",
            "elapsed_seconds": 5.0,
            "return_code": 124,
            "command_args": ["llama-cli"],
        },
    ):
        executed = execute_quality(
            plan,
            timeout=5,
            dataset_dir="/dataset",
            max_tasks=1,
            context_size=512,
            max_tokens=32,
        )

    result = executed["models"][0]["configurations"][0]["result"]
    assert result["status"] == "PARTIAL_FAILURE"
    assert result["task_pass_rate"] == 0.0
    assert result["tasks"][0]["status"] == "REMOTE_TIMEOUT"


def test_execute_boundary_stops_coarse_and_inserts_fine_probes():
    plan = build_plan(
        [{"id": "model", "name": "model.gguf", "path": "/model", "size_bytes": 1}],
        "boundary",
    )
    plan["models"][0]["configurations"] = [
        {"device": "Vulkan0", "tensor_split": None, "mode": "full"}
    ]

    def result_for_context(**kwargs):
        context = kwargs["context_size"]
        return {
            "status": "SUCCESS" if context <= 1280 else "OOM",
            "context_size": context,
            "retrieved": False,
            "prompt_ts": 1.0,
            "gen_ts": 2.0,
            "elapsed_seconds": 1.0,
        }

    with patch(
        "authoritative_bench.run_context_test", side_effect=result_for_context
    ) as runner:
        executed = execute_boundary(
            plan,
            timeout=15,
            context_sizes=[512, 1024, 2048, 4096],
            boundary_step=256,
        )

    result = executed["models"][0]["configurations"][0]["result"]
    tested = [call.kwargs["context_size"] for call in runner.call_args_list]
    assert tested == [512, 1024, 2048, 1280, 1536]
    assert result["maximum_allocatable_context"] == 1280
    assert result["first_failed_context"] == 1536
    assert [probe["phase"] for probe in result["probes"]] == [
        "coarse",
        "coarse",
        "coarse",
        "fine",
        "fine",
    ]
    assert executed["execution_status"] == "completed_boundary"
    assert executed["authoritative"] is False


def test_execute_boundary_is_inconclusive_on_non_capacity_failure():
    plan = build_plan(
        [{"id": "model", "name": "model.gguf", "path": "/model", "size_bytes": 1}],
        "boundary",
    )
    plan["models"][0]["configurations"] = [
        {"device": "Vulkan0", "tensor_split": None, "mode": "full"}
    ]
    with patch(
        "authoritative_bench.run_context_test",
        side_effect=[
            {
                "status": "SUCCESS",
                "context_size": 512,
                "retrieved": False,
                "prompt_ts": 1.0,
                "gen_ts": 2.0,
                "elapsed_seconds": 1.0,
            },
            {
                "status": "SSH_TIMEOUT",
                "context_size": 1024,
                "retrieved": False,
                "prompt_ts": 0.0,
                "gen_ts": 0.0,
                "elapsed_seconds": 5.0,
            },
        ],
    ):
        executed = execute_boundary(
            plan,
            timeout=5,
            context_sizes=[512, 1024, 2048],
            boundary_step=256,
        )

    result = executed["models"][0]["configurations"][0]["result"]
    assert result["status"] == "INCONCLUSIVE"
    assert result["inconclusive_status"] == "SSH_TIMEOUT"
    assert result["maximum_allocatable_context"] == 512
    assert result["first_failed_context"] is None


def test_execute_boundary_keeps_capacity_separate_from_retrieval():
    plan = build_plan(
        [{"id": "model", "name": "model.gguf", "path": "/model", "size_bytes": 1}],
        "boundary",
    )
    plan["models"][0]["configurations"] = [
        {"device": "Vulkan0", "tensor_split": None, "mode": "full"}
    ]
    with patch(
        "authoritative_bench.run_context_test",
        return_value={
            "status": "SUCCESS",
            "context_size": 512,
            "retrieved": False,
            "prompt_ts": 1.0,
            "gen_ts": 2.0,
            "elapsed_seconds": 1.0,
        },
    ):
        executed = execute_boundary(
            plan,
            timeout=5,
            context_sizes=[512],
            boundary_step=256,
        )

    result = executed["models"][0]["configurations"][0]["result"]
    assert result["maximum_allocatable_context"] == 512
    assert result["retrieved"] is False


def test_execute_retrieval_runs_all_positions_and_repetitions():
    plan = build_plan(
        [{"id": "model", "name": "model.gguf", "path": "/model", "size_bytes": 1}],
        "retrieval",
    )
    plan["models"][0]["configurations"] = [
        {"device": "Vulkan0", "tensor_split": None, "mode": "full"}
    ]
    results = [
        {
            "status": "SUCCESS",
            "context_size": 512,
            "retrieved": index != 5,
            "prompt_ts": 10.0,
            "gen_ts": 20.0,
            "elapsed_seconds": 1.0,
        }
        for index in range(6)
    ]
    with patch("authoritative_bench.run_context_test", side_effect=results) as runner:
        executed = execute_retrieval(
            plan,
            timeout=15,
            context_size=512,
            repetitions=2,
            reliability_threshold=0.80,
        )

    result = executed["models"][0]["configurations"][0]["result"]
    assert runner.call_count == 6
    assert [call.kwargs["needle_position"] for call in runner.call_args_list] == [
        0.10,
        0.10,
        0.50,
        0.50,
        0.90,
        0.90,
    ]
    assert result["attempts_expected"] == 6
    assert result["retrieval_rate"] == 5 / 6
    assert result["retrieved"] is True
    assert executed["execution_status"] == "completed_retrieval"
    assert executed["authoritative"] is False


def test_execute_retrieval_marks_partial_failure_but_preserves_attempts():
    plan = build_plan(
        [{"id": "model", "name": "model.gguf", "path": "/model", "size_bytes": 1}],
        "retrieval",
    )
    plan["models"][0]["configurations"] = [
        {"device": "Vulkan0", "tensor_split": None, "mode": "full"}
    ]
    with patch(
        "authoritative_bench.run_context_test",
        side_effect=RuntimeError("tokenizer down"),
    ):
        executed = execute_retrieval(
            plan,
            timeout=5,
            context_size=512,
            repetitions=1,
            reliability_threshold=0.80,
        )

    result = executed["models"][0]["configurations"][0]["result"]
    assert result["status"] == "PARTIAL_FAILURE"
    assert result["attempts_completed"] == 3
    assert all(attempt["status"] == "TOKENIZER_ERROR" for attempt in result["attempts"])


def test_execute_smoke_records_measured_and_load_only_results():
    models = [
        {"id": "small", "name": "small.gguf", "path": "/small", "size_bytes": 10},
        {"id": "large", "name": "large.gguf", "path": "/large", "size_bytes": 3_000_000_000},
    ]
    plan = build_plan(models, "smoke")
    retrieval_result = {
        "status": "SUCCESS",
        "context_size": 512,
        "retrieved": True,
        "prompt_ts": 5.0,
        "gen_ts": 10.0,
        "elapsed_seconds": 2.0,
    }
    load_result = {
        "stage": "load_probe",
        "status": "SUCCESS",
        "load_ok": True,
        "elapsed_seconds": 1.0,
        "return_code": 0,
        "command_args": ["llama-cli"],
    }

    with patch(
        "authoritative_bench.run_context_test", return_value=retrieval_result.copy()
    ) as context_run, patch(
        "authoritative_bench.run_load_probe", return_value=load_result
    ) as load_run:
        executed = execute_smoke(plan, timeout=17)

    assert executed["execution_status"] == "completed_smoke"
    assert context_run.call_count == 3
    assert load_run.call_count == 2
    assert executed["models"][0]["configurations"][0]["result"]["stage"] == (
        "retrieval_performance"
    )


def test_execute_smoke_preserves_tokenizer_failure_and_continues():
    plan = build_plan(
        [{"id": "small", "name": "small.gguf", "path": "/small", "size_bytes": 10}],
        "smoke",
    )
    with patch(
        "authoritative_bench.run_context_test",
        side_effect=[RuntimeError("tokenizer unavailable"), {
            "status": "SUCCESS",
            "context_size": 512,
            "retrieved": False,
            "prompt_ts": 1.0,
            "gen_ts": 2.0,
            "elapsed_seconds": 3.0,
        }],
    ):
        executed = execute_smoke(plan, timeout=9)

    results = [config["result"] for config in executed["models"][0]["configurations"]]
    assert results[0]["status"] == "TOKENIZER_ERROR"
    assert results[1]["status"] == "SUCCESS"
    assert executed["execution_status"] == "completed_smoke"


def test_load_probe_uses_bounded_minimal_inference():
    model = {"path": "/model.gguf"}
    config = {"device": "Vulkan0", "tensor_split": None}
    with patch(
        "authoritative_bench.Runner.run_local_vulkan",
        return_value={
            "success": False,
            "status": "OOM",
            "elapsed_seconds": 1.5,
            "return_code": 1,
            "command_args": ["llama-cli"],
        },
    ) as runner:
        result = run_load_probe(model, config, timeout=11)

    assert result["status"] == "OOM"
    assert result["load_ok"] is False
    assert runner.call_args.kwargs["context_length"] == 128
    assert runner.call_args.kwargs["max_tokens"] == 1
    assert runner.call_args.kwargs["timeout"] == 11


def test_renderer_has_required_columns_and_manifest_for_every_row(tmp_path: Path):
    plan = build_plan(
        [{"id": "model", "name": "model.gguf", "path": "/models/model.gguf", "size_bytes": 1}],
        "smoke",
    )
    text = render_matrix(plan, "manifest.json")

    for heading in (
        "Cache",
        "Allocatable context",
        "Reliable context",
        "Retrieval rate",
        "Prompt t/s",
        "Generation t/s",
        "Elapsed",
        "Task pass rate",
        "Manifest",
    ):
        assert heading in text
    assert "Smoke / Non-Authoritative" in text
    assert text.count("`manifest.json`") == 3

    manifest, report = write_artifacts(plan, tmp_path)
    assert manifest.exists()
    assert report.exists()


def test_renderer_shows_smoke_measurements_without_claiming_task_quality():
    plan = build_plan(
        [{"id": "model", "name": "model.gguf", "path": "/model", "size_bytes": 1}],
        "smoke",
    )
    plan["execution_status"] = "completed_smoke"
    plan["models"][0]["configurations"][0]["result"] = {
        "stage": "retrieval_performance",
        "status": "SUCCESS",
        "context_size": 512,
        "retrieved": True,
        "prompt_ts": 7.5,
        "gen_ts": 12.5,
        "elapsed_seconds": 3.0,
    }

    text = render_matrix(plan, "smoke.json")
    assert "| 512 | 512 | 100% | 7.5 | 12.5 | 3.0s | not_run |" in text


def test_renderer_shows_complete_suite_row():
    plan = build_plan(
        [{"id": "model", "name": "model.gguf", "path": "/model", "size_bytes": 1}],
        "suite",
    )
    plan["models"][0]["configurations"][0]["result"] = {
        "stage": "suite",
        "status": "SUCCESS",
        "maximum_allocatable_context": 1024,
        "maximum_reliable_context": 512,
        "allocatable_is_lower_bound": False,
        "retrieval_rate": 0.80,
        "prompt_ts": 12.0,
        "gen_ts": 22.0,
        "elapsed_seconds": 40.0,
        "task_pass_rate": 0.50,
    }

    text = render_matrix(plan, "suite.json")
    assert "End-to-End Suite / Non-Authoritative" in text
    assert "| 1024 | 512 | 80% | 12.0 | 22.0 | 40.0s | 50% |" in text


def test_renderer_shows_matched_performance_only():
    plan = build_plan(
        [{"id": "model", "name": "model.gguf", "path": "/model", "size_bytes": 1}],
        "performance",
    )
    plan["models"][0]["configurations"][0]["result"] = {
        "stage": "performance",
        "status": "SUCCESS",
        "prompt_ts": 12.0,
        "gen_ts": 22.0,
        "elapsed_seconds": 4.0,
    }

    text = render_matrix(plan, "performance.json")
    assert "Matched Performance / Non-Authoritative" in text
    assert "| not_run | not_run | not_run | 12.0 | 22.0 | 4.0s | not_run |" in text


def test_renderer_shows_quality_without_claiming_context_or_performance():
    plan = build_plan(
        [{"id": "model", "name": "model.gguf", "path": "/model", "size_bytes": 1}],
        "quality",
    )
    plan["models"][0]["configurations"][0]["result"] = {
        "stage": "quality",
        "status": "SUCCESS",
        "task_pass_rate": 0.5,
        "elapsed_seconds": 12.0,
    }

    text = render_matrix(plan, "quality.json")
    assert "Task Quality / Non-Authoritative" in text
    assert "| not_run | not_run | not_run | not_run | not_run | 12.0s | 50% |" in text


def test_renderer_shows_boundary_without_claiming_retrieval():
    plan = build_plan(
        [{"id": "model", "name": "model.gguf", "path": "/model", "size_bytes": 1}],
        "boundary",
    )
    plan["models"][0]["configurations"][0]["result"] = {
        "stage": "boundary",
        "status": "SUCCESS",
        "context_size": 1280,
        "maximum_allocatable_context": 1280,
        "retrieved": False,
        "elapsed_seconds": 30.0,
    }

    text = render_matrix(plan, "boundary.json")
    assert "Context Boundary / Non-Authoritative" in text
    assert "| >=1280 | not_run | not_run | not_run | not_run | 30.0s | not_run |" in text


def test_renderer_shows_aggregate_retrieval_rate():
    plan = build_plan(
        [{"id": "model", "name": "model.gguf", "path": "/model", "size_bytes": 1}],
        "retrieval",
    )
    plan["models"][0]["configurations"][0]["result"] = {
        "stage": "repeated_retrieval",
        "status": "SUCCESS",
        "context_size": 1024,
        "retrieved": True,
        "retrieval_rate": 0.80,
        "prompt_ts": 5.0,
        "gen_ts": 9.0,
        "elapsed_seconds": 30.0,
    }

    text = render_matrix(plan, "retrieval.json")
    assert "| 1024 | 1024 | 80% | 5.0 | 9.0 | 30.0s | not_run |" in text
    assert "Retrieval Slice / Non-Authoritative" in text


def test_full_plan_report_is_not_claimed_as_authoritative():
    plan = build_plan(
        [{"id": "model", "name": "model.gguf", "path": "/model", "size_bytes": 1}],
        "full",
    )

    text = render_matrix(plan, "full.json")
    assert "Planned / Non-Authoritative" in text
    assert "not_run" in text
