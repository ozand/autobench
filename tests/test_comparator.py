import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from compare_models import generate_markdown_matrix


def test_generate_markdown_matrix_structure():
    mock_runs = [
        {
            "metadata": {
                "model": "local",
                "model_path": "/home/opencode/llama.cpp/models/qwen2.5-0.5b-instruct-q8_0.gguf",
                "device": "Vulkan0",
                "dataset": "validation",
            },
            "stats": {
                "total_elapsed_seconds": 12.34,
                "deterministic_pass_rate": 60.0,
                "avg_prompt_speed_ts": 140.5,
                "avg_generation_speed_ts": 35.2,
                "avg_judge_score": 0.0,
            },
        },
        {
            "metadata": {
                "model": "cl/gemini-2.5-flash",
                "device": "Vulkan0",
                "dataset": "validation",
            },
            "stats": {
                "total_elapsed_seconds": 18.5,
                "deterministic_pass_rate": 100.0,
                "avg_prompt_speed_ts": 2500.0,
                "avg_generation_speed_ts": 500.0,
                "avg_judge_score": 4.8,
            },
        },
    ]

    matrix = generate_markdown_matrix(mock_runs)

    # Assert formatting constructs
    assert "| Target Model / Device | Pass Rate (%) |" in matrix
    assert "`qwen2.5-0.5b-instruct-q8_0.gguf` (Vulkan0)" in matrix
    assert "`cl/gemini-2.5-flash`" in matrix
    assert "60.0%" in matrix
    assert "100.0%" in matrix
    assert "4.80 / 5.0" in matrix
    assert "12.34s" in matrix
    assert "18.50s" in matrix
