# AutoBench: Local GGUF LLM Evaluation Framework

AutoBench is an automated evaluation framework for benchmarking local GGUF Large Language Models running on hardware backends (e.g., `llama.cpp` over Vulkan/CUDA).

## Architecture

- **`authoritative_bench.py`**: Executes structured benchmark matrices across capacity, retrieval, performance, and task quality.
- **`inventory_bench.py`**: Resumable inventory runner for candidate model directories.
- **`context_bench.py`**: Evaluates context window scaling and VRAM/RAM OOM boundaries.
- **`compare_models.py` / `fetch_models.py`**: Benchmark comparison and model acquisition tools.
- **`src/runner.py`**: Robust `llama-cli` execution wrapper with failure classification.
- **`src/judge.py`**: Quality evaluation via deterministic schema checks and LLM-as-a-Judge.

## Quickstart

```bash
# Run dry-run / plan
python authoritative_bench.py --plan-only

# Execute smoke benchmark
python authoritative_bench.py --smoke
```

## License

MIT
