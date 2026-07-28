# AGENTS.md — AutoBench Agent Operating Guidelines

## Workspace Overview
AutoBench is an automated LLM evaluation framework designed to benchmark local GGUF models on dedicated GPU hardware (`k7000`).

## Development & Execution Paradigm (Option A)

1. **Local Development Scope**:
   - All code editing, dataset crafting (`datasets/`), and unit testing (`tests/`) MUST take place locally in `T:\Code\autobench`.
   - Before pushing or running remote tasks, run fast unit tests locally:
     ```bash
     python -m pytest
     ```

2. **Remote GPU Execution Scope**:
   - Heavy LLM benchmarks and hardware GPU evaluations MUST be executed remotely on host `k7000` (`opencode@100.67.171.58`).
   - Use the remote runner script to synchronize local code and trigger remote execution seamlessly:
     ```bash
     # Run a benchmark script on k7000
     python scripts/run_remote.py inventory_bench.py

     # Run custom commands or tests on k7000
     python scripts/run_remote.py --cmd "pytest"

     # Run benchmark and fetch back generated run manifests/results
     python scripts/run_remote.py context_bench.py --sync-results
     ```

3. **Git Hygiene**:
   - Do not commit large binary GGUF weights or raw temporary test artifacts.
   - Maintain PEP 8 code style and Black formatting.
