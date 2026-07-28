# AGENTS.md — AutoBench agent guidelines

## Workspace Overview
AutoBench is an automated LLM evaluation framework designed to benchmark GGUF models on local hardware.

## Guidelines
- Follow standard Python code style (PEP 8, Black formatting).
- Use `pytest` for running unit tests in `tests/`.
- Maintain strict deterministic evaluation logic and machine-readable output schemas.
- Do not commit large binary weights or local run raw artifacts to git.
