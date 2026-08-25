# Qwen3.5-4B-Q4_K_M Research Notes

- Source URL: https://huggingface.co/Qwen/Qwen3.5-4B
- Source URL: https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md
- Retrieved via Surf CLI: 2026-08-25

## Model Identity
- Checkpoint: `Qwen/Qwen3.5-4B`
- GGUF: `Qwen3.5-4B-Q4_K_M.gguf`
- Quantization: `Q4_K_M`
- Model class: Qwen3.5 4B class

## Runtime Applicability
- Target runtime: llama.cpp GGUF through Vulkan.
- Single-GPU configurations are load-only when the artifact exceeds the local fit threshold.
- Authoritative multi-GPU mode is Vulkan `-sm layer` with split `1,1`; tensor/row split is excluded.
- Exact local artifact is present on k7000 and requires bounded load/context probing.

## Unresolved Assumptions
- Converted GGUF architecture/runtime compatibility and local context capacity require receipt-backed evidence.
- Historical rows contain only preflight/boundary failures and are non-authoritative.
