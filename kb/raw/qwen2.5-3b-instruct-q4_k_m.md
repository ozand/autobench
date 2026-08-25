# Qwen2.5-3B-Instruct-Q4_K_M Research Notes

- Source URL: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
- Source URL: https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md
- Retrieved via Surf CLI: 2026-08-25

## Model Identity
- Checkpoint: `Qwen/Qwen2.5-3B-Instruct`
- GGUF: `qwen2.5-3b-instruct-q4_k_m.gguf`
- Quantization: `Q4_K_M`
- Model class: Qwen2.5 3B instruct

## Runtime Applicability
- Target runtime: llama.cpp GGUF through Vulkan.
- Single-GPU baselines are load-only when the artifact exceeds the local fit threshold.
- Authoritative multi-GPU mode is Vulkan `-sm layer` with split `1,1`; tensor/row split is excluded.
- Exact artifact is present on k7000 and requires bounded load/context evidence.

## Unresolved Assumptions
- Converted GGUF runtime and local context capacity require receipt-backed probes.
- Historical rows are partial failures with zero metrics and are non-authoritative.
