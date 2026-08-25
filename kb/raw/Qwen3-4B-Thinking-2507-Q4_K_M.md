# Qwen3-4B-Thinking-2507-Q4_K_M Research Notes

- Source URL: https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507
- Source URL: https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md
- Retrieved via Surf CLI: 2026-08-24

## Model Identity
- Checkpoint: `Qwen/Qwen3-4B-Thinking-2507`
- GGUF: `Qwen3-4B-Thinking-2507-Q4_K_M.gguf`
- Quantization: `Q4_K_M`
- Model class: Qwen3 4B thinking model

## Runtime Applicability
- The local execution path is llama.cpp GGUF through Vulkan.
- Authoritative multi-GPU testing uses Vulkan `-sm layer`; tensor/row split is excluded.
- The exact GGUF is present on the k7000 model store and requires bounded load/context validation.
- Upstream model metadata does not establish local VRAM fit or context capacity.

## Unresolved Assumptions
- Exact converted GGUF architecture metadata and local context capacity require receipt-backed probes.
- Historical rows contain only boundary/preflight failures and provide no trustworthy speed or Retrieval metrics.
