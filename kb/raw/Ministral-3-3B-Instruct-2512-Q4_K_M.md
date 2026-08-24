# Ministral-3-3B-Instruct-2512-Q4_K_M Research Notes

- Source URL: https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512
- Source URL: https://github.com/ggml-org/llama.cpp/wiki/Home
- Source URL: https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md
- Retrieved via Surf CLI: 2026-08-24

## Model Identity
- Checkpoint: `mistralai/Ministral-3-3B-Instruct-2512`
- GGUF: `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf`
- Quantization: `Q4_K_M`
- Model family: Ministral 3 instruct, 3B class

## Runtime Applicability
- llama.cpp provides the Vulkan backend and standard GGUF execution path.
- The local Vulkan policy uses `-sm layer` for dual-GPU execution; tensor/row split is excluded from authoritative Vulkan results.
- The exact local GGUF identity and size must be verified on k7000 before inference.
- The declared upstream context and generation limits do not establish local VRAM capacity; bounded context probes remain required.

## Unresolved Assumptions
- Exact architecture metadata and context capacity for this GGUF conversion require local receipt and bounded runtime confirmation.
- The prior record contains only non-authoritative preflight/partial-failure rows and does not establish speed or Retrieval metrics.
