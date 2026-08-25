# Qwen3.5-2B-Q4_K_M Research Notes

- Source URL: https://huggingface.co/Qwen/Qwen3.5-2B
- Source URL: https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md
- Retrieved via Surf CLI: 2026-08-25

## Model Identity
- Checkpoint: `Qwen/Qwen3.5-2B`
- GGUF: `Qwen3.5-2B-Q4_K_M.gguf`
- Architecture: `qwen35`
- Parameters: approximately 2.45B
- Quantization: `Q4_K_M`
- Native context limit used for planning: 32768 tokens

## Runtime Applicability
- Vulkan is the target llama.cpp backend.
- Multi-GPU authoritative mode is Vulkan `-sm layer` with split `1,1`; tensor/row split is excluded.
- The exact artifact is present on k7000.
- Single-GPU capacity and context limits require bounded local probes; upstream claims do not establish local fit.

## Unresolved Assumptions
- Converted GGUF runtime behavior and tokenizer compatibility require receipt-backed execution evidence.
- Historical rows contain only SUCCESS-like zero/incomplete metrics and timeouts; they are non-authoritative.
