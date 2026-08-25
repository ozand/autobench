# Gemma-2-2B-IT-Q4_K_M Research Notes

- Source URL: https://huggingface.co/google/gemma-2-2b-it
- Source URL: https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md
- Retrieved via Surf CLI: 2026-08-25

## Model Identity
- Checkpoint: `google/gemma-2-2b-it`
- GGUF: `gemma-2-2b-it-Q4_K_M.gguf`
- Architecture: `gemma2`
- Parameters: approximately 2.61B
- Quantization: `Q4_K_M`
- Native context limit used for planning: 8192 tokens

## Runtime Applicability
- Vulkan is the target llama.cpp backend.
- Single-GPU load-only baselines are required because the weights are close to the 2GB per-device limit and leave limited KV capacity.
- Authoritative multi-GPU mode is Vulkan `-sm layer` with split `1,1`; tensor/row split is excluded.
- The exact artifact is present on k7000 and requires bounded load/context probes.

## Unresolved Assumptions
- Local converted GGUF behavior and context capacity require receipt-backed execution evidence.
- Historical rows contain incomplete success-like metrics and timeouts; they are non-authoritative.
