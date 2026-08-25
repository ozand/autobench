# Gemma-4-E4B-it-Q4_K_M Research Notes

- Source URL: https://huggingface.co/google/gemma-4-E4B-it
- Source URL: https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md
- Retrieved via Surf CLI: 2026-08-25

## Model Identity
- Checkpoint: `google/gemma-4-E4B-it`
- GGUF: `gemma-4-E4B-it-Q4_K_M.gguf`
- Quantization: `Q4_K_M`
- Model class: Gemma 4 E4B instruct

## Runtime Applicability
- Target runtime: llama.cpp GGUF through Vulkan.
- Single-GPU baselines are load-only when the artifact exceeds the local fit threshold.
- Authoritative multi-GPU mode is Vulkan `-sm layer` with split `1,1`; tensor/row split is excluded.
- The exact artifact is present on k7000 and requires bounded load/context probes.

## Unresolved Assumptions
- Gemma 4 architecture/conversion compatibility with the local Vulkan build requires direct evidence.
- Historical rows are boundary/preflight failures and contain no verified metrics.
