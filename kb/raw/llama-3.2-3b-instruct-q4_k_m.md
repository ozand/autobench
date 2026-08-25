# Llama-3.2-3B-Instruct-Q4_K_M Research Notes

- Source URL: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
- Source URL: https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md
- Retrieved via Surf CLI: 2026-08-25

## Model Identity
- Checkpoint: `meta-llama/Llama-3.2-3B-Instruct`
- GGUF: `llama-3.2-3b-instruct-q4_k_m.gguf`
- Architecture: `llama`
- Parameters: approximately 3.21B
- Quantization: `Q4_K_M`
- Native context limit: 131072 tokens

## Runtime Applicability
- Target runtime: llama.cpp GGUF through Vulkan.
- Single-GPU load-only baselines are required because weights are approximately 1.93GB and saturate a 2GB GPU before KV allocation.
- Authoritative multi-GPU mode is Vulkan `-sm layer` with split `1,1`; tensor/row split is excluded.
- Exact artifact is present on k7000 and requires bounded dual-GPU load/context evidence.

## Unresolved Assumptions
- Local converted GGUF runtime and context behavior require receipt-backed probes.
- Historical rows contain mixed OOM, timeout, and incomplete success-like results; all remain non-authoritative.
