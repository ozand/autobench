# Llama-3.2-1B-Instruct-Q4_K_M Research Notes (Issue #41)

## Provenance
- Source URLs: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
- Source URL: https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md
- Retrieved/verified via Surf CLI: 2026-09-10.
- The upstream Meta repository is access-gated in the current browser session; model metadata below is retained from the existing sanitized note and public repository API metadata. Exact local capability is not inferred from upstream claims.

## Model Identity
- Checkpoint: `meta-llama/Llama-3.2-1B-Instruct`
- GGUF file: `Llama-3.2-1B-Instruct-Q4_K_M.gguf`
- Quantization: `Q4_K_M`
- Architecture: Llama architecture, 16 layers, 32 attention heads, 8 key/value heads (GQA), intermediate size 8192, vocabulary size 128256 (retained from the prior sanitized model note).
- Native context metadata: 131072 tokens (retained from the prior sanitized model note; exact config file was access-gated during this refresh).
- Current target artifact observation: 807694368 bytes, SHA-256 `3f5a22426976ab26cfe84dba63c1d08391717abb1af893e10f1b2968d862dcc1`.

## Backend and capacity assumptions
- Upstream llama.cpp documentation describes `layer` split as the compatible pipeline-parallel multi-GPU mode; this does not prove exact target-build/device support.
- Target Vulkan support and local capacity remain unresolved until Stage 3 target preflight and execution.
- Single-GPU and dual-GPU paths are both applicable to this fitting model; the Issue #41 bounded plan uses matched Vulkan0/Vulkan1 baselines plus one dual-GPU layer configuration.
- Tensor/row split is excluded by project policy and the target Vulkan testbed limitation.
- KV policy starts with default f16 K/V; no blind KV sweep is authorized.

## Prior evidence and limitations
- Issue #43 used an Issue #1-governed two-job diagnostic envelope. Its boundary results were inconclusive, Retrieval had no authoritative aggregate, and the dual-GPU layer configuration was not executed.
- Issue #43 speed, Retrieval, quality, and capacity observations remain historical/non-authoritative and cannot authorize Issue #41 publication.
- The current Issue #41 run must use a fresh receipt bound to this artifact and a reviewed dry-run before inference.
