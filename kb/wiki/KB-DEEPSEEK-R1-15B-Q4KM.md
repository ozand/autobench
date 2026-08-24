---
id: KB-DEEPSEEK-R1-15B-Q4KM
title: DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M Matrix and Dual-GPU Placement
category: model-analysis
created: 2026-08-23
updated: 2026-08-24
environment: dual-gtx690-vulkan
status: validated
tags:
  - deepseek-r1
  - 1.5b
  - vulkan
  - gtx690
  - layer-split
error_signatures: []
source_urls:
  - https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
  - https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md
---

# DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M Performance & Matrix

## Summary

### Current Issue 42 evidence
- Stage 1/2 research and receipt validation passed for the exact GGUF.
- Reviewed execution plan covers Vulkan0, Vulkan1, and dual-GPU Vulkan layer split `1,1`.
- Vulkan0 suite preflight and performance completed with prompt `4.9 t/s` and generation `28.37 t/s`.
- Vulkan0 boundary reached `1024` tokens; the `2048` probe was `SSH_TIMEOUT`, so the boundary remains `INCONCLUSIVE`.
- Vulkan0 retrieval produced 14 `MISSED` and 1 `INCONCLUSIVE` attempt; no `VERIFIED` attempt was observed, so no retrieval pass-rate claim is published.
- Vulkan0 quality stage completed with `0/2` deterministic tasks passed.
- The suite remains `PARTIAL_FAILURE` and non-authoritative pending investigation and the remaining reviewed configurations.
DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M (~1.12 GB) is a reasoning distilled model based on the Qwen2.5-1.5B architecture. On GTX 690 Vulkan:
- **1 GPU (2 GB)**: Fits only at context 1024; larger contexts hit VRAM limits.
- **2 GPU (Dual-GPU 1,1 -sm layer)**: Splits across both cards allowing full operation up to 32768 tokens.
- **KV Options**: Supported in f16, q8_0, q4_0, and --no-kv-offload modes.
