---
id: KB-DEEPSEEK-R1-15B-Q4KM
title: DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M Matrix and Dual-GPU Placement
category: model-analysis
created: 2026-08-23
updated: 2026-08-23
environment: dual-gtx690-vulkan
status: validated
tags:
  - deepseek-r1
  - 1.5b
  - vulkan
  - gtx690
  - layer-split
error_signatures: []
---

# DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M Performance & Matrix

## Summary
DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M (~1.12 GB) is a reasoning distilled model based on the Qwen2.5-1.5B architecture. On GTX 690 Vulkan:
- **1 GPU (2 GB)**: Fits only at context 1024; larger contexts hit VRAM limits.
- **2 GPU (Dual-GPU 1,1 -sm layer)**: Splits across both cards allowing full operation up to 32768 tokens.
- **KV Options**: Supported in f16, q8_0, q4_0, and --no-kv-offload modes.
