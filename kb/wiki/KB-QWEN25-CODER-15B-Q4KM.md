---
id: KB-QWEN25-CODER-15B-Q4KM
title: Qwen2.5-Coder-1.5B-Instruct-Q4_K_M Benchmark Protocol and Operational Baseline
category: benchmark
created: 2026-08-22
updated: 2026-08-22
status: approved
tags:
  - qwen2.5
  - coder
  - 1.5b
  - vulkan
  - gtx690
environment:
  target_host: k7000
  gpu: Dual GTX 690 (2x GK104, 2GB VRAM each)
  driver: Vulkan 1.3
  backend: llama.cpp Vulkan
error_signatures:
  - ErrorOutOfDeviceMemory
  - does not support split buffers
---

## Summary
Execution and boundary evaluation of `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` (~1065.6 MiB) on the dual-GPU Vulkan testbed.

## Hardware Placement
- **Single-GPU (Vulkan0/1)**: Operates at 1024 context, hits VRAM limits on >=2048 context.
- **Dual-GPU (-sm layer 1,1)**: Distributes 28 layers (14/14), enabling reliable execution across standard context.
