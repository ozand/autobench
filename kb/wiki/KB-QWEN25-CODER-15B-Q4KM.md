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

## Issue 58 execution evidence
- Exact receipt was repaired for the current governing issue and validated fail-closed.
- Research and KB/QMD gates passed; no matching error lesson was found for the exact boundary timeout.
- The reviewed dry-run produced two executable inventory jobs: Vulkan0 and Vulkan1 full suites. The model-specific plan also retains the applicable dual-GPU layer `1,1` configuration; tensor/row split remains excluded.
- Vulkan0: preflight `SUCCESS`; performance parsed at `5.1` prompt t/s and `26.35` generation t/s; Retrieval attempts: 5 `VERIFIED`, 3 `MISSED`, 1 `INCONCLUSIVE`; quality `2/2`; boundary `INCONCLUSIVE` because the first 1024 probe ended `BOUNDARY_SSH_TIMEOUT`.
- Vulkan1: preflight `SUCCESS`; performance parsed at `5.1` prompt t/s and `25.9` generation t/s; Retrieval attempts: 3 `VERIFIED`, 6 `MISSED`, 0 `INCONCLUSIVE`; quality `2/2`; boundary `INCONCLUSIVE` because the first 1024 probe ended `BOUNDARY_SSH_TIMEOUT`.
- These values are diagnostic only. No authoritative speed, Retrieval, or quality publication was made. A follow-up investigation is required before dual-GPU inference or promotion.
