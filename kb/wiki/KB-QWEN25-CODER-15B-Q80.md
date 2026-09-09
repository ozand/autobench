---
id: KB-QWEN25-CODER-15B-Q80
title: Qwen2.5-Coder-1.5B-Instruct-Q8_0 Vulkan Diagnostic
category: model-testing
status: verified
created: 2026-08-23
updated: 2026-08-23
tags:
  - qwen25
  - coder
  - 1.5b
  - q8_0
  - vulkan
  - k7000
environment:
  target_device: GTX 690 (Vulkan 1.0)
  driver_vendor: NVIDIA Kepler
error_signatures:
  - ErrorOutOfDeviceMemory
  - vk::PhysicalDevice::createDevice
---

# Qwen2.5-Coder-1.5B-Instruct-Q8_0 Vulkan Diagnostic

## Verified Model Facts
- Model: `qwen2.5-coder-1.5b-instruct-q8_0.gguf`
- File Size: 1,894,532,160 bytes (~1806.8 MiB)
- Architecture: `qwen2`
- Multi-GPU Support: `-sm layer` only.
- Hardware Boundary: 1.81 GB weights exceed single GPU 2GB allocatable buffer limits. Dual-GPU (1,1 layer split) provides 4GB total headroom for execution.

## Issue 59 execution evidence
- Exact Q8_0 receipt was updated to the current fail-closed schema and validated for Issue 59.
- Research and KB/QMD gates passed; lexical discovery returned the exact model note.
- Reviewed dry-run produced two executable fitting single-GPU jobs: Vulkan0 and Vulkan1. The applicable dual-GPU layer `1,1` configuration remains explicitly planned; tensor/row split is excluded.
- Vulkan0: preflight `SUCCESS`; performance `5.2` prompt t/s and `17.2` generation t/s; Retrieval: 2 `VERIFIED`, 6 `MISSED`, 1 `INCONCLUSIVE`; quality `2/2`; boundary `INCONCLUSIVE` after `BOUNDARY_SSH_TIMEOUT` at context `1024`.
- Vulkan1: preflight `SUCCESS`; performance `5.2` prompt t/s and `17.0` generation t/s; Retrieval: 3 `VERIFIED`, 6 `MISSED`, 0 `INCONCLUSIVE`; quality `2/2`; boundary `INCONCLUSIVE` after `BOUNDARY_SSH_TIMEOUT` at context `1024`.
- All speed, Retrieval, and quality values are diagnostic only. No authoritative publication was made. A follow-up investigation is required before dual-GPU inference or promotion.


## Issue #41 Follow-up Plan
- Exact target checksum: `507de59046601282ba768a9789900e6ccf60ed93ddf346730b7c68eb0715bc47`.
- The old Issue #59 receipt is not reused because it is governed by Issue #59 and does not bind the current target checksum.
- New plan: one serial dual-GPU layer job only, with `Vulkan0,Vulkan1`, `-sm layer`, `-ts 1,1`, f16 KV, and context 1024 for boundary, performance, Retrieval, and quality.
- Single-GPU baselines are documented as inapplicable for this model/workload because the model weights nearly fill one 2 GB partition and prior boundary probes failed at the minimum tested context.
- No public result is authorized by this note; a dedicated Issue #41 receipt and reviewed dry-run are required first.
