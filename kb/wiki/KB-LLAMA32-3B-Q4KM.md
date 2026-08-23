---
id: KB-LLAMA32-3B-Q4KM
title: Llama-3.2-3B-Instruct-Q4_K_M Vulkan Diagnostic
category: model-testing
status: verified
created: 2026-08-23
updated: 2026-08-23
tags:
  - llama32
  - 3b
  - vulkan
  - k7000
environment:
  target_device: GTX 690 (Vulkan 1.0)
  driver_vendor: NVIDIA Kepler
error_signatures:
  - ErrorOutOfDeviceMemory
  - vk::PhysicalDevice::createDevice
---

# Llama-3.2-3B-Instruct-Q4_K_M Vulkan Diagnostic

## Verified Model Facts
- Model: `llama-3.2-3b-instruct-q4_k_m.gguf`
- File Size: 2,019,377,600 bytes (~1925.8 MiB)
- Architecture: `llama` (3.21B parameters)
- Multi-GPU Support: `-sm layer` only.
- Single-GPU Limit: Single 2GB GPU will OOM at all context lengths due to ~1.93 GB weight size. Dual-GPU (1,1 layer split) enables execution.
