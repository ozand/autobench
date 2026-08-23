---
id: KB-GEMMA2-2B-Q4KM
title: Gemma-2-2B-IT-Q4_K_M Vulkan Diagnostic
category: model-testing
status: verified
created: 2026-08-23
updated: 2026-08-23
tags:
  - gemma2
  - 2b
  - vulkan
  - k7000
environment:
  target_device: GTX 690 (Vulkan 1.0)
  driver_vendor: NVIDIA Kepler
error_signatures:
  - ErrorOutOfDeviceMemory
  - vk::PhysicalDevice::createDevice
---

# Gemma-2-2B-IT-Q4_K_M Vulkan Diagnostic

## Verified Model Facts
- Model: `gemma-2-2b-it-Q4_K_M.gguf`
- File Size: 1,708,582,752 bytes (~1629.4 MiB)
- Architecture: `gemma2`
- Multi-GPU Support: `-sm layer` only.
- Single-GPU Limit: Single 2GB GPU will OOM at context >= 2048 due to 1.63 GB weight footprint. Dual-GPU (1,1 layer split) enables execution.
