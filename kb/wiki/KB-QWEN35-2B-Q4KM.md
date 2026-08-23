---
id: KB-QWEN35-2B-Q4KM
title: Qwen3.5-2B-Q4_K_M Vulkan Diagnostic
category: model-testing
status: verified
created: 2026-08-23
updated: 2026-08-23
tags:
  - qwen35
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

# Qwen3.5-2B-Q4_K_M Vulkan Diagnostic

## Verified Model Facts
- Model: `Qwen3.5-2B-Q4_K_M.gguf`
- File Size: 1,280,827,392 bytes (~1221.5 MiB)
- Multi-GPU Support: `-sm layer` only (tensor split unsupported).
- Context scaling: Single-GPU fits 1024; Dual-GPU scales through 32768.
