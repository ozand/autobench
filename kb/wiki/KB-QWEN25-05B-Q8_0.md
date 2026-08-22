---
id: KB-QWEN25-05B-Q8_0
title: "Qwen2.5-0.5B-Instruct-Q8_0 Vulkan Benchmark Protocol"
category: research
tags: [qwen2.5, q8_0, gguf, vulkan, dual-gpu, k7000]
status: active
created: 2026-08-22
updated: 2026-08-22
environment:
  os: Linux
  shell: bash
  tools: [llama.cpp, Vulkan, Surf CLI]
error_signatures: []
---

# Qwen2.5-0.5B-Instruct-Q8_0 Vulkan Benchmark Protocol

## Overview
Canonical Stage 1-4 protocol documentation for qwen2.5-0.5b-instruct-q8_0.gguf (~644.4 MiB).

## Architectural & Quantization Profile
- Architecture: 24 layers, 14 query heads, 2 KV heads, 0.49B parameters.
- Quantization: Q8_0 (~644.4 MiB).
- Context Window: 32,768 tokens declared (evaluated at 1024, 2048, 4096).
- Dual-GPU Split: -sm layer only (-sm tensor unsupported on Vulkan).

## References
- [Qwen2.5-0.5B-Instruct Q8_0 Raw Notes](../raw/qwen2.5-0.5b-instruct-q8_0.md)
- [Qwen2.5-0.5B-Instruct Model Card](../raw/qwen2.5-0.5b-instruct-model-card.md)
- [llama.cpp Multi-GPU Reference](../raw/llama-cpp-multi-gpu.md)
