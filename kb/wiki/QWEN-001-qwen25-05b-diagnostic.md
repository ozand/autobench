---
id: QWEN-001
title: "Qwen2.5-0.5B-Instruct diagnostic research baseline"
category: research
tags: [qwen2.5, gguf, llama-cpp, context-window, vulkan, multi-gpu]
status: active
created: 2026-08-17
updated: 2026-08-17
environment:
  os: any
  shell: any
  tools: [Surf CLI, llama.cpp, QMD]
error_signatures: []
---

# Qwen2.5-0.5B-Instruct diagnostic research baseline

## Overview

This note records source-backed assumptions that must be checked before
continuing the bounded AutoBench diagnostic for the concrete
`qwen2.5-0.5b-instruct-q4_k_m.gguf` model. It separates upstream facts from
local hardware observations and unresolved assumptions.

## Key facts

- The official checkpoint model card was retrieved through Surf CLI on
  2026-08-17.
- The card states 0.49B parameters, 24 layers, 14 query heads, and 2 KV heads.
- The checkpoint-specific context declaration is 32,768 tokens, with generation
  up to 8,192 tokens.
- The card's broader Qwen2.5 family statement says long context up to 128K;
  this is not the limit to use for this concrete checkpoint.
- Official llama.cpp documentation describes `layer` as the default and most
  compatible multi-GPU mode; `tensor` is experimental and must be tracked
  separately.
- A `1,1` tensor split follows the order of the selected devices and represents
  an equal split according to the documentation.

## Applicability and uncertainty

- Upstream documentation does not establish local GGUF metadata, Vulkan
  capability, VRAM availability, or runtime success.
- The local diagnostic must verify device capability, model loading, boundary
  behavior, and workload completion independently.
- The final tested context is conditional on the probe range and timeout; a
  timeout is inconclusive unless an explicit capacity failure is observed.
- The size-based single-GPU planning threshold is inventory scheduling policy,
  not evidence that a dual-GPU diagnostic is technically irrelevant.

## References

- [Qwen2.5-0.5B-Instruct model-card notes](../raw/qwen2.5-0.5b-instruct-model-card.md)
- [llama.cpp multi-GPU notes](../raw/llama-cpp-multi-gpu.md)
- [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [llama.cpp multi-GPU guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md)
