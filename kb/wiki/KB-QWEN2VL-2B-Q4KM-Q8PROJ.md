---
id: KB-QWEN2VL-2B-Q4KM-Q8PROJ
title: Qwen2-VL-2B Q4_K_M and Q8 projector preflight pair
category: model-analysis
status: reviewed
created: 2026-09-12
updated: 2026-09-12
tags:
  - qwen2-vl
  - ocr
  - multimodal
  - gguf
  - projector
  - vulkan
  - k7000
environment:
  target: k7000
  backend: llama.cpp Vulkan
  vram: two 2GB partitions
error_signatures: []
source_urls:
  - https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct
  - https://huggingface.co/ggml-org/Qwen2-VL-2B-Instruct-GGUF/tree/main
  - https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md
---

# Qwen2-VL-2B Q4_K_M and Q8 projector preflight pair

## Overview
Issue #122 acquired and verified one exact public multimodal pair for a future zero-inference preflight. This note does not authorize image/OCR inference or claim target compatibility.

## Verified local artifact identity

| Role | Basename | Bytes | SHA-256 |
|---|---|---:|---|
| text model | `Qwen2-VL-2B-Instruct-Q4_K_M.gguf` | 986046944 | `5745685d2e607a82a0696c1118e56a2a1ae0901da450fd9cd4f161c6b62867d7` |
| projector | `mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf` | 709883360 | `a0ad91f00a7a80dcf84d719a61b00ee2e07b71794f4ee2dfa81a254621a8c418` |

- Public ggml-org LFS metadata and target-side file hashes match for both artifacts.
- The combined static size is 1,695,930,304 bytes.

## Compatibility boundary
- The model is a Qwen2-VL multimodal model under Apache-2.0.
- Official llama.cpp documentation describes model/projector multimodal input, but does not prove this exact pair works on the target Vulkan build.
- Two separate 2GB Vulkan allocations constrain fit. The static pair total does not account for projector placement, image tokens, KV cache, graph buffers, or runtime overhead.
- Accepted ADR-003 requires the separate multimodal preflight contract. The text runner and existing text receipts remain unchanged.

## Explicitly not established
- No real image was acquired, opened, processed, or inferred on.
- No multimodal model load, target preflight, OCR result, performance, quality, context, or memory result exists.
- A reviewed zero-inference dry-run/preflight remains required before any future OCR inference approval.

## References
- [Raw pair research](../raw/qwen2-vl-2b-q4km-q8proj.md)
