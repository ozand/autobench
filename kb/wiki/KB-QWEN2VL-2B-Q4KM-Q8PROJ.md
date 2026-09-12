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
  - https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/raw/main/config.json
  - https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/raw/main/README.md
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

## Upstream facts
- The Apache-2.0 checkpoint is `Qwen2VLForConditionalGeneration` / `qwen2_vl`: 28 text layers, hidden size 1536, 12 attention heads, 2 KV heads, and a 32768-position configuration.
- Its vision configuration reports depth 32, 16 heads, patch size 14, spatial merge size 2, and temporal patch size 2.
- The upstream README describes dynamic image resolution and a default per-image visual-token range of 4–16384. Its 256–1280-token sample is an upstream Transformers budget example, not an approved target workload or Vulkan capacity claim.
- Official llama.cpp documentation names this pre-quantized GGUF family and uses a text model plus `--mmproj` artifact contract. It says projector GPU offload is the default and `--no-mmproj-offload` disables it.

## Local observations
- Issue #122 acquired the exact pair and verified both target-side sizes and SHA-256 values.
- Issue #122 executed one target-side metadata-only pair preflight. It was `VALID` with `inference_invoked=false` and terminal classification `PREFLIGHT_VALID / ZERO_INFERENCE`.
- That preflight did not load the model/projector, open an image, or test Vulkan/multimodal execution.

## Readiness boundary and unresolved assumptions
- Official llama.cpp multimodal documentation does not prove this exact pair works on the target Vulkan build.
- Two separate 2GB Vulkan partitions constrain fit. The static pair total is 1,695,930,304 bytes; this is not a per-device allocation measurement and excludes projector placement, image tokens, KV cache, graph buffers, and runtime overhead.
- Default projector GPU offload makes it unsafe to assume that text-model `-sm layer` splitting also splits or relocates the projector.
- The existing preflight contract is intentionally artifact/metadata-only. It cannot validate a non-persisted local image reference, load a model, emit OCR terminal classes, or establish inference eligibility.
- Accepted ADR-003 requires a separate multimodal runner and receipt contract; the text runner and existing text receipts remain unchanged.
- Issue #125 is zero-inference planning only. Future image/OCR inference requires a separately governed execution issue, a reviewed implementation/inference plan, a fresh approved pre-inference dry-run, and explicit owner approval.

## References
- [Raw pair research](../raw/qwen2-vl-2b-q4km-q8proj.md)
