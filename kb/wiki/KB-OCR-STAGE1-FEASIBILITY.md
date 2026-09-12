---
id: KB-OCR-STAGE1-FEASIBILITY
title: OCR Stage 1 feasibility and k7000 candidate shortlist
category: model-analysis
status: reviewed
created: 2026-09-12
updated: 2026-09-12
tags:
  - ocr
  - vision
  - multimodal
  - glm-ocr
  - qwen2-vl
  - smolvlm2
  - vulkan
  - k7000
environment:
  target: k7000
  backend: llama.cpp Vulkan
  vram: two 2GB partitions
error_signatures: []
source_urls:
  - https://huggingface.co/zai-org/GLM-OCR
  - https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct
  - https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct
  - https://huggingface.co/openbmb/MiniCPM-V-2_6
  - https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md
  - https://huggingface.co/ggml-org/Qwen2-VL-2B-Instruct-GGUF/tree/main
  - https://huggingface.co/ggml-org/SmolVLM2-2.2B-Instruct-GGUF/tree/main
---

# OCR Stage 1 feasibility and k7000 candidate shortlist

## Overview
Issue #114 is a research-only OCR selection step. No model artifact was acquired and no image/OCR workload was executed. Public metadata and official llama.cpp documentation do not prove local compatibility, memory fit, or OCR quality.

## Key facts
- k7000 has two separate 2GB Vulkan allocations, not one uniform 4GB allocation pool. A later candidate must fit model layers, projector, image tokens/KV cache, scratch/graph buffers, and runtime overhead on the individual devices.
- Current AutoBench runner is text-only. A later image/projector runner contract requires a separate Proposed ADR and owner approval before implementation.
- Official llama.cpp multimodal documentation, reviewed on 2026-09-12, supports image input with a multimodal projector and lists public GGUF paths for Qwen2-VL-2B and SmolVLM2-2.2B.

## Candidate shortlist

### 1. Qwen2-VL-2B-Instruct — priority conditional candidate
- Official base model license: Apache-2.0.
- Official llama.cpp documentation lists `ggml-org/Qwen2-VL-2B-Instruct-GGUF`.
- Reported public artifact sizes: Q4_K_M text model 986,046,944 bytes; Q8 projector 709,883,360 bytes; f16 projector 1,331,656,160 bytes. They are metadata inputs, not locally verified artifacts.
- Q4_K_M plus Q8 projector totals 1,695,930,304 bytes (about 1.58 GiB) before KV/scratch/runtime overhead.
- Status: conditional only. Prefer the Q8 projector for a future preflight because projector placement/splitting across the two Vulkan devices is unresolved.

### 2. SmolVLM2-2.2B-Instruct — secondary conditional candidate
- Official base model license: Apache-2.0.
- The official card reports OCRBench and DocVQA values, which are upstream reported metrics only.
- Official llama.cpp documentation lists `ggml-org/SmolVLM2-2.2B-Instruct-GGUF`.
- Reported public artifact sizes: Q4_K_M text model 1,112,602,656 bytes; Q8 projector 592,523,200 bytes; total 1,705,125,856 bytes (about 1.59 GiB) before runtime overhead. They are metadata inputs, not locally verified artifacts.
- Status: conditional only. The upstream 5.2GB video-inference statement is not evidence for a quantized single-image Vulkan fit.

### GLM-OCR — excluded from the current runnable shortlist
- Official model card: MIT license, `glm_ocr` architecture, multimodal OCR model with reported 0.9B parameters.
- The reviewed card describes Transformers, vLLM, SGLang, Ollama, and an SDK pipeline. It does not provide a public GGUF/projector path or official llama.cpp compatibility evidence.
- Status: excluded for current k7000 runnable selection; a later conversion/runtime compatibility investigation would be separate.

### MiniCPM-V-2.6 — excluded from the current shortlist
- Public model metadata identifies OCR/vision capability but access is gated/auto and it uses custom code.
- No reviewed public ggml-org GGUF/projector path or official llama.cpp multimodal documentation entry was found.
- Status: excluded pending separate public conversion and runtime compatibility evidence.

## Recommendation
- Future implementation gate: select Qwen2-VL-2B Q4_K_M with a Q8 projector as the first candidate for a later compatibility decision.
- Retain SmolVLM2-2.2B Q4_K_M with Q8 projector as a secondary candidate.
- Do not acquire artifacts or run OCR until a new implementation issue, a Proposed multimodal ADR, owner approval, exact-artifact research, and a reviewed hardware plan exist.

## References
- [OCR Stage 1 raw research](../raw/ocr-stage1-feasibility.md)
