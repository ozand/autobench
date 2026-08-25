---
id: KB-GEMMA4-E2B-Q4KM
title: Gemma-4-E2B-it-Q4_K_M Vulkan Diagnostic Plan
category: model-analysis
status: validated
created: 2026-08-25
updated: 2026-08-25
tags:
  - gemma4
  - e2b
  - vulkan
  - gtx690
  - stage4
environment:
  host: k7000
  gpu: GTX 690 (2x GK104)
  backend: Vulkan
source_urls:
  - https://huggingface.co/google/gemma-4-E2B-it
  - https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md
error_signatures: []
---

# Gemma-4-E2B-it-Q4_K_M

## Research and plan
- Exact GGUF identity: `gemma-4-E2B-it-Q4_K_M.gguf`.
- Backend: Vulkan.
- Single-GPU baselines: Vulkan0 and Vulkan1 load-only because the artifact is above the local fit threshold.
- Multi-GPU mode: Vulkan layer split `1,1`; tensor/row split is excluded.
- Context progression: bounded `1024 -> 2048 -> 4096 -> 8192`, subject to first-failure stop.
- KV policy: f16 baseline before one justified alternative.

## Historical disposition
Previous rows were `PREFLIGHT_EXECUTION_ERROR`; both are non-authoritative and contain no verified metrics.

## Issue 52 execution evidence
- Stage 1/2 evidence passed and the exact receipt validated.
- Dry-run planned three jobs: Vulkan0 load-only, Vulkan1 load-only, and dual-GPU layer `1,1` full.
- Vulkan0 load probe: `SUCCESS`.
- Vulkan1 load probe: `SUCCESS`.
- Dual-GPU layer preflight: `EXECUTION_ERROR` with return code `134`; boundary, Retrieval, performance, and quality were not attempted.
- Final disposition: `NON_AUTHORITATIVE`; no speed or Retrieval values are published.

## Safety
No raw prompts, responses, runtime output, private paths, credentials, or host identifiers are persisted.
