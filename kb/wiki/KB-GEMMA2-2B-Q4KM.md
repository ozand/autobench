---
id: KB-GEMMA2-2B-Q4KM
title: Gemma-2-2B-IT-Q4_K_M Vulkan Diagnostic Plan
category: model-analysis
status: validated
created: 2026-08-25
updated: 2026-08-25
tags:
  - gemma2
  - 2b
  - vulkan
  - gtx690
  - stage4
environment:
  host: k7000
  gpu: GTX 690 (2x GK104)
  backend: Vulkan
source_urls:
  - https://huggingface.co/google/gemma-2-2b-it
  - https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md
error_signatures: []
---

# Gemma-2-2B-IT-Q4_K_M

## Research and plan
- Exact GGUF identity: `gemma-2-2b-it-Q4_K_M.gguf`.
- Backend: Vulkan.
- Single-GPU baselines: Vulkan0 and Vulkan1 load-only.
- Multi-GPU mode: Vulkan layer split `1,1`; tensor/row split is excluded.
- Context progression: bounded `1024 -> 2048 -> 4096 -> 8192`, subject to first-failure stop.
- KV policy: f16 baseline before one justified alternative.

## Historical disposition
Previous rows were `SUCCESS` without complete metrics or `TIMEOUT`; all are non-authoritative.

## Issue 51 execution evidence
- Stage 1/2 evidence passed and the repaired receipt validated.
- Dry-run planned two fitting single-GPU jobs, Vulkan0 and Vulkan1; dual-GPU layer remains separately planned because the current inventory path uses the fitting-model two-job envelope.
- Vulkan0 boundary succeeded at `1024`; the `2048` probe ended `SSH_TIMEOUT`, so boundary is `INCONCLUSIVE`.
- Vulkan0 performance: prompt `3.2 t/s`, generation `15.9 t/s`.
- Vulkan0 Retrieval: 4 `VERIFIED`, 5 `INCONCLUSIVE`; no authoritative pass-rate claim.
- Vulkan0 quality: `2/2` tasks passed.
- Vulkan1 boundary succeeded at `1024`; the `2048` probe ended `SSH_TIMEOUT`, so boundary is `INCONCLUSIVE`.
- Vulkan1 performance: prompt `3.2 t/s`, generation `16.2 t/s`.
- Vulkan1 Retrieval: 8 `VERIFIED`, 1 `INCONCLUSIVE`; no authoritative pass-rate claim.
- Vulkan1 quality: `2/2` tasks passed.
- Overall Issue 51 disposition: `PARTIAL_FAILURE` / non-authoritative.

## Safety
No raw prompts, responses, runtime output, private paths, credentials, or host identifiers are persisted.
