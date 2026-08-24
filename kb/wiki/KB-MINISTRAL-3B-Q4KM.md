---
id: KB-MINISTRAL-3B-Q4KM
title: Ministral-3-3B-Instruct-2512-Q4_K_M Vulkan Diagnostic Plan
category: model-analysis
status: validated
created: 2026-08-24
updated: 2026-08-24
tags:
  - ministral
  - 3b
  - vulkan
  - gtx690
  - stage4
environment:
  host: k7000
  gpu: GTX 690 (2x GK104)
  backend: Vulkan
source_urls:
  - https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512
  - https://github.com/ggml-org/llama.cpp/wiki/Home
  - https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md
error_signatures: []
---

# Ministral-3-3B-Instruct-2512-Q4_K_M

## Research and plan
- Exact GGUF identity: `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf`.
- Backend: Vulkan.
- Required single-GPU baselines: Vulkan0 and Vulkan1.
- Required multi-GPU mode: Vulkan layer split `1,1`; tensor/row split is excluded.
- Context progression: bounded `1024 -> 2048 -> 4096 -> 8192`, subject to researched limits and first-failure stop.
- KV policy: establish f16 baseline before one justified alternative.

## Historical disposition
The previous two rows were `PARTIAL_FAILURE` and `PREFLIGHT_OOM`; both are non-authoritative. They provide no verified speed or Retrieval metric.

## Issue 44 execution evidence
- Stage 1/2 evidence passed and the exact receipt validated.
- Dry-run planned three jobs: Vulkan0 load-only, Vulkan1 load-only, and dual-GPU layer `1,1` full.
- Vulkan0 single-GPU load probe: `OOM`.
- Vulkan1 single-GPU load probe: `OOM`.
- Dual-GPU layer preflight: `SUCCESS`, but the first boundary probe at context `1024` ended `SSH_TIMEOUT`; boundary is `INCONCLUSIVE` and the workload was unsupported.
- No performance, quality, or Retrieval metrics were published because no comparable workload completed.
- Final disposition: `NON_AUTHORITATIVE`.

## Local execution status
Stage 1/2 evidence and the reviewed Issue 44 receipt are persisted. The execution result is retained as sanitized evidence in `docs/issue44-ministral-evidence.json`.

## Safety
No raw prompts, responses, runtime output, private paths, credentials, or host identifiers are persisted.
