---
id: KB-PHI-4-MINI-Q4KM
title: Phi-4-mini-instruct-Q4_K_M Vulkan Diagnostic Plan
category: model-analysis
status: validated
created: 2026-08-24
updated: 2026-08-24
tags:
  - phi
  - 4b
  - vulkan
  - gtx690
  - stage4
environment:
  host: k7000
  gpu: GTX 690 (2x GK104)
  backend: Vulkan
source_urls:
  - https://huggingface.co/microsoft/Phi-4-mini-instruct
  - https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md
error_signatures: []
---

# Phi-4-mini-instruct-Q4_K_M

## Research and plan
- Exact GGUF identity: `Phi-4-mini-instruct-Q4_K_M.gguf`.
- Backend: Vulkan.
- Single-GPU baselines: Vulkan0 and Vulkan1 load-only because the artifact is above the local single-GPU fit threshold.
- Multi-GPU mode: Vulkan layer split `1,1`; tensor/row split is excluded.
- Context progression: bounded `1024 -> 2048 -> 4096 -> 8192`, subject to the first-failure stop.
- KV policy: f16 baseline before one justified alternative.

## Historical disposition
The previous rows were `PARTIAL_FAILURE`; both are non-authoritative and contain no verified metrics.

## Issue 46 execution evidence
- Stage 1/2 evidence passed and the exact receipt validated.
- Dry-run planned three jobs: Vulkan0 load-only, Vulkan1 load-only, and dual-GPU layer `1,1` full.
- Vulkan0 load probe: `OOM`.
- Vulkan1 load probe: `OOM`.
- Dual-GPU layer preflight: `SUCCESS`, but the first boundary probe at context `1024` ended `SSH_TIMEOUT`; boundary is `INCONCLUSIVE` and the workload is unsupported.
- Performance, quality, and Retrieval were not attempted because no comparable workload budget remained.
- Final disposition: `NON_AUTHORITATIVE`; no speed or Retrieval values are published.

## Safety
No raw prompts, responses, runtime output, private paths, credentials, or host identifiers are persisted.
