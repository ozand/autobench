---
id: KB-QWEN35-2B-Q4KM
title: Qwen3.5-2B-Q4_K_M Vulkan Diagnostic Plan
category: model-analysis
status: validated
created: 2026-08-25
updated: 2026-08-25
tags:
  - qwen3.5
  - 2b
  - vulkan
  - gtx690
  - stage4
environment:
  host: k7000
  gpu: GTX 690 (2x GK104)
  backend: Vulkan
source_urls:
  - https://huggingface.co/Qwen/Qwen3.5-2B
  - https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md
error_signatures: []
---

# Qwen3.5-2B-Q4_K_M

## Research and plan
- Exact GGUF identity: `Qwen3.5-2B-Q4_K_M.gguf`.
- Backend: Vulkan.
- Single-GPU baselines: Vulkan0 and Vulkan1 load-only because the artifact is above the local single-GPU fit threshold.
- Multi-GPU mode: Vulkan layer split `1,1`; tensor/row split is excluded.
- Context progression: bounded `1024 -> 2048 -> 4096 -> 8192`, subject to the first-failure stop.
- KV policy: f16 baseline before one justified alternative.

## Historical disposition
Previous rows were `SUCCESS` without complete metrics or `TIMEOUT`; all are non-authoritative.

## Issue 49 execution evidence
- Stage 1/2 evidence passed and the repaired receipt validated.
- Dry-run planned two fitting single-GPU jobs, Vulkan0 and Vulkan1; the dual-GPU layer configuration remains separately planned because the current inventory path uses the fitting-model two-job envelope.
- Vulkan0 boundary succeeded at `1024`; the `2048` probe ended `SSH_TIMEOUT`, so boundary is `INCONCLUSIVE`.
- Vulkan0 performance: prompt `4.6 t/s`, generation `19.45 t/s`.
- Vulkan0 Retrieval: 8 `MISSED`, 1 `INCONCLUSIVE`, 0 `VERIFIED`; no authoritative pass-rate claim.
- Vulkan0 quality: `0/2` tasks passed.
- Vulkan1 boundary succeeded at `1024`; the `2048` probe ended `SSH_TIMEOUT`, so boundary is `INCONCLUSIVE`.
- Vulkan1 performance: prompt `4.6 t/s`, generation `19.9 t/s`.
- Vulkan1 Retrieval: 1 `VERIFIED`, 8 `MISSED`; no authoritative pass-rate claim.
- Vulkan1 quality: `0/2` tasks passed.
- Overall Issue 49 disposition: `PARTIAL_FAILURE` / non-authoritative.

## Safety
No raw prompts, responses, runtime output, private paths, credentials, or host identifiers are persisted.
