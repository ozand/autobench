---
id: KB-LLAMA32-1B-Q4KM
title: Llama-3.2-1B-Instruct-Q4_K_M Issue #41 protocol and historical diagnostics
category: models
status: reviewed
created: 2026-08-22
updated: 2026-09-10
tags:
  - llama
  - q4_k_m
  - vulkan
  - k7000
  - issue41
environment: dual-gtx690-vulkan
error_signatures:
  - SSH execution timed out
source_urls:
  - https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
  - https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md
---

# Llama-3.2-1B-Instruct-Q4_K_M

## Current Issue #41 research and execution gate
- Current target artifact observation: 807694368 bytes, SHA-256 `3f5a22426976ab26cfe84dba63c1d08391717abb1af893e10f1b2968d862dcc1`.
- Upstream model metadata identifies a 1.23B-parameter Llama 3.2 model with GQA; the upstream model repository is access-gated in the current browser session, so exact target-build support and local capacity remain unresolved until target preflight.
- llama.cpp documents `-sm layer` as pipeline-parallel multi-GPU mode; project policy excludes tensor/row split on this Vulkan testbed.
- Issue #41 plan: matched Vulkan0 and Vulkan1 baselines plus dual-GPU `Vulkan0,Vulkan1`, `-sm layer`, `-ts 1,1`; default f16 K/V; bounded contexts 1024 for all publication stages in the same-context follow-up.
- A new Issue #41 receipt must bind the current artifact exactly; the previous Issue #1 receipt is not authorization.

## Historical Issue #43 evidence (retained, non-authoritative)
- Issue #43 was governed by Issue #1 and used two fitting single-GPU jobs; the dual-GPU layer configuration was not executed.
- Vulkan0 and Vulkan1 boundary probes were inconclusive; prior speed, Retrieval, quality, and capacity observations remain diagnostic only.
- No historical row is promoted or rewritten by the Issue #41 preparation.

## Unresolved assumptions
- Exact target Vulkan preflight result.
- Current single-GPU boundary and Retrieval behavior.
- Current dual-GPU layer performance, Retrieval, quality, and capacity.


## Issue #41 execution evidence
- The dedicated Issue #41 receipt validated the current target artifact and the remote dry-run passed before inference.
- The reviewed three-job execution completed with `SUCCESS` for all three jobs.
- Vulkan0 at context 1024: performance 6.8 prompt t/s and 34.1333 generation t/s; Retrieval 13 `VERIFIED`, 2 `MISSED`, 0 `INCONCLUSIVE`; quality 0/2; boundary SUCCESS at 1024 (lower bound only).
- Vulkan1 at context 1024: performance 6.8 prompt t/s and 33.6333 generation t/s; Retrieval 10 `VERIFIED`, 5 `MISSED`, 0 `INCONCLUSIVE`; quality 1/2; boundary SUCCESS at 1024 (lower bound only).
- Dual-GPU layer `1,1`: performance and quality at context 1024, but boundary and Retrieval at context 4096. This mixed-context result is `HELD_NON_AUTHORITATIVE`; a dedicated same-context context-1024 rerun is required before dual-GPU publication.
- No historical Issue #43 row is promoted or rewritten, and no public database/report write is authorized by this evidence record.
