---
id: KB-SMOLLM2-17B-Q4KM
category: model-characterization
created: '2026-08-23'
environment: k7000-dual-gtx690
error_signatures:
- smollm2-1.7b-instruct-q4_k_m.gguf
status: validated
tags:
- smollm2
- 1.7b
- vulkan
- authoritative
title: SmolLM2-1.7B-Instruct-Q4_K_M Model Characterization
updated: '2026-08-23'
---

# SmolLM2-1.7B-Instruct-Q4_K_M Benchmark & Characterization

## Model Overview
- **Identifier**: `smollm2-1.7b-instruct-q4_k_m.gguf`
- **Parameters**: 1.71B
- **Quantization**: `Q4_K_M`
- **Architecture**: `llama`

## Execution Plan & Evidence
- **Single-GPU**: Supported on Vulkan0 and Vulkan1 for short contexts (1024-2048).
- **Dual-GPU**: `-sm layer 1,1` is mandatory for extended context to balance weight layers across two 2GB GPUs.
- **Context Progression**: Bounded testing across 1024, 2048, 4096, 8192, 16384, 32768.
- **KV-Cache**: Baseline f16, q8_0, q4_0, and host memory offload (--no-kv-offload).

## Issue 61 execution evidence
- The exact receipt was updated to the current fail-closed schema and validated for Issue 61.
- Research and KB/QMD gates passed. The exact timeout symptom had no matching project KB lesson.
- Reviewed dry-run produced two executable fitting single-GPU full suites: Vulkan0 and Vulkan1. The applicable dual-GPU layer `1,1` configuration remains explicitly planned; tensor/row split is excluded.
- Vulkan0: preflight `SUCCESS`; parsed performance `3.9` prompt t/s and `25.45` generation t/s; Retrieval: 6 `VERIFIED`, 0 `MISSED`, 3 `INCONCLUSIVE`; quality `2/2`; boundary `INCONCLUSIVE` after `BOUNDARY_SSH_TIMEOUT` at context `1024`.
- Vulkan1: preflight `SUCCESS`; parsed performance `3.9` prompt t/s and `24.7` generation t/s; Retrieval: 7 `VERIFIED`, 1 `MISSED`, 1 `INCONCLUSIVE`; quality `1/2`; boundary `INCONCLUSIVE` after `BOUNDARY_SSH_TIMEOUT` at context `1024`.
- Speed, Retrieval, and quality values are diagnostic only. No authoritative publication was made. Further dual-GPU inference requires a reviewed follow-up after investigating the boundary timeout.
- Vulkan1 completed after the interrupted process: performance `3.9` prompt t/s and `24.7` generation t/s; Retrieval 7 `VERIFIED`, 1 `MISSED`, 1 `INCONCLUSIVE`; quality `1/2`; boundary remained `INCONCLUSIVE` with `BOUNDARY_SSH_TIMEOUT` at context `1024`.
- The dual-GPU layer configuration was not started because the remote SSH transport then failed during banner exchange. The transport symptom is recorded as `SSH_TIMEOUT`, not as a model or backend result.
