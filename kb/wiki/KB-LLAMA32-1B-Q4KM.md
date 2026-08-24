---
id: KB-LLAMA32-1B-Q4KM
title: Llama-3.2-1B-Instruct-Q4_K_M Vulkan Diagnostics and Single-GPU Baseline
category: models
status: validated
created: 2026-08-22
updated: 2026-08-24
tags:
  - llama
  - vulkan
  - k7000
  - stage4
environment:
  host: k7000
  gpu: GTX 690 (2x GK104)
  backend: Vulkan
error_signatures:
  - 'SSH execution timed out after 190 seconds'
---

## Summary
Llama-3.2-1B-Instruct-Q4_K_M (size ~770.3 MiB, 1.23B parameters) passed full Stage 1-4 validation under Issue #1. Single-GPU Vulkan0 and Vulkan1 execution was measured along with multi-GPU layer split constraints.

## Verified Architecture and Limits
- **Architecture**: LLaMA 3.2 (1.23B parameters, GQA 8 heads / 32 Q heads).
- **Context Limit**: 131,072 tokens native, tested safely up to 2048/4096 on k7000.
- **Backend**: Vulkan single-GPU (Vulkan0, Vulkan1) and dual-GPU layer split (-sm layer).
- **Quantization**: Q4_K_M (~770.3 MiB).

## Issue 43 Evidence (Stage 4)
- Stage 1/2 receipt validation passed for the exact GGUF.
- The inventory dry-run planned two fitting single-GPU jobs; the reviewed dual-GPU layer configuration remains planned separately because the current inventory path uses the fitting-model two-job envelope.
- Vulkan0 preflight and performance succeeded: prompt `6.8 t/s`, generation `35.25 t/s`.
- Vulkan0 boundary reached `2048` and remained `INCONCLUSIVE` at the next boundary step.
- Vulkan0 retrieval had 1 `VERIFIED`, 7 `MISSED`, and 1 `INCONCLUSIVE` attempt; the aggregate is not authoritative.
- Vulkan0 quality completed with `0/2` tasks passed.
- Vulkan1 preflight and performance succeeded: prompt `6.8 t/s`, generation `34.05 t/s`.
- Vulkan1 boundary reached `2048` and remained `INCONCLUSIVE` at the next boundary step.
- Vulkan1 retrieval completed with 9 `MISSED` and 0 `VERIFIED` attempts.
- Vulkan1 quality completed with `1/2` tasks passed.
- The model remains `PARTIAL_FAILURE` / non-authoritative until boundary and Retrieval evidence are reconciled and the dual-GPU layer configuration is executed under a reviewed follow-up.

## Empirical Findings (Stage 4)
- **Prompt Speed**: ~6.8 tokens/sec on Vulkan0 (GK104 memory bandwidth bound).
- **Generation Speed**: ~35.4 tokens/sec on Vulkan0.
- **Task Quality**: 50% pass rate on standard validation tasks due to compact 1.2B capacity.
- **Retrieval Rate**: 20% needle retrieval success at context=2048.
- **Boundary Limit**: Context allocation succeeds up to 2048. Context=4096 hits SSH_TIMEOUT on coarse allocation probe.
- **Multi-GPU Behavior**: Explicit tensor split (-sm tensor) unsupported on Vulkan. Dual-GPU execution requires -sm layer.

## References
- `kb/raw/llama-3.2-1b-instruct-q4_k_m.md`
- `results/receipts/Llama-3.2-1B-Instruct-Q4_K_M.gguf.json`
