---
id: KB-QWEN25-05B-Q4KM
title: "Qwen2.5-0.5B-Instruct-Q4_K_M Execution and Layer Split Characteristics"
category: research
tags: [qwen2.5, gguf, q4_k_m, llama-cpp, vulkan, gtx690, dual-gpu]
status: reviewed
created: 2026-08-22
updated: 2026-09-03
environment:
  os: Linux remote GPU host (k7000)
  shell: POSIX shell
  tools: [Surf CLI, llama.cpp, AutoBench, QMD]
error_signatures:
  - "device does not support split buffers"
  - "SSH_TIMEOUT"
  - "BOUNDARY_SSH_TIMEOUT"
---

# Qwen2.5-0.5B-Instruct-Q4_K_M Execution and Multi-GPU Layer Split

## Model Identity & Checkpoint
- Model name: `qwen2.5-0.5b-instruct-q4_k_m.gguf`
- Architecture: `qwen2` (24 layers, GQA 14 Q heads, 2 KV heads, dim 896)
- File size: `491400032` bytes
- Exact SHA-256: `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`
- Quantization: `Q4_K_M`
- Native context: up to 32768 tokens; bounded benchmark protocol uses `[1024, 2048, 4096]`.

## Testbed Sizing & VRAM Analysis (GTX 690 / Vulkan)
- GPU: NVIDIA GeForce GTX 690 (2x GK104 cores, Vulkan0 and Vulkan1)
- VRAM per core: ~1980 MiB usable heap
- Model weight in VRAM: ~468.6 MiB
- KV cache per context (f16):
  - 1024: ~24 MB
  - 2048: ~48 MB
  - 4096: ~96 MB
  - 8192: ~192 MB (high risk of timeout during long compute)
- Single-GPU fit: Model + KV comfortably fits within 1 core up to 4096 tokens.

## Multi-GPU & Execution Policies
- **Split mode:**
  - Row / tensor split (`-sm row` or `-ts` tensor) is strictly unsupported (`device does not support split buffers`).
  - Layer split (`-sm layer`) is supported.
- **Layer distribution (`-ts 1,1`):**
  - 24 transformer layers split symmetrically: 12 layers on Vulkan0, 12 layers on Vulkan1.
  - Weight VRAM per GPU in dual-layer mode: ~235 MiB.
- **Context & Timeout Strategy:**
  - Bounded context progression: `1024 -> 2048 -> 4096`.
  - Primary timeout: 600 seconds per suite job (ADR-001/ADR-002 policy).
  - Fallback 1200 seconds is only authorized conditionally after an initial non-terminal 600s timeout.

## Governing Protocol & Receipts
- Raw notes: `kb/raw/qwen2.5-0.5b-instruct-q4_k_m.md`
- Dedicated receipt: `results/receipts/qwen2.5-0.5b-instruct-q4_k_m.issue41.json`
- Follow-up plan: `docs/issue41-qwen-q4km-followup-plan.json`
- Historical diagnostics: Issue #55 observed SSH_TIMEOUT on 8192 context boundary probe. The Issue #41 follow-up bounds contexts to 1024/2048/4096 to prevent hang while establishing authoritative dual-GPU layer baselines.
