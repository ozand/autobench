---
id: KB-LLAMA32-1B-Q4KM
title: "Llama-3.2-1B-Instruct-Q4_K_M Validation and Multi-GPU Analysis"
category: research
tags: [llama-3.2, 1b, gguf, q4_k_m, vulkan, gtx690, okf]
status: reviewed
created: 2026-08-22
updated: 2026-08-22
environment:
  os: Linux remote GPU host (k7000)
  shell: POSIX shell
  tools: [Surf CLI, llama.cpp, AutoBench, QMD]
error_signatures:
  - "device does not support split buffers"
  - "SSH execution timed out after 190 seconds"
---

# Llama-3.2-1B-Instruct-Q4_K_M Validation & Multi-GPU Analysis

## Model Overview
- **Identifier**: `Llama-3.2-1B-Instruct-Q4_K_M.gguf`
- **Size**: ~770.3 MiB
- **Architecture**: Llama3 (16 layers, 32 attention heads, 8 KV heads, GQA, RoPE context 131,072)

## Verified Hardware Execution (k7000 GTX 690)
1. **Single-GPU (Vulkan0 / Vulkan1)**:
   - Fits entirely in VRAM: 17/17 layers offloaded.
   - Prompt Speed: ~12.2 t/s
   - Generation Speed: ~28.5 t/s
   - Allocation limit: 4096 tokens (8192 hits execution timeout).
   - Repeated needle retrieval @ 4096: 66.7% (10/15 correct).
   - Quality deterministic pass rate: 100% (2/2).
2. **Multi-GPU Behavior**:
   - Tensor split (`-sm tensor`) is unsupported on Vulkan (`device does not support split buffers`).
   - Layer split (`-sm layer`) with `-ts 1,1` is supported when required.
