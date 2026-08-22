---
id: KB-QWEN25-05B-Q8_0
title: "Qwen2.5-0.5B-Instruct-Q8_0 Validation and Multi-GPU Analysis"
category: research
tags: [qwen2.5, gguf, q8_0, vulkan, gtx690, okf]
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

# Qwen2.5-0.5B-Instruct-Q8_0 Validation & Multi-GPU Analysis

## Model Overview
- **Identifier**: `qwen2.5-0.5b-instruct-q8_0.gguf`
- **Size**: ~644.4 MiB (675,710,816 bytes)
- **Architecture**: Dense Transformer (24 layers, 14 attention heads, 2 KV heads, GQA, RoPE context 32,768)

## Verified Hardware Execution (k7000 GTX 690)
1. **Single-GPU (Vulkan0)**:
   - Fits entirely in VRAM: 33/33 layers offloaded (`-ngl 33`).
   - Speed: ~18.4 t/s prompt eval, ~34.8 t/s generation.
   - Quality deterministic pass rate: 100% (2/2).
   - Allocation limit: 4096 tokens (8192 hits execution timeout during coarse boundary probe).
   - Repeated needle retrieval @ 4096: 46.7% (7/15 correct) due to capacity limits of 0.49B parameter size.
2. **Single-GPU (Vulkan1)**:
   - Fits entirely in VRAM: 33/33 layers offloaded (`-ngl 33`).
   - Speed: ~18.3 t/s prompt eval, ~34.3 t/s generation.
   - Quality deterministic pass rate: 50% (1/2).
   - Repeated needle retrieval @ 4096: 60.0% (9/15 correct).
3. **Multi-GPU Behavior**:
   - Tensor split (`-sm tensor`) is unsupported on Kepler Vulkan (`device does not support split buffers`).
   - Layer split (`-sm layer`) with `-ts 1,1` is supported when needed for larger models, but unnecessary for this size class (<700 MiB).
