---
id: KB-QWEN25-05B-Q4KM
title: "Qwen2.5-0.5B-Instruct-Q4_K_M Execution and Layer Split Characteristics"
category: research
tags: [qwen2.5, gguf, q4_k_m, llama-cpp, vulkan, gtx690, dual-gpu]
status: reviewed
created: 2026-08-22
updated: 2026-08-22
environment:
  os: Linux remote GPU host (k7000)
  shell: POSIX shell
  tools: [Surf CLI, llama.cpp, AutoBench, QMD]
error_signatures:
  - "device does not support split buffers"
  - "SSH_TIMEOUT"
  - "BOUNDARY_SSH_TIMEOUT"
---

# Qwen2.5-0.5B-Instruct-Q4_K_M Runbook & Knowledge Base

## Summary
`qwen2.5-0.5b-instruct-q4_k_m.gguf` is an instruction-tuned 0.49B causal language model in Q4_K_M quantization (468.64 MiB).

## Architectural Attributes
- Total Layers: 24
- Attention Heads: 14 Q heads, 2 KV heads (Grouped Query Attention)
- Context Limit: 32,768 native tokens
- Embedding Tie: Yes

## Hardware Execution Matrix (GTX 690 / Vulkan)
1. **Single-GPU (Vulkan0 / Vulkan1)**:
   - Command flags: `-ngl 33 -dev Vulkan0` (or `Vulkan1`)
   - VRAM footprint: ~491 MB weights + context
   - Max offloaded layers: 24/24
   - KV Cache offload: Enabled
   - Allocation: Reaches 4096 context. Context 8192 hits boundary timeout (>190s).
   - Retrieval Pass Rate: 66.7% (10/15 passes at context 4096).
   - Prompt Speed: ~17.9 - 18.0 t/s
   - Generation Speed: ~34.1 - 34.5 t/s

2. **Dual-GPU Layer Split (-sm layer)**:
   - Split ratio: `1,1` (12 layers on Vulkan0, 12 layers on Vulkan1)
   - Command flags: `-ngl 33 -sm layer -ts 1,1`
   - Expected behavior: Clean execution across both 2GB VRAM segments.

3. **Unsupported Configurations**:
   - `-sm tensor`: Strictly unsupported on Vulkan (`device does not support split buffers`). Classified as `UNSUPPORTED_BACKEND`.

4. **Classification & Findings**:
   - Model is fully functional on single-GPU Vulkan up to 4096 context with 100% task quality pass rate on standard validation tasks.
   - Retrieval degradation observed at positions 0.5 and 0.9 under context 4096 due to model capacity limits (0.49B parameters).
   - Multi-GPU tensor split is rejected by Vulkan runtime; dual-GPU layer split is the approved multi-GPU mode.
