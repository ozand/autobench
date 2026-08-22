---
id: KB-QWEN25-05B-Q4KM
title: Qwen2.5-0.5B-Instruct-Q4_K_M Execution and Layer Split Characteristics
category: model-runbook
model_basename: qwen2.5-0.5b-instruct-q4_k_m.gguf
quantization: Q4_K_M
parameters: 0.49B
layers: 24
architecture: qwen2
context_window: 32768
status: reviewed
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

2. **Dual-GPU Layer Split (-sm layer)**:
   - Split ratio: `1,1` (12 layers on Vulkan0, 12 layers on Vulkan1) or `0.5,0.5`
   - Command flags: `-ngl 33 -sm layer -ts 1,1`
   - Expected behavior: Clean execution across both 2GB VRAM segments.

3. **Unsupported Configurations**:
   - `-sm tensor`: Strictly unsupported on Vulkan (`device does not support split buffers`). Classified as `UNSUPPORTED_BACKEND`.

4. **Context Profiles**:
   - Tested contexts: 512, 1024, 2048, 4096, 8192
   - Quality benchmark prompts: standard `smoke`, `boundary`, `performance`, `quality`, `retrieval`
