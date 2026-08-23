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
