---
id: KB-QWEN25-CODER-3B-Q4KM
title: Qwen2.5-Coder-3B-Instruct-Q4_K_M Vulkan Diagnostic
category: model-analysis
status: validated
created: 2026-08-26
updated: 2026-08-26
tags:
  - qwen2.5
  - coder
  - 3b
  - q4_k_m
  - vulkan
environment:
  target_host: k7000
  gpu: GTX 690 dual-GPU Vulkan testbed
  backend: llama.cpp Vulkan
source_urls:
  - https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct/raw/main/config.json
  - https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md
error_signatures: []
---

# Qwen2.5-Coder-3B-Instruct-Q4_K_M

## Upstream facts
- Architecture: `qwen2` / `Qwen2ForCausalLM`.
- 36 hidden layers, 16 attention heads, 2 KV heads, hidden size 2048.
- Declared context limit: 32768 tokens.
- Quantization under test: `Q4_K_M`.
- llama.cpp documents `layer` as the default and most compatible multi-GPU split mode.

## Local plan
- Matched Vulkan0 and Vulkan1 baselines.
- Dual-GPU `-sm layer -ts 1,1`; tensor/row split excluded.
- Bounded context progression: `1024 -> 2048 -> 4096 -> 8192`, stopping at the first unexpected result.
- f16 KV baseline before any justified alternative.

## Historical disposition
The existing two rows are partial failures with zero metrics and remain non-authoritative.

## Issue 60 execution evidence
- Exact receipt validated under Issue 60 after the research and KB/QMD gates passed.
- Dry-run produced three jobs: Vulkan0 load-only, Vulkan1 load-only, and dual-GPU layer `1,1` full.
- Vulkan0 load probe: `SUCCESS`.
- Vulkan1 load probe: `SUCCESS`.
- Dual-GPU layer preflight: `SUCCESS`, but the first boundary probe at context `1024` ended `SSH_TIMEOUT`; the boundary is `INCONCLUSIVE` and the workload is unsupported.
- Performance, Retrieval, and quality were not attempted for the dual-GPU configuration.
- Final disposition: `NON_AUTHORITATIVE`; no metrics are published.

## Safety
No raw prompts, responses, runtime output, private paths, credentials, or host identifiers are persisted.
