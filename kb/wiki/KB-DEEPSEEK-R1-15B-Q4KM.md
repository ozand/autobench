---
id: KB-DEEPSEEK-R1-15B-Q4KM
title: DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M Matrix and Dual-GPU Placement
category: model-analysis
created: 2026-08-23
updated: 2026-09-10
environment: dual-gtx690-vulkan
status: validated
tags:
  - deepseek-r1
  - 1.5b
  - vulkan
  - gtx690
  - layer-split
error_signatures: []
source_urls:
  - https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
  - https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md
---

# DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M Performance & Matrix

## Summary

### Issue #41 refresh
- Current target-side artifact observation: 1117321312 bytes, SHA-256 `f3bdf9cf31dee4b57ae4e455a1cb0d01b5c2c1b50d72d3112141c195506c2840`.
- The old Issue #42 receipt is governed by Issue #42 and does not authorize an Issue #41 run. A new Issue #41 receipt must bind the current artifact exactly.
- Official configuration confirms `qwen2` architecture, 28 layers, 12 attention heads, 2 KV heads, and native `max_position_embeddings=131072`; this is upstream context metadata, not proof of local capacity.
- Current Issue #41 objective: one bounded same-context dual-GPU layer run at context 1024, with f16 KV and no tensor/row split.


### Historical Issue 42 evidence
- Stage 1/2 research and receipt validation passed for the exact GGUF under the former Issue #42 contract.
- Reviewed execution plan covers Vulkan0, Vulkan1, and dual-GPU Vulkan layer split `1,1`.
- Vulkan0 suite preflight and performance completed with prompt `4.9 t/s` and generation `28.37 t/s`.
- Vulkan0 boundary reached `1024` tokens; the `2048` probe was `SSH_TIMEOUT`, so the boundary remains `INCONCLUSIVE`.
- Vulkan0 retrieval produced 14 `MISSED` and 1 `INCONCLUSIVE` attempt; no `VERIFIED` attempt was observed, so no retrieval pass-rate claim is published.
- Vulkan0 quality stage completed with `0/2` deterministic tasks passed.
- The historical suite remains `PARTIAL_FAILURE` and non-authoritative. It is not reused as Issue #41 publication evidence.
## Historical local observations (not current capability proof)
The following observations were recorded in earlier local work and are retained for provenance only; they do not authorize or substitute for the current Issue #41 run:
- Prior note classified single-GPU operation as short-context fit and larger-context pressure; exact current target capability remains to be established.
- Prior note reported dual-GPU `-sm layer -ts 1,1` operation through context 32768; this is not a current Issue #41 capacity claim.
- Prior note listed f16, q8_0, q4_0, and `--no-kv-offload` as possible KV modes; the current bounded plan selects only the default f16 baseline.
