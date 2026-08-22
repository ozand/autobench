# KV-Cache Quantization & Host-Resident Placement Plan

This guide supports the context/KV decision in Stage 3 of the mandatory
[`model-testing-protocol.md`](model-testing-protocol.md). It does not authorize a
full flag matrix. Establish the default/f16 baseline, then select only the
smallest evidence-backed KV alternative needed to test the model-specific
context hypothesis. Preview and review the exact plan before inference.

## 1. Context & Motivation
On dual-GPU and memory-constrained architectures like NVIDIA Kepler (GTX 690, 2x 2GB VRAM), VRAM is primarily consumed by:
1. Model weights (GGUF tensors offloaded via `-ngl`).
2. Key-Value (KV) cache allocated for the requested context window (`-c`).

By default, llama.cpp allocates KV cache in full 16-bit float (`f16`). Quantizing KV cache to `q8_0` or `q4_0` significantly lowers memory consumption per token, enabling longer context lengths without causing out-of-memory errors on 2GB GPUs.

## 2. Supported llama.cpp Flags
- `-ctk <type>`: KV cache type for K (e.g. `f16`, `q8_0`, `q4_0`).
- `-ctv <type>`: KV cache type for V (e.g. `f16`, `q8_0`, `q4_0`).
- `--no-kv-offload`: Keep the KV cache in host RAM rather than GPU VRAM.

## 3. Evaluation Matrix

| Configuration | Flag String | Target Context | Expected VRAM Impact | Quality / Perplexity Impact |
|---|---|---|---|---|
| **Default Baseline** | `-ctk f16 -ctv f16` | 1024 / 2048 | Baseline VRAM | Reference |
| **8-bit Balanced** | `-ctk q8_0 -ctv q8_0` | 2048 / 4096 | ~50% KV VRAM reduction | Negligible degradation (<0.1% perplexity delta) |
| **4-bit Maximum Compression** | `-ctk q4_0 -ctv q4_0` | 4096 / 8192 | ~75% KV VRAM reduction | Minor degradation; suitable for retrieval / summarization |
| **Host-Resident KV Cache** | `--no-kv-offload` | 8192+ | 100% KV offloaded to RAM | Higher token generation latency; zero GPU KV footprint |

## 4. Runner Integration
The AutoBench `Runner.run_local_vulkan` supports explicit parameters:
```python
Runner.run_local_vulkan(
    prompt="...",
    cache_type_k="q8_0",
    cache_type_v="q4_0",
    no_kv_offload=False,
)
```

## 5. Deployment & Execution Protocol
1. Complete Stages 1 and 2 of
   [`model-testing-protocol.md`](model-testing-protocol.md) for the exact GGUF.
2. In the reviewed Stage 3 plan, define a bounded context progression chosen
   from `1024 -> 2048 -> 4096 -> 8192`, without exceeding the researched
   checkpoint limit.
3. Establish the default/f16 KV baseline first. Select `q8_0`, `q4_0`, or
   host-resident placement only when a stated capacity or latency hypothesis
   requires it; do not sweep every option.
4. Run local source validation for runner changes, including
   `test_runner_supports_kv_quantization_and_host_offload` when applicable.
5. Preview the exact remote plan with `--dry-run`, verify job count and flags,
   then stop for explicit review. Dry-run success does not authorize inference.
6. Execute the approved comparisons serially and stop on the first unexpected
   result. Follow `problem-investigation-plan.md` instead of trying another KV
   mode blindly.
7. Review speed, memory, quality, and terminal classification before updating
   the model OKF note or baseline profiles.
