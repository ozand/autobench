# KV-Cache Quantization & Host-Resident Placement Plan

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
1. Dry-run validation locally via unit tests (`test_runner_supports_kv_quantization_and_host_offload`).
2. When remote host connectivity is available, execute bounded diagnostic comparisons across `q8_0`, `q4_0`, and host-resident placement.
3. Review speed (t/s) and memory consumption tradeoffs before updating baseline profiles.
