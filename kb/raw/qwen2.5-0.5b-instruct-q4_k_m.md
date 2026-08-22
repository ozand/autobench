# Qwen2.5-0.5B-Instruct-GGUF Model Facts (Sanitized)

## Provenance
- Upstream: Qwen / Alibaba Cloud
- Base model: Qwen/Qwen2.5-0.5B
- Instruct model: Qwen/Qwen2.5-0.5B-Instruct
- Quantized GGUF repo: `Qwen/Qwen2.5-0.5B-Instruct-GGUF`
- License: Apache-2.0

## Architecture
- Architecture family: `qwen2` (Causal LM with RoPE, SwiGLU, RMSNorm, Attention QKV bias, tied word embeddings)
- Parameter count: 0.49B total, 0.36B non-embedding
- Layers: 24
- Attention heads (GQA): 14 Q heads, 2 KV heads
- Context length: Full 32,768 tokens (generation up to 8,192 tokens)
- Native RoPE base frequency: 1,000,000

## Quantization Details
- Model: `qwen2.5-0.5b-instruct-q4_k_m.gguf`
- Quantization type: `Q4_K_M` (medium k-quant: 4-bit attention & feed-forward mixes)
- File size: ~491 MB on disk (468.64 MiB)
- Fits fully in a single 2GB VRAM partition (`Vulkan0` or `Vulkan1` on GTX 690 / K7000 testbed)

## Testbed Specifics (k7000 / GTX 690)
- GPUs: 2x GK104 (2048 MB VRAM per GPU)
- Vulkan devices: `Vulkan0`, `Vulkan1`
- Layer split behavior:
  - 24 layers total.
  - Single GPU: `ngl=33` (all 24 layers offloaded to 1 GPU, VRAM fits with room for context).
  - Dual GPU: `-sm layer` with split `1,1` (12 layers / 12 layers).
  - `-sm tensor` unsupported on Vulkan (`device does not support split buffers`).
- KV offload: supported on single and dual GPU for standard context sizes (up to 8192 tested safely).
