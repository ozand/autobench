# Qwen2.5-0.5B-Instruct-Q4_K_M Model Facts (Sanitized)

## Provenance
- Upstream: Qwen / Alibaba Cloud
- Base model: Qwen/Qwen2.5-0.5B
- Instruct model: Qwen/Qwen2.5-0.5B-Instruct
- Quantized GGUF repo: `Qwen/Qwen2.5-0.5B-Instruct-GGUF`
- Local file: `qwen2.5-0.5b-instruct-q4_k_m.gguf`
- Size: `491400032` bytes
- SHA-256: `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`
- License: Apache-2.0

## Architecture
- Architecture family: `qwen2` (Causal LM with RoPE, SwiGLU, RMSNorm, Attention QKV bias, tied word embeddings)
- Parameter count: 0.49B total, 0.36B non-embedding
- Layers: 24
- Attention heads (GQA): 14 Q heads, 2 KV heads
- Head dimension: 64
- Hidden dimension: 896
- Intermediate dimension: 4864
- Vocabulary size: 151936
- Native training context length: 32768 tokens (supports up to 32k; documented benchmark tested range 1024 to 4096 on k7000 dual-GPU testbed)

## Quantization Details
- Quantization type: `Q4_K_M`
- Target model file size: 491,400,032 bytes (~468.6 MiB)
- Fits entirely within VRAM on a single 2GB GTX 690 physical core (~1.95 GiB usable Vulkan device memory).

## Hardware & Backend Applicability (GTX 690 / Vulkan)
- GPU architecture: Kepler (GK104), dual physical GPU cores (Vulkan0 and Vulkan1)
- VRAM per core: 2048 MB physical (~1980 MB usable Vulkan heap)
- Split mode:
  - Multi-GPU `-sm layer` is supported.
  - `-sm row` / tensor split is strictly unsupported due to `device does not support split buffers`.
- KV Cache sizing:
  - Context 1024: ~24 MB
  - Context 2048: ~48 MB
  - Context 4096: ~96 MB
  - Context 8192: ~192 MB (previously observed SSH_TIMEOUT on single-core diagnostic runs in Issue 55 due to prolonged compute; bounded plan targets 1024 -> 2048 -> 4096)
- KV Cache quantization: f16 baseline established first; q8_0/q4_0 excluded in this increment.

## Research Sources
- Upstream model card: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
- Upstream GGUF repository: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF
- Model config: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/raw/main/config.json
- llama.cpp multi-GPU documentation: https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md
