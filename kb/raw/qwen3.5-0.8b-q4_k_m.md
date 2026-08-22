# Qwen3.5-0.8B-Q4_K_M Research Notes

## Source Metadata
- Model identifier: Qwen/Qwen3.5-0.8B (GGUF: Qwen3.5-0.8B-Q4_K_M.gguf)
- Target architecture: Qwen3.5 dense transformer architecture with GQA/MQA
- Parameter count: ~0.8B (800M) parameters
- Native context length: 32,768 tokens (max generation 8,192 tokens)
- Quantization method: Q4_K_M (4-bit medium k-quant)
- Size: ~507.8 MiB

## Hardware & Backend Applicability (GTX 690 / Vulkan)
- Single-GPU VRAM requirement: ~508 MiB weights + ~128 MiB KV cache at 2K = ~636 MiB (fits easily in 2048 MiB per GK104 die).
- Vulkan driver compatibility: Fully supported with spirv/vulkan shaders.
- Multi-GPU suitability: Single-GPU fits whole model in full offload (-ngl 99). Dual-GPU split via `-sm layer` is feasible with 1:1 or optimal layer split. Explicit tensor split (`-sm tensor`) is unsupported on Vulkan due to missing split buffer extensions.
- KV Cache: Standard f16/q8_0/q4_0 KV caches supported on Vulkan.

## Expected Workloads & Boundaries
- Workload: standard inventory matrix (2048 context target profile).
- Quality tasks: general question answering, reasoning, code generation, summarization, needle retrieval.
