# Qwen2.5-0.5B-Instruct-Q8_0 Research Notes

## Source Provenance
- Source: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
- Date: 2026-08-22
- Tools: Surf CLI, HuggingFace Hub metadata, llama.cpp GGUF specification

## Checkpoint & Quantization Details
- Model: Qwen2.5-0.5B-Instruct
- Quantization: Q8_0 (8-bit quantization with symmetric range and per-block scales)
- Parameters: 0.49B
- Architecture: Qwen2ForCausalLM (24 transformer layers, hidden_size 896, 14 Q heads, 2 KV heads)
- Context Limit: 32,768 tokens (generation limit 8,192 tokens)
- Local File Size: ~644.4 MiB

## Dual-GPU & Vulkan Architecture Constraints
- Dual-GPU: GTX 690 (2x Kepler GK104, 2GB VRAM per GPU).
- On Vulkan backend, -sm tensor fails with split buffer errors. Only -sm layer is supported.
- Due to small footprint (644.4 MiB), model fits entirely in single-GPU VRAM (Vulkan0 or Vulkan1).
