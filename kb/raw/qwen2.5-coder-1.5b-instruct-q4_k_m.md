# Qwen2.5-Coder-1.5B-Instruct-Q4_K_M Research Notes

## Model Identity
- **Model Checkpoint**: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- **GGUF File**: `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf`
- **Architecture**: `qwen2` (Transformer decoder, GQA, RoPE)
- **Parameters**: 1.54B (active parameters: 1.54B)
- **Quantization**: `Q4_K_M`
- **Native Context Limit**: 32768 tokens

## Hardware & Backend Limits
- **Vulkan Support**: Fully supported by llama.cpp Vulkan backend.
- **Multi-GPU Splitting**: Only `-sm layer` supported on Vulkan testbed (Kepler GTX 690 lacks split buffer hardware support).
- **Single-GPU Fit**: Weighs ~1.11 GB; exceeds safe 2GB single GPU budget on contexts > 1024; dual-GPU layer split (1,1) required for long context.
