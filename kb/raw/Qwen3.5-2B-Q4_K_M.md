# Qwen3.5-2B-Q4_K_M Research Notes

## Model Identity
- **Model Checkpoint**: `Qwen/Qwen3.5-2B`
- **GGUF File**: `Qwen3.5-2B-Q4_K_M.gguf`
- **Architecture**: `qwen35`
- **Parameters**: 2.45B
- **Quantization**: `Q4_K_M`
- **Native Context Limit**: 32768 tokens

## Hardware & Backend Limits
- **Vulkan Support**: Supported via llama.cpp Vulkan backend.
- **Multi-GPU**: Only `-sm layer` supported.
- **Single-GPU Fit**: Weighs ~1.22 GB; fits in 2GB VRAM only on short contexts; requires dual-GPU layer split or --no-kv-offload for 2048+.
