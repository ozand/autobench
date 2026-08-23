# Qwen2.5-Coder-1.5B-Instruct-Q8_0 Research Notes

## Model Identity
- **Model Checkpoint**: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- **GGUF File**: `qwen2.5-coder-1.5b-instruct-q8_0.gguf`
- **Architecture**: `qwen2`
- **Parameters**: 1.54B (full 8-bit quantized weights)
- **Quantization**: `Q8_0`
- **Native Context Limit**: 32768 tokens

## Hardware & Backend Limits
- **Vulkan Support**: Supported via llama.cpp Vulkan backend.
- **Multi-GPU Splitting**: Only `-sm layer` supported.
- **Single-GPU Fit**: Model weights alone are ~1.81 GB (1,894,532,160 bytes). On a 2 GB VRAM GPU, less than 200 MB remains for KV cache and scratch buffers, causing immediate OOM on single GPU at context >= 1024. Dual-GPU (1,1 layer split across 2x2GB) is mandatory.
