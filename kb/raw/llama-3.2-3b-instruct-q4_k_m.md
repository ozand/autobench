# Llama-3.2-3B-Instruct-Q4_K_M Research Notes

## Model Identity
- **Model Checkpoint**: `meta-llama/Llama-3.2-3B-Instruct`
- **GGUF File**: `llama-3.2-3b-instruct-q4_k_m.gguf`
- **Architecture**: `llama` (Grouped Query Attention, 128k native RoPE context)
- **Parameters**: 3.21B
- **Quantization**: `Q4_K_M`
- **Native Context Limit**: 131072 tokens

## Hardware & Backend Limits
- **Vulkan Support**: Supported via llama.cpp Vulkan backend.
- **Multi-GPU Splitting**: Only `-sm layer` supported.
- **Single-GPU Fit**: Model weights alone are ~1.93 GB (2,019,377,600 bytes). This completely saturates a single 2 GB VRAM GPU before KV cache allocation, causing OOM on single GPU. Dual-GPU `-sm layer 1,1` (4 GB total) is strictly required.
