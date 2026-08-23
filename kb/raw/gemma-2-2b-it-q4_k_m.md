# Gemma-2-2B-IT-Q4_K_M Research Notes

## Model Identity
- **Model Checkpoint**: `google/gemma-2-2b-it`
- **GGUF File**: `gemma-2-2b-it-Q4_K_M.gguf`
- **Architecture**: `gemma2` (Sliding window attention + Global attention, logit soft-capping)
- **Parameters**: 2.61B (active: 2.61B)
- **Quantization**: `Q4_K_M`
- **Native Context Limit**: 8192 tokens

## Hardware & Backend Limits
- **Vulkan Support**: Supported via llama.cpp Vulkan backend.
- **Multi-GPU Splitting**: Only `-sm layer` supported.
- **Single-GPU Fit**: Model weights alone are ~1.63 GB. On a 2 GB VRAM GPU, it cannot allocate sufficient KV cache for context > 1024; dual-GPU layer split `-sm layer 1,1` is required for full inference.
