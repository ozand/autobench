# SmolLM2-1.7B-Instruct-Q4_K_M Research Notes

## Model Identity
- **Model Checkpoint**: `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- **GGUF File**: `smollm2-1.7b-instruct-q4_k_m.gguf`
- **Architecture**: `llama` (SmolLM2 transformer, Grouped Query Attention)
- **Parameters**: 1.71B (active parameters: 1.71B)
- **Quantization**: `Q4_K_M`
- **Native Context Limit**: 8192 tokens

## Hardware & Backend Limits
- **Vulkan Support**: Yes, supported by llama.cpp Vulkan backend.
- **Multi-GPU Splitting**: Only `-sm layer` supported on Vulkan testbed due to lack of split buffers on Kepler GTX 690.
- **Single-GPU Fit**: Weighs ~1.05 GB; fits in 2GB single GPU memory with short context; requires layer splitting or --no-kv-offload above 4096.
