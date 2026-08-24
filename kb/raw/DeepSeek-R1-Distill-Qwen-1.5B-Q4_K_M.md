# DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M Research Notes

- Source URL: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
- Source URL: https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md
- Retrieved via Surf CLI: 2026-08-24

## Model Identity
- **Model Checkpoint**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- **GGUF File**: `DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf`
- **Architecture**: `qwen2` (DeepSeek R1 reasoning distill based on Qwen2.5-1.5B base)
- **Parameters**: 1.78B (active parameters: 1.78B)
- **Quantization**: `Q4_K_M`
- **Native Context Limit**: 131072 tokens (tested progression up to 32768)

## Hardware & Backend Limits
- **Vulkan Support**: Yes, supported by llama.cpp Vulkan backend.
- **Multi-GPU Splitting**: Only `-sm layer` supported on Vulkan testbed due to lack of split buffers on Kepler GTX 690.
- **Single-GPU Fit**: Weighs ~1.12 GB; fits in 2GB single GPU memory with short context; requires layer splitting or --no-kv-offload above 4096.
