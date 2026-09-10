# DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M Research Notes

- Source URL: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
- Source URL: https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md
- Retrieved via Surf CLI: 2026-09-10
- This note was refreshed for the Issue #41 follow-up; the prior Issue #42 evidence remains historical and non-authoritative.

## Model Identity
- **Model Checkpoint**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- **GGUF File**: `DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf`
- **Architecture**: `qwen2` (DeepSeek R1 reasoning distill based on Qwen2.5-1.5B base)
- **Parameters**: 1.78B (active parameters: 1.78B)
- **Quantization**: `Q4_K_M`
- **Native Context Limit**: 131072 tokens (official `config.json`; prior local testing reached 32768)
- **Architecture details (official `config.json`)**: 28 hidden layers, 12 attention heads, 2 key/value heads, hidden size 1536, intermediate size 8960, sliding-window field 4096 with `use_sliding_window=false`

## Hardware & Backend Limits
- **Vulkan Support**: Upstream llama.cpp documentation describes Vulkan multi-GPU support, but exact target-build/device capability remains unresolved until local preflight; this is not inferred from the source alone.
- **Multi-GPU Splitting**: Only `-sm layer` supported on Vulkan testbed due to lack of split buffers on Kepler GTX 690.
- **Single-GPU Fit**: Weighs ~1.12 GB in the prior local note; short-context fit is reported, but current target capability must be established by the bounded run.
- **Current target artifact**: `DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf`; target-side observation on 2026-09-10 was 1117321312 bytes with SHA-256 `f3bdf9cf31dee4b57ae4e455a1cb0d01b5c2c1b50d72d3112141c195506c2840`. This exact binding is for the current Issue #41 receipt only.
