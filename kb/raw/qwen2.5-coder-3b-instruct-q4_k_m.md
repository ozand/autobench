# Qwen2.5-Coder-3B-Instruct-Q4_K_M Research Notes

- Source URL: https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct/raw/main/config.json
- Source URL: https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md
- Retrieved via Surf CLI: 2026-08-26

## Model Identity
- Checkpoint: `Qwen/Qwen2.5-Coder-3B-Instruct`
- GGUF: `qwen2.5-coder-3b-instruct-q4_k_m.gguf`
- Architecture: `qwen2` / `Qwen2ForCausalLM`
- Layers: 36
- Attention: 16 heads, 2 KV heads; hidden size 2048
- Declared context limit: 32768 tokens
- Quantization: `Q4_K_M`

## Runtime Applicability
- Target runtime: llama.cpp GGUF through Vulkan.
- llama.cpp documents `layer` as the default and most compatible multi-GPU split mode.
- The GTX 690 Vulkan testbed excludes tensor/row split because split buffers are unsupported; authoritative multi-GPU mode is `-sm layer`.
- Exact artifact identity and local memory behavior require receipt-backed probes.

## Local Prior State
- Historical rows for this exact artifact are partial failures with zero metrics and non-authoritative classification.
