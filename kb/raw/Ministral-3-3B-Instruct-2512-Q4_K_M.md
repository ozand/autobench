# Ministral-3-3B-Instruct-2512-Q4_K_M Research Notes (Issue #41)

## Provenance
- Source URL: https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512
- Source URL: https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512/raw/main/config.json
- Source URL: https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512/raw/main/generation_config.json
- Source URL: https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md
- Retrieved via Surf CLI: 2026-09-10.
- The prior Issue #44 result remains historical/non-authoritative and cannot authorize Issue #41 inference.

## Model Identity and upstream facts
- Checkpoint: `mistralai/Ministral-3-3B-Instruct-2512`
- GGUF: `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf`
- Current target artifact observation: 2146497824 bytes, SHA-256 `fd46fc371ff0509bfa8657ac956b7de8534d7d9baaa4947975c0648c3aa397f4`.
- Official `config.json`: `Mistral3ForConditionalGeneration`, text model type `ministral3`, 26 text layers, hidden size 3072, intermediate size 9216, 32 attention heads, 8 KV heads, vocabulary 131072, native text `max_position_embeddings=262144`, YaRN parameters with original position limit 16384. The checkpoint also has a vision tower and multimodal projector.
- Official `generation_config.json`: `max_length=262144`. These upstream limits do not prove GGUF conversion or exact target-build support.

## Backend and capacity assumptions
- llama.cpp documents `-sm layer` as the compatible pipeline-parallel multi-GPU mode; project policy excludes tensor/row split on this Vulkan testbed.
- Target Vulkan/llama.cpp support, GGUF conversion compatibility, and local capacity remain unresolved until target preflight.
- The prior Issue #44 run observed single-GPU OOM and a dual-layer boundary SSH timeout; these are preserved diagnostics, not a proof that a current dual-layer same-context run cannot succeed.
- Issue #41 first establishes the default f16 K/V baseline in a bounded dual-GPU `Vulkan0,Vulkan1`, `-sm layer`, `-ts 1,1` run. No blind KV sweep is authorized.
