# Qwen2.5-0.5B-Instruct-Q8_0 Research Notes

## Source Provenance
- Official checkpoint card: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
- Official checkpoint configuration: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/raw/main/config.json
- Official generation configuration: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/raw/main/generation_config.json
- Official llama.cpp multi-GPU guide: https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md
- Retrieval date: 2026-09-01
- Method: Surf CLI isolated research plus sanitized target-artifact and runtime probes

## Upstream checkpoint facts
- Checkpoint: Qwen2.5-0.5B-Instruct, instruction-tuned Qwen2 model.
- Parameter count: approximately 0.49B.
- Architecture: `Qwen2ForCausalLM`; 24 hidden layers, 14 attention heads, and 2 key/value heads; hidden size 896.
- Declared context limit: 32,768 tokens (`max_position_embeddings`).
- The official generation configuration declares sampling defaults and does not replace the checkpoint context limit.
- The exact target GGUF is the Q8_0 artifact named `qwen2.5-0.5b-instruct-q8_0.gguf`; upstream files describe the source checkpoint, while the converted GGUF remains a separate artifact.

## Runtime and split facts
- The llama.cpp guide documents `none` for single-device execution and `layer` as the default and most compatible multi-GPU mode.
- In layer mode, GPUs own contiguous layer ranges and the associated KV-cache layers.
- `--tensor-split` proportions follow device order; `1,1` is the balanced two-device ratio.
- The guide treats tensor split as experimental and does not guarantee it for every backend. The k7000 authoritative policy therefore uses layer mode only; tensor/row split is excluded.
- Relevant llama.cpp flags are `-dev`, `-sm layer`, `-ts 1,1`, `-ngl`, `-c`, `-ctk`, and `-ctv`. The default KV types are f16; this increment uses the f16 baseline and does not sweep alternatives.

## Current target-artifact observations
- A fresh target inventory probe reported the exact basename with size `675710816` bytes.
- A fresh target checksum probe reported SHA-256 `ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e`.
- The runtime listed two Vulkan devices, each with 2048 MiB reported memory.
- These are local capability and identity observations, not upstream claims. Historical database rows contain a second size value and are retained as legacy evidence; they do not authorize promotion of those rows.

## Scope and uncertainty for Issue #41
- Governing increment: Issue #41, source `defect found while implementing #41`.
- The prior Issue #56 evidence is diagnostic: both single-GPU 8192 probes ended in SSH timeout and the dual-GPU layer configuration was not executed.
- The next bounded measurement must use the exact artifact identity above, matched Vulkan0/Vulkan1 baselines, and one dual-GPU `-sm layer -ts 1,1` path.
- The 8192 result remains an experiment until measured. An SSH timeout is `INCONCLUSIVE`, not proof of OOM or successful capacity.
- No asymmetric split, tensor/row split, KV quantization, or host-resident KV comparison is authorized in this increment.
- A 600-second primary timeout is selected in the reviewed plan as a bounded execution budget. A 1200-second fallback is not automatic and would require a separately recorded terminal/non-terminal decision if the primary run has no terminal result.
