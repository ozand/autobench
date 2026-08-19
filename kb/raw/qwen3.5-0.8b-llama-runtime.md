# Qwen3.5-0.8B and llama.cpp runtime — sanitized source notes

- Sources:
  - https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md
  - https://github.com/ggml-org/llama.cpp/issues/27237
  - https://github.com/ggml-org/llama.cpp/pull/27132
- Retrieved: 2026-08-19

## Source-backed facts

- llama.cpp documents `layer` as the default and most compatible multi-GPU split; `tensor` is experimental.
- The multi-GPU guide says tensor split is not implemented for several hybrid/state-space architectures, while layer split is the default path to prefer for compatibility.
- Qwen3.5 uses a hybrid architecture with Gated DeltaNet/linear-attention components according to the official model card and config.
- llama.cpp upstream search shows active Qwen3.5 hybrid/DeltaNet Vulkan and conversion work, including a report of Vulkan output sensitivity to batch size for Qwen3.5-27B and a draft conversion fix for qwen3_5 hybrid linear-attention tensors.

## Local observations

- The discovered GGUF `Qwen3.5-0.8B-Q4_K_M.gguf` loaded and generated successfully on both Vulkan0 and Vulkan1 at context 1024.
- Embedded tokenizer smoke test succeeded with 13 tokens.
- The standard AutoBench suite boundary and quality stages completed, but retrieval and performance stages were recorded as `TOKENIZER_ERROR` on both devices. This is a stage-specific failure, not evidence that model loading or Vulkan execution failed.

## Interpretation and unresolved assumptions

- The local tokenizer failure is not yet attributed to an upstream conversion defect; the suite redacts the detailed exception. A direct tokenizer and short generation smoke test succeeded, so the first investigation target is the generated long-context/calibration prompt path or a Qwen3.5 hybrid tokenizer/runtime interaction.
- No dual-GPU workload is authorized until the single-device stage failure is investigated. If later attempted, use default `layer`, not experimental `tensor`, unless separately justified.
- Vision capability is not evaluated by the text-only AutoBench suite.
