---
id: QWEN35-001
title: "Qwen3.5-0.8B Q4_K_M Vulkan diagnostic"
category: research
tags: [qwen3.5, gguf, q4_k_m, llama-cpp, vulkan, hybrid-attention, tokenizer]
status: active
created: 2026-08-19
updated: 2026-08-19
environment:
  os: Linux remote GPU host
  shell: POSIX shell
  tools: [Surf CLI, llama.cpp, AutoBench, QMD]
error_signatures:
  - "TOKENIZER_ERROR"
---

# Qwen3.5-0.8B Q4_K_M Vulkan diagnostic

## Scope

Concrete artifact: `Qwen3.5-0.8B-Q4_K_M.gguf`. Governing benchmark work is
tracked under the authoritative-matrix issue. The text-only Vulkan diagnostic
was run separately on Vulkan0 and Vulkan1 with route validation enabled.

## Upstream facts

- The official Qwen3.5-0.8B model card declares 0.8B parameters and native
  context length 262,144 tokens.
- The model is a multimodal causal model with a vision encoder. This diagnostic
  exercises only text generation; it is not a vision-capability evaluation.
- The configuration identifies a Qwen3.5 hybrid architecture with linear
  attention/Gated DeltaNet components, full-attention intervals, 24 language
  layers, and 2 KV heads.
- llama.cpp documents `layer` as the default and most compatible multi-GPU
  split. Tensor split is experimental and has architecture limitations.
- Upstream llama.cpp currently contains active Qwen3.5 hybrid conversion and
  Vulkan-related work. This establishes a relevant compatibility risk, not a
  diagnosis of the local artifact.

## Local observations

- The GGUF loaded successfully on both Vulkan0 and Vulkan1 at context 1024.
- An embedded-tokenizer smoke test succeeded.
- A short direct generation smoke test succeeded on Vulkan0 with positive
  prompt and generation rates.
- The bounded AutoBench suite completed load/preflight, boundary, and quality
  stages on both single-GPU configurations. Allocation reached the requested
  1024 context, but this is only a lower bound because the probe range stopped
  there.
- The suite's retrieval and performance stages were recorded as
  `TOKENIZER_ERROR` on both devices. The persisted artifact intentionally
  redacts the detailed exception.
- A direct matched context test using the same Qwen3.5 GGUF successfully built
  and executed a calibrated prompt at context 1024 with 16 or 64 output-token
  reservation. This confirms that the tokenizer and execution path can work for
  bounded prompts, but does not repair the standard performance-stage result.
- A bounded retrieval rerun at context 1024 completed all three positions, but
  retrieval correctness was 0/3. Performance measured approximately 12.3
  prompt tokens/s and 36.0 generation tokens/s in that diagnostic slice.

## Classification

- Vulkan load/preflight: SUCCESS.
- Boundary allocation at 1024: SUCCESS, lower bound.
- Quality stage: SUCCESS in the suite artifact.
- Standard retrieval stage: PARTIAL_FAILURE / TOKENIZER_ERROR in the suite;
  bounded rerun executed but retrieval correctness was 0%.
- Standard performance stage: TOKENIZER_ERROR in the suite and in the bounded
  performance-stage invocation.
- Overall model result: PARTIAL_FAILURE, with tokenizer/calibration/runtime
  compatibility unresolved. It is not OOM and not a backend-load failure.

## Investigation and stop decision

Local KB/QMD lookup did not provide a matching Qwen3.5-specific tokenizer
lesson. Surf research found relevant upstream Qwen3.5 hybrid conversion and
Vulkan work, including an active report of Vulkan output sensitivity to batch
size on a larger Qwen3.5 model. Because the suite's detailed exception is
redacted and the direct bounded path succeeds, evidence is insufficient to
attribute the failure to a single upstream defect. The governed one-variable
rerun allowance was consumed by the bounded retrieval/performance investigation.
No parameter sweep, dual-GPU workload, or next-model workload was started.

## References

- [Qwen3.5-0.8B model-card notes](../raw/qwen3.5-0.8b-model-card.md)
- [Qwen3.5-0.8B llama.cpp runtime notes](../raw/qwen3.5-0.8b-llama-runtime.md)
- [Official Qwen3.5-0.8B model card](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- [Qwen3.5 configuration](https://huggingface.co/Qwen/Qwen3.5-0.8B/raw/main/config.json)
- [llama.cpp multi-GPU guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md)
- [llama.cpp Qwen3.5 Vulkan report](https://github.com/ggml-org/llama.cpp/issues/27237)
- [llama.cpp Qwen3.5 conversion work](https://github.com/ggml-org/llama.cpp/pull/27132)
