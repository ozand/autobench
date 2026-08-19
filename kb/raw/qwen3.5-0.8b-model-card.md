# Qwen3.5-0.8B model card — sanitized source notes

- Source: https://huggingface.co/Qwen/Qwen3.5-0.8B/raw/main/README.md
- Retrieved: 2026-08-17
- Source identity: Qwen/Qwen3.5-0.8B, Apache-2.0.

## Source-backed facts

- The checkpoint contains 0.8B parameters.
- It is a causal language model with a vision encoder and unified vision-language architecture.
- The language model has 24 layers, hidden size 1024, vocabulary size 248,320, and tied input/output embeddings.
- The model card declares native context length 262,144 tokens.
- Qwen3.5-0.8B operates in non-thinking mode by default; thinking mode is separately configurable.
- The model card describes current serving examples for Transformers, vLLM, SGLang, and KTransformers. It does not establish llama.cpp GGUF/Vulkan compatibility.

## Applicability

The locally discovered artifact is a GGUF quantization named `Qwen3.5-0.8B-Q4_K_M.gguf`. The model card is authoritative for model identity and declared architecture/context, but not for the local GGUF conversion or k7000 Vulkan support. Those remain hardware/runtime questions.

## Unresolved assumptions

- GGUF metadata and conversion fidelity must be checked on the target host before inference.
- The multimodal vision path is outside the text-only AutoBench workload; benchmark results must not be interpreted as vision capability results.
- The declared 262,144-token context is an upstream claim and requires local boundary evidence; it is not a promise for this Vulkan host.
