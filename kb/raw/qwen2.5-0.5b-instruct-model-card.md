# Qwen2.5-0.5B-Instruct model-card source notes

- Source URL: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/raw/main/README.md
- Retrieved via Surf CLI: 2026-08-17
- Source type: official Hugging Face model-card README
- Scope: concrete Qwen2.5-0.5B-Instruct checkpoint, not the whole Qwen2.5 family

## Sanitized source-backed facts

- The repository contains the instruction-tuned 0.5B Qwen2.5 model.
- The card states 0.49B parameters and 0.36B non-embedding parameters.
- The card states 24 layers, 14 query attention heads, and 2 key/value heads.
- The card states full context length of 32,768 tokens and generation of 8,192
  tokens.
- The card separately describes the wider Qwen2.5 series as supporting long
  context up to 128K; that family statement must not replace the
  checkpoint-specific 32,768 value.
- The card links the Qwen project, documentation, and llama.cpp-related usage
  paths for quantized derivatives.

## Applicability and uncertainty

- These are upstream model-card facts, not proof of local GGUF metadata,
  Vulkan support, available VRAM, or successful execution on k7000.
- The concrete checkpoint context declaration must be checked against the
  actual GGUF and validated by bounded hardware probes before scheduling a
  larger run.
- The card documents the original Transformers checkpoint; quantization and
  conversion can introduce runtime-specific behavior.
