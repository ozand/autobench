# Phi-4-mini-instruct-Q4_K_M Research Notes

- Source URL: https://huggingface.co/microsoft/Phi-4-mini-instruct
- Source URL: https://aka.ms/phi4-feb2025
- Source URL: https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md
- Retrieved via Surf CLI: 2026-08-24

## Model Identity
- Checkpoint: `microsoft/Phi-4-mini-instruct`
- GGUF: `Phi-4-mini-instruct-Q4_K_M.gguf`
- Quantization: `Q4_K_M`
- Model class: Phi-4 Mini instruct, 4B class

## Runtime Applicability
- The local execution path is llama.cpp GGUF through Vulkan.
- Authoritative multi-GPU testing uses Vulkan `-sm layer`; tensor/row split is excluded.
- The exact GGUF is present on the k7000 model store and requires bounded load/context validation.
- Upstream context and architecture claims do not prove local VRAM fit or runtime compatibility.

## Unresolved Assumptions
- Exact converted GGUF architecture metadata and local context capacity require receipt-backed probes.
- Historical rows are non-authoritative partial failures and provide no trustworthy speed or Retrieval metrics.
