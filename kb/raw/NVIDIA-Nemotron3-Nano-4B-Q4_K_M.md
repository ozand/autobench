# NVIDIA-Nemotron3-Nano-4B-Q4_K_M Research Notes

- Source URL: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF
- Source URL: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/
- Source URL: https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md
- Retrieved via Surf CLI: 2026-08-24

## Model Identity
- GGUF: `NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf`
- Upstream GGUF repository: `nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF`
- Quantization: `Q4_K_M`
- Model class: Nemotron 3 Nano, 4B class

## Runtime Applicability
- llama.cpp GGUF execution and Vulkan backend are the local runtime path.
- Authoritative multi-GPU testing uses Vulkan `-sm layer` with an explicit split; tensor/row split is excluded.
- The exact local artifact is present on the k7000 model store and must be tested through bounded load/context probes.
- Upstream model metadata does not prove local VRAM fit or context capacity.

## Unresolved Assumptions
- Exact architecture metadata and local context limits for this conversion require receipt-backed execution evidence.
- Historical rows contain only preflight failures and provide no verified speed or Retrieval data.
