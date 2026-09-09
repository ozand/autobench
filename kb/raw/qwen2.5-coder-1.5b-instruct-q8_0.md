# Qwen2.5-Coder-1.5B-Instruct-Q8_0 Research Notes (Issue #41)

## Model Identity
- Model checkpoint: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- GGUF file: `qwen2.5-coder-1.5b-instruct-q8_0.gguf`
- Architecture: `qwen2`
- Parameters: 1.54B
- Quantization: `Q8_0`
- Native context limit: 32768 tokens
- Target artifact size: `1894532160` bytes
- Target artifact SHA-256: `507de59046601282ba768a9789900e6ccf60ed93ddf346730b7c68eb0715bc47`

## Hardware & Backend Limits
- Vulkan backend: supported by the target llama.cpp build.
- Multi-GPU: only `-sm layer` is permitted by project policy; tensor/row split is excluded because Vulkan reports unsupported split buffers.
- Single-GPU fit: not applicable for the reviewed full workload. The 1.89 GB weights leave insufficient headroom on a 2 GB Vulkan device for the required context and scratch allocations.
- Dual-GPU: `Vulkan0,Vulkan1`, `-sm layer`, `-ts 1,1` is the reviewed configuration.
- KV policy: f16 baseline only; no KV sweep is authorized.

## Prior Evidence
- Issue #59 used a broader 3-job plan. Both single-GPU boundary probes became inconclusive at context 1024 (`BOUNDARY_SSH_TIMEOUT`), and its speed/Retrieval/quality values were diagnostic only.
- The current Issue #41 follow-up therefore removes inapplicable single-GPU jobs and uses one same-context dual-GPU job at context 1024 for all publication stages.
- Prior Issue #41 Qwen Coder Q4_K_M results demonstrate that mixed context values must not be written as one flat authoritative row; this follow-up avoids that mismatch.

## Sources
- Reported upstream model source: https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct
- Reported llama.cpp multi-GPU source: https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md
- Local verification required: target artifact checksum, receipt validation, zero-inference dry-run, and one serial dual-GPU execution.
