# llama.cpp multi-GPU documentation source notes

- Source URL: https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md
- Retrieved via Surf CLI: 2026-08-17
- Source type: official llama.cpp documentation
- Scope: multi-GPU split modes and context/KV-cache behavior

## Sanitized source observations

- `none` confines execution to one selected GPU.
- `layer` is the default and most compatible multi-GPU split mode.
- In layer mode, each GPU holds a contiguous layer slice and the KV cache
  for a layer lives on the GPU that owns that layer.
- `tensor` is experimental tensor parallelism and splits weights and KV across
  participating GPUs; it is not guaranteed for every backend or architecture.
- `--tensor-split` proportions follow the order of `--device`; `1,1` gives an
  equal split across two devices.
- The documentation identifies context size as a direct memory-pressure control
  and recommends lowering it when multi-GPU allocation fails.

## Applicability and uncertainty

- Documentation describes llama.cpp behavior, not the local k7000 build,
  Vulkan backend support, or actual device names.
- Local `--list-devices`, load, boundary, and workload probes remain required.
- The default layer path should be preferred for a diagnostic dual-GPU test;
  tensor mode must remain separately marked experimental.
- A successful model load does not prove a successful long-context workload.
