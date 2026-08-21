# Multi-GPU Split Modes and Limits (k7000)

## Overview

AutoBench executes local GGUF models across Vulkan devices on the `k7000` hardware configuration (dual GTX 690 / 2GB per GPU slice). This document defines the supported and unsupported multi-GPU split modes, row/tensor split constraints, and KV-cache placement limitations.

## llama.cpp Split Modes

llama.cpp provides three primary split modes (`-sm` flag):
1. `none`: Single GPU execution (e.g., `-dev Vulkan0` or `-dev Vulkan1`).
2. `layer`: Default multi-GPU mode. Layers are distributed contiguously across available GPUs according to `-ts` (tensor-split) proportions.
3. `tensor` / `row`: Experimental tensor parallelism (row/column weight and KV-cache splitting).

## Technical Findings on Vulkan Backend

### 1. Unsupported Row/Tensor Split Buffers
- **Observation:** When attempting tensor/row splitting (`-sm tensor` with `-dev Vulkan0,Vulkan1`), llama.cpp fails during initialization with:
  ```
  ggml-vulkan: device does not support split buffers
  ```
- **Root Cause:** The Vulkan backend in llama.cpp requires specific buffer sharing and memory features not supported on the target hardware/driver configuration.
- **Classification:** AutoBench classifies this terminal failure as `UNSUPPORTED_BACKEND` (and at boundary/preflight stages as `PREFLIGHT_UNSUPPORTED_BACKEND` / `BOUNDARY_UNSUPPORTED_BACKEND`).
- **Policy Decision:** Explicit tensor split (`-sm tensor`) is **excluded** from authoritative inventory matrices. Only single-GPU (`none`) and multi-GPU `layer` mode are supported.

### 2. KV-Cache Placement Limits
- In `layer` split mode, the KV cache for a given transformer layer is allocated entirely on the GPU that owns that layer.
- Because each GPU on the target host has 2GB VRAM, larger context lengths (e.g. >= 2048) or higher batch sizes can exhaust individual VRAM slices, resulting in `OOM` (or `PREFLIGHT_OOM` / `BOUNDARY_OOM`).
- Context budget adaptation (see `src/statuses.py` and `tests/test_context_bench.py`) ensures that benchmark workloads adapt to context sizes supported within available VRAM slices.

## Summary Policy Table

| Mode | Flag | Supported on Vulkan (k7000) | Authoritative Benchmark Status |
| :--- | :--- | :--- | :--- |
| Single GPU | `-sm none` / `-dev Vulkan0` | Yes | Eligible (subject to memory capacity) |
| Layer Split | `-sm layer` / `-dev Vulkan0,Vulkan1` | Yes | Eligible (layer distribution) |
| Tensor / Row Split | `-sm tensor` / `-dev Vulkan0,Vulkan1` | **No** (`does not support split buffers`) | **Excluded** (Diagnostic only) |
