# GTX 690 Dual-GPU Layer Split Optimization and Ratios (k7000)

## Overview

This guide defines the multi-GPU layer split ratios (`-sm layer -ts <ratio>`) for executing GGUF models across the dual GK104 cores of the NVIDIA GeForce GTX 690 (2x 2 GB VRAM) on the `k7000` testbed.

## Hardware & Runtime Architecture

- **Hardware:** NVIDIA GeForce GTX 690 (PCIe dual GK104 GPUs, 2048 MB VRAM per GPU slice).
- **Backend:** Vulkan (`-dev Vulkan0,Vulkan1`).
- **Split Mode:** Layer split (`-sm layer`). Row/tensor split (`-sm tensor`) is unsupported by the Vulkan backend (`does not support split buffers`).
- **Main Device Overhead:** `Vulkan0` acts as the primary device hosting graph compute buffers, token embeddings, and final output logits. Consequently, `Vulkan0` experiences an inherent 50–150 MB buffer overhead compared to `Vulkan1`.

## Split Ratio Matrix & Evaluation

| Split Ratio (`-ts`) | GPU0 Share | GPU1 Share | Operational Behavior & Characteristics | Use Case Recommendation |
| :--- | :--- | :--- | :--- | :--- |
| `1,1` | 50% | 50% | **Balanced Allocation.** Even layer partition. Due to GPU0 main-device overhead, GPU0 reaches VRAM limits earlier than GPU1. | Standard default for symmetric workloads and models fitting comfortably within ~3.2 GB total VRAM. |
| `1,2` | ~33% | ~67% | **GPU1-Biased Asymmetric Allocation.** Offloads more model layers to GPU1, leaving additional VRAM headroom on GPU0 for context buffer, KV cache, and output tokens. | **Recommended for larger contexts (>=2048)** or models where GPU0 encounters OOM on `1,1`. |
| `2,1` | ~67% | ~33% | **GPU0-Biased Asymmetric Allocation.** Loads the majority of layers on GPU0. Increases memory pressure on GPU0. | Specialized for fast-first-layer pipelining or when GPU1 has shared display buffer constraints. Not recommended for tight VRAM models. |
| `2,3` | 40% | 60% | **Fine-grained GPU1 bias.** Provides moderate relief to GPU0 without over-allocating layers onto GPU1. | Recommended near memory boundaries (e.g. 2B–3B quantized models). |
| `3,2` | 60% | 40% | **Fine-grained GPU0 bias.** Provides slight GPU0 priority. | Boundary testing and pipeline profiling. |

## Model Family & Size Class Recommendations

| Size Class | Example Models | Viability on GTX 690 (2x 2GB) | Recommended Config / Split Ratio | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **0.5B – 0.8B** | `Qwen2.5-0.5B`, `Qwen3.5-0.8B` | Single-GPU Viable | Single GPU (`-dev Vulkan0`) or Dual `1,1` | Single GPU preferred to avoid inter-GPU PCIe sync overhead. |
| **1.0B – 1.5B** | `Qwen2.5-1.5B`, `DeepSeek-R1-Distill-Qwen-1.5B` | Single/Dual Boundary | Dual GPU (`-sm layer -ts 1,1`) | Fits comfortably across both GPUs; `1,1` yields balanced layer distribution. |
| **2.0B – 3.0B** | `Qwen2.5-3B`, `Llama-3.2-3B` | Dual GPU Required | Dual GPU (`-sm layer -ts 1,2` or `2,3`) | Use `1,2` or `2,3` to relieve GPU0 main-buffer VRAM pressure and prevent early OOM. |
| **> 4.0B** | `Mistral-7B`, `Llama-3.1-8B` | Beyond 4GB VRAM | CPU/Offload or Unsupported | Exceeds 4GB combined VRAM; requires host-resident KV or CPU offload. |

## Failure Modes & Error Diagnostics

1. **`does not support split buffers` (`UNSUPPORTED_BACKEND`):**
   Occurs if `-sm tensor` is passed on Vulkan. **Remedy:** Enforce `-sm layer`.
2. **`Vulkan out of memory` (`BOUNDARY_OOM` / `PREFLIGHT_OOM`):**
   Occurs when layer allocation + KV cache exceeds 2048 MB on a GPU slice.
   **Remedy:** Switch from `1,1` to `1,2` (to free GPU0) or reduce context length via workload budget adaptation.
3. **No Split Silently Injected:**
   AutoBench runner explicitly records `-sm layer` and `-ts <ratio>` in command metadata to ensure 100% deterministic, reproducible benchmark runs.
