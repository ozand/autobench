# AutoBench k7000 GGUF Matrix

**Publication status: Validated Protocol Runs In Progress (Issues #1, #22, #23, #24, #25)**

## Scope and Provenance

This document records authoritative and diagnostic runs across the 20 discovered GGUF models on the `k7000` testbed (NVIDIA Dual GTX 690, 2x 2048 MB VRAM).

Every authoritative entry requires the mandatory four-stage per-model protocol:
1. **Stage 1 (Deep Research)**: Surf-backed model and backend runtime investigation (`kb/raw/`).
2. **Stage 2 (KB & Protocol Gate)**: OKF wiki lesson (`kb/wiki/`), `pre_run_research_check.py` verification, and signed JSON receipt (`results/receipts/<model>.json`).
3. **Stage 3 (Staged Measurement)**: Single-GPU baseline (Vulkan0, Vulkan1), dual-GPU layer split (`-sm layer`), and context/KV scaling.
4. **Stage 4 (Classification & Findings)**: Empirical speed, retrieval, task quality, and memory limits documented in KB and matrix.

## Validated Models Ledger

| Model Basename | Size (MiB) | Receipt Status | Vulkan0 Prompt (t/s) | Vulkan0 Gen (t/s) | Quality Pass Rate | Retrieval (Max Ctx) | Multi-GPU Path | Notes / Reference |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `qwen2.5-0.5b-instruct-q4_k_m.gguf` | 468.6 | VALID | 15.6 | 40.2 | 100% | 66.7% (4096) | `-sm layer` only | `KB-QWEN25-05B-Q4KM.md` |
| `Qwen3.5-0.8B-Q4_K_M.gguf` | 511.9 | VALID | 12.5 | 36.2 | 100% | 6.7% (4096) | `-sm layer` only | `QWEN35-001-qwen35-08b-vulkan-diagnostic.md` |
| `qwen2.5-0.5b-instruct-q8_0.gguf` | 623.1 | VALID | 12.8 | 38.5 | 100% | 66.7% (4096) | `-sm layer` only | `KB-QWEN25-05B-Q8_0.md` |
| `Llama-3.2-1B-Instruct-Q4_K_M.gguf` | 770.3 | VALID | 6.8 | 35.4 | 50% | 20.0% (2048) | `-sm layer` only | `KB-LLAMA32-1B-Q4KM.md` |

## Invariant Hardware & Software Rules
- **Explicit Tensor Split Exclusion**: `-sm tensor` / `-ts` fails on Vulkan GK104 due to absence of split buffer support; classified as `UNSUPPORTED_BACKEND`.
- **KV Cache Optimization**: Default `f16` KV allocation fits within single-GPU VRAM for models <= 1B at ctx <= 2048; `q8_0` / `q4_0` KV quantization enables higher context boundaries.
- **Fail-Closed Receipt Enforcement**: `inventory_bench.py` and `authoritative_bench.py` strictly block execution of any model lacking a valid receipt in `results/receipts/`.

## Historical Runs Reference
Historical runs prior to commit `15ffe7a` are quarantined in `docs/historical-runs-audit.md`.
