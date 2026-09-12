# Qwen2-VL-2B-Instruct Q4_K_M + Q8 Projector Research (Issue #122)

## Scope and provenance
- Governing issue: #122; exact-pair acquisition, research, and zero-inference preflight only.
- Retrieved and locally re-verified: 2026-09-12.
- Public source repository: https://huggingface.co/ggml-org/Qwen2-VL-2B-Instruct-GGUF/tree/main
- Multimodal interface source reviewed: https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md
- No image was acquired, opened, decoded, processed, or used for inference.
- No OCR inference, benchmark workload, performance measurement, database/report publication, or broad artifact acquisition was performed.

## Exact acquired pair

| Role | Basename | Bytes | SHA-256 |
|---|---|---:|---|
| text model | `Qwen2-VL-2B-Instruct-Q4_K_M.gguf` | 986046944 | `5745685d2e607a82a0696c1118e56a2a1ae0901da450fd9cd4f161c6b62867d7` |
| projector | `mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf` | 709883360 | `a0ad91f00a7a80dcf84d719a61b00ee2e07b71794f4ee2dfa81a254621a8c418` |

- The public ggml-org metadata supplied the expected size and LFS SHA-256 for each file.
- After acquisition, target-side `sha256sum` and file size matched the expected value for both artifacts.
- Static pair size is 1,695,930,304 bytes. This does not prove fit on the two individual 2 GB Vulkan allocations after projector placement, image tokens, KV cache, graph buffers, and runtime overhead.

## Upstream model and runtime facts
- `Qwen/Qwen2-VL-2B-Instruct` is an Apache-2.0 `qwen2_vl` image-text-to-text model.
- The public model configuration reports a 28-layer text model and a vision component; this is upstream architecture metadata, not target capability evidence.
- Official llama.cpp multimodal documentation reports a model plus `--mmproj` artifact contract and lists `ggml-org/Qwen2-VL-2B-Instruct-GGUF` as a public vision-model path.
- The current AutoBench text runner remains out of scope. The accepted ADR-003 contract is implemented only as a separate zero-inference multimodal preflight path.

## Current local observation
- Exact pair acquisition and local hash verification succeeded.
- No target-side multimodal model load, image validation against a real image, image processing, or OCR inference has been attempted.

## Unresolved gates
- Per-device Vulkan memory fit and projector placement are unverified.
- Multimodal CLI/backend compatibility for this exact pair is unverified.
- OCR quality, image-token budget, prompt contract, performance, and capacity are unverified.
- A reviewed zero-inference preflight/dry-run is required before any future OCR inference decision.
