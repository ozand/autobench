# Qwen2-VL-2B-Instruct Q4_K_M + Q8 Projector Research (Issue #122)

## Scope and provenance
- Governing issue: #122; exact-pair acquisition, research, and zero-inference preflight only.
- Retrieved and locally re-verified: 2026-09-12.
- Public source repository: https://huggingface.co/ggml-org/Qwen2-VL-2B-Instruct-GGUF/tree/main
- Multimodal interface source reviewed: https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md
- Upstream checkpoint card/configuration reviewed: https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct and https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/raw/main/config.json
- Qwen2-VL upstream README reviewed: https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/raw/main/README.md
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
- The public configuration identifies `Qwen2VLForConditionalGeneration` / `qwen2_vl`, with 28 text layers, hidden size 1536, 12 attention heads, 2 KV heads, and `max_position_embeddings` 32768. Its vision configuration reports depth 32, 16 heads, patch size 14, spatial merge size 2, and temporal patch size 2.
- The upstream README describes dynamic image resolution and a default visual-token range of 4–16384. Its 256–1280-token example is an upstream Transformers input-budget example, not a llama.cpp/Vulkan limit.
- Official llama.cpp multimodal documentation reports a model plus `--mmproj` artifact contract, lists `ggml-org/Qwen2-VL-2B-Instruct-GGUF` among pre-quantized vision models, and states that the projector is GPU-offloaded by default; `--no-mmproj-offload` disables that default.
- The current AutoBench text runner remains out of scope. The accepted ADR-003 contract is implemented only as a separate zero-inference multimodal preflight path.

## Current local observation
- Exact pair acquisition and local hash verification succeeded.
- Issue #122 completed one target-side metadata-only preflight for this exact pair. Its hash-bound sanitized evidence classified the result `PREFLIGHT_VALID / ZERO_INFERENCE`.
- That preflight validated pair identity and a synthetic metadata descriptor only. It did not load a model or projector, open an image, invoke an OCR workload, or establish runtime compatibility.

## Readiness analysis (Issue #125)
- The two artifacts total 1,695,930,304 bytes. A 2,147,483,648-byte Vulkan partition would have only 451,553,344 bytes remaining if both artifacts were resident on one partition, before runtime, graph, KV, image-token, and allocator costs. This arithmetic is a risk indicator, not a fit measurement.
- Because llama.cpp documents projector GPU offload as the default, layer split of text-model layers cannot be assumed to split or relocate the projector. The target placement behavior is unknown.
- The current preflight CLI validates artifacts and metadata only; it deliberately performs no model load or image-reference validation. It cannot provide an OCR inference receipt or classify runtime failures.

## Unresolved gates
- Per-device Vulkan memory fit and projector placement are unverified.
- Multimodal CLI/backend compatibility for this exact pair is unverified.
- A separately governed runner must validate a non-persisted local image reference and emit the OCR terminal classes specified by ADR-003; it does not exist yet.
- OCR quality, image-token budget, prompt contract, performance, capacity, and a safe image-fixture contract are unverified.
- Issue #125 is planning/review only. Any image/OCR inference requires separate explicit owner approval, a fresh approved pre-inference dry-run, and a new execution-governing issue.
