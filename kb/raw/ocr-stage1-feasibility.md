# OCR Stage 1 Feasibility Research (Issue #114)

## Scope and provenance
- Governing issue: #114, research and candidate selection only.
- Retrieved: 2026-09-12.
- Sources are public model cards/configuration metadata, public ggml-org GGUF file metadata, and official llama.cpp documentation. They are reported compatibility facts, not proof of k7000 runtime support, OCR accuracy, or VRAM fit.
- Exact artifact-size metadata sources: https://huggingface.co/ggml-org/Qwen2-VL-2B-Instruct-GGUF/tree/main and https://huggingface.co/ggml-org/SmolVLM2-2.2B-Instruct-GGUF/tree/main.
- llama.cpp multimodal documentation was reviewed at https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md on 2026-09-12.
- No weights were downloaded. No image/OCR inference, remote OCR command, deployment, or benchmark publication was performed.

## Runtime constraint
- The target has two 2 GB Vulkan partitions (4 GB aggregate), not one uniform 4 GB allocation pool.
- A practical candidate must accommodate model layers, multimodal projector, image tokens/KV cache, graph/scratch buffers, and runtime overhead on the individual device allocations. Aggregate static file sizes do not prove fit.
- Current AutoBench runner is text-only. It has no projector/image arguments or OCR workload/receipt contract. A later multimodal implementation requires a separate Proposed ADR and owner approval.

## GLM-OCR — excluded from the current runnable shortlist
- Official model: `zai-org/GLM-OCR`; model card declares MIT license, `glm_ocr` architecture, image-to-text task, and approximately 0.9B parameters.
- The card describes a CogViT visual encoder, cross-modal connector, GLM-0.5B language decoder, and a document pipeline that also uses PP-DocLayoutV3 (Apache-2.0).
- The official card documents Transformers, vLLM, SGLang, Ollama, and its SDK. No public GGUF/projector artifact or official llama.cpp `glm_ocr` compatibility evidence was found in the reviewed llama.cpp multimodal documentation.
- Status: **excluded for current k7000 runnable candidate selection**, despite a small reported parameter count. A later conversion/runtime compatibility assessment would be a separate task.

## Qwen2-VL-2B-Instruct — priority conditional candidate
- Official model: `Qwen/Qwen2-VL-2B-Instruct`; Apache-2.0; `qwen2_vl` image-text-to-text architecture.
- Official llama.cpp multimodal documentation lists `ggml-org/Qwen2-VL-2B-Instruct-GGUF` as a ready-to-use vision-model path.
- Reported public ggml-org artifact metadata: Q4_K_M text GGUF is 986,046,944 bytes; Q8 multimodal projector is 709,883,360 bytes; f16 projector is 1,331,656,160 bytes. These file sizes are selection inputs, not independently verified local artifacts.
- Static total: Q4_K_M + Q8 projector is 1,695,930,304 bytes (about 1.58 GiB). Q4_K_M + f16 projector is 2,317,703,104 bytes (about 2.16 GiB).
- Status: **priority conditional candidate**. The Q8-projector combination is the conservative starting artifact assumption for a later preflight. The f16 projector has a higher per-device OOM risk because projector placement/splitting on Vulkan is not yet verified.

## SmolVLM2-2.2B-Instruct — secondary conditional candidate
- Official model: `HuggingFaceTB/SmolVLM2-2.2B-Instruct`; Apache-2.0; image-text-to-text architecture.
- The official model card reports OCRBench 72.9 and DocVQA 79.98, but these reported benchmark values are not local OCR evidence.
- Official llama.cpp multimodal documentation lists `ggml-org/SmolVLM2-2.2B-Instruct-GGUF`.
- Reported public ggml-org artifact metadata: Q4_K_M text GGUF is 1,112,602,656 bytes and Q8 projector is 592,523,200 bytes; static total is 1,705,125,856 bytes (about 1.59 GiB). The f16 projector combination totals 1,984,906,336 bytes (about 1.85 GiB). These file sizes are selection inputs, not independently verified local artifacts.
- The upstream card states 5.2 GB GPU RAM for video inference; that does not establish fit for a quantized, single-image Vulkan path.
- Status: **secondary conditional candidate**. Exact visual-token, projector-placement, and Vulkan memory behavior remain unresolved.

## MiniCPM-V-2.6 — excluded pending new public compatibility evidence
- Official model: `openbmb/MiniCPM-V-2_6`; model metadata is gated/auto and advertises custom code plus OCR/vision capabilities.
- No reviewed official llama.cpp multimodal documentation entry, public ggml-org GGUF/projector path, or target artifact exists.
- Status: **excluded for this Stage 1 shortlist**. Reconsider only with separate public conversion and runtime compatibility evidence.

## Recommendation
1. Select **Qwen2-VL-2B-Instruct Q4_K_M with Q8 projector** as the sole priority candidate for the future multimodal implementation decision.
2. Retain **SmolVLM2-2.2B Q4_K_M with Q8 projector** as a secondary conditional comparison candidate.
3. Exclude GLM-OCR and MiniCPM-V-2.6 from current runnable selection because their llama.cpp GGUF/projector compatibility is not established.
4. Before any model acquisition or OCR inference: create a separate implementation issue, write a Proposed ADR for the multimodal runner/receipt/image contract, obtain owner approval, and complete exact-artifact Stage 1/2 research.
