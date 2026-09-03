---
id: KB-QWEN25-05B-Q8_0
title: "Qwen2.5-0.5B-Instruct-Q8_0 Issue 41 Follow-up"
category: research
tags: [qwen2.5, gguf, q8_0, vulkan, gtx690, okf, issue41]
status: reviewed
created: 2026-08-22
updated: 2026-09-01
environment:
  os: Linux remote GPU host
  shell: POSIX shell
  tools: [Surf CLI, llama.cpp, AutoBench, QMD]
error_signatures:
  - "device does not support split buffers"
  - "SSH execution timed out"
---

# Qwen2.5-0.5B-Instruct-Q8_0 Issue 41 Follow-up

## Source-backed upstream facts

See the exact-model source note [`kb/raw/qwen2.5-0.5b-instruct-q8_0.md`](../raw/qwen2.5-0.5b-instruct-q8_0.md).

- The official checkpoint is Qwen2.5-0.5B-Instruct, approximately 0.49B parameters.
- The official configuration identifies `Qwen2ForCausalLM` with 24 layers, 14 attention heads, 2 KV heads, and hidden size 896.
- The declared checkpoint context limit is 32,768 tokens.
- Official llama.cpp documentation describes `layer` as the default and most compatible multi-GPU split. `1,1` is a balanced tensor-split proportion for the ordered devices, but this project uses it with `-sm layer` only.
- Default KV cache types are f16. This increment establishes the f16 baseline and does not perform a blind KV option sweep.

## Exact artifact and target observations

- Target basename: `qwen2.5-0.5b-instruct-q8_0.gguf`.
- Fresh target byte size: `675710816`.
- Fresh target SHA-256: `ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e`.
- Target runtime lists Vulkan0 and Vulkan1, each with 2048 MiB reported memory.
- These observations bind the follow-up artifact but do not prove benchmark success.

## Local historical observations

- Issue #56 classified the exact model as partial/non-authoritative: Vulkan0 and Vulkan1 completed through context 4096, while both 8192 probes ended in SSH timeout; the dual-GPU layer path was not executed.
- Historical database rows include a conflicting legacy size value and remain retained as historical evidence. They are not silently promoted or used as the identity for this follow-up.
- Prior tensor/row-split evidence is excluded from authoritative Vulkan coverage because the target testbed reports split-buffer incompatibility.

## Reviewed follow-up scope

- Governing issue: GitHub Issue #41.
- Objective: resolve the outstanding 8192 boundary uncertainty and obtain the missing dual-GPU layer evidence for this exact Q8_0 artifact.
- Configuration groups: matched Vulkan0 single-device baseline, matched Vulkan1 single-device baseline, and Vulkan0,Vulkan1 `-sm layer -ts 1,1`.
- KV policy: f16 baseline only (`-ctk f16 -ctv f16` where explicit flags are recorded).
- Context policy: baseline at 1024 on all three groups, then 2048, 4096, and 8192 only on the dual-layer path. Expected execution unit: 3 serial suite jobs; these contain 6 allocation probes (one Vulkan0, one Vulkan1, and four dual-layer probes), plus a separately counted preflight/tokenizer smoke.
- Fixed workload policy: identical prompt/output budgets and logical workload across matched comparisons; output is isolated to sanitized summaries.
- Timeout policy: 600 seconds primary for this increment. The 1200-second fallback is conditional and must be separately reviewed and recorded only after a primary run has no terminal result.
- Stop rule: stop at the first unexpected boundary, transport, OOM, execution, artifact, or privacy result. Preserve the first class and do not blindly retry or continue later stages.

## Issue #41 execution outcome

- The approved runner completed all 3 suite jobs and 6 allocation probes with the primary 600-second timeout; no fallback was used and no stop-rule surprise occurred.
- Vulkan0 and Vulkan1 completed the 1024 baseline. The dual-GPU layer `1,1` path completed 1024, 2048, 4096, and 8192 allocation probes.
- The runner measured performance and quality at the suite operational context (1024), while the dual-layer repeated Retrieval stage used the allocatable context (8192). This is a mixed-context result and is retained as diagnostic/non-authoritative rather than promoted to a report row.
- Sanitized execution evidence: `docs/issue41-qwen-q8-followup-evidence.json`.

## Same-context publication follow-up

- A separate Issue #41 same-context run used the exact reviewed artifact, Vulkan0,Vulkan1 `-sm layer -ts 1,1`, f16 KV, and declared context 1024 for boundary, performance, Retrieval, and quality.
- All four stages completed successfully at context 1024. Performance was 18.1 prompt tokens/s and 25.3667 generation tokens/s across three measured repetitions after one discarded warm-up.
- Retrieval completed 15 attempts: 9 VERIFIED, 6 MISSED, and 0 INCONCLUSIVE (0.6 recorded rate). Quality passed 2/2.
- One authoritative row was published after separate evidence review. Capacity is recorded only as a lower bound at 1024; a reliable context was not established because Retrieval rate was below the 0.8 threshold.
- Sanitized publication review: `docs/issue41-qwen-q8-same-context-publication-review.json`.
- Sanitized runtime evidence: `results/runs/issue41-qwen-q8-same-context/summary.json`.

## Unresolved assumptions

- Upstream checkpoint metadata does not prove this converted GGUF's local runtime behavior.
- Single-GPU fit and dual-layer usefulness must be established by the bounded target run; size alone does not exclude the dual-layer measurement.
- Existing historical metrics are not evidence for this new Issue #41 scope. Publication requires fresh parsed metrics, complete execution evidence, explicit Retrieval outcomes, and a sanitized publication review.
