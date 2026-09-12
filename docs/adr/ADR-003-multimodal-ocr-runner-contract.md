# ADR-003: Isolate Multimodal OCR Runner and Receipt Contracts from Text Benchmarks

**Status**: Accepted
**Date**: 2026-09-12
**Authors**: AutoBench maintainers
**Supersedes**: None
**Related**: ADR-002 (issue-scoped runner contract), Issue #114, Issue #116

## Context

Issue #114 completed Stage 1 OCR feasibility research. It identifies
`Qwen2-VL-2B-Instruct` Q4_K_M plus a Q8 multimodal projector as a conditional
future candidate. It does not establish exact artifact availability, Vulkan
compatibility, per-device fit, projector placement, OCR quality, or performance
on k7000.

The current AutoBench runner (`src/runner.py`) is a text-only `llama-cli`
contract. Existing model receipts bind one text GGUF artifact, while the current
sanitization policy prohibits persisting raw prompts, responses, runtime output,
private paths, and payloads. Image OCR requires distinct artifact pairing,
image-reference validation, a multimodal CLI boundary, and evidence rules. If
those are added as optional fields to the text runner, a multimodal change can
silently alter or weaken the verified text benchmark contract.

Official llama.cpp documentation reports multimodal model/projector/image input
support. This is reported upstream compatibility evidence only; it does not
prove target-host support. Issue #116 is intentionally decision-only: it must
draft this record but must not acquire artifacts, implement code, execute image
workloads, or accept the decision.

## Decision

When and only when this ADR is accepted, future OCR implementation must use a
**separate multimodal OCR runner and receipt contract**, rather than adding
optional image/projector arguments to the existing text runner.

The future contract must:

1. Bind an exact text-model artifact and exact multimodal-projector artifact by
   basename, size, and SHA-256 before execution.
2. Accept image input through an implementation-defined, non-persisted image
   reference. The resulting receipt may record only a sanitized image descriptor
   (format, bounded dimensions/byte class, and validation outcome), never image
   content, base64 data, original filename, absolute path, or source URL.
3. Reject missing, unsupported, oversized, ambiguous, or non-local image
   references before invocation. Validation failures must be terminal evidence
   classes, not inferred OCR failures.
4. Preserve the existing text receipt and runner interfaces unchanged.
5. Produce only sanitized receipt/summary metadata: exact artifact bindings,
   image-validation outcome, bounded context/KV/split configuration, command
   family, terminal status, and OCR outcome class. It must exclude raw images,
   prompts, responses, OCR text, stdout/stderr, absolute paths, credentials, and
   unsanitized exception payloads.
6. Keep OCR evidence classes distinct: `IMAGE_INPUT_REJECTED`,
   `PROJECTOR_MISMATCH`, `UNSUPPORTED_BACKEND`, `OOM`, `CONTEXT_OVERFLOW`,
   `METRIC_PARSE_FAILED`, `OCR_MISSED`, `OCR_INCONCLUSIVE`, and `SUCCESS`.
7. Require a future model-specific research gate, artifact preflight, reviewed
   dry-run, and explicit owner approval before OCR inference. A successful
   dry-run does not authorize image inference or publication.

This ADR does **not** authorize model/projector download, conversion, caching,
runner implementation, image processing, image/OCR inference, benchmark
execution, deployment, database/report mutation, or publication. It does not
claim Qwen2-VL fits the k7000 hardware.

## Alternatives Considered

### Extend `src/runner.py` with optional projector/image arguments

Rejected. It couples a new multimodal artifact and input trust boundary to the
verified text runner, risks regressions for existing text models, and makes
receipt validation ambiguous.

### Use a Python Transformers, vLLM, or external SDK path

Rejected. It would not exercise the intended local llama.cpp/Vulkan target
stack and would introduce a new runtime/dependency boundary before target
compatibility is established.

### Use ad-hoc shell commands for OCR diagnostics

Rejected. Shell commands alone cannot provide fail-closed artifact/input
binding, sanitized receipts, reproducibility, or evidence classification.

### Treat GLM-OCR as the initial implementation candidate

Rejected for the current path. Stage 1 found no public GGUF/projector artifact
or official llama.cpp compatibility evidence for its `glm_ocr` architecture.
It remains a separate future conversion/runtime question.

## Consequences

### What gets easier

- Text benchmark behavior remains isolated from OCR-specific change.
- A later OCR preflight can reject wrong model/projector/image inputs before
  inference.
- Reviewers can distinguish image validation, runtime compatibility, OCR
  correctness, and performance evidence.

### What gets harder

- A separate runner, receipt validator, image descriptor, and OCR test fixtures
  must be implemented and maintained.
- Exact model/projector artifact acquisition and per-device memory behavior must
  be verified before any workload.
- Sanitized OCR evaluation cannot retain raw images or transcribed content,
  limiting later root-cause inspection.

### What does not change

- Existing text runner, text receipts, historical benchmark rows, SQLite schema,
  and reports stay unchanged.
- `-sm layer` remains the only authoritative multi-GPU mode on the Vulkan
  testbed.
- No model is considered hardware-compatible from parameter count or file-size
  metadata alone.

## Test Contract

| Claim in Decision | Test | Currently |
|---|---|---|
| Text runner remains unchanged | Existing `tests/test_runner.py` and text benchmark suite | passing before implementation |
| Multimodal receipt requires paired model/projector identity | Future receipt validation test with synthetic metadata | not yet written |
| Raw image/prompt/response/path payloads are excluded | Future sanitization test with synthetic image descriptor and prohibited fields | not yet written |
| Invalid image reference fails before invocation | Future image-reference validation test with missing/unsupported/oversized synthetic descriptors | not yet written |
| Multimodal invocation is distinct from text invocation | Future multimodal runner command-construction test | not yet written |
| Existing text benchmark suite does not regress | `python -m pytest -q` | passing before implementation |

## Owner Approval Gate

This ADR remains **Proposed**. The owner must explicitly approve and accept
ADR-003 before any follow-on implementation Issue, artifact acquisition,
model/projector download, OCR/image inference, deployment, benchmark workload,
or publication begins. Acceptance is a separate decision from approving this
draft.

## Rollback

Before acceptance, revert this document and its index entry. After acceptance,
reversal requires a follow-on ADR that supersedes ADR-003; it must not silently
merge multimodal behavior into the text runner. No benchmark data is changed by
this proposal.

## References

- Issue #114: OCR Stage 1 feasibility and candidate selection
- Issue #116: multimodal runner contract and Proposed ADR draft
- `kb/raw/ocr-stage1-feasibility.md`
- `kb/wiki/KB-OCR-STAGE1-FEASIBILITY.md`
- `docs/model-testing-protocol.md`
- `docs/adr/ADR-002-qwen-q8-bounded-runner-contract.md`
- https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md

## Addendum (2026-09-12)

The owner explicitly accepted ADR-003 in Issue #116 on 2026-09-12; the
lifecycle update was merged in PR #118. This addendum supersedes only the
historical lifecycle wording in the Context, `## Owner Approval Gate`,
`## Rollback`, and the Issue #116 reference label below: the statements that
this record “remains Proposed,” requires future acceptance, or describes
pre-acceptance rollback are no longer operative. The canonical status is
`Accepted`.

Acceptance authorizes a separate implementation issue under this contract. It
does not waive any continuing gates: exact-artifact research and pairing,
sanitized receipt validation, reviewed zero-inference dry-run, explicit
pre-inference review, evidence-first classification, and separate publication
review remain mandatory. Accepted ADR lifecycle changes now require a
follow-on superseding ADR; this record is not reverted in place. The ADR's
Decision and non-goals are unchanged.

The Issue #116 reference denotes the completed decision-phase and acceptance
history, not a still-pending Proposed-ADR draft.
