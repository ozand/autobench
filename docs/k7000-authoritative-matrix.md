# AutoBench k7000 GGUF Matrix

**Publication status: diagnostic-only; no authoritative ranking is available.**

## Scope and provenance

This document summarizes the completed k7000 inventory executed under the
validated non-tensor policy. It is derived from sanitized inventory manifests
and the report-only diagnostics from issues #7 and #14.

- Inventory scope: 20 discovered GGUF models and 50 authoritative job identities.
- Configuration policy: Vulkan0 and Vulkan1 single-GPU configurations; legacy
  dual-GPU layer-split configurations only where applicable.
- Explicit tensor split (`-sm tensor`) is excluded from authoritative inventory
  use. Tensor evidence remains diagnostic-only because the bounded validation
  returned terminal execution failures on the tested configuration.
- Workload policy: context sizes 512 and 1024, boundary step 256, one retrieval
  repetition, one performance repetition, one warmup, one quality task, 128
  prompt tokens, and 16 output tokens.
- Execution was resumable and model-contiguous. The cumulative status reached
  50 terminal job identities with zero pending and zero failed scheduler jobs.

## Authoritative comparison eligibility

No rows qualify for authoritative performance/quality ranking.

The completed non-tensor suite rows are excluded because they contain partial
or terminal stage outcomes, including retrieval failures or tokenizer errors,
preflight failures, OOM, model-load errors, and boundary/transport timeouts.
A successful load probe is retained as a load diagnostic, not as a comparable
performance result.

## Diagnostic outcomes

The sanitized report preserves separate technical categories:

- load/preflight success and failure;
- OOM;
- model-load failure;
- execution failure;
- tokenizer failure;
- boundary timeout;
- partial suite failure;
- non-comparable or reduced-workload exclusions.

No generated prompt, response, stdout/stderr, raw output, credential,
operational host target, absolute model path, or unsanitized runtime payload is
published here.

## Limitations

- This is not a model ranking and must not be used to select a production model
  configuration.
- Historical tensor/performance-era manifests and mixed-policy artifacts are
  excluded from authoritative comparison.
- The current workload produced no complete comparable performance/quality rows.
- A future authoritative publication requires a clean rerun whose boundary,
  retrieval, performance, and quality stages all complete under one pinned
  policy.

## Source references

- Sanitized inventory report: local ignored artifact `results/inventory/report.md`
- Inventory execution: GitHub issue #7
- Report-only diagnostics: GitHub issue #14
- Tensor validation and policy decision: GitHub issue #5
