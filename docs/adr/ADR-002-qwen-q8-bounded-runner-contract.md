# ADR-002: Add an Explicit Issue-Scoped Runner for the Qwen Q8 Follow-up

**Status**: Accepted
**Date**: 2026-09-01
**Authors**: AutoBench maintainers
**Supersedes**: None
**Related**: Issue #41, ADR-001

## Context

The Issue #41 follow-up for `qwen2.5-0.5b-instruct-q8_0.gguf` requires a small,
reviewable execution contract that the existing size-gated inventory planner
cannot express. The model fits on one GPU, but the approved increment still
requires matched Vulkan0 and Vulkan1 baseline jobs plus one dual-GPU layer job.
The existing CLI has no zero-inference dry-run and its generic planner omits
layer mode for fitting models.

The review also found two trust-boundary gaps: the reviewed f16 KV policy was
not forwarded to runner calls, and default receipt lookup could select a valid
Issue #1 receipt when an Issue #41 receipt was also present. These gaps can make
a plausible run differ from its reviewed plan or bind it to the wrong issue.

## Decision

1. Add one simple, Issue-41-scoped runner script with an explicit `--dry-run`
   preview and three serial suite jobs: Vulkan0 at 1024, Vulkan1 at 1024, and
   Vulkan0,Vulkan1 layer `1,1` at 1024 -> 2048 -> 4096 -> 8192.
2. Represent f16 KV explicitly in the configuration and pass `-ctk f16 -ctv
   f16` through every load, boundary, performance, Retrieval, and quality call
   in this increment.
3. Make receipt verification fail closed when more than one candidate receipt
   exists for a model unless the caller supplies an explicit receipt path, and
   allow the caller to require a matching governing issue.
4. Keep the runner serial and bounded. It must stop after the first unexpected
   result and persist only sanitized summaries.

This decision explicitly does not add broad inventory execution, asymmetric
split sweeps, tensor/row split support, automatic timeout fallback, or generic
publication promotion.

## Consequences

### What gets easier

- Reviewers can preview the exact three-job scope without inference.
- The executed KV policy is visible in command metadata and remains comparable
  with the reviewed plan.
- An Issue #1 receipt cannot silently authorize an Issue #41 run.

### What gets harder

- The Issue #41 runner is a small additional entry point to maintain.
- Receipt directories containing legacy and issue-scoped files require an
  explicit path rather than implicit selection.
- A stop-first-surprise run may leave later jobs pending and requires Stage 4
  classification before continuation.

### What does not change

- Existing historical rows remain retained and non-authoritative unless they
  independently pass the publication gate.
- Layer mode remains the only authoritative multi-GPU mode on the k7000 Vulkan
  testbed.
- ADR-001 remains scoped to Qwen2.5 Coder and is not extended by this record.

## Alternatives Considered

- **Reuse `inventory_bench.py` unchanged** — rejected because its size-gated
  configuration envelope cannot express the approved fitting-model layer job
  and it has no suitable zero-inference preview for this scope.
- **Expand the generic inventory planner immediately** — rejected because it
  would broaden a one-model issue increment and risk changing unrelated model
  scheduling semantics.
- **Rely on implicit f16 defaults** — rejected because the reviewed plan must
  be evidenced by the actual runner command arguments.
- **Prefer one receipt filename over another** — rejected because a valid
  legacy receipt can then silently authorize the wrong governing issue.

## Test Contract

| Claim in Decision | Test | Currently |
|---|---|---|
| The issue runner previews exactly three jobs without inference | Focused runner dry-run test and remote `--dry-run` command | not yet written |
| Explicit f16 KV flags reach every runner invocation | Focused call-argument tests for suite stages | not yet written |
| Ambiguous default receipt lookup fails closed | Protocol receipt ambiguity and governing-issue tests | not yet written |
| The runner stops after the first unexpected result | Focused serial orchestration test | not yet written |

## References

- `docs/issue41-qwen-q8-followup-plan.json`
- `results/receipts/qwen2.5-0.5b-instruct-q8_0.issue41.json`
- `docs/model-testing-protocol.md`
- `docs/kv-cache-optimization.md`
- `docs/gtx690-layer-split-optimization.md`
