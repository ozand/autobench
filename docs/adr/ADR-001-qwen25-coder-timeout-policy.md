# ADR-001: Use a 600-Second Primary Timeout for Qwen2.5 Coder Benchmark Runs

**Status**: Proposed
**Date**: 2026-08-31
**Authors**: AutoBench maintainers
**Supersedes**: None
**Related**: Issue #62, Issue #63, Issue #41

## Context

The Qwen2.5 Coder Q4_K_M benchmark has shown a real runtime-latency boundary
on the dedicated Vulkan test host. Under the original 300-second timeout, the standard dual-GPU layer
workload at context 8192 reached `BOUNDARY_SSH_TIMEOUT`. The same workload
completed in 597.36 seconds when the timeout was increased to 600 seconds.
Three lower-utilization diagnostic probes also completed, but those probes were
not comparable to the standard workload and were not published as authoritative.

The evidence is **measured** for the observed timeout behavior and **decision
required** for the benchmark contract. This ADR records a timeout policy before
any dependent publication work proceeds.

The policy applies to the bounded Qwen2.5 Coder benchmark work governed by Issue
#62 and the dependent publication review in Issue #63. It does not authorize a
broad inventory rerun or promote diagnostic measurements.

## Decision

1. Use **600 seconds** as the primary timeout for the Qwen2.5 Coder benchmark
   suite covered by the approved publication review.
2. Permit a **1200-second fallback only when a 600-second run times out without
   producing a terminal result**.
3. Record any fallback execution as separate evidence. It must not be silently
   combined with or substituted for the primary 600-second result.
4. Keep timeout policy, workload, model configuration, and evidence provenance
   explicit in the reviewed plan and publication receipt.
5. This ADR is **Requires ADR before implementation**: dependent timeout-policy
   implementation or publication work must not proceed until this ADR is
   accepted.

This decision explicitly does **not** change the model, device selection, split
mode, KV policy, Retrieval semantics, or authoritative publication criteria.

## Consequences

### What gets easier

- High-latency Qwen2.5 Coder runs have one explicit primary timeout.
- Reviewers can distinguish a completed 600-second run from a conditional
  fallback run.
- The publication review can compare evidence against a documented contract.

### What gets harder

- A primary run may occupy the execution environment for up to 600 seconds.
- A conditional fallback can extend a single investigation to 1200 seconds.
- Results from different timeout policies must remain separated for comparison.

### What does not change

- Diagnostic values remain non-authoritative until the complete publication gate
  is satisfied.
- Historical rows are retained and are not silently replaced or promoted.
- Tensor/row split remains excluded from the Vulkan testbed.
- A timeout or incomplete stage still requires explicit classification.

## Alternatives Considered

- **Keep the 300-second timeout** — rejected because the measured standard
  workload reached the timeout before producing a terminal result at context
  8192.
- **Use 1200 seconds automatically** — rejected because it changes the normal
  contract and spends additional execution time without the required trigger.
- **Run a broad inventory rerun immediately** — rejected because it expands the
  scope beyond the bounded Qwen2.5 Coder decision and known failures remain
  unresolved.
- **Promote diagnostic metrics without this policy** — rejected because isolated
  or non-comparable diagnostics do not satisfy authoritative publication.

## Test Contract

| Claim in Decision | Test | Currently |
|---|---|---|
| The standard Qwen2.5 Coder 8192 workload can complete within 600 seconds | Reviewed Qwen2.5 Coder suite with the 600-second timeout | passing: completed in 597.36 seconds |
| A 1200-second fallback is used only after a 600-second run has no terminal result | Receipt and plan review for the dependent execution | not yet written |
| Primary and fallback results remain separate evidence | Publication receipt review and artifact audit | not yet written |
| Diagnostic metrics are not promoted automatically | Existing publication classification and report checks | passing |

## Rollback

To roll back this proposed decision, reject or withdraw ADR-001 and retain the
previous timeout contract. Before acceptance, no runtime or public artifact
changes are made by this record. Any already-published benchmark data must be
handled by a separate reviewed publication decision; this ADR does not erase
historical evidence.

## References

- Issue #62: `docs(adr): record Qwen2.5 Coder benchmark timeout policy`
- Issue #63: `publish(benchmark): review and selectively publish Qwen2.5 Coder Q4_K_M dual-GPU suite`
- Issue #41: `rerun(benchmark): republish verified speed and Retrieval data for affected recent models`
- `docs/issue41-qwen-coder-suite-timeout600-plan.json`
- `docs/issue41-qwen-coder-suite-timeout600.json`
- `docs/issue41-qwen-coder-boundary-8192-timeout600.json`
