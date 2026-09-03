# AutoBench — ADR Index

Architecture Decision Records for `ozand/autobench`. Naming convention:
`ADR-NNN-kebab-case-slug.md` (three-digit padding). Each ADR follows the shape:
Title / Status / Date / Authors / Supersedes / Related → Context → Decision →
Consequences → Alternatives Considered → Test Contract → Rollback → References.

| ADR | Title | Status |
|-----|-------|--------|
| [ADR-001](./ADR-001-qwen25-coder-timeout-policy.md) | Use a 600-Second Primary Timeout for Qwen2.5 Coder Benchmark Runs | Accepted — Implemented |
| [ADR-002](./ADR-002-qwen-q8-bounded-runner-contract.md) | Add an Explicit Issue-Scoped Runner for the Qwen Q8 Follow-up | Accepted |

## How to amend

- Edits to a Proposed ADR happen in PRs that reference the ADR number.
- A ratified ADR (`Status: Accepted`) is amended by a follow-on ADR
  (`Supersedes: ADR-NNN`), never edited in place.
- New ADRs append to the series — they do not renumber.
- Cross-references use the ADR number, not the slug, so renames don't break links.
- Withdrawn numbers stay burned and are listed below rather than reused.

## Burned numbers

| ADR | Reason |
|-----|--------|
| — | — |
