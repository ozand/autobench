# ADR-004: Use ordinary evidence for owner-approved non-sensitive OCR smoke fixtures

**Status**: Accepted
**Date**: 2026-09-15
**Authors**: AutoBench maintainers
**Supersedes**: ADR-003
**Related**: Issue #155, Issue #138

## Context

ADR-003 applied one receipt/evidence privacy boundary to every OCR image. In
practice, that boundary forced path-hiding, handoff, and staging work even when
the owner had explicitly supplied a non-sensitive test document for a short OCR
smoke. The additional machinery delayed the runtime test without improving its
safety.

The owner has now directed AutoBench to use non-sensitive documents and normal
testing processes for this smoke workflow. Credentials, tokens, private keys,
and documents not explicitly approved as non-sensitive remain out of scope.

## Decision

For an exact fixture that the owner explicitly identifies as non-sensitive in
the active GitHub Issue, AutoBench uses ordinary reviewed test evidence:

- the approved fixture's paths, filenames, commands, test logs, prompts, OCR
  output, and content may be retained in the Issue, PR, and ordinary test
  artifacts; existing strict multimodal receipts remain unchanged;
- no special path-hiding, private handoff, redacted receipt, or privacy-only
  staging mechanism is required;
- the reviewed runner still binds the exact approved model/configuration and
  performs exactly one authorized process invocation.

This decision does not authorize a retry, configuration sweep, second image,
benchmark expansion, bypass of the GitHub Issue/review/test workflow, or the
exposure of credentials, tokens, private keys, private target/model locations,
unrelated runtime/environment data, or fixtures not explicitly approved as
non-sensitive.

## Consequences

### What gets easier

- A non-sensitive OCR smoke can use conventional staging and diagnostic logs.
- The next test can be reproduced from its Issue/PR evidence without bespoke
  privacy infrastructure.
- Review focuses on the actual model/backend result rather than fixture-path
  indirection.

### What gets harder

- The active Issue must state that the exact fixture is non-sensitive before
  the fast path is used.
- Reviewers must still distinguish ordinary non-sensitive test data from
  credentials and genuinely private documents.

### What does not change

- `src/runner.py` and the text benchmark pipeline remain separate from OCR.
- One configuration, one process, bounded timeout, stop-first classification,
  no retry, and the governing Issue/PR/test gates remain mandatory.
- Sensitive inputs and credentials retain their existing protections.

## Alternatives Considered

### Apply the strict privacy contract to every fixture

Rejected. It adds path-hiding and handoff work to an explicitly non-sensitive
fixture, preventing fast testing without adding useful protection.

### Remove all input and evidence restrictions

Rejected. Credentials, tokens, private keys, and undesignated documents require
protection even when a particular smoke fixture does not.

## Test Contract

| Claim in Decision | Test | Currently |
|---|---|---|
| Core one-run OCR constraints remain enforced | `tests/test_multimodal_target_launcher.py` | passing |
| Fixture staging remains bounded and reviewed | `tests/test_run_remote.py` | passing |
| Text runner remains unchanged | `tests/test_runner.py` | passing |
| Owner issue explicitly identifies fixture as non-sensitive | Issue review before invocation | manual gate |

## References

- Issue #155
- Issue #138
- ADR-003
- `docs/model-testing-protocol.md`
