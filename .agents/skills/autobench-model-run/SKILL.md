---
name: autobench-model-run
description: Use for every manual AutoBench GGUF model diagnostic or inventory model run. Research the exact model and runtime through Surf CLI first, persist sanitized provenance in the project KB/QMD, execute one bounded model-specific plan, and investigate failures from authoritative sources instead of retrying by parameter sweep.
compatibility: Requires the AutoBench checkout, Surf CLI with Chromium, project-local OKF/QMD tools, GitHub issue governance, and the reviewed remote execution workflow.
---

# AutoBench manual model run

Execute one model at a time with an explicit evidence trail. Automation may run
individual commands, but it must not turn this into an unreviewed batch or blind
retry loop.

## When to use

Use for every new GGUF model, context boundary probe, rerun after a failure, or
model-specific configuration change. Before starting, identify the governing
GitHub issue and read the project `AGENTS.md` plus the relevant local skills.

## Per-model sequence

1. **Scope and gate.** Record the exact GGUF basename, issue, target backend,
   devices, split mode, context/workload budget, timeout, and expected job count.
   Run a dry-run/plan preview. Stop if the plan contains an unexpected model,
   configuration, size class, or job count.
2. **Research with Surf CLI.** In an isolated Surf window, inspect the official
   model card and relevant official runtime/backend documentation. Verify model
   identity, declared context and generation limits, architecture, conversion
   assumptions, split-mode semantics, and relevant flags. Do not infer local
   hardware capability from web research.
3. **Persist sanitized knowledge.** Write source notes to `kb/raw/` and curate a
   strict OKF note in `kb/wiki/`. Include source URLs, retrieval dates,
   source-backed claims, local observations, applicability, and unresolved
   assumptions. Exclude raw page output, prompts, responses, credentials,
   absolute/private paths, host identifiers, commands, and runtime payloads.
4. **Refresh and verify QMD.** Run the project-local QMD update and a lexical
   search for the model/runtime facts. Review the note and links before any
   inference. If the installed tooling differs from its documentation, record
   the observed interface and do not invent a successful validation step.
5. **Run stages separately.** Execute only this model through the bounded
   applicable configuration set: load/preflight, boundary, retrieval,
   performance, and quality. Keep Vulkan0, Vulkan1, and explicitly requested
   dual-GPU attempts distinct. Preserve comparability and mark reduced budgets.
6. **Stop on unexpected behavior.** On the first failure or surprising result,
   stop the current model sequence. Do not sweep context sizes, timeouts,
   split ratios, flags, prompts, or retries looking for a passing result.
7. **Investigate by evidence.** Classify the result first, then search the
   workspace KB with `kb-lookup` and the project QMD using the exact stable
   error text. If no adequate lesson exists, use isolated Surf CLI windows to
   consult authoritative model/runtime documentation and issue trackers.
   Inspect benchmark budget, prompt construction, parser, and stage aggregation
   before blaming the model. Form one falsifiable hypothesis, change at most
   one justified variable, and run one bounded rerun. If the symptom may be
   device-specific, request the same bounded reproduction on the approved RTX
   2080 Super 8 GB host through the omarchy agent, keeping artifacts separate.
   If evidence remains insufficient, report `INCONCLUSIVE` and continue the
   documented investigation plan rather than silently advancing to another
   model.
8. **Capture and publish.** Update the model's sanitized KB note with the
   observed cause, source consulted, and rerun rationale. Use `kb-capture` for
   a new recurring or non-obvious fix. Publish only sanitized artifacts and
   concise stage-level results.

## Failure classes

Keep these distinct and never collapse them into generic failure or blocked:

- `OOM` / `CONTEXT_OVERFLOW`: observed capacity failure;
- `SSH_TIMEOUT` / transport timeout: inconclusive until investigated;
- `TOKENIZER_ERROR`: tokenizer or token-counting failure;
- `UNSUPPORTED_BACKEND`: runtime/backend capability limitation;
- `EXECUTION_ERROR`: other runtime execution failure;
- `WORKLOAD_UNSUPPORTED`: no valid workload budget;
- `PARTIAL_FAILURE`: some stages completed, others did not;
- `INCONCLUSIVE`: evidence is insufficient for a stronger classification.

## Non-negotiable rules

- Never run a full inventory as a substitute for a per-model review.
- Never treat a web claim as proof of k7000 support.
- Never use brute-force retries or parameter permutations as diagnosis.
- Never persist raw Surf output, raw runtime output, prompts, responses,
  manifests, credentials, private paths, or host identifiers.
- Before every future model workload, invoke this skill and
  `autobench-pre-run-research`.

## Related skills

- `autobench-pre-run-research` — source research and pre-run note gate.
- `kb-lookup` — required first lookup after an error.
- `kb-capture` — record a new reusable fix.
- `qmd-operator` — update and query the project-local knowledge index.
