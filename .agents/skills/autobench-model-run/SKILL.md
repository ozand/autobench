---
name: autobench-model-run
description: Use this skill for every AutoBench GGUF workload, including manual diagnostics, inventory-selected models, context probes, reruns, and configuration experiments. Enforce the four-stage protocol in order: exact-model Surf research, sanitized OKF/QMD gate, reviewed single-GPU/dual-layer/context-KV plan and dry-run, then evidence-first classification and sanitized publication. Never run inventory merely to reduce stale jobs or treat dry-run success as inference approval.
compatibility: Requires the AutoBench checkout, Surf CLI with Chromium, project-local OKF/QMD tools, GitHub issue governance, and the reviewed remote execution workflow.
---

# AutoBench manual model run

Execute one exact GGUF at a time through the canonical protocol in
`docs/model-testing-protocol.md`. Automation may execute approved individual
commands, but it must not turn the work into an unreviewed batch, completion
counter exercise, or blind retry loop.

## When to use

Use before every new model workload, filtered inventory invocation, context
boundary probe, rerun after a failure, or model-specific configuration change.
Also use when resuming a model whose research, runtime, backend, artifact, or
benchmark policy may have changed.

## Mandatory state machine

```text
1. Deep Research (Surf CLI)
        -> 2. KB & Gate Validation (OKF / QMD)
        -> 3. Reviewed Staged Measurement
        -> 4. Evidence-First Classification & Publication
```

Never reorder or skip these stages. A successful `--dry-run`, existing manifest,
completed inventory job, or prior research for a different quantization does not
advance the state machine.

## Stage 1 — Deep Research

### Prerequisite

Verify the governing open GitHub Issue and record the exact GGUF basename,
quantization, backend, and experiment objective. Do not run `--dry-run` yet.

### Required evidence

Invoke `autobench-pre-run-research`. In isolated Surf CLI windows, inspect the
official checkpoint card/configuration and official llama.cpp/backend material.
Verify identity, architecture, layers/attention/KV heads when published,
declared context and generation limits, conversion assumptions, split semantics,
quantization/backend compatibility, and relevant runtime flags. Persist sanitized
source notes with URLs and retrieval dates in `kb/raw/`.

### Stop condition

Stop when the exact artifact cannot be tied to authoritative evidence, sources
are unavailable/stale, or model-specific sanitized notes do not exist.

## Stage 2 — KB and gate validation

### Prerequisite

Stage 1 evidence exists for this exact GGUF and current runtime assumptions.

### Required evidence

Curate a strict OKF note in `kb/wiki/`, separating upstream facts, local
observations, and unresolved assumptions. Then run:

```text
python scripts/pre_run_research_check.py
qmd update -c autobench-kb
qmd search "<model> <context> <backend>" -c autobench-kb --no-rerank
```

Record the passing checker and discoverable model/runtime note.

### Stop condition

Stop before planning, dry-run, deployment, or inference if validation, links,
sanitation, QMD refresh, or lexical discovery fails. Chat history is not a
substitute for repository evidence.

## Stage 3 — Reviewed staged measurement

### Prerequisite

Stages 1 and 2 are complete and reviewed.

### Required evidence

Write one bounded model-specific plan naming the GGUF, issue, backend, devices,
contexts, split mode/ratios, KV policy, workload budgets, timeout, output
isolation, and expected job count. Explicitly decide:

1. **Single GPU:** matched Vulkan0 and Vulkan1 baselines, normally at context
   1024.
2. **Dual GPU:** authoritative mode is `-sm layer`; begin with `-ts 1,1` when
   applicable. Add `1,2` or `2,3` only for a documented memory hypothesis.
   Tensor/row split is excluded on this Vulkan testbed. Record an evidence-backed
   exclusion if dual GPU is not applicable.
3. **Context/KV:** choose a bounded progression from
   `1024 -> 2048 -> 4096 -> 8192` below the researched checkpoint limit.
   Establish default/f16 before one justified `q8_0`, `q4_0`, or
   `--no-kv-offload` comparison. Record exclusions instead of silently omitting
   this group.

Preview the exact plan with `--dry-run`. Verify model identity, size class,
configuration order, output directory, and expected job count.

A successful dry-run does **not** authorize inference. Stop and obtain explicit
review of the KB evidence and dry-run plan before removing `--dry-run`.

After approval, execute serially from the smallest discriminating workload:
load/tokenizer smoke where applicable, Vulkan0 baseline, Vulkan1 baseline,
applicable dual-layer configuration, then bounded context/KV steps. Preserve
logical workload comparability.

### Stop condition

Stop on the first unexpected model/configuration/job count, privacy or artifact
risk, checkout/transport failure, artifact-write failure, or surprising stage
outcome. Do not continue later configurations. Do not sweep contexts, split
ratios, KV flags, prompts, timeouts, or retries. Advance to Stage 4.

## Stage 4 — Evidence-first classification and publication

### Prerequisite

Stage 3 produced a planned result or the first unexpected outcome.

### Required evidence

Classify the first result without guessing. On an error or surprise, follow
`docs/problem-investigation-plan.md`: use `kb-lookup`, refresh/search QMD with
the stable symptom, research authoritative sources through Surf, inspect prompt,
budget, parser, and aggregation logic, form one falsifiable hypothesis, and run
at most one justified bounded rerun. Use the approved comparison host only for
the same exact artifact/logical reproduction when the cause may be device
specific.

Update the model OKF note with stage outcomes, lower bounds, exclusions,
investigation, conclusions, and unresolved uncertainty. Use `kb-capture` for a
new recurring/non-obvious fix. Sync and publish only selected sanitized evidence.

### Completion condition

Do not call the model complete because a command succeeded, jobs say completed,
or stale counts decreased. Complete Stage 4 only when each planned/affected
stage succeeded, has an evidence-backed terminal classification, or is recorded
as `INCONCLUSIVE` with one bounded next action. Do not select another model
before this condition is met unless the user explicitly changes scope.

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

## Gotchas that caused prior protocol drift

- `inventory_bench.py` is only an execution mechanism. Never invoke it merely to
  reduce stale/pending counts or interpret its default two-job plan as a complete
  model investigation.
- Do not preview first and research later. The dry-run belongs after the OKF/QMD
  gate because the research determines valid context and configuration bounds.
- Do not silently omit dual-layer or context/KV decisions because the model fits
  one GPU. Explicitly test them when applicable or record an evidence-backed
  exclusion in the reviewed plan and OKF note.
- Do not run every documented split ratio or KV mode. The protocol requires
  coverage decisions, not a brute-force parameter matrix.

## Non-negotiable rules

- Never run a full inventory as a substitute for per-model review.
- Never treat a web claim as proof of target-host support.
- Never persist raw Surf/runtime output, prompts, responses, manifests,
  credentials, private paths, host identifiers, or unsanitized commands/errors.
- Invoke `autobench-pre-run-research` before Stage 1 work and `kb-lookup` after
  any command failure or unexpected behavior.

## Related skills

- `autobench-pre-run-research` — mandatory Stage 1 and Stage 2 gate.
- `kb-lookup` — required first lookup after an error.
- `kb-capture` — record a new reusable fix.
- `qmd-operator` — update and query the project-local knowledge index.
