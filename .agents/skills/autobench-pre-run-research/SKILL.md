---
name: autobench-pre-run-research
description: Use this skill before every AutoBench GGUF workload, filtered inventory run, context probe, rerun, or configuration experiment. Complete exact-model Surf research and sanitized raw/OKF notes, pass the repository checker, refresh and query QMD, then hand off to a reviewed single-GPU/dual-layer/context-KV plan. Block dry-run and inference when evidence is missing, stale, undiscoverable, or unsanitized.
compatibility: Requires Surf CLI, a Chromium-based browser, and the project-local kb/ and QMD tooling.
---

# AutoBench pre-run research

This skill implements Stages 1 and 2 of
`docs/model-testing-protocol.md` and hands the model to its reviewed Stage 3
measurement plan and Stage 4 evidence-first classification. It is a per-model
gate, not a one-time project setup. Upstream research establishes safe
assumptions; it does not prove local capability or authorize inference.

## When to use

Use immediately before every new GGUF model workload, inventory-selected run,
context boundary probe, configuration experiment, or resumed investigation
whose checkpoint, GGUF conversion, llama.cpp revision, backend, device inventory,
or benchmark policy changed materially.

## Stage 1 — Deep Research with Surf CLI

### Prerequisite

Verify the governing open GitHub Issue. Record the exact GGUF basename,
quantization, target backend, intended device/configuration groups, and
experiment objective. Do not run `--dry-run`, deploy, or infer yet.

### Required process and evidence

1. Open each authoritative source in an isolated Surf window:

   ```text
   surf window.new "<source-url>"
   surf --window-id <window-id> wait.network
   surf --window-id <window-id> page.text
   ```

2. Inspect the official checkpoint model card/configuration and official
   llama.cpp/backend documentation. Extract only stable facts relevant to the
   concrete artifact:
   - checkpoint identity and parameter count;
   - architecture, layer count, query/KV heads when published;
   - declared context and generation limits;
   - conversion/GGUF assumptions and quantization compatibility;
   - split-mode semantics, backend limits, and relevant runtime flags.
3. Write sanitized notes under `kb/raw/`, including source URLs, retrieval dates,
   concise source-backed claims, applicability, and uncertainty. Never copy raw
   page output into the repository.

### Stop condition

Stop before Stage 2, planning, dry-run, deployment, or inference if the exact
model identity is unresolved, authoritative sources are unavailable, the
research is stale for the current runtime/artifact, or sanitized provenance has
not been persisted.

## Stage 2 — KB and gate validation

### Prerequisite

Stage 1 is complete for the exact GGUF and current runtime assumptions.

### Required process and evidence

1. Curate a strict OKF note under `kb/wiki/` with source links and separate:
   - upstream facts;
   - local observations;
   - unresolved assumptions.
2. Validate the repository notes:

   ```text
   python scripts/pre_run_research_check.py
   ```

3. Refresh and query the project collection:

   ```text
   qmd update -c autobench-kb
   qmd search "<model> <context> <backend>" -c autobench-kb --no-rerank
   ```

   If the collection is not registered, add it with
   `qmd collection add kb --name autobench-kb` before updating.
4. Record that the checker passed, links/sanitization were reviewed, and the
   exact model/runtime note is discoverable.

### Stop condition

Stop before any dry-run or inference if validation fails, links are broken, QMD
is unavailable/stale, the model-specific search does not find the note, or raw
prompts, responses, logs, browser output, private paths, host identifiers,
credentials, commands, or runtime payloads appear in persisted knowledge.
Repository evidence is mandatory; memory or prior chat is insufficient.

## Handoff to Stage 3

After Stage 2 passes, invoke `autobench-model-run` and prepare the reviewed
model-specific plan. Research must inform explicit decisions for:

- matched Vulkan0 and Vulkan1 baselines;
- applicable dual-GPU `-sm layer` testing and justified ratios;
- bounded context progression and default/f16, `q8_0`, `q4_0`, or
  `--no-kv-offload` KV policy;
- workload budgets, timeout, output isolation, and expected job count.

Only now may the exact plan be previewed with `--dry-run`. A successful dry-run
does **not** authorize inference. Stop and obtain explicit review of both the KB
evidence and the previewed plan before removing `--dry-run`.

## Failure-response rules

If execution later fails or surprises:

1. Preserve the first observed class.
2. Invoke `kb-lookup`; refresh/search QMD using the exact stable symptom.
3. Use isolated Surf windows for authoritative model/runtime investigation only
   after local KB search.
4. Form one falsifiable hypothesis and permit at most one justified bounded
   rerun with one changed variable.
5. Update the model OKF note with the cause, source, rationale, result, and
   unresolved uncertainty. Use `kb-capture` for a new reusable fix.

`OOM`, `CONTEXT_OVERFLOW`, `SSH_TIMEOUT`, `TOKENIZER_ERROR`,
`UNSUPPORTED_BACKEND`, `EXECUTION_ERROR`, and `WORKLOAD_UNSUPPORTED` are
distinct evidence classes. A timeout remains inconclusive until investigated.

## Gotchas

- Research for Qwen2.5 Q4_K_M does not automatically cover Q8_0, another GGUF
  conversion, Qwen3.5, or a changed llama.cpp/backend revision. Verify scope and
  refresh assumptions explicitly.
- Do not place dry-run before research. That prior ordering created a bypass in
  which a plausible two-job inventory preview was mistaken for approval.
- Do not treat size-based inventory policy as evidence that dual-layer or
  context/KV analysis is irrelevant. The reviewed plan must test or explicitly
  exclude each group with evidence.
- Treat a result equal to the last requested context probe as a lower bound
  unless the researched checkpoint target was reached and the report says so.
- Prefer llama.cpp `layer` split for multi-GPU diagnostics. Keep tensor/row mode
  separate and excluded from the authoritative Vulkan matrix on this testbed.

## Rules

- Complete this skill separately for each exact model workload.
- Refresh notes when the checkpoint, artifact conversion, runtime revision,
  backend, devices, or benchmark policy changes materially.
- Never use a generic batch note to authorize multiple models.
- Never store raw Surf/runtime output or sensitive local/remote details.
