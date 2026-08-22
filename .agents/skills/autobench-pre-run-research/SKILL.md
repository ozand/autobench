---
name: autobench-pre-run-research
description: Use before any AutoBench GGUF workload to verify the concrete model and llama.cpp/backend assumptions with Surf CLI, persist sanitized source-backed notes in the local OKF/QMD knowledge base, and block execution until the bounded plan is reviewed.
compatibility: Requires Surf CLI, a Chromium-based browser, and the project-local kb/ and qmd tooling.
---

# AutoBench pre-run research

## When to use

Use immediately before every new GGUF model workload, context-boundary probe,
or configuration experiment. This is a per-model gate, not a one-time project
setup. Do not use upstream research as a substitute for local capability checks.

## Required process

### Per-model gate

Run this sequence separately for each model. Do not prepare one generic note
for a batch and then launch the batch without model-specific review.

1. Confirm the active GitHub Issue, exact GGUF basename, target context, backend,
   device list, split mode, and timeout budget.
2. Open each authoritative public source in an isolated Surf window:

   ```text
   surf window.new "<source-url>"
   surf --window-id <window-id> wait.network
   surf --window-id <window-id> page.text
   ```

   Use the official checkpoint model card and the official llama.cpp runtime
   documentation. Never reuse an unverified browser tab or copy raw browser
   output into the repository.
3. Extract only stable facts: checkpoint identity, declared context/generation
   limits, architecture constraints, split-mode semantics, and relevant runtime
   flags. Record source URL and retrieval date.
4. Write sanitized source notes to `kb/raw/` and a strict OKF summary to
   `kb/wiki/`. Separate upstream facts, local observations, and unresolved
   assumptions. Do not store prompts, responses, credentials, private paths,
   host identifiers, raw logs, or raw Surf output.
5. Refresh and query the project-local QMD collection. Search the local KB
   before asking Surf to investigate a problem, and cite the result in the
   run note:

   ```text
   qmd update -c autobench-kb
   qmd search "<model> <context> <backend>" -c autobench-kb --no-rerank
   ```

   If the collection is not registered, create it with `qmd collection add kb
   --name autobench-kb` before updating.
6. Run the zero-inference repository gate and review its output:

   ```text
   python scripts/pre_run_research_check.py
   qmd update
   qmd search "<model> <context> <backend>" -c autobench-kb --no-rerank
   ```

   Research may define the upper probe target, but only hardware results
   establish what the GGUF actually supports. The checker never launches a
   model and is safe to run before review.
7. During execution, stop at the first unexpected failure. Do not perform blind
   parameter sweeps, retries, or workaround permutations. Classify the failure,
   search the local KB, and investigate the authoritative upstream cause with
   Surf CLI before changing one parameter for one bounded rerun.
8. Record the outcome, cause, source consulted, and rationale for any rerun in
   the model note. Capture a new recurring/non-obvious fix with `kb-capture`
   when no existing lesson covers it.

## Failure-response rules

- `OOM`, `CONTEXT_OVERFLOW`, `SSH_TIMEOUT`, `TOKENIZER_ERROR`,
  `UNSUPPORTED_BACKEND`, `EXECUTION_ERROR`, and `WORKLOAD_UNSUPPORTED` are
  distinct evidence classes. Preserve the first observed cause.
- A timeout is inconclusive until an authoritative source-backed diagnosis and
  one bounded rerun establish more. It is never an invitation to sweep timeouts,
  context sizes, split ratios, or flags.
- Search order after a failure: local `qmd`/KB -> official runtime/model source
  through Surf CLI -> one justified rerun. If evidence is insufficient, stop.

## Rules

- Treat a result equal to the last requested probe as a lower bound unless the
  probe reached the documented checkpoint target and the report says so.
- Treat transport or SSH timeout as inconclusive; do not relabel it as OOM.
- Prefer llama.cpp's default `layer` split for diagnostic multi-GPU runs.
  Keep `tensor` separate and explicitly experimental.
- Refresh notes when the checkpoint, GGUF conversion, llama.cpp revision,
  backend, device inventory, or benchmark policy changes.
