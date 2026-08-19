# Problem investigation plan

This document defines the governed response when an AutoBench model run
produces an error, unexpected result, or disagreement between stages. The goal
is to resolve the cause and complete the model investigation, not to treat a
stop as a final answer.

## Scope

The plan applies to every model-specific workload and to every follow-up probe
for a failed or surprising stage. It must be used before moving to another
model.

## Evidence-first sequence

1. **Preserve the first result.** Keep the original sanitized manifest and
   classify the observed stage outcome without guessing. Keep `OOM`,
   `CONTEXT_OVERFLOW`, `TOKENIZER_ERROR`, `UNSUPPORTED_BACKEND`,
   `EXECUTION_ERROR`, `WORKLOAD_UNSUPPORTED`, `PARTIAL_FAILURE`, and
   `INCONCLUSIVE` distinct. A stop is a safety boundary for the current
   experiment, not completion of the investigation.
2. **Search the project knowledge base.** Search the exact stable error text and
   nearby terms in the workspace KB and project notes. Use the `kb-lookup`
   procedure first after any command or tool failure; read the matched lesson's
   Resolution section before retrying.
3. **Use local QMD.** Refresh the project collection and run lexical and, when
   useful, semantic searches over the model/runtime notes:
   `qmd update -c autobench-kb`, followed by `qmd search` or `qmd query`.
   Record whether a matching lesson exists or the search is inconclusive.
4. **Use Surf CLI for authoritative internet research.** In isolated Surf
   windows inspect the official model card, configuration/conversion material,
   llama.cpp/backend documentation, and relevant upstream issues or pull
   requests. Search only for the stable symptom and model/runtime combination;
   do not copy raw browser output into the repository. Persist only sanitized
   source-backed conclusions with URLs and retrieval dates.
5. **Check the benchmark itself.** Inspect the workload resolver, token budget,
   prompt construction, parser, command flags, and stage aggregation. Run
   local unit tests or a no-inference reproduction where possible. A reported
   tokenizer error must not be accepted as a tokenizer defect until the
   embedded tokenizer, prompt construction, and direct short generation are
   checked separately.
6. **Form one falsifiable hypothesis.** Examples: insufficient prompt budget,
   conversion mismatch, backend operation gap, response parser mismatch, or
   device-specific memory limitation. Change at most one justified variable.
7. **Run one bounded diagnostic rerun.** Use the smallest test that can
   distinguish the hypothesis. Preserve the original result and clearly mark
   the rerun as diagnostic/non-comparable. Do not sweep contexts, timeouts,
   split ratios, flags, prompts, or retries.
8. **Compare hardware when the cause may be device-specific.** If the same
   model and symptom can be tested on the approved comparison host with an
   RTX 2080 Super 8 GB, run the same sanitized bounded smoke or reproduction
   there in parallel with the k7000 investigation. The comparison host is a
   diagnostic control, not a replacement for the k7000 result. Keep host
   labels, manifests, and commands separate; do not publish private host
   identifiers or raw runtime output.
9. **Resolve or record the remaining uncertainty.** If the hypothesis is
   confirmed, fix the benchmark/configuration or document the runtime/model
   limitation, test the fix, and rerun the complete affected stage sequence.
   If the evidence is insufficient, record `INCONCLUSIVE` with the exact
   missing evidence and the next concrete diagnostic; do not silently proceed
   to another model.
10. **Capture and publish.** Update the model OKF note with the symptom, KB/QMD
    searches, Surf sources, hypothesis, rerun rationale, comparison-host
    result, and final classification. Use `kb-capture` for a new recurring or
    non-obvious fix. Publish only sanitized selected artifacts.

## RTX 2080 Super comparison protocol

The comparison test is authorized only for a model already under investigation
and only after the exact model, GGUF basename, workload, and expected job count
are reviewed. The omarchy agent must receive a bounded command and return a
sanitized status only. Before inference verify:

- the exact GGUF is available on omarchy;
- the intended llama.cpp/backend is available;
- the command is the same logical workload as on k7000;
- the route/supervision gate and model-specific research requirements remain
  satisfied where applicable;
- the result directory is isolated from k7000 artifacts.

Run, in order: tokenizer smoke, short generation smoke, then the minimal
reproduction of the failing stage. Do not run a broad inventory or unrelated
model. If the comparison host succeeds while k7000 fails, investigate a
k7000/Vulkan/device-specific cause. If both fail, investigate model/GGUF,
benchmark logic, or common runtime compatibility. If omarchy cannot run the
same backend or exact artifact, classify the comparison as unavailable rather
than treating it as a negative result.

## Completion condition

A problematic model is complete only when each affected stage is either:

- resolved and rerun successfully;
- explicitly classified with source-backed evidence; or
- documented as `INCONCLUSIVE` with a bounded next action and no unsupported
  claim.

Moving to the next model is allowed only after this condition is met or the
user explicitly changes scope.
