# Mandatory per-model GGUF testing protocol

This document is the canonical agent-facing protocol for every AutoBench GGUF
workload. It applies to manual diagnostics, inventory-selected runs, context
probes, reruns, and configuration experiments. Follow the four stages below in
order for one exact GGUF. No command, successful dry-run, existing inventory
entry, or prior research for another quantization authorizes skipping a stage.

## State machine

```text
1. Deep Research (Surf CLI)
        -> 2. KB & Gate Validation (OKF / QMD)
        -> 3. Reviewed Staged Measurement
        -> 4. Evidence-First Classification & Publication
```

A model may advance only when the current stage's required evidence exists and
its stop conditions are clear. Complete Stage 4 before selecting another model.
A full or filtered `inventory_bench.py` invocation is an execution mechanism,
not an alternative workflow and not evidence that these gates were satisfied.

## Stage 1 — Deep Research with Surf CLI

### Prerequisite

- A verified open GitHub Issue governs the work.
- The exact GGUF basename, quantization, target backend, and intended experiment
  are known.

### Required evidence

Use isolated Surf CLI windows to inspect authoritative sources for the concrete
checkpoint and runtime. Record sanitized notes under `kb/raw/` with URLs and
retrieval dates. At minimum establish:

- checkpoint identity, parameter count, layer count, attention/KV-head layout,
  architecture, declared context length, and generation limit when published;
- GGUF/conversion assumptions relevant to the selected artifact;
- official llama.cpp split semantics and the selected backend's known limits;
- compatibility evidence for the quantization, Vulkan path, and relevant flags.

Separate upstream facts from assumptions. Web research never proves local
hardware capability.

### Stop conditions

Stop before QMD, planning, dry-run, deployment, or inference when the exact
model cannot be verified, authoritative sources are unavailable, the notes are
stale for a materially changed checkpoint/runtime/backend, or sanitized
provenance has not been written.

## Stage 2 — KB and gate validation

### Prerequisite

Stage 1 is complete for the exact GGUF and current runtime assumptions.

### Required evidence

1. Curate an OKF note under `kb/wiki/` that separates:
   - upstream facts;
   - local observations;
   - unresolved assumptions.
2. Validate and index the notes:

   ```bash
   python scripts/pre_run_research_check.py
   qmd update -c autobench-kb
   qmd search "<model> <context> <backend>" -c autobench-kb --no-rerank
   ```

3. Record that the checker passed and that the model/runtime facts are
   discoverable. If the collection is absent, register it before updating.

### Stop conditions

Stop before any `--dry-run` or inference if the checker fails, links are broken,
QMD is unavailable or stale, the search cannot find the model-specific note, or
private/raw runtime material is present. Fix the knowledge gate first; do not
continue on the basis of memory or chat history.

## Stage 3 — Reviewed staged measurement

### Prerequisite

Stages 1 and 2 are complete and their evidence has been reviewed.

### Required evidence: bounded plan

Write and review one model-specific plan before previewing it. The plan must
name the exact GGUF, issue, backend, devices, split mode/ratio, context values,
KV-cache policy, workload budgets, timeout, output directory, and expected job
count. It must explicitly decide all three measurement groups:

1. **Single GPU baseline**
   - Vulkan0 at the bounded baseline context, normally 1024.
   - Vulkan1 at the same logical workload.
2. **Dual GPU layer split**
   - Use only `-sm layer`; Vulkan tensor/row split is excluded from the
     authoritative matrix on this testbed.
   - Start from `-ts 1,1` when dual-GPU testing is applicable.
   - Include an asymmetric ratio such as `1,2` or `2,3` only when model size,
     main-device overhead, or observed memory evidence justifies it.
   - Do not run the full ratio list as a parameter sweep. Explicitly document a
     reason when dual-GPU testing is not applicable.
3. **Context and KV-cache scaling**
   - Define a bounded progression chosen from `1024 -> 2048 -> 4096 -> 8192`,
     never exceeding the researched checkpoint limit.
   - Establish the default/f16 baseline before testing `q8_0`, `q4_0`, or
     `--no-kv-offload`.
   - Treat each KV policy change as a justified experiment, not a permutation
     sweep. Document why a context or KV mode is included or excluded.

Run the exact command with `--dry-run` and verify model basenames, size classes,
configurations, order, output isolation, and expected job count. A successful
`--dry-run` does **not** authorize inference. Stop and obtain explicit review of
the KB evidence and dry-run plan before removing `--dry-run`.

### Execution order and stop conditions

After approval, execute serially from the smallest discriminating workload:
load/tokenizer smoke when applicable, Vulkan0 baseline, Vulkan1 baseline,
applicable dual-layer configuration, then bounded context/KV steps. Preserve
comparable workloads across devices.

Stop immediately on an unexpected model, configuration, job count, privacy or
artifact-write risk, checkout/transport failure, or surprising stage outcome.
Do not continue to later configurations and do not switch to another model.
Do not sweep contexts, split ratios, KV flags, prompts, timeouts, or retries.
Proceed to Stage 4 investigation.

## Stage 4 — Evidence-first classification and publication

### Prerequisite

Stage 3 produced either a planned result or the first unexpected outcome.

### Required evidence

- Preserve distinct terminal classes such as `OOM`, `CONTEXT_OVERFLOW`,
  `TOKENIZER_ERROR`, `UNSUPPORTED_BACKEND`, `EXECUTION_ERROR`,
  `WORKLOAD_UNSUPPORTED`, `PARTIAL_FAILURE`, and `INCONCLUSIVE`.
- For an unexpected result, follow `docs/problem-investigation-plan.md`:
  preserve the first result, use `kb-lookup`, refresh/search QMD, research the
  exact stable symptom with Surf, inspect benchmark logic, form one falsifiable
  hypothesis, and run at most one justified bounded diagnostic rerun.
- Use the approved comparison host only for the same exact artifact and logical
  reproduction when the cause may be device-specific. Keep its evidence
  separate from the target testbed.
- Update the model OKF note with stage outcomes, investigation, source-backed
  conclusions, exclusions, lower bounds, and unresolved uncertainty.
- Synchronize only sanitized artifacts. Review before intentionally publishing
  selected reports or committing documentation.

### Stop and completion conditions

Do not call a model complete merely because jobs are marked completed, stale
counts decreased, or a command exited successfully. Stage 4 is complete only
when every planned or affected stage is successful, explicitly classified with
evidence, or recorded as `INCONCLUSIVE` with one bounded next action. Do not
start another model before this condition is met unless the user explicitly
changes scope.

Never persist or publish raw prompts, responses, stdout/stderr, browser output,
manifests, credentials, private host details, absolute model paths, or
unsanitized exception/command text.

## Required agent entry points

Agents must read this document together with:

- `.agents/skills/autobench-model-run/SKILL.md` for every model workload;
- `.agents/skills/autobench-pre-run-research/SKILL.md` before Stage 1;
- `docs/pre-run-research.md` for research-note requirements;
- `docs/problem-investigation-plan.md` after any error or unexpected result;
- `docs/gtx690-layer-split-optimization.md` for layer-split decisions;
- `docs/kv-cache-optimization.md` for context/KV decisions.
