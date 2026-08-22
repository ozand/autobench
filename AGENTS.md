# AGENTS.md — AutoBench Operating Guide

## Required reading

Before any AutoBench GGUF workload, read these files in order:

1. `docs/model-testing-protocol.md` — canonical four-stage model protocol.
2. `.agents/skills/autobench-model-run/SKILL.md` — mandatory execution skill.
3. `.agents/skills/autobench-pre-run-research/SKILL.md` — mandatory research gate.
4. `docs/pre-run-research.md` — KB/OKF/QMD evidence requirements.
5. `docs/problem-investigation-plan.md` — required response to failures or surprises.
6. `docs/gtx690-layer-split-optimization.md` and
   `docs/kv-cache-optimization.md` — required planning references for dual-GPU,
   context, and KV-cache decisions.

## Project boundary

AutoBench is a standalone Python project. The local checkout is the source of
truth for code, tests, datasets, and reviewed documentation. Dedicated remote
hosts provide hardware-dependent execution only.

Key entry points:

- `authoritative_bench.py` — benchmark suites and manifests.
- `inventory_bench.py` — resumable execution mechanism; it does not replace the
  per-model research, review, or completion protocol.
- `context_bench.py` — context-boundary evaluation.
- `src/runner.py` — llama.cpp execution and failure classification.
- `scripts/run_remote.py` — reviewed commit deployment and remote execution.
- `kb/raw/` and `kb/wiki/` — sanitized research provenance and curated OKF notes.

## GitHub Issue gate

Before changing code, documentation, configuration, tests, deployment state, or
starting a new benchmark increment, match or create a verified open GitHub Issue
and mark it `in progress` when work starts. Do not expand an active Issue beyond
its acceptance criteria. Add a sanitized completion comment and close it only
after its criteria are verified.

## Mandatory per-model protocol

Use both `autobench-model-run` and `autobench-pre-run-research` for every manual
diagnostic, inventory-selected model, context probe, rerun, or configuration
experiment. Complete these four stages in order for one exact GGUF:

1. **Deep Research (Surf CLI).** Verify the exact checkpoint, architecture,
   context/generation limits, quantization/conversion assumptions, and official
   llama.cpp/backend behavior. Persist sanitized source notes in `kb/raw/`.
2. **KB & Gate Validation (OKF / QMD).** Curate the model note in `kb/wiki/`,
   run `python scripts/pre_run_research_check.py`, update `autobench-kb`, and
   verify a lexical model/runtime search before creating or previewing a run.
3. **Reviewed Staged Measurement.** Prepare one bounded plan that explicitly
   decides single-GPU Vulkan0/Vulkan1 baselines, applicable dual-GPU `-sm layer`
   ratios, and bounded context/KV-cache progression. Preview it with `--dry-run`,
   verify exact models/configurations/job count, then obtain explicit review
   before removing `--dry-run`.
4. **Evidence-First Classification & Publication.** Preserve distinct outcomes,
   investigate the first surprise through KB/QMD/Surf and one falsifiable
   bounded rerun, update the OKF note, sync sanitized artifacts, and intentionally
   publish only selected reviewed evidence.

The prerequisites, evidence, stop conditions, and completion criteria in
`docs/model-testing-protocol.md` are non-negotiable. A successful `--dry-run`,
a clean deployment, an existing inventory plan, or a decreased stale-job count
does **not** authorize inference or prove completion.

### Required stage-3 plan decisions

- Test Vulkan0 and Vulkan1 with the same bounded baseline workload.
- Use only `-sm layer` for authoritative multi-GPU work. Tensor/row split is
  excluded on this Vulkan testbed because split buffers are unsupported.
- Start with `-ts 1,1` when dual-GPU testing applies; add an asymmetric ratio
  such as `1,2` or `2,3` only when a documented model/memory hypothesis requires
  it. Never sweep the full ratio list.
- Select a bounded context progression from `1024 -> 2048 -> 4096 -> 8192`
  without exceeding the researched checkpoint limit.
- Establish the default/f16 KV baseline before a justified `q8_0`, `q4_0`, or
  `--no-kv-offload` comparison. Never sweep KV options blindly.
- If a group is inapplicable, document the evidence-backed exclusion in the
  model plan and OKF note; do not silently omit it.

### Mandatory stops

Stop before inference when research is missing/stale, the KB checker or QMD gate
fails, the plan has not been reviewed, or dry-run scope/counts differ. Stop the
current model at the first unexpected result, privacy/artifact risk, checkout or
transport failure, or artifact-write failure. Do not advance to later configs or
another model until Stage 4 resolves or explicitly classifies the outcome.

Never run `inventory_bench.py` merely to reduce pending/stale counts. A filtered
inventory command remains governed by all four stages and may execute only the
reviewed model-specific configurations.

## Development and deployment workflow

1. Edit tracked source and documentation only in the local checkout.
2. Run `python -m pytest` locally for source changes. For documentation-only
   protocol changes, run the issue-specific documentation checks and confirm no
   unintended source/config/result changes.
3. Review `git diff` and commit intentionally. Never let automation create a
   generic commit or stage the entire repository.
4. Push the reviewed commit before deployment. Remote execution must be pinned
   to the exact local commit and fast-forward-only.
5. Validate the remote environment when applicable:

   ```bash
   python scripts/run_remote.py -- pytest -q
   python scripts/run_remote.py --deploy-only
   ```

6. After the mandatory research and plan review, preview the exact bounded
   hardware command with `--dry-run`. Verify exact GGUF basename, size class,
   devices, split/KV/context choices, output directory, and expected job count.
7. Execute only the approved command, then copy ignored results with
   `--sync-results`. Review and publish only selected sanitized reports.

## Safety rules

- The runner must refuse dirty local or remote checkouts.
- Remote updates must be fast-forward-only and pinned to the exact local commit.
- Do not edit tracked source files directly on a remote execution host.
- Do not use `git add .`, automatic commits, force pushes, destructive resets,
  or shell-string command interpolation in deployment automation.
- Keep GGUF weights, credentials, raw logs, manifests, and generated run output
  out of Git unless explicitly sanitized, selected, and approved.
- Sanitize before serialization, not only while rendering. Persist no raw
  prompts, responses, stdout/stderr, browser output, private remote targets,
  absolute model paths, host identifiers, or unsanitized exception/command text.
- Treat `OOM`, `CONTEXT_OVERFLOW`, `SSH_TIMEOUT`, `TOKENIZER_ERROR`,
  `UNSUPPORTED_BACKEND`, `EXECUTION_ERROR`, `WORKLOAD_UNSUPPORTED`,
  `PARTIAL_FAILURE`, and `INCONCLUSIVE` as distinct evidence classes.
- Use SSH key authentication in batch mode; never embed passwords or private
  keys in scripts.

## Verification commands

- Local source gate: `python -m pytest`
- Research-note gate: `python scripts/pre_run_research_check.py`
- QMD refresh: `qmd update -c autobench-kb`
- Remote environment gate: `python scripts/run_remote.py -- pytest -q`
- Deployment-only gate: `python scripts/run_remote.py --deploy-only`

No workspace-wide or inventory-wide completion claim may be made from one
model's checks.
