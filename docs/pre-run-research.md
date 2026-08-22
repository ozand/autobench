# Pre-run research gate

AutoBench hardware runs must begin with a bounded research pass. Research
provides assumptions and provenance; it does not prove local device capability
or authorize an unbounded workload.

## Required sequence

1. Confirm the active GitHub Issue and the exact model/configuration scope.
2. Use Surf CLI in an isolated browser window to inspect authoritative public
   sources for the concrete model checkpoint and the runtime/backend behavior.
3. Record retrieval dates, source URLs, concise source-backed claims, and
   applicability/uncertainty notes in `kb/raw/`.
4. Curate a strict OKF Markdown note under `kb/wiki/`, preserving links to the
   raw source notes and separating upstream facts from local observations.
5. Update the local QMD collection and verify discoverability with a lexical
   search. Use `qmd update -c autobench-kb`; embedding is optional for a small
   bounded note set.
6. Run the user-testable, zero-inference gate from the repository root:

   ```bash
   python scripts/pre_run_research_check.py
   qmd update
   qmd search "<model> <context> <backend>" -c autobench-kb --no-rerank
   ```

   The checker validates the OKF frontmatter, local links, and forbidden
   sensitive/runtime payload markers. It does not launch a model.
7. Validate links with the installed bootstrap package's actual CLI interface
   before committing. The current installed package exposes validation as a
   Python API rather than the README's documented subcommand, so do not assume
   `kb-bootstrap validate --dir kb` is supported.
8. Review the note for secrets, private paths, host identifiers, raw browser
   output, prompts, responses, and unsanitized runtime payloads.
9. Only after the note is reviewed, preview the hardware plan and run it through
   the pinned remote-execution workflow.

## Evidence classes

- **Upstream fact:** explicitly stated by the cited official model or runtime
  documentation.
- **Local capability observation:** result of a bounded k7000 device/load/
  boundary/workload probe.
- **Unresolved assumption:** a claim requiring a local check; it must not be
  presented as a capability guarantee.

## Qwen2.5-0.5B example

For the current diagnostic, the official checkpoint card states a 32,768-token
full context and 8,192-token generation limit. The official llama.cpp guide
states that `layer` is the default, most compatible multi-GPU split, while
`tensor` is experimental. The resulting diagnostic must still test the actual
GGUF and Vulkan devices, and a timeout must remain inconclusive until rerun with
an appropriate budget.

## User-testable gate

The repository's simple acceptance command is:

```bash
python scripts/pre_run_research_check.py
```

It checks every Markdown file under `kb/raw/` and `kb/wiki/`, requires the
curated OKF fields, verifies relative links, and rejects private paths, host
identifiers, and raw runtime payload keys. It is deliberately independent of
Surf, QMD embeddings, SSH, and llama.cpp, so it is safe to run during review.

## Error and unexpected-result investigation

A stop after an unexpected result is a safety boundary for the current
experiment, not the end of the investigation. Follow the full procedure in
[`problem-investigation-plan.md`](problem-investigation-plan.md). In short:
preserve and classify the first result, search the workspace KB with the exact
stable error text, refresh and query project QMD, use isolated Surf CLI windows
for authoritative model/runtime research, inspect the benchmark's own budget
and parser logic, form one falsifiable hypothesis, and run one bounded
justified diagnostic rerun. Use `kb-lookup` before retrying and `kb-capture`
when a new recurring or non-obvious fix is found.

When the symptom may be device-specific, use the approved RTX 2080 Super 8 GB
comparison host through the omarchy agent. Keep the same model and logical
workload, use an isolated result directory, return only sanitized status, and
never replace the k7000 evidence with the comparison result. A comparison
cannot authorize a broad inventory or an unrelated model.

## Scope and freshness

Research notes are bounded to the active Issue and model. Refresh them when the
checkpoint, GGUF conversion, llama.cpp revision, backend, device inventory, or
benchmark policy changes materially. Do not use the notes as a substitute for
runtime evidence or as justification for a full inventory run. The checker is a
preparation gate only; a passing check does not authorize inference by itself.
