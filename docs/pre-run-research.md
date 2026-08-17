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
6. Validate links with the installed bootstrap package's actual CLI interface
   before committing. The current installed package exposes validation as a
   Python API rather than the README's documented subcommand, so do not assume
   `kb-bootstrap validate --dir kb` is supported.
7. Review the note for secrets, private paths, host identifiers, raw browser
   output, prompts, responses, and unsanitized runtime payloads.
8. Only after the note is reviewed, preview the hardware plan and run it through
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

## Scope and freshness

Research notes are bounded to the active Issue and model. Refresh them when the
checkpoint, GGUF conversion, llama.cpp revision, backend, device inventory, or
benchmark policy changes materially. Do not use the notes as a substitute for
runtime evidence or as justification for a full inventory run.
