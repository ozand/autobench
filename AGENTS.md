# AGENTS.md — AutoBench Operating Guide

## Project boundary

AutoBench is a standalone Python project. The local checkout is the source of
truth for code, tests, datasets, and reviewed documentation. Dedicated remote
hosts provide hardware-dependent execution only.

## Required workflow

1. Edit code only in the local checkout.
2. Run `python -m pytest` locally.
3. Review `git diff` and commit intentionally. Never let automation create a
   generic commit or stage the entire repository on your behalf.
4. Deploy the reviewed commit and run it remotely with:

   ```bash
   python scripts/run_remote.py -- pytest -q
   python scripts/run_remote.py --sync-results -- \
     python inventory_bench.py --status
   ```

5. Treat generated benchmark artifacts as remote execution output. Copy them
   back with `--sync-results`; review and publish only intentionally selected,
   sanitized reports.

## Safety rules

- The runner must refuse dirty local or remote checkouts.
- Remote updates must be fast-forward-only and pinned to the exact local commit.
- Do not edit tracked source files directly on the remote execution host.
- Do not use `git add .`, automatic commits, force pushes, destructive remote
  resets, or shell-string command interpolation in deployment automation.
- Keep GGUF weights, credentials, raw logs, manifests, and generated run output
  out of Git unless explicitly sanitized and approved.
- Use SSH key authentication in batch mode; do not embed passwords or private
  keys in scripts.

## Verification

- Local gate: `python -m pytest`
- Remote environment gate:

  ```bash
  python scripts/run_remote.py -- pytest -q
  ```

- Deployment-only check:

  ```bash
  python scripts/run_remote.py --deploy-only
  ```
