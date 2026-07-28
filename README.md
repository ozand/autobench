# AutoBench: Local GGUF LLM Evaluation Framework

AutoBench evaluates local GGUF language models across capacity, retrieval,
performance, and task quality. Development happens in the local Git checkout;
hardware-dependent runs execute on a dedicated remote host.

## Architecture

- `authoritative_bench.py` — reproducible benchmark suites and manifests.
- `inventory_bench.py` — resumable model-inventory execution.
- `context_bench.py` — context scaling and memory-boundary evaluation.
- `compare_models.py` / `fetch_models.py` — comparison and model acquisition.
- `src/runner.py` — `llama-cli` execution and failure classification.
- `src/judge.py` — deterministic and model-assisted quality evaluation.
- `scripts/run_remote.py` — safe Git/SSH deployment and remote execution.

## Local setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux:   source .venv/bin/activate
python -m pip install -e .
python -m pytest
```

## Local development and remote execution

The local repository is authoritative. Commit reviewed changes locally, then
use the runner to push the commit, fast-forward the clean remote checkout, and
execute through its virtual environment.

```bash
# Verify the remote environment
python scripts/run_remote.py -- pytest -q

# Deploy without executing a benchmark
python scripts/run_remote.py --deploy-only

# Run a status command and copy ignored results back afterwards
python scripts/run_remote.py --sync-results -- \
  python inventory_bench.py --status

# Run a smoke benchmark
python scripts/run_remote.py --sync-results -- \
  python authoritative_bench.py --smoke
```

Runner options must appear before `--`; the command and its arguments appear
after `--`. The runner deliberately refuses dirty local and remote checkouts
and never creates commits automatically.

Default deployment target:

- SSH: `opencode@100.67.171.58`
- Checkout: `/home/opencode/code/autobench`

Override these without editing code:

```bash
set AUTOBENCH_REMOTE_HOST=user@example-host
set AUTOBENCH_REMOTE_DIR=/srv/autobench
set AUTOBENCH_EXPECTED_ORIGIN=https://github.com/example/autobench.git
```

On POSIX shells use `export` instead of `set`.

## Result policy

Generated manifests, inventory state, run JSON, and comparison reports are
ignored by default. Use `--sync-results` to copy them into local `results/`,
then deliberately select sanitized, authoritative reports for publication.

## License

MIT
