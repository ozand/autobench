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
- `inventory_report.py` — report-only diagnostic and comparable views from inventory manifests.

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

### Bounded Vulkan tensor validation

Use exactly one explicitly named small GGUF (`<= 1.9 GB`) and one explicitly
named large GGUF (`> 1.9 GB`). The dedicated mode plans six serial jobs and
never runs the full inventory suite.

```bash
# Required preview: zero inference and zero job manifests
python scripts/run_remote.py -- \
  python inventory_bench.py --tensor-validation \
    --models "<SMALL_GGUF>,<LARGE_GGUF>" \
    --dry-run --expected-jobs 6 \
    --output-dir results/issue-5-tensor-validation

# Execute only after the preview shows two models and exactly six jobs
python scripts/run_remote.py --sync-results -- \
  python inventory_bench.py --tensor-validation \
    --models "<SMALL_GGUF>,<LARGE_GGUF>" \
    --expected-jobs 6 --context-size 512 --prompt-tokens 128 \
    --max-tokens 16 --warmups 1 --performance-repetitions 1 \
    --timeout 180 --overall-timeout 1800 \
    --output-dir results/issue-5-tensor-validation
```

The preview must report one model in each size class and six jobs. Stop before
inference if the counts or classes differ. Persisted artifacts are sanitized
before writing: raw prompts, responses, stdout/stderr, private model paths,
remote targets, and detailed exception payloads are excluded or reduced to
safe metadata. Do not execute or sync results unless local tests, the remote
test gate, and deployment-only gate all succeed.

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

To build a report without rerunning inference:

```bash
python inventory_report.py --input results/inventory --output results/inventory/report.md
```

The report keeps load/preflight diagnostics, capacity diagnostics, comparable
performance/quality metrics, and stage coverage in separate tables. It is
never a replacement for the authoritative final matrix.

## License

MIT
