# Benchmark report status

Historical `matrix_*.md` and `context_matrix_*.md` files created before GitHub
issue #26 are **preliminary and unverified**. They mix benchmark revisions,
ambiguous legacy failure labels, unmatched workloads, and insufficient retrieval
repetitions. Do not use them as the authoritative local-model comparison.

Generated reports remain local-only. The reproducible replacement starts with:

```bash
python3 Project/servers/k7000/autobench/authoritative_bench.py --plan-only
```

The current `--full` mode writes the complete execution contract but does not
yet run inference. It and smoke/plan reports are explicitly non-authoritative.
Only a later completed execution whose manifest satisfies the pinned policy may
be called authoritative.
