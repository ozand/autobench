#!/usr/bin/env python3
"""Build a complete sanitized benchmark result report from the SQLite store."""

import argparse
import csv
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


METRIC_FIELDS = (
    "prompt_tokens_per_sec",
    "eval_tokens_per_sec",
    "retrieval_rate",
    "quality_pass_rate",
)
ROW_FIELDS = (
    "id",
    "model_name",
    "size_bytes",
    "device",
    "tensor_split",
    "context_length",
    "kv_quant",
    "kv_offload",
    *METRIC_FIELDS,
    "status",
    "publication_class",
    "provenance",
    "created_at",
)


def metric_quality(row: sqlite3.Row) -> str:
    if row["publication_class"] == "AUTHORITATIVE" and all(
        row[field] is not None for field in METRIC_FIELDS
    ):
        return "COMPLETE_AUTHORITATIVE"
    if row["publication_class"] == "AMBIGUOUS":
        return "INCOMPLETE_SUCCESS"
    return "NOT_PUBLISHED"


def load_rows(db_path: str) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = []
    for source in conn.execute(
        "SELECT * FROM model_benchmarks ORDER BY model_name, context_length, id"
    ):
        row = {field: source[field] for field in ROW_FIELDS}
        row["metric_quality"] = metric_quality(source)
        rows.append(row)
    conn.close()
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    result = []
    for model in sorted({row["model_name"] for row in rows}):
        model_rows = [row for row in rows if row["model_name"] == model]
        authoritative = [
            row for row in model_rows if row["publication_class"] == "AUTHORITATIVE"
        ]
        metrics = {}
        for field in METRIC_FIELDS:
            values = [row[field] for row in authoritative if row[field] is not None]
            metrics[field] = {
                "count": len(values),
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "average": sum(values) / len(values) if values else None,
            }
        result.append(
            {
                "model_name": model,
                "test_count": len(model_rows),
                "authoritative_count": len(authoritative),
                "classification_counts": dict(
                    Counter(row["publication_class"] for row in model_rows)
                ),
                "status_counts": dict(Counter(row["status"] for row in model_rows)),
                "metric_counts": {
                    field: sum(row[field] is not None for row in model_rows)
                    for field in METRIC_FIELDS
                },
                "authoritative_metric_summary": metrics,
            }
        )
    return result


def write_report(rows: list[dict], output_dir: str) -> dict:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    summaries = summarize(rows)
    generated_at = datetime.now(timezone.utc).isoformat()
    report = {
        "schema_version": 1,
        "generated_at": generated_at,
        "scope": "all rows in results/benchmarks.db",
        "metric_semantics": {
            "prompt_tokens_per_sec": "prompt processing speed in tokens per second",
            "eval_tokens_per_sec": "generation speed in tokens per second",
            "retrieval_rate": "recorded legacy retrieval rate in the range 0..1",
            "quality_pass_rate": "recorded legacy quality pass rate in the range 0..1",
            "null_metrics": "not published as authoritative evidence",
        },
        "totals": {
            "models": len(summaries),
            "tests": len(rows),
            "classification_counts": dict(
                Counter(row["publication_class"] for row in rows)
            ),
        },
        "models": summaries,
        "results": rows,
    }
    (directory / "benchmarks_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    with (directory / "benchmarks_report.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=(*ROW_FIELDS, "metric_quality"))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# AutoBench — полный отчёт результатов",
        "",
        f"Сгенерировано: `{generated_at}`",
        "",
        "Отчёт содержит все строки из `results/benchmarks.db`, включая успешные, неоднозначные и неавторитетные тесты.",
        "Нулевые/отсутствующие метрики не трактуются как измеренные значения.",
        "",
        "## Сводка",
        "",
        f"- Моделей: **{len(summaries)}**",
        f"- Тестов: **{len(rows)}**",
        f"- Классификация: `{dict(Counter(row['publication_class'] for row in rows))}`",
        "",
        "## Метрики по моделям",
        "",
        "| Модель | Тестов | Authoritative | Prompt min/avg/max | Gen min/avg/max | Retrieval min/avg/max | Quality min/avg/max | Статусы |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in summaries:
        values = []
        for field in METRIC_FIELDS:
            metric = item["authoritative_metric_summary"][field]
            values.append(
                "—"
                if metric["count"] == 0
                else f"{metric['min']:.3f}/{metric['average']:.3f}/{metric['max']:.3f}"
            )
        statuses = ", ".join(
            f"{status}: {count}" for status, count in sorted(item["status_counts"].items())
        )
        lines.append(
            f"| `{item['model_name']}` | {item['test_count']} | {item['authoritative_count']} | "
            + " | ".join(values)
            + f" | {statuses} |"
        )

    lines.extend(
        [
            "",
            "## Все тестовые строки",
            "",
            "Полный построчный набор доступен в `benchmarks_report.csv` и `benchmarks_report.json`.",
            "",
            "| ID | Модель | Устройство | Контекст | KV | Prompt t/s | Gen t/s | Retrieval | Quality | Статус | Класс | Качество метрик |",
            "|---:|---|---|---:|---|---:|---:|---:|---:|---|---|---|",
        ]
    )
    for row in rows:
        def display(field):
            value = row[field]
            return "—" if value is None else str(value)

        kv = f"{row['kv_quant']}{'_no_offload' if row['kv_offload'] else ''}"
        lines.append(
            f"| {row['id']} | `{row['model_name']}` | `{row['device']}` | {row['context_length']} | "
            f"{kv} | {display('prompt_tokens_per_sec')} | {display('eval_tokens_per_sec')} | "
            f"{display('retrieval_rate')} | {display('quality_pass_rate')} | `{row['status']}` | "
            f"`{row['publication_class']}` | `{row['metric_quality']}` |"
        )
    (directory / "benchmarks_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="results/benchmarks.db")
    parser.add_argument("--output-dir", default="docs")
    args = parser.parse_args()
    report = write_report(load_rows(args.db), args.output_dir)
    print(json.dumps(report["totals"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
