#!/usr/bin/env python3
"""Audit and export benchmark rows without deleting historical evidence."""

import argparse
import json
import sqlite3
from pathlib import Path

from matrix_runner import audit_existing_rows, ensure_schema, export_dashboard


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="results/benchmarks.db")
    parser.add_argument("--json", default="docs/benchmarks_data.json")
    parser.add_argument("--report", default="docs/benchmark-data-audit.json")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    ensure_schema(conn)
    counts = audit_existing_rows(conn)
    exported = export_dashboard(conn, args.json)
    conn.close()

    report = {
        "policy": {
            "authoritative": "SUCCESS with positive prompt and generation speeds",
            "ambiguous": "SUCCESS without both positive speed metrics",
            "non_authoritative": "Any non-SUCCESS execution status",
            "deletion": "none; historical rows are retained",
        },
        "classification_counts": counts,
        "exported_rows": exported,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
