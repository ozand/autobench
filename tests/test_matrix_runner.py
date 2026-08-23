import json
import sqlite3

from scripts.matrix_runner import audit_existing_rows, classify_row, export_dashboard, parse_output


def test_parse_output_missing_metrics_is_not_success():
    result = parse_output("completed without rate summary", "", 0)
    assert result["status"] == "METRIC_PARSE_FAILED"
    assert result["prompt_ts"] == 0.0
    assert result["gen_ts"] == 0.0


def test_classify_row_requires_both_positive_speeds():
    assert classify_row("SUCCESS", 10.0, 20.0) == "AUTHORITATIVE"
    assert classify_row("SUCCESS", 0.0, 20.0) == "AMBIGUOUS"
    assert classify_row("TIMEOUT", 0.0, 0.0) == "NON_AUTHORITATIVE"


def test_audit_retains_rows_and_exports_classification(tmp_path):
    db = tmp_path / "bench.db"
    output = tmp_path / "data.json"
    conn = sqlite3.connect(db)
    conn.execute(
        """CREATE TABLE model_benchmarks (
        id INTEGER PRIMARY KEY, model_name TEXT NOT NULL, size_bytes INTEGER,
        device TEXT NOT NULL, tensor_split TEXT, context_length INTEGER NOT NULL,
        kv_quant TEXT NOT NULL, kv_offload BOOLEAN NOT NULL,
        prompt_tokens_per_sec REAL, eval_tokens_per_sec REAL,
        retrieval_rate REAL, quality_pass_rate REAL, status TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"""
    )
    conn.executemany(
        """INSERT INTO model_benchmarks
        (model_name, device, context_length, kv_quant, kv_offload,
         prompt_tokens_per_sec, eval_tokens_per_sec, status)
        VALUES (?, 'Vulkan0', 1024, 'f16', 0, ?, ?, ?)""",
        [("measured.gguf", 10.0, 20.0, "SUCCESS"),
         ("ambiguous.gguf", 0.0, 0.0, "SUCCESS"),
         ("failed.gguf", None, None, "TIMEOUT")],
    )
    conn.commit()
    counts = audit_existing_rows(conn)
    assert counts == {"AUTHORITATIVE": 1, "AMBIGUOUS": 1, "NON_AUTHORITATIVE": 1}
    assert export_dashboard(conn, str(output)) == 3
    conn.close()

    rows = json.loads(output.read_text(encoding="utf-8"))
    assert {row["publication_class"] for row in rows} == {
        "AUTHORITATIVE", "AMBIGUOUS", "NON_AUTHORITATIVE"
    }
    assert all("provenance" in row for row in rows)
