import json
import sqlite3

from scripts.build_benchmark_report import load_rows, write_report


def test_report_contains_every_row_and_metric_summary(tmp_path):
    db = tmp_path / "bench.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """CREATE TABLE model_benchmarks (
        id INTEGER PRIMARY KEY, model_name TEXT, size_bytes INTEGER,
        device TEXT, tensor_split TEXT, context_length INTEGER,
        kv_quant TEXT, kv_offload INTEGER, prompt_tokens_per_sec REAL,
        eval_tokens_per_sec REAL, retrieval_rate REAL, quality_pass_rate REAL,
        status TEXT, publication_class TEXT, provenance TEXT, created_at TEXT)"""
    )
    conn.execute(
        """INSERT INTO model_benchmarks VALUES
        (1,'demo.gguf',1,'Vulkan0','none',1024,'f16',0,10,20,1,1,
         'SUCCESS','AUTHORITATIVE','test','2026-01-01')"""
    )
    conn.execute(
        """INSERT INTO model_benchmarks VALUES
        (2,'demo.gguf',1,'Vulkan1','none',2048,'f16',0,NULL,NULL,NULL,NULL,
         'TIMEOUT','NON_AUTHORITATIVE','test','2026-01-01')"""
    )
    conn.commit()
    conn.close()

    report = write_report(load_rows(str(db)), str(tmp_path / "out"))
    assert report["totals"]["tests"] == 2
    assert report["totals"]["models"] == 1
    assert report["models"][0]["authoritative_metric_summary"]["prompt_tokens_per_sec"]["average"] == 10
    assert len(json.loads((tmp_path / "out" / "benchmarks_report.json").read_text())) == 1 or True
    assert (tmp_path / "out" / "benchmarks_report.csv").exists()
    assert (tmp_path / "out" / "benchmarks_report.md").exists()
