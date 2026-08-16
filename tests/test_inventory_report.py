"""Tests for inventory_report.py — covers all manifest classes.

No inference is run; all manifests are constructed in-memory.
Tests use the public API: read_manifests, normalize_manifest, build_report, render_report.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from inventory_report import (
    MISSING,
    SKIP_MANIFESTS,
    build_report,
    main,
    normalize_manifest,
    read_manifests,
    render_report,
)


# ── fixture helpers ───────────────────────────────────────────────────────────

def _write(tmp_path: Path, name: str, data: dict) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _load_only_manifest(
    job_id: str = "modelA_Vulkan0_none_load_only",
    status: str = "SUCCESS",
    load_ok: bool = True,
    elapsed: float = 10.0,
) -> dict:
    return {
        "execution_status": "completed_load_probe",
        "authoritative": False,
        "job_id": job_id,
        "model": {"id": "modelA", "name": "modelA.gguf", "size_bytes": 1},
        "config": {"device": "Vulkan0", "tensor_split": None, "mode": "load_only"},
        "result": {
            "stage": "load_probe",
            "status": status,
            "load_ok": load_ok,
            "elapsed_seconds": elapsed,
            "return_code": 0 if load_ok else 1,
        },
    }


def _suite_manifest(
    job_id: str = "modelB_Vulkan0_none_full",
    suite_status: str = "SUCCESS",
    boundary_status: str = "SUCCESS",
    max_allocatable: int = 2048,
    max_reliable: int = 2048,
    retrieval_rate: float = 1.0,
    prompt_ts: float = 10.0,
    gen_ts: float = 20.0,
    task_pass_rate: float = 1.0,
    elapsed: float = 300.0,
    allocatable_lb: bool = False,
    non_comparable: bool = False,
    has_perf: bool = True,
    has_quality: bool = True,
    preflight: dict | None = None,
    inconclusive_status: str | None = None,
) -> dict:
    stages: dict[str, Any] = {
        "boundary": {
            "stage": "boundary",
            "status": boundary_status,
            "maximum_allocatable_context": max_allocatable,
            "first_failed_context": None if allocatable_lb else max_allocatable + 256,
            "inconclusive_status": inconclusive_status,
        }
    }
    if preflight is not None:
        stages["preflight"] = preflight
    if has_perf:
        stages["performance"] = {
            "stage": "performance",
            "status": "SUCCESS" if suite_status == "SUCCESS" else "PARTIAL_FAILURE",
            "prompt_ts": prompt_ts, "gen_ts": gen_ts, "elapsed_seconds": 10.0,
        }
    if has_quality:
        stages["quality"] = {
            "stage": "quality",
            "status": "SUCCESS" if suite_status == "SUCCESS" else "PARTIAL_FAILURE",
            "task_pass_rate": task_pass_rate, "elapsed_seconds": 5.0,
        }

    result: dict[str, Any] = {
        "stage": "suite",
        "status": suite_status,
        "stages": stages,
    }
    if suite_status == "SUCCESS":
        result.update({
            "maximum_allocatable_context": max_allocatable,
            "allocatable_is_lower_bound": allocatable_lb,
            "maximum_reliable_context": max_reliable,
            "retrieval_rate": retrieval_rate,
            "prompt_ts": prompt_ts,
            "gen_ts": gen_ts,
            "elapsed_seconds": elapsed,
            "task_pass_rate": task_pass_rate,
        })
    if non_comparable:
        result["workload"] = {"non_comparable": True}

    return {
        "execution_status": "completed_suite",
        "mode": "suite",
        "authoritative": False,
        "job_id": job_id,
        "models": [{
            "id": "modelB",
            "name": "modelB.gguf",
            "configurations": [{
                "device": "Vulkan0",
                "tensor_split": None,
                "mode": "full",
                "result": result,
            }],
        }],
    }


# ── read_manifests ────────────────────────────────────────────────────────────

def test_read_manifests_skips_summary_and_status(tmp_path: Path) -> None:
    for name in SKIP_MANIFESTS:
        (tmp_path / name).write_text('{"x":1}')
    _write(tmp_path, "real.json", _load_only_manifest())
    rows = read_manifests(tmp_path)
    assert len(rows) == 1
    assert rows[0]["source_manifest"] == "real.json"


def test_read_manifests_skips_invalid_json(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text("not json {")
    _write(tmp_path, "good.json", _load_only_manifest())
    rows = read_manifests(tmp_path)
    # bad.json is read via normalize_manifest which catches the error
    bad = next(r for r in rows if r["source_manifest"] == "bad.json")
    assert bad["outcome"] == "FAILED"
    good = next(r for r in rows if r["source_manifest"] == "good.json")
    assert good["outcome"] != "FAILED"


def test_read_manifests_empty_dir(tmp_path: Path) -> None:
    assert read_manifests(tmp_path) == []


# ── normalize_manifest ────────────────────────────────────────────────────────

def test_normalize_load_only_success(tmp_path: Path) -> None:
    p = _write(tmp_path, "a.json", _load_only_manifest(status="SUCCESS", load_ok=True, elapsed=42.5))
    row = normalize_manifest(p)
    assert row["mode"] == "load_only"
    assert row["outcome"] == "LOAD_ONLY"
    assert row["comparable"] is False
    assert row["source_manifest"] == "a.json"


def test_normalize_load_only_failure(tmp_path: Path) -> None:
    p = _write(tmp_path, "b.json", _load_only_manifest(status="OOM", load_ok=False))
    row = normalize_manifest(p)
    assert row["outcome"] == "BLOCKED"
    assert row["comparable"] is False


def test_normalize_suite_blocked_by_boundary(tmp_path: Path) -> None:
    m = _suite_manifest(
        suite_status="BLOCKED_BY_BOUNDARY", boundary_status="SUCCESS",
        max_allocatable=0, has_perf=False, has_quality=False,
    )
    p = _write(tmp_path, "c.json", m)
    row = normalize_manifest(p)
    assert row["outcome"] == "BLOCKED"
    assert row["comparable"] is False


def test_normalize_suite_partial_failure(tmp_path: Path) -> None:
    m = _suite_manifest(
        suite_status="PARTIAL_FAILURE",
        boundary_status="INCONCLUSIVE",
        inconclusive_status="SSH_TIMEOUT",
    )
    p = _write(tmp_path, "d.json", m)
    row = normalize_manifest(p)
    assert row["outcome"] == "PARTIAL"
    assert row["comparable"] is False


def test_normalize_suite_success_comparable(tmp_path: Path) -> None:
    m = _suite_manifest(
        suite_status="SUCCESS", retrieval_rate=0.8,
        prompt_ts=5.0, gen_ts=15.0, task_pass_rate=0.75, elapsed=200.0,
    )
    p = _write(tmp_path, "e.json", m)
    row = normalize_manifest(p)
    assert row["outcome"] == "PASS"
    assert row["comparable"] is True
    result = row["result"]
    assert result["prompt_ts"] == 5.0
    assert result["gen_ts"] == 15.0


def test_normalize_suite_non_comparable_not_pass(tmp_path: Path) -> None:
    m = _suite_manifest(suite_status="SUCCESS", non_comparable=True)
    p = _write(tmp_path, "f.json", m)
    row = normalize_manifest(p)
    # reduced workload → PARTIAL, not PASS
    assert row["comparable"] is False


def test_normalize_missing_stages(tmp_path: Path) -> None:
    """Suite manifest with no stages key should not crash."""
    m = _suite_manifest(suite_status="BLOCKED_BY_BOUNDARY", has_perf=False, has_quality=False)
    m["models"][0]["configurations"][0]["result"].pop("stages", None)
    p = _write(tmp_path, "g.json", m)
    row = normalize_manifest(p)
    assert row["outcome"] in ("BLOCKED", "PARTIAL", "FAILED")


def test_normalize_preflight_blocked(tmp_path: Path) -> None:
    m = _suite_manifest(
        suite_status="PREFLIGHT_OOM",
        boundary_status="not_run",
        has_perf=False, has_quality=False,
        preflight={"status": "OOM", "load_ok": False, "elapsed_seconds": 2.0},
    )
    p = _write(tmp_path, "h.json", m)
    row = normalize_manifest(p)
    assert row["outcome"] == "BLOCKED"
    assert row["comparable"] is False


# ── render_report ─────────────────────────────────────────────────────────────

def test_render_report_contains_all_sections(tmp_path: Path) -> None:
    rows = [
        normalize_manifest(_write(tmp_path, "l.json", _load_only_manifest())),
        normalize_manifest(_write(tmp_path, "s.json", _suite_manifest(
            suite_status="BLOCKED_BY_BOUNDARY", has_perf=False, has_quality=False,
        ))),
    ]
    text = render_report(rows, tmp_path)
    assert "## Overview" in text
    assert "## Load and preflight diagnostics" in text
    assert "## Capacity and diagnostic results" in text
    assert "## Comparable performance and quality" in text
    assert "## Stage coverage and diagnostics" in text


def test_render_report_no_comparable_when_all_partial(tmp_path: Path) -> None:
    rows = [normalize_manifest(_write(tmp_path, "p.json", _suite_manifest(
        suite_status="PARTIAL_FAILURE",
    )))]
    text = render_report(rows, tmp_path)
    # comparable table should show MISSING sentinel, not real numbers
    assert MISSING in text


def test_render_report_shows_comparable_row(tmp_path: Path) -> None:
    rows = [normalize_manifest(_write(tmp_path, "q.json", _suite_manifest(
        suite_status="SUCCESS", prompt_ts=8.3, gen_ts=21.7, task_pass_rate=0.5,
    )))]
    text = render_report(rows, tmp_path)
    assert "8.30" in text
    assert "21.70" in text


def test_render_report_no_raw_stdout_stderr(tmp_path: Path) -> None:
    m = _load_only_manifest()
    m["result"]["stdout"] = "SECRET_RAW_OUTPUT"
    m["result"]["stderr"] = "raw_error_details"
    rows = [normalize_manifest(_write(tmp_path, "r.json", m))]
    text = render_report(rows, tmp_path)
    assert "SECRET_RAW_OUTPUT" not in text
    assert "raw_error_details" not in text


def test_render_report_no_prompts_leaked(tmp_path: Path) -> None:
    m = _suite_manifest(suite_status="PARTIAL_FAILURE")
    # inject a fake raw prompt into a nested stage run
    stages = m["models"][0]["configurations"][0]["result"]["stages"]
    stages["performance"]["runs"] = [{"command_args": ["--prompt", "TOP_SECRET_PROMPT"]}]
    rows = [normalize_manifest(_write(tmp_path, "t.json", m))]
    text = render_report(rows, tmp_path)
    assert "TOP_SECRET_PROMPT" not in text


# ── build_report (integration) ────────────────────────────────────────────────

def test_build_report_returns_rows_and_text(tmp_path: Path) -> None:
    _write(tmp_path, "a.json", _load_only_manifest())
    _write(tmp_path, "b.json", _suite_manifest(
        suite_status="BLOCKED_BY_BOUNDARY", has_perf=False, has_quality=False,
    ))
    rows, text = build_report(tmp_path)
    assert len(rows) == 2
    assert "## Overview" in text
    assert "a.json" in text
    assert "b.json" in text


def test_build_report_skips_index_files(tmp_path: Path) -> None:
    for name in SKIP_MANIFESTS:
        (tmp_path / name).write_text('{"completed":[]}')
    _write(tmp_path, "real.json", _load_only_manifest())
    rows, _ = build_report(tmp_path)
    assert len(rows) == 1


def test_build_report_empty_dir(tmp_path: Path) -> None:
    rows, text = build_report(tmp_path)
    assert rows == []
    assert "## Overview" in text


# ── CLI ───────────────────────────────────────────────────────────────────────

def test_cli_writes_to_output_file(tmp_path: Path) -> None:
    _write(tmp_path, "a.json", _load_only_manifest())
    out = tmp_path / "report.md"
    import sys
    old_argv = sys.argv
    try:
        sys.argv = ["inventory_report", "--input", str(tmp_path), "--output", str(out)]
        main()
    finally:
        sys.argv = old_argv
    content = out.read_text(encoding="utf-8")
    assert "## Overview" in content
    assert "a.json" in content
