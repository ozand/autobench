#!/usr/bin/env python3
"""Build a sanitized diagnostic report from completed AutoBench manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SKIP_MANIFESTS = {"summary.json", "status.json"}
MISSING = "—"


def _text(value: Any, default: str = MISSING) -> str:
    """Return a short scalar suitable for a Markdown cell."""
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (dict, list)):
        return default
    return str(value).replace("|", "\\|").replace("\n", " ")[:160]


def _number(value: Any) -> Any:
    """Keep numeric metrics numeric while rejecting booleans and containers."""
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _model_config(data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract the compatible model/config envelope used by inventory manifests."""
    model = data.get("model")
    config = data.get("config")
    if not isinstance(model, dict):
        models = data.get("models")
        model = models[0] if isinstance(models, list) and models and isinstance(models[0], dict) else {}
    if not isinstance(config, dict):
        configs = model.get("configurations", []) if isinstance(model, dict) else []
        config = configs[0] if configs and isinstance(configs[0], dict) else {}
    return model, config


def _result(data: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    result = data.get("result")
    if not isinstance(result, dict):
        result = config.get("result")
    return result if isinstance(result, dict) else {}


def _stage_state(stage: Any) -> str:
    if not isinstance(stage, dict):
        return "missing"
    status = str(stage.get("status", "")).upper()
    if status in {"SUCCESS", "COMPLETED"}:
        return "measured"
    if status in {"NOT_RUN", "SKIPPED"}:
        return "not_run"
    if status in {"INCONCLUSIVE", "BLOCKED", "BLOCKED_BY_BOUNDARY"}:
        return "blocked"
    return "failed"


def _reason(result: dict[str, Any], stages: dict[str, Any], execution_status: str) -> str:
    status = str(result.get("cause_status") or result.get("source_status") or result.get("status") or execution_status)
    boundary = stages.get("boundary")
    if isinstance(boundary, dict):
        status = str(boundary.get("inconclusive_status") or boundary.get("cause_status") or status)
    if status == "SUCCESS":
        return "complete"
    if status.startswith("PREFLIGHT_"):
        return status
    if status in {"OOM", "REMOTE_TIMEOUT", "SSH_TIMEOUT", "UNSUPPORTED_BACKEND", "EXECUTION_ERROR", "MODEL_LOAD_ERROR"}:
        return status
    if status in {"BLOCKED_BY_BOUNDARY", "INCONCLUSIVE"}:
        return status
    if execution_status == "failed":
        return "execution_failed"
    return status or "unknown"


def _classify(data: dict[str, Any], config: dict[str, Any], result: dict[str, Any]) -> tuple[str, bool, str]:
    execution_status = str(data.get("execution_status", "unknown"))
    mode = str(config.get("mode", ""))
    stages = result.get("stages") if isinstance(result.get("stages"), dict) else {}
    if execution_status == "failed" or not result:
        return "FAILED", False, _reason(result, stages, execution_status)
    if mode == "load_only" or result.get("stage") == "load_probe":
        return ("LOAD_ONLY" if result.get("load_ok") else "BLOCKED"), False, _reason(result, stages, execution_status)
    status = str(result.get("status", ""))
    boundary = stages.get("boundary")
    boundary_status = str(boundary.get("status", "")) if isinstance(boundary, dict) else ""
    if status.startswith("PREFLIGHT_") or status == "BLOCKED_BY_BOUNDARY" or boundary_status == "INCONCLUSIVE":
        return "BLOCKED", False, _reason(result, stages, execution_status)
    workload = result.get("workload")
    reduced = isinstance(workload, dict) and bool(workload.get("non_comparable"))
    required = ("boundary", "retrieval", "performance", "quality")
    complete = all(_stage_state(stages.get(name)) == "measured" for name in required)
    if status != "SUCCESS" or not complete or reduced:
        return "PARTIAL", False, "reduced_workload" if reduced else _reason(result, stages, execution_status)
    return "PASS", True, "complete"


def normalize_manifest(path: Path) -> dict[str, Any]:
    """Normalize one manifest into sanitized rows used by the report renderer."""
    base = {"source_manifest": path.name, "authoritative": False}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {**base, "model": MISSING, "job_id": path.stem, "device": MISSING, "tensor_split": MISSING, "mode": MISSING, "execution_status": "invalid_manifest", "result_status": MISSING, "outcome": "FAILED", "comparable": False, "reason": "invalid_manifest", "stages": {}}
    if not isinstance(data, dict):
        return {**base, "model": MISSING, "job_id": path.stem, "device": MISSING, "tensor_split": MISSING, "mode": MISSING, "execution_status": "invalid_manifest", "result_status": MISSING, "outcome": "FAILED", "comparable": False, "reason": "invalid_manifest", "stages": {}}
    model, config = _model_config(data)
    result = _result(data, config)
    outcome, comparable, reason = _classify(data, config, result)
    stages = result.get("stages") if isinstance(result.get("stages"), dict) else {}
    return {
        **base,
        "model": model.get("id") or model.get("name") or MISSING,
        "job_id": data.get("job_id") or path.stem,
        "device": config.get("device", MISSING),
        "tensor_split": config.get("tensor_split", MISSING),
        "mode": config.get("mode") or ("load_only" if result.get("stage") == "load_probe" else "full"),
        "execution_status": data.get("execution_status", "unknown"),
        "result_status": result.get("status", MISSING),
        "outcome": outcome,
        "comparable": comparable,
        "reason": reason,
        "authoritative": bool(data.get("authoritative", False)),
        "stages": stages,
        "result": result,
    }


def read_manifests(input_dir: Path) -> list[dict[str, Any]]:
    """Read measurement manifests, excluding scheduler bookkeeping files."""
    return [normalize_manifest(path) for path in sorted(input_dir.glob("*.json")) if path.name not in SKIP_MANIFESTS]


def _cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return _text(value)


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(_cell(value) for value in row) + " |" for row in rows)
    return lines


def render_report(rows: list[dict[str, Any]], input_dir: Path) -> str:
    """Render separate diagnostic and comparable tables without raw process output."""
    lines = [
        "# AutoBench Inventory Report",
        "",
        f"Source directory: `{input_dir}`",
        "",
        "> This report is diagnostic. It does not publish the authoritative model ranking.",
        "",
        "## Overview",
        "",
    ]
    overview = [[row["model"], row["job_id"], row["device"], row["tensor_split"], row["mode"], row["execution_status"], row["result_status"], row["outcome"], row["comparable"], row["authoritative"], row["source_manifest"]] for row in rows]
    lines += _table(["Model", "Job ID", "Device", "Tensor split", "Mode", "Execution", "Result", "Outcome", "Comparable", "Authoritative", "Source manifest"], overview)
    lines += ["", "## Load and preflight diagnostics", ""]
    load_rows = []
    for row in rows:
        result = row.get("result", {})
        if row["mode"] == "load_only" or result.get("stage") == "load_probe" or "preflight" in row.get("stages", {}):
            preflight = row.get("stages", {}).get("preflight", result)
            load_rows.append([row["model"], row["job_id"], row["device"], preflight.get("status", row["result_status"]), preflight.get("elapsed_seconds"), preflight.get("return_code"), row["reason"], row["comparable"], row["source_manifest"]])
    lines += _table(["Model", "Job ID", "Device", "Status", "Load seconds", "Return code", "Reason", "Comparable", "Source manifest"], load_rows or [[MISSING] * 9])
    lines += ["", "## Capacity and diagnostic results", ""]
    capacity_rows = []
    for row in rows:
        if row["mode"] != "load_only":
            result = row.get("result", {})
            capacity_rows.append([row["model"], row["job_id"], row["device"], result.get("maximum_allocatable_context"), result.get("maximum_reliable_context"), result.get("retrieval_rate"), row["outcome"], row["reason"], row["source_manifest"]])
    lines += _table(["Model", "Job ID", "Device", "Allocatable context", "Reliable context", "Retrieval rate", "Outcome", "Reason", "Source manifest"], capacity_rows or [[MISSING] * 9])
    lines += ["", "## Comparable performance and quality", ""]
    comparable_rows = []
    for row in rows:
        if row["comparable"]:
            result = row["result"]
            comparable_rows.append([row["model"], row["job_id"], row["device"], result.get("prompt_ts"), result.get("gen_ts"), result.get("task_pass_rate"), result.get("elapsed_seconds"), row["source_manifest"]])
    lines += _table(["Model", "Job ID", "Device", "Prompt tok/s", "Generation tok/s", "Task pass rate", "Elapsed seconds", "Source manifest"], comparable_rows or [[MISSING] * 8])
    lines += ["", "## Stage coverage and diagnostics", ""]
    stage_rows = []
    for row in rows:
        stages = row.get("stages", {})
        stage_rows.append([row["model"], row["job_id"], _stage_state(stages.get("boundary")), _stage_state(stages.get("retrieval")), _stage_state(stages.get("performance")), _stage_state(stages.get("quality")), row["reason"], row["result_status"], row["source_manifest"]])
    lines += _table(["Model", "Job ID", "Boundary", "Retrieval", "Performance", "Quality", "Reason", "Result status", "Source manifest"], stage_rows or [[MISSING] * 9])
    lines.append("")
    return "\n".join(lines)


def build_report(input_dir: Path) -> tuple[list[dict[str, Any]], str]:
    """Read manifests and return normalized rows plus Markdown output."""
    rows = read_manifests(input_dir)
    return rows, render_report(rows, input_dir)


def load_manifests(input_dir: Path) -> list[dict[str, Any]]:
    """Load valid measurement manifests while skipping scheduler bookkeeping."""
    manifests: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("*.json")):
        if path.name in SKIP_MANIFESTS:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            data["_source_file"] = path.name
            manifests.append(data)
    return manifests


def _manifest_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a manifest into one row per load probe or suite configuration."""
    source = manifest.get("_source_file", MISSING)
    if manifest.get("execution_status") == "completed_load_probe" or manifest.get("config", {}).get("mode") == "load_only":
        result = manifest.get("result") if isinstance(manifest.get("result"), dict) else {}
        model = manifest.get("model") if isinstance(manifest.get("model"), dict) else {}
        config = manifest.get("config") if isinstance(manifest.get("config"), dict) else {}
        return [{
            "model": model.get("id") or model.get("name") or MISSING,
            "job_id": manifest.get("job_id", Path(str(source)).stem),
            "device": config.get("device", MISSING),
            "tensor_split": config.get("tensor_split", MISSING),
            "mode": "load_only",
            "load_status": result.get("status", MISSING),
            "load_ok": bool(result.get("load_ok")),
            "elapsed_s": result.get("elapsed_seconds"),
            "source": source,
            "authoritative": bool(manifest.get("authoritative", False)),
        }]
    models = manifest.get("models") if isinstance(manifest.get("models"), list) else []
    rows: list[dict[str, Any]] = []
    for model in models:
        if not isinstance(model, dict):
            continue
        for config in model.get("configurations", []):
            if not isinstance(config, dict):
                continue
            result = config.get("result") if isinstance(config.get("result"), dict) else {}
            stages = result.get("stages") if isinstance(result.get("stages"), dict) else {}
            preflight = stages.get("preflight") if isinstance(stages.get("preflight"), dict) else {}
            boundary = stages.get("boundary") if isinstance(stages.get("boundary"), dict) else {}
            workload = result.get("workload") if isinstance(result.get("workload"), dict) else {}
            rows.append({
                "model": model.get("id") or model.get("name") or MISSING,
                "job_id": manifest.get("job_id", Path(str(source)).stem),
                "device": config.get("device", MISSING),
                "tensor_split": config.get("tensor_split", MISSING),
                "mode": config.get("mode", "full"),
                "suite_status": result.get("status", MISSING),
                "boundary_status": boundary.get("status", "not_run"),
                "boundary_inconclusive": boundary.get("inconclusive_status"),
                "has_boundary": bool(boundary),
                "has_performance": isinstance(stages.get("performance"), dict),
                "has_quality": isinstance(stages.get("quality"), dict),
                "preflight_status": preflight.get("status", "not_run"),
                "preflight_ok": preflight.get("load_ok"),
                "prompt_ts": None if workload.get("non_comparable") else result.get("prompt_ts"),
                "gen_ts": None if workload.get("non_comparable") else result.get("gen_ts"),
                "task_pass_rate": None if workload.get("non_comparable") else result.get("task_pass_rate"),
                "non_comparable": bool(workload.get("non_comparable")),
                "source": source,
                "authoritative": bool(manifest.get("authoritative", False)),
            })
    return rows


def extract_rows(manifests: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return separate load/preflight and suite rows for report sections."""
    load_rows: list[dict[str, Any]] = []
    suite_rows: list[dict[str, Any]] = []
    for manifest in manifests:
        for row in _manifest_rows(manifest):
            (load_rows if row.get("mode") == "load_only" else suite_rows).append(row)
    return load_rows, suite_rows


def _md_rows(headers: list[str], rows: list[list[Any]], empty: str = "No comparable rows") -> str:
    if not rows:
        return empty
    return "\n".join(_table(headers, rows))


def render_overview(load_rows: list[dict[str, Any]], suite_rows: list[dict[str, Any]]) -> str:
    rows = load_rows + suite_rows
    values = [[r.get("model"), r.get("job_id"), r.get("device"), r.get("mode"), r.get("load_status", r.get("suite_status")), r.get("source"), r.get("authoritative")] for r in rows]
    return "## Overview\n\n" + _md_rows(["Model", "Job ID", "Device", "Mode", "Status", "Source manifest", "Authoritative"], values, "No manifests")


def render_load_preflight(load_rows: list[dict[str, Any]], suite_rows: list[dict[str, Any]]) -> str:
    values = [[r.get("model"), r.get("job_id"), r.get("device"), r.get("load_status"), r.get("elapsed_s"), r.get("source")] for r in load_rows]
    values += [[r.get("model"), r.get("job_id"), r.get("device"), r.get("preflight_status"), MISSING, r.get("source")] for r in suite_rows if r.get("preflight_status") != "not_run"]
    return "## Load / Preflight\n\n" + _md_rows(["Model", "Job ID", "Device", "Status", "Elapsed seconds", "Source manifest"], values, "No load or preflight diagnostics")


def render_capacity(suite_rows: list[dict[str, Any]]) -> str:
    values = [[r.get("model"), r.get("job_id"), r.get("device"), r.get("suite_status"), r.get("boundary_inconclusive") or r.get("boundary_status"), r.get("source")] for r in suite_rows]
    return "## Capacity / Diagnostic\n\n" + _md_rows(["Model", "Job ID", "Device", "Suite status", "Boundary reason", "Source manifest"], values, "No capacity diagnostics")


def render_performance_quality(suite_rows: list[dict[str, Any]]) -> str:
    values = [[r.get("model"), r.get("job_id"), r.get("prompt_ts"), r.get("gen_ts"), f"{r.get('task_pass_rate') * 100:.0f}%" if isinstance(r.get("task_pass_rate"), (int, float)) else MISSING, r.get("source")] for r in suite_rows if r.get("suite_status") == "SUCCESS" and not r.get("non_comparable") and r.get("has_performance") and r.get("has_quality")]
    return "## Comparable Performance / Quality\n\n" + _md_rows(["Model", "Job ID", "Prompt tok/s", "Generation tok/s", "Task pass rate", "Source manifest"], values)


def render_stage_coverage(suite_rows: list[dict[str, Any]]) -> str:
    values = [[r.get("model"), r.get("job_id"), "measured" if r.get("has_boundary") else "missing", "measured" if r.get("has_performance") else "missing", "measured" if r.get("has_quality") else "missing", r.get("suite_status"), r.get("source")] for r in suite_rows]
    return "## Stage Coverage / Diagnostics\n\n" + _md_rows(["Model", "Job ID", "Boundary", "Performance", "Quality", "Status", "Source manifest"], values, "No suite diagnostics")


def generate_report(input_dir: Path) -> str:
    """Generate the complete report from a directory without running inference."""
    load_rows, suite_rows = extract_rows(load_manifests(input_dir))
    sections = [
        f"# AutoBench Inventory Report\n\nManifests read: {len(load_rows) + len(suite_rows)}",
        render_overview(load_rows, suite_rows),
        render_load_preflight(load_rows, suite_rows),
        render_capacity(suite_rows),
        render_performance_quality(suite_rows),
        render_stage_coverage(suite_rows),
    ]
    return "\n\n".join(sections) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_pos", nargs="?", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    input_dir = args.input or args.input_pos or Path("results/inventory")
    report = generate_report(input_dir)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"Report: {args.output}")
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
