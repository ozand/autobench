"""Automated conformance tests for the four-stage per-model testing protocol.

Validates that canonical protocol definitions, AGENTS.md instructions, and
agent skills maintain consistent stages, evidence requirements, stop conditions,
anti-bypass rules, and hardware limits.
"""

from pathlib import Path
import re
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

CANONICAL_PROTOCOL_PATH = REPO_ROOT / "docs" / "model-testing-protocol.md"
AGENTS_MD_PATH = REPO_ROOT / "AGENTS.md"
SKILL_MODEL_RUN_PATH = REPO_ROOT / ".agents" / "skills" / "autobench-model-run" / "SKILL.md"
SKILL_PRE_RESEARCH_PATH = REPO_ROOT / ".agents" / "skills" / "autobench-pre-run-research" / "SKILL.md"

GOVERNED_DOCS = [
    CANONICAL_PROTOCOL_PATH,
    AGENTS_MD_PATH,
    SKILL_MODEL_RUN_PATH,
    SKILL_PRE_RESEARCH_PATH,
]

STAGE_ORDER = [
    "Stage 1",
    "Stage 2",
    "Stage 3",
    "Stage 4",
]


def test_governed_files_exist():
    for path in GOVERNED_DOCS:
        assert path.is_file(), f"Governed file missing: {path}"


def test_canonical_protocol_stage_order_and_naming():
    text = CANONICAL_PROTOCOL_PATH.read_text(encoding="utf-8")
    pos = 0
    for stage in STAGE_ORDER:
        idx = text.find(stage, pos)
        assert idx != -1, f"Stage '{stage}' not found after index {pos} in {CANONICAL_PROTOCOL_PATH}"
        pos = idx + len(stage)


def test_anti_bypass_dry_run_rule_in_governed_docs():
    """All governed docs must assert that --dry-run does NOT authorize/advance inference."""
    for path in [CANONICAL_PROTOCOL_PATH, AGENTS_MD_PATH, SKILL_MODEL_RUN_PATH]:
        text = path.read_text(encoding="utf-8")
        clean_text = text.replace("**", "").replace("__", "")
        assert "dry-run" in clean_text.lower() or "dry run" in clean_text.lower(), f"Missing dry-run mention in {path}"
        has_rule = (
            "not authorize" in clean_text.lower()
            or "does not authorize" in clean_text.lower()
            or "does not advance" in clean_text.lower()
            or "authorizes skipping" in clean_text.lower()
        )
        assert has_rule, f"Missing anti-bypass dry-run rule in {path}"


def test_layer_only_dual_gpu_rule_in_governed_docs():
    """All governed docs must enforce -sm layer and exclude tensor/row splits from authoritative matrix."""
    for path in [CANONICAL_PROTOCOL_PATH, AGENTS_MD_PATH, SKILL_MODEL_RUN_PATH]:
        text = path.read_text(encoding="utf-8")
        assert "-sm layer" in text, f"Missing '-sm layer' rule in {path}"
        assert "tensor" in text.lower(), f"Missing tensor split exclusion note in {path}"


def test_stage_stop_conditions_present_in_model_run_and_protocol():
    """Governed execution docs must contain explicit stop conditions."""
    for path in [CANONICAL_PROTOCOL_PATH, SKILL_MODEL_RUN_PATH]:
        text = path.read_text(encoding="utf-8")
        assert "stop condition" in text.lower() or "stop conditions" in text.lower(), f"Missing stop conditions in {path}"


def test_sanitization_rules_present_in_all_governed_docs():
    """All governed docs must mention sanitization and forbidden raw artifacts."""
    for path in GOVERNED_DOCS:
        text = path.read_text(encoding="utf-8")
        assert "sanitize" in text.lower() or "sanitized" in text.lower(), f"Missing sanitization rules in {path}"


def test_no_unreviewed_sweeps_rule():
    """Governed execution guidance must explicitly forbid parameter sweeps."""
    for path in [CANONICAL_PROTOCOL_PATH, AGENTS_MD_PATH, SKILL_MODEL_RUN_PATH]:
        text = path.read_text(encoding="utf-8")
        assert "sweep" in text.lower(), f"Missing anti-parameter-sweep rule in {path}"
