#!/usr/bin/env python3
"""Validate the sanitized pre-run research gate without launching inference."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

REQUIRED_FRONTMATTER = {
    "id:",
    "title:",
    "category:",
    "tags:",
    "status:",
    "created:",
    "updated:",
    "environment:",
    "error_signatures:",
}

FORBIDDEN_PATTERNS = {
    "private_path": re.compile(r"(?i)(?:/home/|[A-Za-z]:[\\/](?:Users|home|srv|tmp|var)[\\/])"),
    "host_identifier": re.compile(
        r"(?i)(?:opencode@|192[.]168[.]\d+[.]\d+|100[.]67[.]\d+[.]\d+)"
    ),
    "sensitive_payload_key": re.compile(
        r"(?i)\b(?:stdout|stderr|raw_output|password|api[_-]?key|credential)\b"
    ),
}


def markdown_files(root: Path) -> list[Path]:
    return sorted(
        path
        for directory in (root / "raw", root / "wiki")
        if directory.exists()
        for path in directory.rglob("*.md")
    )


def frontmatter(text: str) -> str:
    if not text.startswith("---\n"):
        return ""
    end = text.find("\n---", 4)
    return text[4:end] if end >= 0 else ""


def check_note(path: Path, require_okf: bool) -> list[str]:
    errors: list[str] = []
    text = path.read_text(encoding="utf-8")
    if require_okf:
        header = frontmatter(text)
        missing = sorted(field for field in REQUIRED_FRONTMATTER if field not in header)
        if missing:
            errors.append(f"missing OKF fields: {', '.join(missing)}")
    for name, pattern in FORBIDDEN_PATTERNS.items():
        if pattern.search(text):
            errors.append(f"contains forbidden {name}")
    for target in re.findall(r"\]\(([^)]+)\)", text):
        if target.startswith(("http://", "https://", "#")):
            if target.startswith(("http://", "https://")) and not urlparse(target).netloc:
                errors.append(f"invalid URL: {target}")
            continue
        if not (path.parent / target).resolve().exists():
            errors.append(f"broken local link: {target}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("kb"))
    args = parser.parse_args()
    root = args.root.resolve()
    raw = sorted((root / "raw").glob("*.md")) if (root / "raw").exists() else []
    wiki = sorted((root / "wiki").glob("*.md")) if (root / "wiki").exists() else []
    if not raw or not wiki:
        print("FAIL: expected kb/raw/*.md and kb/wiki/*.md notes", file=sys.stderr)
        return 1

    failures: list[str] = []
    for path in markdown_files(root):
        failures.extend(f"{path}: {error}" for error in check_note(path, path in wiki))
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1

    print(f"PASS: {len(raw)} raw note(s), {len(wiki)} OKF note(s), links and sanitization verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
