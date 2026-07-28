#!/usr/bin/env python3
"""Safely deploy local AutoBench commits and execute them on a remote host.

The local Git repository is the source of truth. This tool never stages or
commits changes. It verifies a clean, tested local commit; pushes it; updates
the remote checkout with a fast-forward-only merge; executes a command; and can
copy ignored benchmark artifacts back to the local workspace.

Examples:
    python scripts/run_remote.py --deploy-only
    python scripts/run_remote.py -- pytest
    python scripts/run_remote.py --sync-results -- python inventory_bench.py --status
    python scripts/run_remote.py --sync-results -- python authoritative_bench.py --smoke
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
from collections.abc import Sequence

DEFAULT_HOST = os.environ.get("AUTOBENCH_REMOTE_HOST", "opencode@100.67.171.58")
DEFAULT_REMOTE_DIR = os.environ.get(
    "AUTOBENCH_REMOTE_DIR", "/home/opencode/code/autobench"
)
SSH_OPTIONS = (
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=10",
    "-o",
    "StrictHostKeyChecking=accept-new",
)


class WorkflowError(RuntimeError):
    """Raised when a deployment safety check fails."""


def repository_root() -> Path:
    """Return the repository root containing this script."""
    return Path(__file__).resolve().parents[1]


def run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a command and raise a readable workflow error on failure."""
    try:
        return subprocess.run(
            list(command),
            cwd=cwd,
            check=True,
            text=True,
            capture_output=capture_output,
        )
    except FileNotFoundError as exc:
        raise WorkflowError(f"Required executable was not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        rendered = subprocess.list2cmdline(list(command))
        detail = (exc.stderr or exc.stdout or "").strip()
        suffix = f"\n{detail}" if detail else ""
        raise WorkflowError(
            f"Command failed with exit code {exc.returncode}: {rendered}{suffix}"
        ) from exc


def git_output(repo: Path, *arguments: str) -> str:
    """Run Git and return stripped stdout."""
    return run(
        ("git", *arguments), cwd=repo, capture_output=True
    ).stdout.strip()


def ensure_clean_local_repository(repo: Path) -> None:
    """Refuse deployment when tracked or untracked local files are pending."""
    status = git_output(repo, "status", "--porcelain")
    if status:
        raise WorkflowError(
            "Local repository is not clean. Review, test, and commit changes "
            "explicitly before remote execution:\n" + status
        )


def ensure_origin_is_expected(repo: Path, expected_origin: str | None) -> None:
    """Optionally verify that deployment uses the intended canonical remote."""
    if not expected_origin:
        return
    actual_origin = git_output(repo, "remote", "get-url", "origin")
    if actual_origin != expected_origin:
        raise WorkflowError(
            f"Unexpected origin URL: {actual_origin!r}; expected {expected_origin!r}."
        )


def ensure_local_main_is_current(repo: Path) -> str:
    """Fetch origin and ensure local main is not behind or diverged."""
    branch = git_output(repo, "branch", "--show-current")
    if branch != "main":
        raise WorkflowError(f"Deployment requires branch 'main'; current branch is {branch!r}.")

    run(("git", "fetch", "origin", "main"), cwd=repo)
    local_sha = git_output(repo, "rev-parse", "HEAD")
    remote_sha = git_output(repo, "rev-parse", "origin/main")
    merge_base = git_output(repo, "merge-base", "HEAD", "origin/main")

    if local_sha == remote_sha:
        return local_sha
    if merge_base == local_sha:
        raise WorkflowError(
            "Local main is behind origin/main. Run 'git pull --ff-only', verify, and retry."
        )
    if merge_base != remote_sha:
        raise WorkflowError(
            "Local main and origin/main have diverged. Resolve the divergence explicitly."
        )

    run(("git", "push", "origin", "main"), cwd=repo)
    return local_sha


def run_local_tests(repo: Path) -> None:
    """Run the local fast test suite before deployment."""
    print("[local] Running test suite...")
    run((sys.executable, "-m", "pytest", "-q"), cwd=repo)


def remote_shell(host: str, script: str, *, capture_output: bool = False) -> str:
    """Run a non-interactive Bash script over SSH."""
    result = run(
        ("ssh", *SSH_OPTIONS, host, "bash", "-lc", shlex.quote(script)),
        capture_output=capture_output,
    )
    return result.stdout.strip() if capture_output else ""


def deploy_commit(host: str, remote_dir: str, expected_sha: str) -> None:
    """Fast-forward the clean remote checkout and refresh its editable install."""
    quoted_dir = shlex.quote(remote_dir)
    script = f"""
set -euo pipefail
cd {quoted_dir}
if [ -n "$(git status --porcelain)" ]; then
    echo 'Remote checkout is dirty; refusing to overwrite it.' >&2
    git status --short >&2
    exit 20
fi
git fetch origin main
git checkout main
git merge --ff-only origin/main
test -x .venv/bin/python || python3 -m venv .venv
.venv/bin/python -m pip install --disable-pip-version-check -q -e .
test "$(git rev-parse HEAD)" = {shlex.quote(expected_sha)}
"""
    print(f"[remote] Deploying commit {expected_sha[:12]} to {host}:{remote_dir}...")
    remote_shell(host, script)


def normalize_remote_command(arguments: Sequence[str], remote_dir: str) -> str:
    """Build a safely quoted remote command using the project virtualenv."""
    if not arguments:
        raise WorkflowError("No remote command supplied. Use '-- <command> [args...]'.")

    command = list(arguments)
    if command[0] in {"python", "python3"}:
        command[0] = ".venv/bin/python"
    elif command[0] == "pytest":
        command = [".venv/bin/python", "-m", "pytest", *command[1:]]

    quoted_dir = shlex.quote(remote_dir)
    quoted_command = shlex.join(command)
    return (
        f"set -euo pipefail; cd {quoted_dir}; "
        f"export AUTOBENCH_EXECUTION_MODE=local; exec {quoted_command}"
    )


def execute_remote(host: str, remote_dir: str, arguments: Sequence[str]) -> None:
    """Execute a command remotely and stream its output."""
    print(f"[remote] Executing: {shlex.join(arguments)}")
    remote_shell(host, normalize_remote_command(arguments, remote_dir))


def sync_results(host: str, remote_dir: str, repo: Path) -> None:
    """Copy remote ignored benchmark artifacts into the local results directory."""
    destination = repo / "results"
    destination.mkdir(parents=True, exist_ok=True)
    source = f"{host}:{remote_dir.rstrip('/')}/results/."
    print(f"[sync] Copying {source} to {destination}...")
    run(("scp", *SSH_OPTIONS, "-r", source, str(destination)))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy a clean local commit and execute it on remote hardware.",
        epilog="Place runner options before '--'; everything after '--' is the remote command.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="SSH user and host")
    parser.add_argument(
        "--remote-dir", default=DEFAULT_REMOTE_DIR, help="Remote Git checkout"
    )
    parser.add_argument(
        "--expected-origin",
        default=os.environ.get("AUTOBENCH_EXPECTED_ORIGIN"),
        help="Optional exact origin URL safety check",
    )
    parser.add_argument(
        "--skip-local-tests",
        action="store_true",
        help="Skip local pytest only when tests were already run for this commit",
    )
    parser.add_argument(
        "--deploy-only", action="store_true", help="Synchronize code without executing"
    )
    parser.add_argument(
        "--sync-results", action="store_true", help="Copy results back after execution"
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    return args


def main() -> int:
    args = parse_arguments()
    repo = repository_root()

    try:
        ensure_clean_local_repository(repo)
        ensure_origin_is_expected(repo, args.expected_origin)
        if not args.skip_local_tests:
            run_local_tests(repo)
        expected_sha = ensure_local_main_is_current(repo)
        deploy_commit(args.host, args.remote_dir, expected_sha)

        if not args.deploy_only:
            execute_remote(args.host, args.remote_dir, args.command)
        if args.sync_results:
            sync_results(args.host, args.remote_dir, repo)
    except WorkflowError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("Workflow completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
