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
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys
from collections.abc import Callable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model_route import CANONICAL_LUNA_ROUTE, route_is_valid, validate_model_route

DEFAULT_HOST = os.environ.get("AUTOBENCH_REMOTE_HOST", "opencode@100.67.171.58")
DEFAULT_REMOTE_DIR = os.environ.get(
    "AUTOBENCH_REMOTE_DIR", "/home/opencode/code/autobench"
)
MAX_STAGED_IMAGE_BYTES = 10 * 1024 * 1024
_REMOTE_STAGING_ROOT = re.compile(r"/[A-Za-z0-9._/-]+")
_REMOTE_STAGED_BASENAME = re.compile(r"\.autobench-ocr-[A-Za-z0-9]{6}\.jpg")

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


def command_requires_model_route(arguments: Sequence[str]) -> bool:
    """Identify commands that can schedule model inference or targeted planning."""
    names = {Path(argument).name for argument in arguments if argument.endswith(".py")}
    benchmark_commands = {
        "authoritative_bench.py",
        "context_bench.py",
        "inventory_bench.py",
        "run_bench.py",
    }
    if not names.intersection(benchmark_commands):
        return False
    non_execution_flags = {"--status", "--help", "-h"}
    return not non_execution_flags.intersection(arguments)


def validate_required_model_route(args: argparse.Namespace) -> dict | None:
    """Require an exact, verified provider/model identity before inference."""
    route_values = (
        args.required_model_route,
        args.configured_model_route,
        args.resolved_provider,
        args.resolved_model,
    )
    required = (
        command_requires_model_route(args.command)
        or args.require_model_route
        or any(value is not None for value in route_values)
        or args.identity_check is not None
    )
    if not required:
        return None
    if any(value is None for value in route_values) or args.identity_check is None:
        raise WorkflowError(
            "Model route validation requires required/configured route, resolved "
            "provider/model, and identity check before workload scheduling."
        )
    try:
        evidence = validate_model_route(
            args.configured_model_route,
            required_route=args.required_model_route,
            resolved_provider=args.resolved_provider,
            resolved_model=args.resolved_model,
            identity_check=args.identity_check,
        )
    except ValueError as exc:
        raise WorkflowError("Model route validation input is invalid.") from exc
    if not route_is_valid(evidence):
        raise WorkflowError(
            "Model route validation failed before workload scheduling: "
            f"{evidence['status']}"
        )
    rendered = json.dumps(evidence, sort_keys=True, separators=(",", ":"))
    print(f"[route] evidence={rendered}")
    if args.route_evidence_output:
        output = Path(args.route_evidence_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return evidence


def execute_remote(host: str, remote_dir: str, arguments: Sequence[str]) -> None:
    """Execute a command remotely and stream its output."""
    print(f"[remote] Executing: {shlex.join(arguments)}")
    remote_shell(host, normalize_remote_command(arguments, remote_dir))


def _private_remote_shell(host: str, script: str) -> None:
    """Execute sensitive OCR orchestration without rendering its paths or command."""
    try:
        result = subprocess.run(
            ("ssh", *SSH_OPTIONS, host, "bash", "-lc", shlex.quote(script)),
            check=True,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise WorkflowError("Required executable was not found: ssh") from exc
    except subprocess.CalledProcessError as exc:
        raise WorkflowError("Reviewed OCR smoke invocation failed.") from exc
    if result.stderr:
        raise WorkflowError("Reviewed OCR smoke invocation produced unexpected output.")


def _validated_staging_source(source: Path) -> Path:
    """Accept one regular bounded JPEG after header-only shape validation."""
    candidate = Path(source)
    if (
        candidate.suffix.lower() != ".jpg" or candidate.is_symlink()
        or not candidate.is_file() or candidate.stat().st_size < 1
        or candidate.stat().st_size > MAX_STAGED_IMAGE_BYTES
    ):
        raise WorkflowError("Designated image is not eligible for staging.")
    try:
        from src.multimodal_target_harness import _jpeg_descriptor
        _jpeg_descriptor(candidate)
    except Exception as exc:
        raise WorkflowError("Designated image is not an eligible JPEG.") from exc
    return candidate


def _validated_remote_staging_root(root: str, remote_dir: str) -> str:
    """Reject ambiguous roots and any lexical location inside the checkout."""
    if not isinstance(root, str) or not _REMOTE_STAGING_ROOT.fullmatch(root):
        raise WorkflowError("Remote staging root is unsafe.")
    staging = PurePosixPath(root)
    checkout = PurePosixPath(remote_dir)
    if "." in staging.parts or ".." in staging.parts:
        raise WorkflowError("Remote staging root is unsafe.")
    if staging == checkout or checkout in staging.parents:
        raise WorkflowError("Remote staging root must be outside the checkout.")
    return root


def _remote_staging_root_check(root: str, remote_dir: str) -> str:
    """Return a fail-closed target-side root verification script without output."""
    quoted_root = shlex.quote(root)
    quoted_checkout = shlex.quote(remote_dir)
    return (
        "set -eu; "
        f"test -d {quoted_root}; test ! -L {quoted_root}; "
        f"test \"$(stat -c %a -- {quoted_root})\" = 700; "
        f"root=$(realpath -e -- {quoted_root}); checkout=$(realpath -e -- {quoted_checkout}); "
        "test \"$root\" = " + quoted_root + "; "
        "case \"$root\" in \"$checkout\"|\"$checkout\"/*) exit 1;; esac"
    )


def _validated_remote_runtime_root(root: str, remote_dir: str, staging_root: str) -> str:
    """Accept only a canonical private directory outside checkout and staging."""
    root = _validated_remote_staging_root(root, remote_dir)
    runtime = PurePosixPath(root)
    staging = PurePosixPath(staging_root)
    if runtime == staging or staging in runtime.parents or runtime in staging.parents:
        raise WorkflowError("OCR runtime root must be separate from staging.")
    return root


def _remote_runtime_root_check(root: str, remote_dir: str, staging_root: str) -> str:
    """Return a quiet target-side check for a protected OCR runtime directory."""
    return _remote_staging_root_check(root, remote_dir) + "; " + _remote_staging_root_check(staging_root, remote_dir)


def _cleanup_staged_jpeg(host: str, staged_image: str, staging_root: str) -> None:
    """Remove only the owned staged image without emitting its opaque path."""
    candidate = PurePosixPath(staged_image)
    if candidate.parent != PurePosixPath(staging_root) or not _REMOTE_STAGED_BASENAME.fullmatch(candidate.name):
        raise WorkflowError("Staged image cleanup is unsafe.")
    _private_remote_shell(
        host,
        f"set -eu; test -f {shlex.quote(staged_image)}; test ! -L {shlex.quote(staged_image)}; rm -f -- {shlex.quote(staged_image)}",
    )


def run_staged_ocr_smoke(
    host: str,
    source: Path,
    staging_root: str,
    remote_dir: str,
    *,
    binary: str,
    model: str,
    projector: str,
    temporary_root: str,
    output_root: str,
    stage_runner: Callable[[str, Path, str, str], str] | None = None,
    remote_runner: Callable[[str, str], None] = _private_remote_shell,
    cleanup_runner: Callable[[str, str, str], None] = _cleanup_staged_jpeg,
) -> None:
    """Stage one JPEG, invoke one reviewed OCR CLI, and always remove the stage."""
    if stage_runner is None:
        stage_runner = stage_designated_jpeg
    staging_root = _validated_remote_staging_root(staging_root, remote_dir)
    temporary_root = _validated_remote_runtime_root(temporary_root, remote_dir, staging_root)
    output_root = _validated_remote_runtime_root(output_root, remote_dir, staging_root)
    temporary_path = PurePosixPath(temporary_root)
    output_path = PurePosixPath(output_root)
    if temporary_path == output_path or temporary_path in output_path.parents or output_path in temporary_path.parents:
        raise WorkflowError("OCR temporary and output roots must be separate.")
    for value in (binary, model, projector):
        if not isinstance(value, str) or not _REMOTE_STAGING_ROOT.fullmatch(value):
            raise WorkflowError("OCR target input is unsafe.")
    remote_runner(host, _remote_runtime_root_check(temporary_root, remote_dir, staging_root))
    remote_runner(host, _remote_runtime_root_check(output_root, remote_dir, staging_root))
    receipt = f"{output_root}/receipt.json"
    diagnostic = f"{output_root}/diagnostic.json"
    remote_runner(host, f"set -eu; test ! -e {shlex.quote(receipt)}; test ! -e {shlex.quote(diagnostic)}")
    staged_image = stage_runner(host, source, staging_root, remote_dir)
    staged_path = PurePosixPath(staged_image)
    owned_stage = staged_path.parent == PurePosixPath(staging_root) and bool(_REMOTE_STAGED_BASENAME.fullmatch(staged_path.name))
    try:
        if not owned_stage:
            raise WorkflowError("Staged image handoff is unsafe.")
        command = shlex.join((
            ".venv/bin/python", "scripts/run_ocr_target_smoke.py",
            "--binary", binary, "--model", model, "--projector", projector,
            "--image", staged_image, "--temporary-root", temporary_root,
            "--output", receipt, "--diagnostic-output", diagnostic,
        ))
        remote_runner(host, f"set -euo pipefail; cd {shlex.quote(remote_dir)}; export AUTOBENCH_EXECUTION_MODE=local; exec {command}")
    finally:
        if owned_stage:
            cleanup_runner(host, staged_image, staging_root)


def stage_designated_jpeg(
    host: str,
    source: Path,
    remote_root: str,
    remote_dir: str,
    *,
    remote_runner: Callable[[str, str], str] | None = None,
    copy_runner: Callable[[Sequence[str]], None] | None = None,
) -> str:
    """Stage one image without logging operational paths or image data.

    The target location is private runtime state. This function emits no path,
    image, transport output, or readiness receipt; its caller records status only.
    """
    source = _validated_staging_source(source)
    remote_root = _validated_remote_staging_root(remote_root, remote_dir)
    remote_runner = remote_runner or (lambda target, script: remote_shell(target, script, capture_output=True))

    def default_copy(command: Sequence[str]) -> None:
        result = subprocess.run(list(command), text=True, capture_output=True)
        if result.returncode:
            raise WorkflowError("Designated image staging transport failed.")

    copy_runner = copy_runner or default_copy
    destination: str | None = None
    try:
        remote_runner(host, _remote_staging_root_check(remote_root, remote_dir))
        # mktemp is the target OS's exclusive-create primitive. The protected,
        # canonical 0700 root makes this returned private name ownership-safe.
        claimed_destination = remote_runner(host, f"set -eu; umask 077; created=$(mktemp -- {shlex.quote(remote_root)}/.autobench-ocr-XXXXXX.jpg); test -f \"$created\"; test ! -L \"$created\"; printf %s \"$created\"")
        if not isinstance(claimed_destination, str):
            raise WorkflowError("Designated image staging failed.")
        remote_path = PurePosixPath(claimed_destination)
        if remote_path.parent != PurePosixPath(remote_root) or not _REMOTE_STAGED_BASENAME.fullmatch(remote_path.name):
            raise WorkflowError("Designated image staging failed.")
        destination = claimed_destination
        remote_runner(host, f"set -eu; test \"$(realpath -e -- {shlex.quote(destination)})\" = {shlex.quote(destination)}; test -f {shlex.quote(destination)}; test ! -L {shlex.quote(destination)}")
        copy_runner(("scp", *SSH_OPTIONS, str(source), f"{host}:{destination}"))
        remote_runner(host, f"set -eu; test -f {shlex.quote(destination)}; test ! -L {shlex.quote(destination)}; chmod 600 {shlex.quote(destination)}")
        return destination
    except Exception as exc:
        if destination is not None:
            try:
                remote_runner(host, f"test -f {shlex.quote(destination)} && test ! -L {shlex.quote(destination)} && rm -f -- {shlex.quote(destination)}")
            except Exception:
                pass
        raise WorkflowError("Designated image staging failed.") from exc


def sync_results(host: str, remote_dir: str, repo: Path) -> None:
    """Copy only ignored benchmark artifacts into the local results directory."""
    destination = repo / "results"
    destination.mkdir(parents=True, exist_ok=True)
    for relative_path in ("inventory", "manifests", "runs"):
        local_path = destination / relative_path
        local_path.mkdir(parents=True, exist_ok=True)
        source = f"{host}:{remote_dir.rstrip('/')}/results/{relative_path}/."
        print(f"[sync] Copying {source} to {local_path}...")
        result = subprocess.run(
            ("scp", *SSH_OPTIONS, "-r", source, str(local_path)),
            text=True,
            capture_output=True,
        )
        if result.returncode != 0 and "No such file or directory" not in result.stderr:
            raise WorkflowError(
                f"Result synchronization failed for {relative_path}: "
                f"{result.stderr.strip()}"
            )

    comparisons = destination / "comparisons"
    comparisons.mkdir(parents=True, exist_ok=True)
    source = f"{host}:{remote_dir.rstrip('/')}/results/comparisons/matrix_*.md"
    result = subprocess.run(
        ("scp", *SSH_OPTIONS, source, str(comparisons)),
        text=True,
        capture_output=True,
    )
    if result.returncode != 0 and "No such file or directory" not in result.stderr:
        raise WorkflowError(
            f"Result synchronization failed for comparisons: {result.stderr.strip()}"
        )


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
    parser.add_argument("--stage-image", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--stage-root", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--run-staged-ocr-smoke", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--ocr-binary", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--ocr-model", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--ocr-projector", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--ocr-temporary-root", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--ocr-output-root", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--require-model-route",
        action="store_true",
        help="Block execution unless provider/model identity evidence is verified",
    )
    parser.add_argument(
        "--required-model-route",
        default=None,
        help=f"Required provider-qualified supervisor route (default example: {CANONICAL_LUNA_ROUTE})",
    )
    parser.add_argument(
        "--configured-model-route",
        default=None,
        help="Provider-qualified route selected by the supervisor",
    )
    parser.add_argument(
        "--resolved-provider",
        default=None,
        help="Sanitized provider identity returned by the resolver/completion",
    )
    parser.add_argument(
        "--resolved-model",
        default=None,
        help="Sanitized model identity returned by the resolver/completion",
    )
    parser.add_argument(
        "--identity-check",
        choices=("verified", "unverified", "auth_failed", "rejected"),
        default=None,
        help="Result of the provider identity completion check",
    )
    parser.add_argument(
        "--route-evidence-output",
        default=None,
        help="Optional local path for sanitized route evidence JSON",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if (args.stage_image is None) != (args.stage_root is None):
        parser.error("--stage-image and --stage-root must be provided together")
    if args.run_staged_ocr_smoke:
        required = (args.stage_image, args.stage_root, args.ocr_binary, args.ocr_model, args.ocr_projector, args.ocr_temporary_root, args.ocr_output_root)
        if any(value is None for value in required):
            parser.error("reviewed OCR smoke requires staging and OCR path arguments")
        if args.deploy_only or args.sync_results or args.command:
            parser.error("reviewed OCR smoke cannot combine with deploy-only, result sync, or a remote command")
    elif args.stage_image is not None:
        if not args.deploy_only:
            parser.error("image staging requires --deploy-only")
        if args.sync_results or args.command:
            parser.error("image staging cannot combine with result sync or a remote command")
    elif any(value is not None for value in (args.ocr_binary, args.ocr_model, args.ocr_projector, args.ocr_temporary_root, args.ocr_output_root)):
        parser.error("OCR path arguments require --run-staged-ocr-smoke")
    return args


def main() -> int:
    args = parse_arguments()
    repo = repository_root()

    try:
        ensure_clean_local_repository(repo)
        validate_required_model_route(args)
        ensure_origin_is_expected(repo, args.expected_origin)
        if not args.skip_local_tests:
            run_local_tests(repo)
        expected_sha = ensure_local_main_is_current(repo)
        deploy_commit(args.host, args.remote_dir, expected_sha)
        if args.run_staged_ocr_smoke:
            run_staged_ocr_smoke(
                args.host, args.stage_image, args.stage_root, args.remote_dir,
                binary=args.ocr_binary, model=args.ocr_model, projector=args.ocr_projector,
                temporary_root=args.ocr_temporary_root, output_root=args.ocr_output_root,
            )
        elif args.stage_image is not None:
            stage_designated_jpeg(args.host, args.stage_image, args.stage_root, args.remote_dir)
        elif not args.deploy_only:
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
