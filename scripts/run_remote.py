#!/usr/bin/env python3
"""
Remote execution runner for AutoBench on host k7000.

Workflow:
1. Checks local git status in T:\\Code\\autobench.
2. Pushes local changes to GitHub (if unpushed / dirty).
3. Connects to k7000 via SSH, pulls latest changes into /home/opencode/code/autobench.
4. Executes target benchmark script inside k7000 virtualenv (.venv).
5. Streams real-time stdout/stderr back to local console.
6. (Optional) Syncs remote results/ directory back to local results/.

Usage:
    python scripts/run_remote.py inventory_bench.py
    python scripts/run_remote.py context_bench.py --sync-results
    python scripts/run_remote.py --cmd "pytest"
"""

import sys
import os
import argparse
import subprocess
import shutil

DEFAULT_HOST = os.environ.get("K7000_HOST", "opencode@100.67.171.58")
REMOTE_DIR = os.environ.get("K7000_AUTOBENCH_DIR", "/home/opencode/code/autobench")


def run_local_cmd(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a local shell command."""
    return subprocess.run(cmd, check=check, text=True, capture_output=True)


def get_repo_root() -> str:
    """Get the root directory of the local autobench repository."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)


def sync_git_local_to_remote(repo_root: str, host: str, auto_commit: bool = True) -> bool:
    """Ensure local changes are committed & pushed, then pulled on remote."""
    print("🔄 Checking local git status...")
    status_res = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True
    )
    
    if status_res.stdout.strip():
        if auto_commit:
            print("📝 Uncommitted local changes detected. Committing...")
            subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
            subprocess.run(
                ["git", "commit", "-m", "chore: auto-sync local changes for remote execution"],
                cwd=repo_root,
                check=True,
            )
        else:
            print("⚠️ Uncommitted changes exist. Please commit or use --auto-commit.")
            return False

    print("🚀 Pushing local changes to GitHub...")
    push_res = subprocess.run(["git", "push"], cwd=repo_root)
    if push_res.returncode != 0:
        print("❌ Failed to push local changes to GitHub.")
        return False

    print(f"📥 Pulling latest changes on {host}...")
    ssh_cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        host,
        f"cd {REMOTE_DIR} && git pull origin main"
    ]
    pull_res = subprocess.run(ssh_cmd)
    if pull_res.returncode != 0:
        print("❌ Remote git pull failed.")
        return False

    return True


def execute_remote_command(host: str, target_cmd: str) -> int:
    """Execute command on k7000 inside venv and stream output."""
    full_remote_cmd = f"cd {REMOTE_DIR} && source .venv/bin/activate && {target_cmd}"
    print(f"⚡ Executing on {host}: {target_cmd}\n" + "-" * 60)
    
    ssh_cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-t",
        host,
        full_remote_cmd,
    ]
    
    proc = subprocess.run(ssh_cmd)
    print("-" * 60)
    return proc.returncode


def sync_results_back(host: str, repo_root: str) -> None:
    """Download updated results from k7000 to local repo."""
    print(f"📥 Syncing remote results/ back to local {repo_root}/results/...")
    local_results = os.path.join(repo_root, "results")
    os.makedirs(local_results, exist_ok=True)
    
    scp_cmd = [
        "scp",
        "-o", "StrictHostKeyChecking=no",
        "-r",
        f"{host}:{REMOTE_DIR}/results/*",
        local_results
    ]
    res = subprocess.run(scp_cmd)
    if res.returncode == 0:
        print("✅ Results synchronized successfully.")
    else:
        print("⚠️ Could not sync results via scp (check if remote results directory contains files).")


def main():
    parser = argparse.ArgumentParser(description="Run AutoBench scripts remotely on k7000.")
    parser.add_argument("script", nargs="?", help="Python script to run (e.g., inventory_bench.py, context_bench.py)")
    parser.add_argument("--cmd", help="Custom full command to run remotely (e.g. 'pytest' or 'python3 authoritative_bench.py --limit 5')")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"SSH host string (default: {DEFAULT_HOST})")
    parser.add_argument("--no-push", action="store_true", help="Skip local git push & remote git pull")
    parser.add_argument("--sync-results", action="store_true", help="Sync results/ directory back after execution")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Additional arguments passed to the script")

    args = parser.parse_args()

    if not args.script and not args.cmd:
        parser.print_help()
        sys.exit(1)

    repo_root = get_repo_root()

    if not args.no_push:
        if not sync_git_local_to_remote(repo_root, args.host):
            sys.exit(1)

    if args.cmd:
        target_cmd = args.cmd
    else:
        extra = " ".join(args.extra_args) if args.extra_args else ""
        target_cmd = f"python3 {args.script} {extra}".strip()

    exit_code = execute_remote_command(args.host, target_cmd)

    if getattr(args, "sync_results", False):
        sync_results_back(args.host, repo_root)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
