"""Reviewed local launcher for one bounded target OCR process.

Tests inject ``popen_factory``. The real subprocess adapter stays dormant until
Issue #138 performs its fresh execution review and calls the launcher once.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Sequence

from src.multimodal_execution_driver import OUTPUT_LIMIT_BYTES, ProcessObservation
from src.multimodal_receipt import sanitize_multimodal_receipt, validate_multimodal_receipt
from src.multimodal_target_wrapper import run_one_target_smoke

TIMEOUT_SECONDS = 120


class MultimodalTargetLauncherError(ValueError):
    """Raised for one-run launcher boundary violations."""


PopenFactory = Callable[..., subprocess.Popen]


def _read_capped(stream, cap: int, bucket: bytearray, overflow: threading.Event, failures: list[Exception], stop_process: Callable[[], None]) -> None:
    """Drain one pipe while retaining no more than cap + one sentinel byte."""
    try:
        while not overflow.is_set():
            block = stream.read(min(4096, cap + 1 - len(bucket)))
            if not block:
                return
            bucket.extend(block)
            if len(bucket) > cap:
                overflow.set()
                stop_process()
                return
    except Exception as exc:
        if not overflow.is_set():
            failures.append(exc)
    finally:
        stream.close()


def one_shot_process_runner(popen_factory: PopenFactory):
    """Create a consumable-once shell-free adapter with concurrent bounded pipes."""
    used = False

    def run(argv: Sequence[str], timeout_seconds: int) -> ProcessObservation:
        nonlocal used
        if used:
            raise RuntimeError("one-run process boundary already consumed")
        used = True
        if type(timeout_seconds) is not int or timeout_seconds != TIMEOUT_SECONDS:
            raise RuntimeError("unapproved timeout")
        try:
            process = popen_factory(list(argv), shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as exc:
            raise RuntimeError("process creation failed") from exc
        stdout, stderr = bytearray(), bytearray()
        overflow = threading.Event()
        failures: list[Exception] = []
        termination_lock = threading.Lock()
        killed = False

        def stop_process() -> None:
            nonlocal killed
            with termination_lock:
                if not killed:
                    killed = True
                    try:
                        process.kill()
                    except OSError:
                        pass

        readers = [
            threading.Thread(target=_read_capped, args=(process.stdout, OUTPUT_LIMIT_BYTES, stdout, overflow, failures, stop_process)),
            threading.Thread(target=_read_capped, args=(process.stderr, OUTPUT_LIMIT_BYTES, stderr, overflow, failures, stop_process)),
        ]
        for reader in readers:
            reader.daemon = True
            reader.start()
        started = time.monotonic()
        returncode: int | None = None
        try:
            while returncode is None:
                if overflow.is_set() or failures or time.monotonic() - started >= timeout_seconds:
                    stop_process()
                    try:
                        returncode = process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        if overflow.is_set():
                            return ProcessObservation(-1, b"x" * (OUTPUT_LIMIT_BYTES + 1), b"")
                        raise TimeoutError
                    break
                try:
                    returncode = process.wait(timeout=0.01)
                except subprocess.TimeoutExpired:
                    continue
        finally:
            for reader in readers:
                reader.join(timeout=1)
            if any(reader.is_alive() for reader in readers):
                for stream in (process.stdout, process.stderr):
                    try:
                        stream.close()
                    except Exception:
                        pass
                for reader in readers:
                    reader.join(timeout=1)
            if any(reader.is_alive() for reader in readers):
                raise RuntimeError("stream drain failed")
        if failures:
            raise RuntimeError("stream reader failed") from failures[0]
        if overflow.is_set():
            # Preserve actual child returncode while exposing only the bounded sentinel.
            return ProcessObservation(returncode if returncode is not None else -1, b"x" * (OUTPUT_LIMIT_BYTES + 1), b"")
        return ProcessObservation(returncode if returncode is not None else -1, bytes(stdout), bytes(stderr))

    return run


def _safe_output_parent(parent: Path) -> Path:
    candidate = Path(parent)
    if not candidate.is_dir() or candidate.is_symlink():
        raise MultimodalTargetLauncherError("receipt directory is unsafe")
    try:
        lexical = os.path.normcase(os.path.normpath(str(candidate.absolute())))
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise MultimodalTargetLauncherError("receipt directory is unsafe") from exc
    if lexical != os.path.normcase(os.path.normpath(str(resolved))):
        raise MultimodalTargetLauncherError("receipt directory has unsafe ancestor")
    return resolved


def write_sanitized_receipt(output: Path, receipt: dict, protected: tuple[Path, ...]) -> None:
    """Atomically create exactly one validated receipt; never overwrite or follow links."""
    validation = validate_multimodal_receipt(receipt)
    if validation["status"] != "MULTIMODAL_RECEIPT_VALID":
        raise MultimodalTargetLauncherError("receipt is invalid")
    safe = sanitize_multimodal_receipt(validation["receipt"])
    path = Path(output)
    parent = _safe_output_parent(path.parent)
    resolved = parent / path.name
    if path.exists() or path.is_symlink() or any(resolved == item.resolve(strict=False) for item in protected):
        raise MultimodalTargetLauncherError("receipt output is unsafe")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".autobench-receipt-", dir=parent)
    temporary = Path(temporary_name)
    published = False
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise MultimodalTargetLauncherError("receipt temporary output is not regular")
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(json.dumps(safe, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        # A hard link is an atomic no-overwrite publication primitive: if another
        # writer created the final name, link fails and its content stays untouched.
        os.link(temporary, resolved)
        published = True
    except FileExistsError as exc:
        raise MultimodalTargetLauncherError("receipt output is unsafe") from exc
    finally:
        if descriptor != -1:
            os.close(descriptor)
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
        if not published and resolved.exists() and not resolved.is_symlink():
            # Only our publication can have reached this point; link does not replace.
            # Leave pre-existing outputs intact (the FileExistsError path above).
            pass


def launch_one_target_smoke(*, binary_path: Path, model_path: Path, projector_path: Path,
                            target_image_path: Path, plan: dict, temporary_root: Path,
                            output: Path, popen_factory: PopenFactory = subprocess.Popen) -> dict:
    """Prepare at most one process, write one sanitized receipt, then return."""
    safe_temporary_root = Path(temporary_root).resolve(strict=True)
    safe_output = Path(output).resolve(strict=False)
    if safe_output == safe_temporary_root or safe_temporary_root in safe_output.parents:
        raise MultimodalTargetLauncherError("receipt output must be outside the temporary root")
    receipt = run_one_target_smoke(
        binary_path=binary_path, model_path=model_path, projector_path=projector_path,
        source_image=target_image_path, plan=plan, temporary_root=temporary_root,
        process_runner=one_shot_process_runner(popen_factory),
    )
    write_sanitized_receipt(safe_output, receipt, (binary_path, model_path, projector_path, target_image_path))
    return receipt
