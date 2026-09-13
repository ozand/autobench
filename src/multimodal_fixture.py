"""Ephemeral, deterministic PNG fixtures for zero-inference contract tests only.

The fixture is created beneath a caller-owned, non-concurrently-mutated temporary
folder and removed on context exit. It validates bounded PNG container structure
and caller-visible metadata without decoding pixels, opening a model, or
invoking a process. This test helper is not a general hostile-filesystem API.
"""

from __future__ import annotations

import os
import stat
import struct
import zlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from src.multimodal_preflight import MAX_IMAGE_BYTES, MAX_IMAGE_DIMENSION, MultimodalPreflightError
from src.multimodal_runner import ImageReferenceRejected, validate_local_image_reference

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_IHDR_LENGTH = 13


class EphemeralFixtureError(ValueError):
    """Raised when an ephemeral PNG fixture violates its bounded contract."""


@dataclass(frozen=True)
class EphemeralPngFixture:
    """A temporary reference and its metadata; never serialize ``path``."""

    path: Path
    descriptor: dict[str, Any]


def _png_chunk(name: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + name
        + payload
        + struct.pack(">I", zlib.crc32(name + payload) & 0xFFFFFFFF)
    )


def deterministic_png_bytes(*, width: int, height: int) -> bytes:
    """Return a deterministic opaque RGBA PNG without reading external input."""
    if type(width) is not int or type(height) is not int or not 1 <= width <= MAX_IMAGE_DIMENSION or not 1 <= height <= MAX_IMAGE_DIMENSION:
        raise EphemeralFixtureError("fixture dimensions exceed contract")
    # One filter byte followed by deterministic fully transparent RGBA pixels per row.
    scanlines = b"".join(b"\x00" + b"\x00\x00\x00\xff" * width for _ in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return _PNG_SIGNATURE + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", zlib.compress(scanlines)) + _png_chunk(b"IEND", b"")


def _trusted_temporary_root(temporary_root: Path) -> Path:
    root = Path(temporary_root)
    if not root.is_dir() or root.is_symlink():
        raise EphemeralFixtureError("temporary fixture root is unsafe")
    return root.resolve(strict=True)


def _validate_temporary_scope(path: Path, temporary_root: Path) -> Path:
    """Resolve containment so a symlinked parent cannot escape the temporary root."""
    try:
        resolved_path = path.resolve(strict=True)
        resolved_path.relative_to(temporary_root)
    except (OSError, ValueError) as exc:
        raise EphemeralFixtureError("fixture reference is outside temporary scope") from exc
    return resolved_path


def _validate_png_container(payload: bytes, *, width: int, height: int) -> None:
    """Validate PNG chunk structure and CRCs without decoding or inspecting pixels."""
    if len(payload) < 45 or payload[:8] != _PNG_SIGNATURE:
        raise EphemeralFixtureError("fixture PNG signature is invalid")
    offset = 8
    saw_ihdr = saw_idat = saw_iend = False
    while offset < len(payload):
        if offset + 12 > len(payload):
            raise EphemeralFixtureError("fixture PNG chunk is truncated")
        length = struct.unpack(">I", payload[offset:offset + 4])[0]
        name = payload[offset + 4:offset + 8]
        end = offset + 12 + length
        if end > len(payload):
            raise EphemeralFixtureError("fixture PNG chunk length is invalid")
        chunk = payload[offset + 8:offset + 8 + length]
        crc = struct.unpack(">I", payload[offset + 8 + length:end])[0]
        if zlib.crc32(name + chunk) & 0xFFFFFFFF != crc:
            raise EphemeralFixtureError("fixture PNG chunk checksum is invalid")
        if not saw_ihdr:
            if name != b"IHDR" or length != _IHDR_LENGTH:
                raise EphemeralFixtureError("fixture PNG header is invalid")
            actual_width, actual_height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(">IIBBBBB", chunk)
            if (
                (actual_width, actual_height) != (width, height)
                or bit_depth != 8
                or color_type != 6
                or compression != 0
                or filtering != 0
                or interlace != 0
            ):
                raise EphemeralFixtureError("fixture PNG header metadata is invalid")
            saw_ihdr = True
        elif name == b"IDAT":
            saw_idat = True
        elif name == b"IEND":
            if length != 0 or not saw_idat or end != len(payload):
                raise EphemeralFixtureError("fixture PNG end marker is invalid")
            saw_iend = True
            break
        offset = end
    if not saw_ihdr or not saw_idat or not saw_iend:
        raise EphemeralFixtureError("fixture PNG structure is incomplete")


def validate_ephemeral_png_fixture(
    reference: Path | str,
    descriptor: Mapping[str, Any],
    *,
    temporary_root: Path,
) -> tuple[Path, dict[str, Any]]:
    """Validate bounded PNG-header metadata and temporary-locality without pixel decoding."""
    root = _trusted_temporary_root(temporary_root)
    try:
        path, safe_descriptor = validate_local_image_reference(reference, descriptor)
    except (ImageReferenceRejected, MultimodalPreflightError) as exc:
        raise EphemeralFixtureError("fixture local-reference validation failed") from exc
    resolved_path = _validate_temporary_scope(path, root)
    # Container parsing checks bounded structure/metadata only; no pixels are decoded.
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor_fd = os.open(resolved_path, flags)
    except OSError as exc:
        raise EphemeralFixtureError("fixture container cannot be safely read") from exc
    with os.fdopen(descriptor_fd, "rb") as handle:
        file_stat = os.fstat(handle.fileno())
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size > MAX_IMAGE_BYTES:
            raise EphemeralFixtureError("fixture container is unsafe")
        payload = handle.read(MAX_IMAGE_BYTES + 1)
    if len(payload) != file_stat.st_size:
        raise EphemeralFixtureError("fixture container changed during read")
    _validate_png_container(payload, width=safe_descriptor["width"], height=safe_descriptor["height"])
    return resolved_path, safe_descriptor


@contextmanager
def ephemeral_png_fixture(
    temporary_root: Path,
    *,
    width: int = 28,
    height: int = 28,
) -> Iterator[EphemeralPngFixture]:
    """Create exactly one deterministic PNG under ``temporary_root`` and always remove it."""
    root = _trusted_temporary_root(temporary_root)
    path = root / "autobench-ephemeral-fixture.png"
    payload = deterministic_png_bytes(width=width, height=height)
    if len(payload) > MAX_IMAGE_BYTES:
        raise EphemeralFixtureError("fixture size exceeds contract")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    created_identity: tuple[int, int] | None = None
    created_path = False
    cleanup_exact_path = False
    try:
        try:
            descriptor_fd = os.open(path, flags, 0o600)
        except FileExistsError as exc:
            raise EphemeralFixtureError("temporary fixture path is ambiguous") from exc
        created_path = True
        cleanup_exact_path = True
        try:
            initial_stat = os.fstat(descriptor_fd)
            created_identity = (initial_stat.st_dev, initial_stat.st_ino)
            with os.fdopen(descriptor_fd, "wb") as handle:
                handle.write(payload)
        except Exception:
            # fdopen owns the descriptor only after it succeeds.
            try:
                os.close(descriptor_fd)
            except OSError:
                pass
            raise
        descriptor = {
            "format": "png",
            "width": width,
            "height": height,
            "size_bytes": len(payload),
        }
        validated_path, safe_descriptor = validate_ephemeral_png_fixture(
            path, descriptor, temporary_root=root
        )
        yield EphemeralPngFixture(path=validated_path, descriptor=safe_descriptor)
    finally:
        # Unlink only the regular file this context created; never follow a replacement link.
        try:
            current = path.lstat()
            is_original_regular_file = (
                created_identity is not None
                and stat.S_ISREG(current.st_mode)
                and (current.st_dev, current.st_ino) == created_identity
            )
            # If fstat failed before identity capture, the documented trusted,
            # non-concurrent root invariant permits unlinking only this exact
            # regular destination; never follow a link or directory.
            is_setup_failure_regular_file = (
                cleanup_exact_path
                and created_identity is None
                and stat.S_ISREG(current.st_mode)
            )
            if is_original_regular_file or is_setup_failure_regular_file:
                path.unlink()
        except FileNotFoundError:
            pass
