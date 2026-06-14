from __future__ import annotations

from hashlib import sha256
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file_sha256(path: str | Path, expected_sha256: str) -> str:
    actual = file_sha256(path)
    if actual.casefold() != expected_sha256.casefold():
        raise ValueError(f"Checksum mismatch for {path}: expected {expected_sha256}, got {actual}")
    return actual
