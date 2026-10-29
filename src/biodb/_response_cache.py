"""Crash-safe, process-locked receipts for small validated API responses."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from filelock import FileLock


def cache_path(cache_dir: str | Path | None, namespace: str, identity: str) -> Path:
    root = (
        Path(cache_dir).expanduser()
        if cache_dir is not None
        else Path("~/.cache/biodb").expanduser()
    )
    return root / namespace / (hashlib.sha256(identity.encode()).hexdigest() + ".json")


@contextlib.contextmanager
def locked(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path.with_suffix(".lock"))):
        yield


def read(path: Path, identity: str) -> Any:
    try:
        receipt = json.loads(path.read_text())
        if receipt["identity"] == identity and receipt["complete"] is True:
            return receipt["response"]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    raise KeyError(identity)


def write(path: Path, identity: str, response: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump({"identity": identity, "complete": True, "response": response}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
