"""Manifest-backed shard acquisition, separate from the vendored bulk readers."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from urllib.parse import unquote, urlparse

from filelock import FileLock

_MANIFEST = ".biodb-manifest.json"


def validate_parquet(path: Path) -> None:
    """Raise if the shard has no readable Parquet footer."""
    import pyarrow.parquet as pq

    pq.read_metadata(path)


def _fingerprint(path: Path) -> dict:
    validate_parquet(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return {"size": path.stat().st_size, "sha256": digest.hexdigest()}


def _atomic_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _read_manifest(path: Path, identity: dict) -> dict | None:
    try:
        manifest = json.loads(path.read_text())
        if not isinstance(manifest, dict):
            return None
        if manifest.get("schema") != 1 or manifest.get("identity") != identity:
            return None
        files = manifest["files"]
        if not isinstance(files, list) or not files:
            return None
        names = set()
        for entry in files:
            name, url = entry["name"], entry["url"]
            if not isinstance(name, str) or Path(name).name != name or name in names:
                return None
            if not name.endswith(".parquet") or not isinstance(url, str):
                return None
            if unquote(Path(urlparse(url).path).name) != name:
                return None
            names.add(name)
        return manifest
    except (OSError, ValueError, KeyError, TypeError):
        return None


def ensure_shards(
    *,
    dataset: str,
    version: str,
    source_url: str,
    official_url: str,
    cache_dir: Path,
    limit_files: int | None,
    force: bool,
    list_files: Callable,
    download: Callable,
) -> list[Path]:
    """Resolve one exact listed membership; a preview never proves completeness.

    The manifest stores the *entire* listing even when only a prefix is fetched.
    Only receipt-validated members are reused. A legacy directory must obtain a
    listing before adoption. The per-dataset OS lock protects manifest updates
    and is automatically released if the process exits.
    """
    if limit_files is not None and (
        isinstance(limit_files, bool) or not isinstance(limit_files, int) or limit_files < 1
    ):
        raise ValueError("limit_files must be a positive integer or None")
    for label, value in [("dataset", dataset), ("version", version)]:
        if not value or Path(value).name != value or value in {".", ".."}:
            raise ValueError(f"Invalid {label}: {value!r}")
    source_url = source_url.rstrip("/")
    root = cache_dir / version
    if source_url != official_url.rstrip("/"):
        digest = hashlib.sha256(source_url.encode()).hexdigest()
        root = root / "_sources" / digest
    root = root / dataset
    root.mkdir(parents=True, exist_ok=True)
    identity = {"dataset": dataset, "version": version, "source_url": source_url}
    manifest_path = root / _MANIFEST
    with FileLock(str(root / ".biodb-cache.lock")):
        manifest = _read_manifest(manifest_path, identity)
        if manifest is None or force:
            urls = sorted(set(list_files(source_url)))
            if not urls:
                raise FileNotFoundError(f"No parquet files found under {source_url}")
            entries = []
            names = set()
            for url in urls:
                name = unquote(Path(urlparse(url).path).name)
                if not name.endswith(".parquet") or Path(name).name != name or name in names:
                    raise ValueError(f"Invalid or duplicate shard filename: {name!r}")
                names.add(name)
                entry = {"name": name, "url": url}
                if force:
                    entry["refresh_required"] = True
                entries.append(entry)
            # Publishing the expected set first means interrupted acquisitions
            # can resume without another directory listing.
            manifest = {"schema": 1, "identity": identity, "files": entries}
            _atomic_manifest(manifest_path, manifest)
        selected = manifest["files"]
        if limit_files is not None:
            selected = selected[:limit_files]
        result = []
        for entry in selected:
            path = root / entry["name"]
            valid = False
            if not force and not entry.get("refresh_required", False):
                try:
                    receipt = _fingerprint(path)
                    valid = "sha256" not in entry or all(
                        receipt[key] == entry.get(key) for key in ("size", "sha256")
                    )
                except (OSError, ValueError):
                    valid = False
            if not valid:
                # Never let an existence-only downloader reuse a damaged final.
                path = download(entry["url"], root, force=True)
                receipt = _fingerprint(path)
            entry.update(receipt)
            entry.pop("refresh_required", None)
            _atomic_manifest(manifest_path, manifest)
            result.append(path)
        return result
