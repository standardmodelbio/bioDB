"""Release-aware I/O shared by transcript data sources (no model dependencies)."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import sqlite3
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import requests

from biodb._downloads import stream_to_file

REFERENCE_CHROMOSOMES = tuple([str(i) for i in range(1, 23)] + ["X", "Y", "MT"])
_VERSION = re.compile(r"^(ENS[A-Z]*\d+)\.(\d+)(.*)$")
_ATTRIBUTE = re.compile(r'(\w+)\s+"([^"]*)"')


def split_id(identifier: str) -> tuple[str, str | None]:
    """Strip only the numeric Ensembl version; preserve locus suffixes (PAR_Y)."""
    match = _VERSION.fullmatch(identifier)
    if match:
        return match[1] + match[3], match[2]
    return identifier, None


def chromosome(value: str) -> str:
    value = value.removeprefix("chr")
    return "MT" if value in {"M", "MT"} else value


def open_text(path: str | Path) -> TextIO:
    path = Path(path)
    return (
        gzip.open(path, "rt", encoding="utf-8")
        if path.suffix == ".gz"
        else path.open(encoding="utf-8")
    )


def digest_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_source(
    url: str,
    destination: str | Path,
    *,
    source: str,
    release: str,
    assembly: str = "GRCh38.p14",
    terms_url: str = "",
    expected_sha256: str | None = None,
    expected_headers: dict[str, str] | None = None,
    progress: bool = True,
) -> Path:
    """Atomically cache original bytes and a source/hash manifest; detect corruption.

    A changed release/URL at an existing destination is rejected. Callers should
    use source/release-qualified cache paths instead of overwriting snapshots.
    """
    path = Path(destination)
    manifest_path = path.with_name(path.name + ".source.json")
    identity = {"url": url, "source": source, "release": str(release), "assembly": assembly}
    if path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if any(manifest.get(k) != v for k, v in identity.items()):
            raise ValueError(f"source identity mismatch at {path}; use a release-qualified cache")
        actual = digest_file(path)
        if actual != manifest["sha256"] or (expected_sha256 and actual != expected_sha256):
            raise ValueError(f"checksum mismatch for cached source {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    try:

        class CheckedSession(requests.Session):
            def get(self, *args: Any, **kwargs: Any) -> requests.Response:
                response = super().get(*args, **kwargs)
                for header, expected in (expected_headers or {}).items():
                    if response.headers.get(header) != expected:
                        response.close()
                        raise ValueError(
                            f"upstream release header {header} does not match {expected!r}"
                        )
                return response

        with CheckedSession() as session:
            stream_to_file(url, partial, progress=progress, session=session)
        actual = digest_file(partial)
        if expected_sha256 and actual != expected_sha256:
            raise ValueError(f"checksum mismatch for downloaded source {url}")
        partial.replace(path)
        manifest = {
            **identity,
            "sha256": actual,
            "size_bytes": path.stat().st_size,
            "acquired_at": datetime.now(timezone.utc).isoformat(),
            "terms_url": terms_url,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    finally:
        partial.unlink(missing_ok=True)
    return path


def iter_gtf_features(path: str | Path) -> Iterator[dict[str, Any]]:
    """Read every GTF feature, preserving phase and repeated attributes."""
    with open_text(path) as fh:
        for number, line in enumerate(fh, 1):
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                raise ValueError(f"GTF line {number}: expected 9 fields")
            chrom, source, feature, start, end, score, strand, phase, attrs = fields
            begin, stop = int(start) - 1, int(end)
            if begin < 0 or stop <= begin or strand not in {"+", "-", "."}:
                raise ValueError(f"GTF line {number}: invalid interval or strand")
            attributes: dict[str, list[str]] = {}
            for key, value in _ATTRIBUTE.findall(attrs):
                attributes.setdefault(key, []).append(value)
            yield {
                "chromosome": chromosome(chrom),
                "source": source,
                "feature_type": feature,
                "start": begin,
                "end": stop,
                "strand": strand,
                "phase": None if phase in {".", "-"} else int(phase),
                "score": None if score in {".", "-"} else float(score),
                "attributes": attributes,
            }


def iter_fasta(path: str | Path) -> Iterator[tuple[str, str]]:
    """Stream complete FASTA records, retaining the source header."""
    header: str | None = None
    chunks: list[str] = []
    with open_text(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks).upper()
                header, chunks = line[1:].strip(), []
            elif line.strip():
                if header is None:
                    raise ValueError("FASTA sequence precedes its header")
                chunks.append(line.strip())
    if header is not None:
        yield header, "".join(chunks).upper()


class SequenceStore:
    """Disk-backed exact native-ID sequence lookup; safe for whole transcriptomes."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.connection = sqlite3.connect(self.path)

    @classmethod
    def create(cls, path: str | Path) -> SequenceStore:
        store = cls(path)
        store.connection.execute(
            "CREATE TABLE IF NOT EXISTS sequences "
            "(kind TEXT, identifier TEXT, header TEXT, sequence TEXT, sha256 TEXT, "
            "PRIMARY KEY(kind, identifier))"
        )
        return store

    def add_fasta(self, path: str | Path, *, kind: str) -> None:
        if kind not in {"rna", "protein"}:
            raise ValueError("sequence kind must be rna or protein")
        batch = []
        for header, sequence in iter_fasta(path):
            identifier = header.split()[0].split("|")[0]
            batch.append(
                (kind, identifier, header, sequence, hashlib.sha256(sequence.encode()).hexdigest())
            )
            if len(batch) == 1000:
                self.connection.executemany("INSERT INTO sequences VALUES (?, ?, ?, ?, ?)", batch)
                self.connection.commit()
                batch.clear()
        self.connection.executemany("INSERT INTO sequences VALUES (?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def get(self, identifier: str, *, kind: str = "rna") -> str | None:
        row = self.connection.execute(
            "SELECT sequence FROM sequences WHERE kind=? AND identifier=?", (kind, identifier)
        ).fetchone()
        return row[0] if row else None

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> SequenceStore:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
