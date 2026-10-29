"""GWAS Catalog raw harmonised statistics from exact manifest entries.

Every file for each study is retained. No filename guessing, significance
threshold, effect reconstruction, allele flipping or coordinate normalization
is performed. Legacy harmonised headers and GWAS-SSF are read by name. P=0,
missing effects and original tokens remain available. Downloads validate gzip
EOF; batch successes and successful zero-row results persist independently.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

import polars as pl

from biodb._acquisition import (
    IngestError,
    IngestResult,
    download,
    file_record,
    ingest,
    iterate,
    load_ingest_result,
)
from biodb._downloads import stream_to_file

BASE_URL = "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics"
MANIFEST_URL = f"{BASE_URL}/harmonised_list.txt"
CACHE_DIR = Path("~/.cache/biodb/gwas_catalog").expanduser()
SOURCE = "gwas_catalog"


def _read_manifest(path, manifest):
    """Validate the entire source manifest before any study selection."""
    rows, seen = [], set()
    recognized = 0
    for line in path.read_text().splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        parsed = urlparse(entry)
        relative = entry[2:] if entry.startswith("./") else entry
        pathname = parsed.path if parsed.scheme else relative
        match = re.search(r"(?:^|/)(GCST\d+)/harmonised/[^/]+\.(?:h\.tsv|tsv)(?:\.gz)?$", pathname)
        if match is None:
            if re.search(r"(?:^|/)GCST\d+/harmonised/[^/]+\.(?:tbi|csi|yaml|yml)$", pathname):
                continue
            raise ValueError(f"Invalid Catalog manifest entry: {entry!r}")
        recognized += 1
        study = match.group(1)
        url = entry if parsed.scheme else f"{BASE_URL}/{relative.lstrip('/')}"
        if url in seen:
            continue
        seen.add(url)
        rows.append(file_record(url, study, SOURCE, manifest_path=relative, manifest_url=manifest))
    if recognized == 0:
        raise ValueError("Catalog manifest has no recognized summary statistics entries")
    return pl.DataFrame(
        rows,
        schema={
            "source": pl.String,
            "study_id": pl.String,
            "file_id": pl.String,
            "url": pl.String,
            "manifest_path": pl.String,
            "manifest_url": pl.String,
        },
    )


def list_datasets(manifest=None, *, study_ids=None, cache_dir=None, force=False):
    """Return one row per exact harmonised file in a manifest, including aliases.

    ``manifest`` is a local text file or URL; omitted uses EBI's manifest.
    Membership and file paths come exclusively from that manifest. A caller
    should save this table to pin a source snapshot before large ingestion.
    Invalid downloaded manifests never replace the previously validated cache.
    """
    manifest = MANIFEST_URL if manifest is None else str(manifest)
    if urlparse(manifest).scheme in {"https", "http"}:
        import hashlib

        root = Path(cache_dir or CACHE_DIR).expanduser()
        path = root / "manifests" / (hashlib.sha256(manifest.encode()).hexdigest() + ".txt")
        if force or not path.exists():
            stream_to_file(
                manifest,
                path,
                resume=True,
                progress=False,
                validate=lambda candidate: _read_manifest(candidate, manifest),
            )
    else:
        path = Path(manifest)
    files = _read_manifest(path, manifest)
    if study_ids is not None:
        requested = [study_ids] if isinstance(study_ids, str) else list(study_ids)
        files = files.filter(pl.col("study_id").is_in(requested))
    return files


def download_dataset(
    dataset, cache_dir=None, *, local_root=None, force=False, resume=True, progress=True
):
    """Acquire one exact ``list_datasets`` row; return its validated local path."""
    return download(
        dataset,
        source=SOURCE,
        cache_dir=cache_dir,
        local_root=local_root,
        force=force,
        resume=resume,
        progress=progress,
    )


def iter_sumstats(dataset_or_path, **kwargs):
    """Yield raw Polars batches; default scope is genomewide, with no P filter.

    Accept a manifest row or local file. Options: batch_size/batch_rows,
    p_max (inclusive), scope='regions' plus regions=['chr1:100-200'], local_root,
    cache_dir. Consumers must exhaust the iterator to validate complete input.
    """
    yield from iterate(dataset_or_path, source=SOURCE, **kwargs)


def ingest_sumstats(datasets, output_dir=None, **kwargs):
    """Durably ingest a manifest table/rows. See ``biodb._acquisition.ingest``.

    Defaults workers=4, batch_size=100000, resume=True, allow_partial=False.
    On any failure IngestError.result exposes committed successes and failures.
    Changed semantic configuration or input identity requires a new run directory.
    """
    return ingest(datasets, output_dir, source=SOURCE, **kwargs)


__all__ = [
    "load_ingest_result",
    "BASE_URL",
    "CACHE_DIR",
    "MANIFEST_URL",
    "IngestError",
    "IngestResult",
    "list_datasets",
    "download_dataset",
    "iter_sumstats",
    "ingest_sumstats",
]
