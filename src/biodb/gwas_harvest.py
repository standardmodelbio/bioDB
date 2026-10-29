"""Raw GWAS Harvest acquisition, recovered from hg-horizon's local source.

The source schema SNP CHR BP A1 A2 FRQ BETA SE Z P N is preserved as float64
measurements and exact text tokens. As in the recovered MungeSumstats source,
A2 is the effect allele and A1 the other allele; neither is asserted to be a
verified genomic ALT/REF. File suffixes do not establish a reference build.
No public default manifest is invented: supply the actual source URL list.
``local_root`` supports source-host processing without network transfer.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

import polars as pl

from biodb._acquisition import (
    IngestError,
    IngestResult,
    download,
    file_record,
    ingest,
    iterate,
    load_ingest_result,
    local_path,
)
from biodb._downloads import stream_to_file

SOURCE = "gwas_harvest"
CACHE_DIR = Path("~/.cache/biodb/gwas_harvest").expanduser()


def study_id_from_url(url):
    """Preserve the complete source filename stem, including ontology/build suffix."""
    return Path(unquote(urlparse(url).path)).name.removesuffix(".tsv.gz").removesuffix(".tsv")


def gcst_of(study_id):
    match = re.search(r"GCST\d+", study_id)
    return match.group(0) if match else None


def resolve_local(url, local_root):
    """Find exact nested/flat source layout; absent input returns None."""
    try:
        return local_path({"url": url}, local_root)
    except FileNotFoundError:
        return None


def _read_manifest(path, manifest):
    """Validate all exact URLs before publishing a cache or selecting studies."""
    rows, seen = [], set()
    for line in path.read_text().splitlines():
        url = line.strip()
        if not url or url.startswith("#"):
            continue
        if urlparse(url).scheme not in {"http", "https", "file"}:
            raise ValueError(f"Expected exact absolute Harvest URL: {url!r}")
        if not urlparse(url).path.endswith((".tsv.gz", ".tsv")):
            raise ValueError(f"Expected summary statistics file: {url!r}")
        study = study_id_from_url(url)
        if url in seen:
            continue
        seen.add(url)
        rows.append(file_record(url, study, SOURCE, gcst_id=gcst_of(study), manifest_url=manifest))
    if not rows:
        raise ValueError("Harvest manifest has no summary statistics entries")
    return pl.DataFrame(
        rows,
        schema={
            "source": pl.String,
            "study_id": pl.String,
            "file_id": pl.String,
            "url": pl.String,
            "gcst_id": pl.String,
            "manifest_url": pl.String,
        },
    )


def list_datasets(manifest, *, study_ids=None, cache_dir=None, force=False):
    """Return all exact source URLs from a newline-delimited Harvest manifest.

    Validate downloaded content before publishing it, so a temporary error page
    cannot poison the cache or replace a valid manifest during a force refresh.
    """
    manifest = str(manifest)
    if urlparse(manifest).scheme in {"https", "http"}:
        path = (
            Path(cache_dir or CACHE_DIR).expanduser()
            / "manifests"
            / (hashlib.sha256(manifest.encode()).hexdigest() + ".txt")
        )
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
        files = files.filter(
            pl.col("study_id").is_in(requested) | pl.col("gcst_id").is_in(requested)
        )
    return files


def download_dataset(
    dataset, cache_dir=None, *, local_root=None, force=False, resume=True, progress=True
):
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
    """Yield raw batches, genomewide/unfiltered by default; validate EOF on exhaustion."""
    yield from iterate(dataset_or_path, source=SOURCE, **kwargs)


def ingest_sumstats(datasets, output_dir=None, **kwargs):
    """Durable file receipts and strict failures; workers=4, batch_size=100000."""
    return ingest(datasets, output_dir, source=SOURCE, **kwargs)


__all__ = [
    "load_ingest_result",
    "CACHE_DIR",
    "IngestError",
    "IngestResult",
    "list_datasets",
    "download_dataset",
    "iter_sumstats",
    "ingest_sumstats",
    "study_id_from_url",
    "gcst_of",
    "resolve_local",
]
