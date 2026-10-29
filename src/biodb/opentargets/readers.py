"""Raw, projection-aware Open Targets convenience readers."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import polars as pl

from biodb.opentargets import _bulk
from biodb.opentargets._bulk import DEFAULT_VERSION, ensure_cached_shards


def scan_dataset(
    dataset: str | None = None,
    *,
    remote_url: str | None = None,
    version: str = DEFAULT_VERSION,
    cache_dir: str | Path | None = None,
    limit_files: int | None = None,
    force: bool = False,
    remote: bool = False,
    columns: list[str] | None = None,
    **download_kwargs,
) -> pl.LazyFrame:
    """Scan validated raw Parquet with lazy projection and predicate pushdown.

    Acquisition uses the same authoritative manifest resolver as ``get_dataset``.
    A file limit is an explicit preview, never a complete dataset. Files must be
    acquired locally before scanning; column/row projections avoid decoding
    unwanted Parquet columns and rows, rather than avoiding shard downloads.
    ``remote=True`` scans the explicit remote listing directly using HTTP range
    reads, avoiding bulk acquisition for narrow projections. It requires network
    access during collection and does not certify a local cache. Additional local
    download options are ``session`` and ``timeout``.
    """
    if remote_url:
        dataset = Path(urlparse(remote_url.rstrip("/")).path).name
    dataset = dataset or "association_overall_direct"
    if remote:
        if force:
            raise ValueError("force is only meaningful for local acquisition")
        if limit_files is not None and (
            isinstance(limit_files, bool) or not isinstance(limit_files, int) or limit_files < 1
        ):
            raise ValueError("limit_files must be a positive integer or None")
        source = (
            remote_url or f"{_bulk.DOWNLOADS_BASE_URL_TEMPLATE.format(version=version)}/{dataset}"
        )
        paths = _bulk._list_parquet_files(source, **download_kwargs)
        if limit_files is not None:
            paths = paths[:limit_files]
        if not paths:
            raise FileNotFoundError(f"No parquet files found under {source}")
        lazy = pl.scan_parquet(paths)
        return lazy if columns is None else lazy.select(columns)
    paths = ensure_cached_shards(
        dataset,
        remote_url=remote_url,
        version=version,
        cache_dir=cache_dir,
        limit_files=limit_files,
        force=force,
        **download_kwargs,
    )
    lazy = pl.scan_parquet(paths)
    return lazy if columns is None else lazy.select(columns)


def get_diseases(*, columns: list[str] | None = None, **kwargs) -> pl.DataFrame:
    """Read raw ``disease`` records, retaining nested fields by default."""
    return scan_dataset("disease", columns=columns, **kwargs).collect()


def get_traits(*, columns: list[str] | None = None, **kwargs) -> pl.DataFrame:
    """Read OT trait/phenotype records from the ``disease`` dataset."""
    return get_diseases(columns=columns, **kwargs)


def get_drugs(*, columns: list[str] | None = None, **kwargs) -> pl.DataFrame:
    """Read raw drug records from the official ``drug_molecule`` dataset."""
    return scan_dataset("drug_molecule", columns=columns, **kwargs).collect()
