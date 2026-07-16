"""Open Targets study-level bulk readers (data-gathering only)."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from biodb.opentargets._bulk import DEFAULT_VERSION, ensure_cached_shards

_DEFAULT_STUDY_COLUMNS = [
    "studyId",
    "studyType",
    "traitFromSource",
    "projectId",
    "nSamples",
]


def get_study(
    *,
    version: str = DEFAULT_VERSION,
    columns: list[str] | None = None,
    cache_dir: str | Path | None = None,
    limit_files: int | None = None,
) -> pl.DataFrame:
    """Read the OT ``study`` dataset as a flat polars DataFrame.

    Parameters
    ----------
    version : str
        OT release (defaults to the pinned :data:`DEFAULT_VERSION`).
    columns : list[str], optional
        Columns to select. Defaults to
        ``["studyId", "studyType", "traitFromSource", "projectId", "nSamples"]``.
    cache_dir, limit_files
        Forwarded to :func:`ensure_cached_shards`.

    Returns
    -------
    polars.DataFrame
    """
    cols = columns if columns is not None else _DEFAULT_STUDY_COLUMNS
    shards = ensure_cached_shards(
        "study", version=version, cache_dir=cache_dir, limit_files=limit_files
    )
    frames = [pl.scan_parquet(p).select(cols) for p in shards]
    return pl.concat(frames).collect()


def attach_study_type(
    df: pl.DataFrame,
    study_df: pl.DataFrame,
    *,
    on: str = "studyId",
) -> pl.DataFrame:
    """Left-join ``studyType`` and ``traitFromSource`` from ``study_df`` onto ``df``."""
    keep = [c for c in (on, "studyType", "traitFromSource") if c in study_df.columns]
    return df.join(study_df.select(keep), on=on, how="left")
