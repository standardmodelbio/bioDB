"""Open Targets variant-level bulk readers (data-gathering only)."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from biodb.opentargets import studies
from biodb.opentargets._bulk import DEFAULT_VERSION, ensure_cached_shards

_CREDIBLE_SET_SCALAR_COLUMNS = [
    "variantId",
    "chromosome",
    "position",
    "beta",
    "pValueMantissa",
    "pValueExponent",
    "standardError",
    "finemappingMethod",
    "studyLocusId",
    "studyId",
    "confidence",
    "credibleSetlog10BF",
]


def get_credible_set(
    *,
    version: str = DEFAULT_VERSION,
    study_type: str | list[str] | None = None,
    cache_dir: str | Path | None = None,
    limit_files: int | None = None,
) -> pl.DataFrame:
    """Read the OT ``credible_set`` bulk Parquet into a flat polars DataFrame.

    The fine-mapping posterior (``pip``) is extracted from the first element of
    the nested ``locus`` list-of-struct column, and ``credibleSetSize`` from its
    length. ``studyType`` is not present in ``credible_set`` itself; pass
    ``study_type`` to join it from the ``study`` dataset and filter.

    Parameters
    ----------
    version : str
        OT release (defaults to the pinned :data:`DEFAULT_VERSION`).
    study_type : str | list[str], optional
        If given, join the ``study`` dataset on ``studyId`` and keep only rows
        whose ``studyType`` matches (e.g. ``"gwas"`` or ``["eqtl", "pqtl"]``).
    cache_dir, limit_files
        Forwarded to :func:`ensure_cached_shards`.

    Returns
    -------
    polars.DataFrame
    """
    shards = ensure_cached_shards(
        "credible_set", version=version, cache_dir=cache_dir, limit_files=limit_files
    )
    lazy = pl.concat([pl.scan_parquet(p) for p in shards])
    out = lazy.select(
        *_CREDIBLE_SET_SCALAR_COLUMNS,
        pl.col("locus").list.first().struct.field("posteriorProbability").alias("pip"),
        pl.col("locus").list.len().alias("credibleSetSize"),
    ).collect()

    if study_type is not None:
        wanted = [study_type] if isinstance(study_type, str) else list(study_type)
        study_df = studies.get_study(version=version, cache_dir=cache_dir, limit_files=limit_files)
        out = studies.attach_study_type(out, study_df)
        out = out.filter(pl.col("studyType").is_in(wanted))
    return out
