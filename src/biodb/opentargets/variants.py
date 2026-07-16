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


_VEP_AGGREGATES = {"max", "mean", "median"}


def get_variant_effects(
    *,
    version: str = DEFAULT_VERSION,
    aggregate: str = "max",
    cache_dir: str | Path | None = None,
    limit_files: int | None = None,
) -> pl.DataFrame:
    """Per-variant VEP-style deleteriousness score from the OT ``variant`` dataset.

    OT ships one ``normalisedScore`` (on a −1…+1 axis) per predictor method
    inside the ``variantEffect`` list-of-struct column, but no single scalar.
    This aggregates them per variant.

    Parameters
    ----------
    aggregate : {"max", "mean", "median"}
        How to combine predictor scores. ``"max"`` (default) takes the most
        deleterious predictor.

    Returns
    -------
    polars.DataFrame with columns ``variantId, vep_score``.
    """
    if aggregate not in _VEP_AGGREGATES:
        raise ValueError(f"aggregate must be one of {sorted(_VEP_AGGREGATES)}, got {aggregate!r}")
    shards = ensure_cached_shards(
        "variant", version=version, cache_dir=cache_dir, limit_files=limit_files
    )
    lazy = pl.concat([pl.scan_parquet(p) for p in shards])
    score = (
        pl.col("variantEffect")
        .list.eval(pl.element().struct.field("normalisedScore"))
        .alias("_scores")
    )
    agg = {
        "max": pl.col("_scores").list.max(),
        "mean": pl.col("_scores").list.mean(),
        "median": pl.col("_scores").list.median(),
    }[aggregate]
    return lazy.select("variantId", score).select("variantId", agg.alias("vep_score")).collect()
