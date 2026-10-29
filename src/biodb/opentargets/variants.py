"""Open Targets raw, lead-variant, and member-level bulk readers."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from biodb.opentargets import studies
from biodb.opentargets._bulk import DEFAULT_VERSION, ensure_cached_shards


def _scan(
    dataset: str, *, columns: list[str] | None = None, remote: bool = False, **kwargs
) -> pl.LazyFrame:
    if remote:
        from biodb.opentargets.readers import scan_dataset

        return scan_dataset(dataset, columns=columns, remote=True, **kwargs)
    paths = ensure_cached_shards(dataset, **kwargs)
    lazy = pl.scan_parquet(paths)
    return lazy if columns is None else lazy.select(columns)


def scan_credible_sets(*, columns: list[str] | None = None, **kwargs) -> pl.LazyFrame:
    """Lazily scan raw credible sets, retaining nested locus members by default.

    ``version``, ``cache_dir``, ``limit_files``, ``force`` and ``remote_url`` are
    forwarded to :func:`ensure_cached_shards`. Projection and caller filters
    remain in the Parquet query plan; acquisition still validates every shard
    selected by the file limit.
    """
    return _scan("credible_set", columns=columns, **kwargs)


def get_credible_sets(*, columns: list[str] | None = None, **kwargs) -> pl.DataFrame:
    """Read raw credible sets, including ``locus`` unless explicitly projected."""
    return scan_credible_sets(columns=columns, **kwargs).collect()


def scan_variant_annotations(*, columns: list[str] | None = None, **kwargs) -> pl.LazyFrame:
    """Scan raw variants with projection/filter pushdown.

    ``remote=True`` uses HTTP range reads against the explicit remote shard list,
    avoiding full downloads for narrow annotation projections.
    """
    return _scan("variant", columns=columns, **kwargs)


def get_variant_annotations(*, columns: list[str] | None = None, **kwargs) -> pl.DataFrame:
    """Read raw variant annotations without collapsing nested predictor fields."""
    return scan_variant_annotations(columns=columns, **kwargs).collect()


def summarize_credible_sets(frame: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    """Add lead ``pip``, ``credibleSetSize`` and ``pipStatus`` to raw records.

    Match the top-level ``variantId`` against *all* ``locus[*].variantId`` values.
    Duplicate matching IDs agree only if their posterior values agree (including
    nulls). Missing/conflicting matches yield null PIP, never a different
    variant's posterior. Status is one of ``matched``, ``lead_not_found``,
    ``conflicting_lead``, ``missing_probability``, ``missing_locus`` or
    ``missing_variant_id``. Raw columns and order are retained; the input frame
    type is preserved and lazy inputs are never collected.
    """
    eager = isinstance(frame, pl.DataFrame)
    lazy = frame.lazy() if eager else frame
    schema = lazy.collect_schema()
    if "variantId" not in schema:
        raise ValueError("credible sets require a top-level variantId column")
    locus_type = schema.get("locus")
    size = (
        pl.col("locus").list.len()
        if isinstance(locus_type, pl.List)
        else pl.lit(None, dtype=pl.UInt32)
    )
    if not isinstance(locus_type, pl.List) or locus_type.inner == pl.Null:
        result = lazy.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("pip"),
            size.alias("credibleSetSize"),
            pl.lit("missing_locus").alias("pipStatus"),
        )
        return result.collect() if eager else result
    if not isinstance(locus_type.inner, pl.Struct) or not {
        "variantId",
        "posteriorProbability",
    }.issubset({field.name for field in locus_type.inner.fields}):
        raise ValueError("locus members require variantId and posteriorProbability")
    prefix = "_biodb_"
    while any(name.startswith(prefix) for name in schema):
        prefix = "_" + prefix
    index, lead, matches = (prefix + suffix for suffix in ("row", "lead", "matches"))
    indexed = lazy.with_row_index(index)
    member_matches = (
        indexed.select(index, pl.col("variantId").alias(lead), "locus")
        .explode("locus")
        .group_by(index)
        .agg(
            pl.col("locus")
            .struct.field("posteriorProbability")
            .cast(pl.Float64)
            .filter(pl.col("locus").struct.field("variantId") == pl.col(lead))
            .alias(matches)
        )
    )
    result = indexed.join(member_matches, on=index, how="left", maintain_order="left")
    first = pl.col(matches).list.first()
    status = (
        pl.when(pl.col("variantId").is_null())
        .then(pl.lit("missing_variant_id"))
        .when(size.fill_null(0) == 0)
        .then(pl.lit("missing_locus"))
        .when(pl.col(matches).list.len() == 0)
        .then(pl.lit("lead_not_found"))
        .when(pl.col(matches).list.n_unique() > 1)
        .then(pl.lit("conflicting_lead"))
        .when(first.is_null() | first.is_nan())
        .then(pl.lit("missing_probability"))
        .otherwise(pl.lit("matched"))
    )
    result = result.with_columns(status.alias("pipStatus"), size.alias("credibleSetSize"))
    result = result.with_columns(
        pl.when(pl.col("pipStatus") == "matched")
        .then(first)
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("pip")
    ).drop(index, matches)
    return result.collect() if eager else result


def _filter_study_type(frame: pl.DataFrame, study_type, **kwargs) -> pl.DataFrame:
    if study_type is None:
        return frame
    if "studyType" not in frame.columns:
        frame = studies.attach_study_type(frame, studies.get_study(**kwargs))
    wanted = [study_type] if isinstance(study_type, str) else list(study_type)
    return frame.filter(pl.col("studyType").is_in(wanted))


def get_credible_set(
    *,
    version: str = DEFAULT_VERSION,
    study_type: str | list[str] | None = None,
    cache_dir: str | Path | None = None,
    limit_files: int | None = None,
    force: bool = False,
) -> pl.DataFrame:
    """Read one flat lead row per credible set with its ID-matched posterior.

    Scalar source columns, including ``zScore`` and any existing ``studyType``,
    are retained. Nested columns are available from :func:`get_credible_sets`.
    Missing or conflicting lead membership returns null ``pip`` with an explicit
    ``pipStatus``. ``study_type`` joins the study table only when needed.
    """
    kwargs = dict(version=version, cache_dir=cache_dir, limit_files=limit_files)
    raw = scan_credible_sets(force=force, **kwargs)
    scalar = [
        name
        for name, dtype in raw.collect_schema().items()
        if not isinstance(dtype, (pl.List, pl.Struct, pl.Array))
    ]
    summary = summarize_credible_sets(raw)
    out = summary.select(
        list(dict.fromkeys(scalar + ["pip", "credibleSetSize", "pipStatus"]))
    ).collect()
    return _filter_study_type(out, study_type, **kwargs)


def get_credible_set_members(
    *,
    version: str = DEFAULT_VERSION,
    study_type: str | list[str] | None = None,
    cache_dir: str | Path | None = None,
    limit_files: int | None = None,
    force: bool = False,
) -> pl.DataFrame:
    """Return one row per non-null locus member with that member's own effects.

    Member fields (including 95%/99% membership flags) are preserved. Top-level
    variant/effect fields receive a ``lead`` prefix; they never fill absent
    member statistics. Empty/null loci yield no member rows. Study/locus metadata
    stays on each row; ``pip`` aliases the member posterior probability.
    """
    kwargs = dict(version=version, cache_dir=cache_dir, limit_files=limit_files)
    raw = scan_credible_sets(force=force, **kwargs)
    schema = raw.collect_schema()
    locus = schema.get("locus")
    if not isinstance(locus, pl.List):
        raise ValueError("credible sets require a locus list column")
    if locus.inner == pl.Null:
        # An all-empty fixture has no inferred member struct. Its schema still
        # exposes member IDs and posterior, without inventing member effects.
        raw = raw.with_columns(
            pl.col("locus").cast(
                pl.List(
                    pl.Struct(
                        [
                            pl.Field("variantId", pl.String),
                            pl.Field("posteriorProbability", pl.Float64),
                        ]
                    )
                )
            )
        )
        locus = raw.collect_schema()["locus"]
    if not isinstance(locus.inner, pl.Struct):
        raise ValueError("locus members must be structs")
    member_fields = {field.name for field in locus.inner.fields}
    lead_fields = {
        "variantId",
        "chromosome",
        "position",
        "referenceAllele",
        "alternateAllele",
        "beta",
        "standardError",
        "zScore",
        "pValueMantissa",
        "pValueExponent",
        "posteriorProbability",
    }
    rename = {
        name: "lead" + name[0].upper() + name[1:]
        for name in schema
        if name != "locus" and (name in member_fields or name in lead_fields)
    }
    out = raw.rename(rename).explode("locus").unnest("locus")
    if "variantId" not in member_fields:
        raise ValueError("locus members require variantId")
    out = out.filter(pl.col("variantId").is_not_null())
    if "posteriorProbability" not in member_fields:
        out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias("posteriorProbability"))
    out = out.with_columns(pl.col("posteriorProbability").alias("pip"))
    for flag in ("is95CredibleSet", "is99CredibleSet"):
        if flag not in member_fields:
            out = out.with_columns(pl.lit(None, dtype=pl.Boolean).alias(flag))
    return _filter_study_type(out.collect(), study_type, **kwargs)


get_credible_set_variants = get_credible_set_members
"""Alias for :func:`get_credible_set_members`."""


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
