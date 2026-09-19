"""ENCODE-rE2G client — the CRISPRi enhancer–gene benchmark and rE2G predictions.

The ENCODE enhancer–gene regulatory interaction effort (Gschwind et al.
2023, "An encyclopedia of enhancer-gene regulatory interactions in the human
genome") harmonised every CRISPR perturbation of a candidate enhancer with a
measured expression readout into one **gold-standard table of tested
element–gene pairs**, positive and negative alike, and released the
`ENCODE-rE2G` predictions trained against it. This module exposes both.

* :func:`download_crispri_benchmark` / :func:`load_crispri_benchmark` —
  the ``EPCrisprBenchmark_ensemble_data_GRCh38.tsv`` table from the
  EngreitzLab ``ENCODE_Test_Dataset_Analysis`` repository, pinned to a
  commit: 10,412 tested pairs in K562 (FlowFISH, Gasperini 2019 and TAP-seq
  screens), 487 of them ``Regulated``, each with the perturbed element, the
  target gene's TSS, the effect size, the adjusted p-value and the power at
  four effect sizes. The negatives are *tested* pairs at the same distances
  as the positives, which is what makes this the best-matched negative set
  in the genome.

* :func:`crispri_pairs` — the benchmark as one row per element–gene pair
  with a ``label`` (1 regulated, 0 tested negative), ``distance`` in bases
  between element midpoint and TSS, and the coordinates a downstream
  interval join needs (0-based half-open, GRCh38).

Cached files live at ``~/.cache/biodb/encode_re2g/``. The benchmark is
CC BY 4.0; nothing is redistributed by this package.

Examples
--------
>>> from biodb.encode_re2g import crispri_pairs
>>> pairs = crispri_pairs()                              # doctest: +SKIP
>>> pairs.filter(pl.col("chrom") == "chr19").height      # doctest: +SKIP
2266
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from biodb._downloads import stream_to_file

logger = logging.getLogger(__name__)

BENCHMARK_REPOSITORY = "EngreitzLab/ENCODE_Test_Dataset_Analysis"
"""GitHub repository that publishes the combined CRISPRi benchmark."""

DEFAULT_REVISION = "main"
"""Git revision of the repository to read. Pin to a commit SHA for
reproducibility; the file's own SHA-256 is recorded on every load."""

BENCHMARK_PATH = "resources/combine_val_data_and_format/EPCrisprBenchmark_ensemble_data_GRCh38.tsv"
"""Path of the benchmark table inside the repository."""

CACHE_DIR = Path("~/.cache/biodb/encode_re2g").expanduser()
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _benchmark_url(revision: str) -> str:
    return f"https://raw.githubusercontent.com/{BENCHMARK_REPOSITORY}/{revision}/{BENCHMARK_PATH}"


def download_crispri_benchmark(
    *,
    revision: str = DEFAULT_REVISION,
    cache_dir: str | Path | None = None,
    force: bool = False,
    progress: bool = True,
) -> Path:
    """Download the combined CRISPRi enhancer–gene benchmark table.

    Parameters
    ----------
    revision : str
        Git revision (branch, tag or commit SHA) of
        :data:`BENCHMARK_REPOSITORY` to read (default :data:`DEFAULT_REVISION`).
    cache_dir : str or Path, optional
        Override the default :data:`CACHE_DIR`.
    force : bool
        If True, re-download even if cached.
    progress : bool
        Forwarded to :func:`biodb._downloads.stream_to_file`.

    Returns
    -------
    pathlib.Path
        Local path to the cached ``.tsv`` file.
    """
    cache = Path(cache_dir or CACHE_DIR).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    dst = cache / f"EPCrisprBenchmark_ensemble_data_GRCh38.{revision.replace('/', '_')}.tsv"
    if dst.exists() and not force:
        return dst
    return stream_to_file(_benchmark_url(revision), dst, progress=progress, desc=dst.name)


def load_crispri_benchmark(
    *,
    revision: str = DEFAULT_REVISION,
    cache_dir: str | Path | None = None,
    force: bool = False,
    progress: bool = True,
) -> pl.DataFrame:
    """Load the benchmark table as published, one row per tested pair.

    Returns
    -------
    polars.DataFrame
        The 25 published columns, typed: ``chrom``, ``chromStart``,
        ``chromEnd`` (0-based half-open element), ``chrTSS``, ``startTSS``,
        ``endTSS`` (the target TSS, null where the source did not report
        one), ``measuredGeneSymbol``, ``EffectSize``, ``Significant``,
        ``Regulated``, ``pValueAdjusted``, ``PowerAtEffectSize{10,15,20,25,50}``,
        ``ValidConnection``, ``CellType``, ``Reference``, ``dataset``,
        ``pair_uid``, ``merged_uid``, ``merged_start``, ``merged_end``.
    """
    path = download_crispri_benchmark(
        revision=revision, cache_dir=cache_dir, force=force, progress=progress
    )
    frame = pl.read_csv(
        path,
        separator="\t",
        null_values=["NA", ""],
        infer_schema_length=20_000,
        schema_overrides={
            "chrom": pl.Utf8,
            "chrTSS": pl.Utf8,
            "startTSS": pl.Int64,
            "endTSS": pl.Int64,
            "chromStart": pl.Int64,
            "chromEnd": pl.Int64,
            "EffectSize": pl.Float64,
            "pValueAdjusted": pl.Float64,
        },
    )
    for column in ("Significant", "Regulated", "ValidConnection"):
        if column in frame.columns and frame.schema[column] == pl.Utf8:
            frame = frame.with_columns(
                pl.col(column).str.to_uppercase().is_in(["TRUE", "T", "1"]).alias(column)
            )
    logger.info("Loaded %d CRISPRi pairs from %s", frame.height, path.name)
    return frame


def crispri_pairs(
    *,
    cell_type: str | None = "K562",
    valid_only: bool = True,
    revision: str = DEFAULT_REVISION,
    cache_dir: str | Path | None = None,
    force: bool = False,
    progress: bool = True,
) -> pl.DataFrame:
    """The benchmark as a labelled element–gene pair table.

    Parameters
    ----------
    cell_type : str, optional
        Keep only this ``CellType`` (default ``"K562"``; ``None`` keeps all).
    valid_only : bool
        Keep only rows the source marks ``ValidConnection`` *and* that carry a
        ``Regulated`` verdict (default True). One Gasperini2019 row
        (RP1-40E16.9, chr6) is ``ValidConnection`` with ``Regulated = NA``; a
        tested pair without a verdict is neither a positive nor a negative.

    Returns
    -------
    polars.DataFrame
        Columns ``chrom``, ``start``, ``end`` (element, 0-based half-open),
        ``gene`` (``measuredGeneSymbol``), ``tss_chrom``, ``tss`` (0-based
        position, null where unreported), ``label`` (1 = ``Regulated``,
        0 = tested negative), ``effect_size``, ``p_adjusted``, ``distance``
        (|TSS − element midpoint| in bases, null when the TSS is), ``cell_type``,
        ``reference``, ``dataset``, ``pair_uid``. Sorted by ``chrom``, ``start``.
    """
    frame = load_crispri_benchmark(
        revision=revision, cache_dir=cache_dir, force=force, progress=progress
    )
    if cell_type is not None:
        frame = frame.filter(pl.col("CellType") == cell_type)
    if valid_only and "ValidConnection" in frame.columns:
        frame = frame.filter(pl.col("ValidConnection") & pl.col("Regulated").is_not_null())
    midpoint = (pl.col("chromStart") + pl.col("chromEnd")) // 2
    return frame.select(
        pl.col("chrom"),
        pl.col("chromStart").alias("start"),
        pl.col("chromEnd").alias("end"),
        pl.col("measuredGeneSymbol").alias("gene"),
        pl.col("chrTSS").alias("tss_chrom"),
        pl.col("startTSS").alias("tss"),
        pl.col("Regulated").cast(pl.Int8).alias("label"),
        pl.col("EffectSize").alias("effect_size"),
        pl.col("pValueAdjusted").alias("p_adjusted"),
        (pl.col("startTSS") - midpoint).abs().alias("distance"),
        pl.col("CellType").alias("cell_type"),
        pl.col("Reference").alias("reference"),
        pl.col("dataset"),
        pl.col("pair_uid"),
    ).sort(["chrom", "start"])
