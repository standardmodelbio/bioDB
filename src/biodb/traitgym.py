"""TraitGym client — causal non-coding variant benchmarks with matched negatives.

`TraitGym <https://huggingface.co/datasets/songlab/TraitGym>`_ (Benegas et
al., Song lab) benchmarks variant-effect predictors on **causal regulatory
variants**: Mendelian (OMIM) and complex-trait (fine-mapped GWAS) positives,
each with negatives matched on consequence, TSS distance and (for complex
traits) allele frequency. Every split ships its variant table and a set of
precomputed feature tables (CADD, GPN-MSA, Enformer, Borzoi, Evo 2, ...)
aligned row for row, plus published AUPRC results.

* :func:`download_split` / :func:`load_variants` — a split's ``test.parquet``:
  ``chrom``, ``pos`` (1-based, GRCh38), ``ref``, ``alt``, ``label``,
  ``consequence``, ``tss_dist``, ``match_group`` and split-specific columns
  (``OMIM`` or ``trait``, ``maf``).
* :func:`load_feature` — one precomputed feature table for a split, aligned
  to the variant rows (e.g. ``"GPN-MSA_absLLR"``, ``"CADD"``).
* :func:`list_features` — feature names available for a split.

Files are read from the Hub's ``resolve`` endpoint at a pinned dataset
revision through plain HTTPS; no Hugging Face client is needed. Cached files
live at ``~/.cache/biodb/traitgym/<revision>/``.

Examples
--------
>>> from biodb.traitgym import load_variants, load_feature
>>> v = load_variants("mendelian_traits_matched_9")            # doctest: +SKIP
>>> v.filter(pl.col("label")).height                            # doctest: +SKIP
338
>>> gpn = load_feature("mendelian_traits_matched_9", "GPN-MSA_absLLR")  # doctest: +SKIP
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl
import requests

from biodb._downloads import stream_to_file

logger = logging.getLogger(__name__)

HUB_DATASET = "songlab/TraitGym"
"""The Hugging Face dataset repository."""

DEFAULT_REVISION = "1fde19555fe8"
"""Dataset revision (commit) to read. Pin a full SHA for reproducibility."""

SPLITS = (
    "mendelian_traits_all",
    "mendelian_traits_matched_9",
    "mendelian_traits_v21_matched_9",
    "mendelian_traits_v22_matched_9",
    "complex_traits_all",
    "complex_traits_matched_9",
    "complex_traits_v22_matched_9",
)
"""Splits published at :data:`DEFAULT_REVISION`. ``*_matched_9`` carry nine
matched negatives per positive; ``*_all`` the full candidate sets."""

CACHE_DIR = Path("~/.cache/biodb/traitgym").expanduser()
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_url(path: str, revision: str) -> str:
    return f"https://huggingface.co/datasets/{HUB_DATASET}/resolve/{revision}/{path}"


def _fetch(
    path: str, *, revision: str, cache_dir: str | Path | None, force: bool, progress: bool
) -> Path:
    cache = Path(cache_dir or CACHE_DIR).expanduser() / revision
    dst = cache / path
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not force:
        return dst
    return stream_to_file(_resolve_url(path, revision), dst, progress=progress, desc=path)


def download_split(
    split: str,
    *,
    revision: str = DEFAULT_REVISION,
    cache_dir: str | Path | None = None,
    force: bool = False,
    progress: bool = True,
) -> Path:
    """Download ``<split>/test.parquet``; returns the local path."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    return _fetch(
        f"{split}/test.parquet",
        revision=revision,
        cache_dir=cache_dir,
        force=force,
        progress=progress,
    )


def load_variants(
    split: str = "mendelian_traits_matched_9",
    *,
    revision: str = DEFAULT_REVISION,
    cache_dir: str | Path | None = None,
    force: bool = False,
    progress: bool = True,
) -> pl.DataFrame:
    """The split's variant table, with ``chrom`` normalised to ``chrN``.

    Returns
    -------
    polars.DataFrame
        ``chrom`` (``chr``-prefixed), ``pos`` (1-based GRCh38), ``ref``,
        ``alt``, ``label`` (bool: causal), ``consequence``, ``tss_dist``,
        ``match_group``, plus the split's own columns (``OMIM``, ``trait``,
        ``maf`` ...). Row order is the published order, which the feature
        tables share.
    """
    path = download_split(
        split, revision=revision, cache_dir=cache_dir, force=force, progress=progress
    )
    frame = pl.read_parquet(path)
    if frame.schema["chrom"] != pl.Utf8:
        frame = frame.with_columns(pl.col("chrom").cast(pl.Utf8))
    frame = frame.with_columns(
        pl.when(pl.col("chrom").str.starts_with("chr"))
        .then(pl.col("chrom"))
        .otherwise(pl.lit("chr") + pl.col("chrom"))
        .alias("chrom")
    )
    logger.info("Loaded %d TraitGym variants from %s", frame.height, split)
    return frame


def list_features(
    split: str = "mendelian_traits_matched_9",
    *,
    revision: str = DEFAULT_REVISION,
    timeout: float = 30.0,
) -> list[str]:
    """Feature names (``<split>/features/<name>.parquet``) published for a split."""
    response = requests.get(
        f"https://huggingface.co/api/datasets/{HUB_DATASET}/tree/{revision}/{split}/features",
        timeout=timeout,
    )
    response.raise_for_status()
    return sorted(
        Path(item["path"]).stem
        for item in response.json()
        if item.get("path", "").endswith(".parquet")
    )


def load_feature(
    split: str,
    feature: str,
    *,
    revision: str = DEFAULT_REVISION,
    cache_dir: str | Path | None = None,
    force: bool = False,
    progress: bool = True,
) -> pl.DataFrame:
    """One precomputed feature table, aligned row for row with :func:`load_variants`."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    path = _fetch(
        f"{split}/features/{feature}.parquet",
        revision=revision,
        cache_dir=cache_dir,
        force=force,
        progress=progress,
    )
    return pl.read_parquet(path)
