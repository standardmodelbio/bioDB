"""GWAS Atlas client — Watanabe et al. gene-level meta-analysis catalog.

[GWAS Atlas](https://atlas.ctglab.nl/) (Watanabe et al. Nat Genet 2019)
is a Vrije Universiteit Amsterdam meta-resource that publishes
per-study gene-level MAGMA p-values across ~4,000 GWAS summary
statistics in a single (gene × study) matrix.

The Atlas distributes its bulk artifacts at
``https://atlas.ctglab.nl/ukb2_sumstats/`` — the two files
this module wraps are:

* ``gwasATLAS_v20191115.txt.gz`` — per-study metadata (PMID, trait,
  domain, sample size, …).
* ``gwasATLAS_v20191115_magma_P.txt.gz`` — the (gene × study) MAGMA
  -log10 p-value matrix.

Both are cached under ``~/.cache/biodb/gwas_atlas/``.

Examples
--------
>>> from biodb.gwas_atlas import load_metadata, load_magma_p
>>> meta = load_metadata()                                  # doctest: +SKIP
>>> magma = load_magma_p()                                  # doctest: +SKIP
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

GWAS_ATLAS_BASE_URL = "https://atlas.ctglab.nl/ukb2_sumstats"
"""Public bulk-download root."""

DEFAULT_VERSION = "20191115"
"""Default GWAS Atlas snapshot date. Bump after testing a new release."""

CACHE_DIR = Path("~/.cache/biodb/gwas_atlas").expanduser()
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _metadata_url(version: str) -> str:
    return f"{GWAS_ATLAS_BASE_URL}/gwasATLAS_v{version}.txt.gz"


def _magma_p_url(version: str) -> str:
    return f"{GWAS_ATLAS_BASE_URL}/gwasATLAS_v{version}_magma_P.txt.gz"


def _download(url: str, dst: Path) -> Path:
    """Stream ``url`` to ``dst`` (creating parents); return ``dst``."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s", url)
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()
    with open(dst, "wb") as f:
        for chunk in response.iter_content(chunk_size=1 << 16):
            f.write(chunk)
    return dst


def download_metadata(
    version: str = DEFAULT_VERSION,
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Download the per-study metadata TSV (gzip) and return its local path."""
    root = Path(cache_dir).expanduser() if cache_dir else CACHE_DIR
    dst = root / f"gwasATLAS_v{version}.txt.gz"
    if dst.exists() and not force:
        return dst
    return _download(_metadata_url(version), dst)


def download_magma_p(
    version: str = DEFAULT_VERSION,
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Download the (gene × study) MAGMA P-value matrix (gzip)."""
    root = Path(cache_dir).expanduser() if cache_dir else CACHE_DIR
    dst = root / f"gwasATLAS_v{version}_magma_P.txt.gz"
    if dst.exists() and not force:
        return dst
    return _download(_magma_p_url(version), dst)


def load_metadata(
    version: str = DEFAULT_VERSION,
    cache_dir: str | Path | None = None,
    force: bool = False,
    **read_kwargs,
) -> pd.DataFrame:
    """Read the per-study metadata as a DataFrame.

    Extra ``**read_kwargs`` are forwarded to :func:`pandas.read_csv`.
    """
    path = download_metadata(version=version, cache_dir=cache_dir, force=force)
    return pd.read_csv(path, sep="\t", **read_kwargs)


def load_magma_p(
    version: str = DEFAULT_VERSION,
    cache_dir: str | Path | None = None,
    force: bool = False,
    index_col: int | str = 0,
    **read_kwargs,
) -> pd.DataFrame:
    """Read the (gene × study) MAGMA -log10 p-value matrix.

    Parameters
    ----------
    index_col : int or str, default 0
        Which column holds the gene IDs. The Atlas uses Ensembl gene IDs
        as the index column.
    **read_kwargs
        Forwarded to :func:`pandas.read_csv`.
    """
    path = download_magma_p(version=version, cache_dir=cache_dir, force=force)
    return pd.read_csv(path, sep="\t", index_col=index_col, **read_kwargs)


def melt_magma_p(magma_p: pd.DataFrame, p_col: str = "score") -> pd.DataFrame:
    """Pivot the (gene × study) wide matrix to a long ``(sourceId, targetId, score)`` frame.

    The result schema mirrors what :func:`biodb.transform.create_gene_association_matrix`
    expects, so callers can plug it straight into the matrix builder.

    Parameters
    ----------
    magma_p : pd.DataFrame
        Wide ``(gene × study)`` frame, gene IDs in the index.
    p_col : str, default ``"score"``
        Output column name for the cell value.
    """
    long = magma_p.reset_index().melt(
        id_vars=magma_p.index.name or magma_p.columns[0],
        var_name="sourceId",
        value_name=p_col,
    )
    long = long.rename(columns={magma_p.index.name or magma_p.columns[0]: "targetId"})
    long = long.dropna(subset=[p_col])
    return long


__all__ = [
    "CACHE_DIR",
    "DEFAULT_VERSION",
    "GWAS_ATLAS_BASE_URL",
    "download_magma_p",
    "download_metadata",
    "load_magma_p",
    "load_metadata",
    "melt_magma_p",
]
