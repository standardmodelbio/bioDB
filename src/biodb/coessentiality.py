"""Co-essentiality network (Wainberg et al. 2021) — gene pairs whose CRISPR
fitness profiles covary across cancer cell lines.

Wainberg et al. (*Nature Genetics* 2021, doi:10.1038/s41588-021-00840-z)
correlated DepMap CRISPR gene-effect profiles across 485 cell lines with
generalised least squares after Cholesky whitening, so that the lineage
structure of the panel does not inflate every correlation, and published the
gene × gene result at ``https://mitra.stanford.edu/bassik/coessentiality/``:
``genes.txt`` (17,634 gene symbols, the matrix order), ``GLS_p.npy`` (the GLS
p-value of each pair, float64, Fortran order) and ``GLS_sign.npy`` (the sign
of the correlation). Their network keeps pairs with a positive sign at
Benjamini–Hochberg FDR < 10 % over all pairs.

* :func:`download_matrices` — the three files (2.5 GB each for the two
  matrices; memory-mapped, never read whole).
* :func:`load_genes` — the symbol order.
* :func:`coessential_pairs` — the network as a pair table at a chosen FDR (or
  raw p threshold), optionally restricted to a gene list, with the BH cutoff
  computed over every upper-triangle pair as the paper did.
* :func:`pair_p_values` — the GLS p and sign for explicit gene pairs.

Licence: the files carry no licence statement; the paper is CC BY 4.0
(Nature Genetics open access). Data are downloaded, never redistributed.

Examples
--------
>>> from biodb.coessentiality import coessential_pairs
>>> net = coessential_pairs(fdr=0.10)                                 # doctest: +SKIP
>>> net.filter(pl.col("gene_a") == "BRCA1")                            # doctest: +SKIP
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl

from biodb._downloads import stream_to_file

logger = logging.getLogger(__name__)

BASE_URL = "https://mitra.stanford.edu/bassik/coessentiality"
"""Where Wainberg et al. host the GLS matrices."""

FILES = ("genes.txt", "GLS_p.npy", "GLS_sign.npy")

CACHE_DIR = Path("~/.cache/biodb/coessentiality").expanduser()
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def download_matrices(
    *, cache_dir: str | Path | None = None, force: bool = False, progress: bool = True
) -> dict[str, Path]:
    """Fetch ``genes.txt``, ``GLS_p.npy`` and ``GLS_sign.npy``; returns their paths."""
    root = Path(cache_dir).expanduser() if cache_dir else CACHE_DIR
    out: dict[str, Path] = {}
    for name in FILES:
        dst = root / name
        if not dst.exists() or force:
            stream_to_file(f"{BASE_URL}/{name}", dst, progress=progress, desc=name)
        out[name] = dst
    return out


def load_genes(*, cache_dir: str | Path | None = None, progress: bool = True) -> list[str]:
    """The gene symbols in matrix order."""
    paths = download_matrices(cache_dir=cache_dir, progress=progress)
    return [line.strip() for line in paths["genes.txt"].read_text().splitlines() if line.strip()]


def _open(paths: dict[str, Path]) -> tuple[np.ndarray, np.ndarray]:
    p = np.load(paths["GLS_p.npy"], mmap_mode="r")
    sign = np.load(paths["GLS_sign.npy"], mmap_mode="r")
    if p.shape != sign.shape or p.ndim != 2 or p.shape[0] != p.shape[1]:
        raise ValueError("GLS_p.npy and GLS_sign.npy must be square and of one shape")
    return p, sign


def _bh_cutoff(p_values: np.ndarray, fdr: float) -> float:
    """The largest p passing Benjamini–Hochberg at `fdr` (0.0 when none does)."""
    ordered = np.sort(p_values[np.isfinite(p_values)])
    if ordered.size == 0:
        return 0.0
    ranks = np.arange(1, ordered.size + 1, dtype=np.float64)
    passing = np.flatnonzero(ordered <= fdr * ranks / ordered.size)
    return float(ordered[passing[-1]]) if passing.size else 0.0


def coessential_pairs(
    *,
    fdr: float | None = None,
    p_threshold: float | None = None,
    positive_only: bool = True,
    genes: Sequence[str] | None = None,
    cache_dir: str | Path | None = None,
    progress: bool = True,
) -> pl.DataFrame:
    """Gene pairs in the co-essential network.

    One of ``fdr`` (Benjamini–Hochberg over every upper-triangle pair of the
    whole matrix; the paper's 10 % when neither is given) and ``p_threshold``
    (a raw GLS p cutoff) applies, never both. ``positive_only`` keeps positively correlated
    pairs, as the published network does. ``genes`` restricts the *output*
    to pairs with both genes in the list; the FDR cutoff is still computed
    over the whole matrix so a subset does not loosen it.

    Returns
    -------
    polars.DataFrame
        ``gene_a``, ``gene_b`` (``gene_a < gene_b``), ``p``, ``sign``.
    """
    if fdr is not None and p_threshold is not None:
        raise ValueError("pass either fdr or p_threshold, not both")
    if fdr is None and p_threshold is None:
        fdr = 0.10
    paths = download_matrices(cache_dir=cache_dir, progress=progress)
    names = load_genes(cache_dir=cache_dir, progress=progress)
    p, sign = _open(paths)
    n = p.shape[0]
    if len(names) != n:
        raise ValueError(f"genes.txt has {len(names)} names for a {n} x {n} matrix")
    # Row by row over the upper triangle: the matrices are Fortran-ordered, so
    # a row is strided, but 17,634 strided reads of a memory map is seconds.
    if p_threshold is None:
        assert fdr is not None
        collected: list[np.ndarray] = []
        for i in range(n - 1):
            collected.append(np.asarray(p[i, i + 1 :], dtype=np.float64))
        cutoff = _bh_cutoff(np.concatenate(collected), fdr)
        logger.info("BH cutoff at FDR %.3f: p <= %.3g", fdr, cutoff)
        del collected
    else:
        cutoff = float(p_threshold)
    wanted = None if genes is None else {g.upper() for g in genes}
    keep_row = (
        np.ones(n, dtype=bool)
        if wanted is None
        else np.asarray([name.upper() in wanted for name in names])
    )
    left: list[str] = []
    right: list[str] = []
    pvals: list[float] = []
    signs: list[float] = []
    for i in range(n - 1):
        if not keep_row[i]:
            continue
        row_p = np.asarray(p[i, i + 1 :], dtype=np.float64)
        row_s = np.asarray(sign[i, i + 1 :], dtype=np.float64)
        mask = row_p <= cutoff
        if positive_only:
            mask &= row_s > 0
        if wanted is not None:
            mask &= keep_row[i + 1 :]
        for j in np.flatnonzero(mask):
            a, b = names[i], names[i + 1 + int(j)]
            if a > b:
                a, b = b, a
            left.append(a)
            right.append(b)
            pvals.append(float(row_p[j]))
            signs.append(float(row_s[j]))
    return pl.DataFrame(
        {"gene_a": left, "gene_b": right, "p": pvals, "sign": signs},
        schema={"gene_a": pl.Utf8, "gene_b": pl.Utf8, "p": pl.Float64, "sign": pl.Float64},
    )


def pair_p_values(
    pairs: Sequence[tuple[str, str]],
    *,
    cache_dir: str | Path | None = None,
    progress: bool = True,
) -> pl.DataFrame:
    """GLS p and sign for explicit gene pairs; genes not in the matrix give null."""
    paths = download_matrices(cache_dir=cache_dir, progress=progress)
    names = load_genes(cache_dir=cache_dir, progress=progress)
    index = {name.upper(): i for i, name in enumerate(names)}
    p, sign = _open(paths)
    out_p: list[float | None] = []
    out_s: list[float | None] = []
    for a, b in pairs:
        i, j = index.get(a.upper()), index.get(b.upper())
        if i is None or j is None or i == j:
            out_p.append(None)
            out_s.append(None)
        else:
            out_p.append(float(p[i, j]))
            out_s.append(float(sign[i, j]))
    return pl.DataFrame(
        {
            "gene_a": [a for a, _ in pairs],
            "gene_b": [b for _, b in pairs],
            "p": out_p,
            "sign": out_s,
        },
        schema={"gene_a": pl.Utf8, "gene_b": pl.Utf8, "p": pl.Float64, "sign": pl.Float64},
    )


__all__ = [
    "BASE_URL",
    "CACHE_DIR",
    "FILES",
    "coessential_pairs",
    "download_matrices",
    "load_genes",
    "pair_p_values",
]
