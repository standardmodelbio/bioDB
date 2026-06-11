"""STRING client — physical PPI edges with continuous combined-score weights.

[STRING](https://string-db.org/) (Search Tool for the Retrieval of
Interacting Genes/Proteins) is the EMBL-EBI / SIB protein-interaction
knowledgebase. This module exposes the bulk downloads:

* :func:`download_physical_links` / :func:`load_physical_links` —
  the ``protein.physical.links`` file (one row per protein-pair with
  a ``combined_score`` 150-999 integrating experimental, database,
  text-mining, co-expression, neighborhood, fusion, and co-occurrence
  evidence). The *physical* sub-network is restricted to edges with
  direct binding evidence — closer to what's usually called "PPI"
  than the full functional-coupling network in ``protein.links``.

* :func:`download_protein_info` / :func:`load_protein_info` —
  the STRING protein info table (STRING ID → ``preferred_name``,
  typically a HGNC gene symbol for human).

* :func:`physical_ppi_edges` — convenience joiner that returns a
  pair-level edge list keyed by ``preferred_name`` (HGNC symbol for
  human, equivalent for other organisms), with the symmetric STRING
  file deduplicated to one row per unordered pair and the integer
  ``combined_score`` rescaled to ``score`` ∈ (0.15, 0.999].

Cached files live at ``~/.cache/biodb/string/``.

Examples
--------
>>> from biodb.string import physical_ppi_edges
>>> edges = physical_ppi_edges()                       # doctest: +SKIP
>>> edges.head()                                       # doctest: +SKIP
"""

from __future__ import annotations

import gzip
import logging
from pathlib import Path

import pandas as pd

from biodb._downloads import stream_to_file

logger = logging.getLogger(__name__)

STRING_DOWNLOAD_BASE = "https://stringdb-downloads.org/download"
"""STRING bulk-download root."""

DEFAULT_VERSION = "12.0"
"""Default STRING release version. Bump when STRING ships a major update."""

DEFAULT_ORGANISM = "9606"
"""NCBI taxon ID. ``9606`` = Homo sapiens; ``10090`` = mouse, etc."""

CACHE_DIR = Path("~/.cache/biodb/string").expanduser()
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _physical_links_url(version: str, organism: str) -> str:
    """Build the URL for the ``protein.physical.links`` file."""
    return (
        f"{STRING_DOWNLOAD_BASE}/protein.physical.links.v{version}/"
        f"{organism}.protein.physical.links.v{version}.txt.gz"
    )


def _protein_info_url(version: str, organism: str) -> str:
    """Build the URL for the ``protein.info`` file."""
    return (
        f"{STRING_DOWNLOAD_BASE}/protein.info.v{version}/{organism}.protein.info.v{version}.txt.gz"
    )


def download_physical_links(
    *,
    version: str = DEFAULT_VERSION,
    organism: str = DEFAULT_ORGANISM,
    cache_dir: str | Path | None = None,
    force: bool = False,
    progress: bool = True,
) -> Path:
    """Download ``protein.physical.links`` for ``organism`` / ``version``.

    Parameters
    ----------
    version : str
        STRING release version (default :data:`DEFAULT_VERSION`).
    organism : str
        NCBI taxon ID (default :data:`DEFAULT_ORGANISM` = ``"9606"``).
    cache_dir : str or Path, optional
        Override the default :data:`CACHE_DIR`.
    force : bool
        If True, re-download even if cached.
    progress : bool
        Forwarded to :func:`biodb._downloads.stream_to_file`.

    Returns
    -------
    pathlib.Path
        Local path to the cached ``.txt.gz`` file.

    Examples
    --------
    >>> download_physical_links()                      # doctest: +SKIP
    """
    cache = Path(cache_dir or CACHE_DIR).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    dst = cache / f"{organism}.protein.physical.links.v{version}.txt.gz"
    if dst.exists() and not force:
        return dst
    return stream_to_file(
        _physical_links_url(version, organism),
        dst,
        progress=progress,
        desc=dst.name,
    )


def download_protein_info(
    *,
    version: str = DEFAULT_VERSION,
    organism: str = DEFAULT_ORGANISM,
    cache_dir: str | Path | None = None,
    force: bool = False,
    progress: bool = True,
) -> Path:
    """Download ``protein.info`` for ``organism`` / ``version``.

    Parameters
    ----------
    version : str
    organism : str
    cache_dir : str or Path, optional
    force : bool
    progress : bool

    Returns
    -------
    pathlib.Path

    Examples
    --------
    >>> download_protein_info()                        # doctest: +SKIP
    """
    cache = Path(cache_dir or CACHE_DIR).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    dst = cache / f"{organism}.protein.info.v{version}.txt.gz"
    if dst.exists() and not force:
        return dst
    return stream_to_file(
        _protein_info_url(version, organism),
        dst,
        progress=progress,
        desc=dst.name,
    )


def load_protein_info(
    *,
    version: str = DEFAULT_VERSION,
    organism: str = DEFAULT_ORGANISM,
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Load the STRING protein info table.

    Parameters
    ----------
    version : str
    organism : str
    cache_dir : str or Path, optional
    force : bool

    Returns
    -------
    pd.DataFrame
        Columns: ``#string_protein_id``, ``preferred_name``,
        ``protein_size``, ``annotation``.

    Examples
    --------
    >>> info = load_protein_info()                     # doctest: +SKIP
    >>> "preferred_name" in info.columns               # doctest: +SKIP
    True
    """
    path = download_protein_info(
        version=version, organism=organism, cache_dir=cache_dir, force=force
    )
    with gzip.open(path, "rt") as fh:
        return pd.read_csv(fh, sep="\t")


def load_physical_links(
    *,
    version: str = DEFAULT_VERSION,
    organism: str = DEFAULT_ORGANISM,
    cache_dir: str | Path | None = None,
    force: bool = False,
    min_combined_score: int = 0,
) -> pd.DataFrame:
    """Load the raw STRING physical links table.

    Parameters
    ----------
    version : str
    organism : str
    cache_dir : str or Path, optional
    force : bool
    min_combined_score : int
        If > 0, drop edges below this raw STRING score before returning.

    Returns
    -------
    pd.DataFrame
        Columns: ``protein1``, ``protein2``, ``combined_score``.

    Examples
    --------
    >>> links = load_physical_links(min_combined_score=700)  # doctest: +SKIP
    """
    path = download_physical_links(
        version=version, organism=organism, cache_dir=cache_dir, force=force
    )
    with gzip.open(path, "rt") as fh:
        df = pd.read_csv(fh, sep=" ")
    if min_combined_score > 0:
        df = df[df["combined_score"] >= min_combined_score].reset_index(drop=True)
    return df


def physical_ppi_edges(
    *,
    version: str = DEFAULT_VERSION,
    organism: str = DEFAULT_ORGANISM,
    min_combined_score: int = 150,
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Return a deduplicated STRING physical PPI edge list keyed by gene symbol.

    Joins :func:`load_physical_links` with :func:`load_protein_info` to
    map STRING IDs to ``preferred_name`` (HGNC for human, equivalent for
    other organisms), normalises pair order so ``gene_a <= gene_b``,
    deduplicates the symmetric STRING entries (keeping the max
    ``combined_score`` per unordered pair), and exposes a normalised
    ``score`` ∈ (0.15, 0.999] alongside the raw integer.

    Parameters
    ----------
    version : str
    organism : str
    min_combined_score : int, default 150
        Drop edges below this raw STRING score. The STRING default
        lowest-confidence cutoff is 150 — anything lower includes
        edges supported by a single low-confidence text-mining hit.
    cache_dir : str or Path, optional
    force : bool

    Returns
    -------
    pd.DataFrame
        Columns: ``gene_a``, ``gene_b``, ``combined_score``, ``score``.
        Sorted by ``combined_score`` descending. Self-edges removed.

    Examples
    --------
    >>> edges = physical_ppi_edges(min_combined_score=700)  # doctest: +SKIP
    >>> "score" in edges.columns                            # doctest: +SKIP
    True
    """
    info = load_protein_info(version=version, organism=organism, cache_dir=cache_dir, force=force)
    id_col = next(c for c in info.columns if "string_protein_id" in c.lower())
    info_map = dict(
        zip(
            info[id_col].astype(str),
            info["preferred_name"].astype(str).str.upper(),
            strict=True,
        )
    )

    links = load_physical_links(
        version=version,
        organism=organism,
        cache_dir=cache_dir,
        force=force,
        min_combined_score=min_combined_score,
    )
    links["gene_a_raw"] = links["protein1"].astype(str).map(info_map)
    links["gene_b_raw"] = links["protein2"].astype(str).map(info_map)
    mask = (
        links["gene_a_raw"].notna()
        & links["gene_b_raw"].notna()
        & (links["gene_a_raw"] != links["gene_b_raw"])
    )
    links = links.loc[mask].copy()

    # Normalize so gene_a <= gene_b lex.
    a = links["gene_a_raw"].to_numpy()
    b = links["gene_b_raw"].to_numpy()
    swap_mask = a > b
    a_norm = a.copy()
    b_norm = b.copy()
    a_norm[swap_mask] = b[swap_mask]
    b_norm[swap_mask] = a[swap_mask]
    links["gene_a"] = a_norm
    links["gene_b"] = b_norm

    out = links.groupby(["gene_a", "gene_b"], as_index=False)["combined_score"].max()
    out["score"] = out["combined_score"] / 1000.0
    out = out.sort_values(
        ["combined_score", "gene_a", "gene_b"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    return out[["gene_a", "gene_b", "combined_score", "score"]]
