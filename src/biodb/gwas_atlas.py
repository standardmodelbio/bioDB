"""GWAS Atlas client — Watanabe et al. gene-level meta-analysis catalog.

[GWAS Atlas](https://atlas.ctglab.nl/) (Watanabe et al. Nat Genet 2019)
is a Vrije Universiteit Amsterdam meta-resource that publishes
per-study gene-level MAGMA p-values across ~4,000 GWAS summary
statistics in a single (gene × study) matrix.

The Atlas does *not* expose direct-URL downloads. Each bulk artifact
goes through a Laravel CSRF-protected form on the homepage that posts
to ``/home/release`` with a per-session ``_token`` and an
``XSRF-TOKEN`` / ``atlas_session`` cookie pair. :func:`_session` performs
that handshake transparently so callers see plain
``download_*`` / ``load_*`` APIs.

Files this module wraps:

* ``gwasATLAS_v20191115.txt.gz`` — per-study metadata (PMID, trait,
  domain, sample size, …).
* ``gwasATLAS_v20191115_magma_P.txt.gz`` — the (gene × study) MAGMA
  -log10 p-value matrix.

Any other file listed on the release page (``_columns.txt.gz``,
``_GC.txt.gz``, ``_magma_sets_P.txt.gz``, ``_riskloci.txt.gz``, …) can be
fetched with :func:`download_file`.

Both are cached under ``~/.cache/biodb/gwas_atlas/``.

Examples
--------
>>> from biodb.gwas_atlas import load_metadata, load_magma_p
>>> meta = load_metadata()                                  # doctest: +SKIP
>>> magma = load_magma_p()                                  # doctest: +SKIP
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd
import requests

from biodb._downloads import stream_to_file

logger = logging.getLogger(__name__)

GWAS_ATLAS_BASE_URL = "https://atlas.ctglab.nl"
"""Public site root. The form-POST endpoint lives at ``/home/release``."""

GWAS_ATLAS_RELEASE_ENDPOINT = f"{GWAS_ATLAS_BASE_URL}/home/release"

DEFAULT_VERSION = "20191115"
"""Default GWAS Atlas snapshot date. Bump after testing a new release."""

CACHE_DIR = Path("~/.cache/biodb/gwas_atlas").expanduser()
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_USER_AGENT = "biodb/0.1 (+https://github.com/bschilder/bioDB)"
_TOKEN_RE = re.compile(r'name="_token"\s+value="([^"]+)"')


def _session(timeout: float = 30) -> tuple[requests.Session, str]:
    """Open a session, scrape a Laravel CSRF ``_token``, return ``(session, token)``.

    Cookies (``XSRF-TOKEN`` + ``atlas_session``) are stored on the session.
    The token is form-bound — pass it as ``data["_token"]`` on the POST to
    :data:`GWAS_ATLAS_RELEASE_ENDPOINT`.
    """
    session = requests.Session()
    session.headers["User-Agent"] = _USER_AGENT
    response = session.get(f"{GWAS_ATLAS_BASE_URL}/", timeout=timeout)
    response.raise_for_status()
    match = _TOKEN_RE.search(response.text)
    if not match:
        raise RuntimeError(
            "Could not locate a Laravel _token field on the GWAS Atlas homepage — "
            "the site layout may have changed; please file an issue at "
            "https://github.com/bschilder/bioDB/issues."
        )
    return session, match.group(1)


def _download(
    filename: str,
    dst: Path,
    timeout: float = 300,
    *,
    progress: bool = True,
) -> Path:
    """Stream the GWAS Atlas file named ``filename`` to ``dst``; return ``dst``.

    Goes through the CSRF-form flow at :data:`GWAS_ATLAS_RELEASE_ENDPOINT`.
    """
    session, token = _session(timeout=timeout)
    logger.info("Downloading %s via %s", filename, GWAS_ATLAS_RELEASE_ENDPOINT)
    return stream_to_file(
        GWAS_ATLAS_RELEASE_ENDPOINT,
        dst,
        timeout=int(timeout),
        progress=progress,
        desc=filename,
        session=session,
        post_data={"_token": token, "file": filename},
    )


def download_file(
    filename: str,
    cache_dir: str | Path | None = None,
    force: bool = False,
    *,
    progress: bool = True,
) -> Path:
    """Download an arbitrary GWAS Atlas release file by filename.

    Useful for the auxiliary files not wrapped by the dedicated helpers
    (``gwasATLAS_v20191115.readme``, ``_columns.txt.gz``, ``_GC.txt.gz``,
    ``_magma_sets_P.txt.gz``, ``_riskloci.txt.gz``, …).

    Parameters
    ----------
    filename : str
        Exact filename shown on https://atlas.ctglab.nl/ (e.g.
        ``"gwasATLAS_v20191115_riskloci.txt.gz"``).
    cache_dir : str or Path, optional
        Cache root. Defaults to :data:`CACHE_DIR`.
    force : bool, default False
        Re-download even if cached.
    progress : bool, default True
        Show a tqdm download bar.
    """
    root = Path(cache_dir).expanduser() if cache_dir else CACHE_DIR
    dst = root / filename
    if dst.exists() and not force:
        return dst
    return _download(filename, dst, progress=progress)


def download_metadata(
    version: str = DEFAULT_VERSION,
    cache_dir: str | Path | None = None,
    force: bool = False,
    *,
    progress: bool = True,
) -> Path:
    """Download the per-study metadata TSV (gzip) and return its local path."""
    return download_file(
        f"gwasATLAS_v{version}.txt.gz",
        cache_dir=cache_dir,
        force=force,
        progress=progress,
    )


def download_magma_p(
    version: str = DEFAULT_VERSION,
    cache_dir: str | Path | None = None,
    force: bool = False,
    *,
    progress: bool = True,
) -> Path:
    """Download the (gene × study) MAGMA P-value matrix (gzip)."""
    return download_file(
        f"gwasATLAS_v{version}_magma_P.txt.gz",
        cache_dir=cache_dir,
        force=force,
        progress=progress,
    )


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


# ─── Per-trait targeted-query API ───────────────────────────────────────────
# GWAS Atlas doesn't publish a structured REST API for per-trait
# lookups, so we mimic one by caching the per-study metadata TSV in
# memory and indexing on the lookup keys callers most often want
# (numeric ``id``, ``Trait`` name, or ``PMID``).

_METADATA_CACHE: pd.DataFrame | None = None


def _get_metadata_cached(
    version: str = DEFAULT_VERSION,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Lazy-load the per-study metadata TSV and cache it in module memory."""
    global _METADATA_CACHE
    if _METADATA_CACHE is None or _METADATA_CACHE.empty:
        _METADATA_CACHE = load_metadata(version=version, cache_dir=cache_dir)
    return _METADATA_CACHE


def query_trait(
    trait: str | int,
    *,
    column: str | None = None,
    version: str = DEFAULT_VERSION,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Look up GWAS Atlas study metadata by trait name, PMID, or numeric id.

    Parameters
    ----------
    trait : str or int
        The lookup value:

        * an ``int`` (or all-digit ``str``) is treated as ``id`` first,
          falling back to ``PMID`` if no ``id`` matches.
        * any other string is treated as a substring filter on ``Trait``.
    column : str, optional
        Force the lookup column. One of ``"id"``, ``"PMID"``, ``"Trait"``,
        or any other column present in the metadata frame.
    version : str
    cache_dir : str or Path, optional

    Returns
    -------
    pandas.DataFrame
        Zero or more matching rows from the metadata TSV (e.g. PMID,
        Trait, Year, N, Domain, Population, Chip, …).
    """
    meta = _get_metadata_cached(version=version, cache_dir=cache_dir)

    if column is not None:
        if column not in meta.columns:
            raise KeyError(
                f"Column {column!r} not in metadata; available: {list(meta.columns)[:8]}…"
            )
        col_data = meta[column].astype(str)
        return meta[col_data.str.contains(str(trait), na=False, case=False)]

    str_val = str(trait).strip()
    if str_val.isdigit():
        as_int = int(str_val)
        if "id" in meta.columns:
            id_col = meta["id"]
            hits = meta[id_col == (as_int if id_col.dtype.kind in "iuf" else str_val)]
            if not hits.empty:
                return hits
        if "PMID" in meta.columns:
            pmid_col = meta["PMID"]
            return meta[pmid_col == (as_int if pmid_col.dtype.kind in "iuf" else str_val)]
        return meta.iloc[0:0]

    if "Trait" in meta.columns:
        return meta[meta["Trait"].astype(str).str.contains(str_val, na=False, case=False)]
    return meta.iloc[0:0]


def list_traits(
    *,
    version: str = DEFAULT_VERSION,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Return the full per-study metadata table (cached after first call)."""
    return _get_metadata_cached(version=version, cache_dir=cache_dir)


__all__ = [
    "CACHE_DIR",
    "DEFAULT_VERSION",
    "GWAS_ATLAS_BASE_URL",
    "GWAS_ATLAS_RELEASE_ENDPOINT",
    "download_file",
    "download_magma_p",
    "download_metadata",
    "list_traits",
    "load_magma_p",
    "load_metadata",
    "melt_magma_p",
    "query_trait",
]
