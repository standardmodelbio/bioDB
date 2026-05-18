"""gProfiler client — gene-set library downloader + REST functional-enrichment.

[gProfiler](https://biit.cs.ut.ee/gprofiler/) is a University of Tartu
gene-set / functional-enrichment service. This module surfaces:

* **Bulk mode** — :func:`download_gmt`, :func:`load_gmt` for the
  published ``gprofiler_full_<organism>.name.gmt`` files (the same
  zip used by the AoU PhenomicLandscape pipeline).
* **API mode (placeholder)** — :func:`gost` is a thin wrapper around
  the gProfiler REST endpoint for live g:Profiler functional enrichment
  (``gost``). Re-uses the ``gprofiler-official`` Python package when
  installed, falls back to a plain ``requests.post``.

Cached files live at ``~/.cache/biodb/gprofiler/``.

Examples
--------
>>> from biodb.gprofiler import download_gmt, load_gmt
>>> path = download_gmt(organism="hsapiens")           # doctest: +SKIP
>>> df = load_gmt(organism="hsapiens")                 # doctest: +SKIP
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from biodb._downloads import stream_to_file

logger = logging.getLogger(__name__)

GPROFILER_GMT_URL_TEMPLATE = (
    "https://biit.cs.ut.ee/gprofiler/static/gprofiler_full_{organism}.name.gmt"
)
"""URL template for the combined per-organism g:Profiler gene-set library.

g:Profiler migrated to a Vue SPA in 2026 and renamed the bulk-download
path: the file is now a plain ``.gmt`` (no longer wrapped in a zip) and
the old double-slash path returns 404. The combined-file pattern is the
one the website's *Download g:Profiler data as a combined GMT file* link
points at. ~41 MB for hsapiens, smaller for other organisms.
"""

GPROFILER_REST_API = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"
"""REST endpoint for live g:Profiler functional enrichment (``gost``)."""

CACHE_DIR = Path("~/.cache/biodb/gprofiler").expanduser()
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def download_gmt(
    organism: str = "hsapiens",
    cache_dir: str | Path | None = None,
    force: bool = False,
    progress: bool = True,
) -> Path:
    """Download ``gprofiler_full_<organism>.name.gmt`` (unzipped) and return path.

    Parameters
    ----------
    organism : str, default ``"hsapiens"``
        gProfiler organism code (``"hsapiens"`` / ``"mmusculus"`` / …).
    cache_dir : str or Path, optional
        Cache root. Defaults to :data:`CACHE_DIR`.
    force : bool, default False
        Re-download even if cached.
    progress : bool, default True
        Show a tqdm download bar (~41 MB for hsapiens).

    Returns
    -------
    pathlib.Path
        Absolute path to the unzipped ``.gmt`` file.
    """
    root = Path(cache_dir).expanduser() if cache_dir else CACHE_DIR
    root.mkdir(parents=True, exist_ok=True)
    gmt_path = root / f"gprofiler_full_{organism}.name.gmt"
    if gmt_path.exists() and not force:
        return gmt_path

    url = GPROFILER_GMT_URL_TEMPLATE.format(organism=organism)
    logger.info("Downloading %s", url)
    return stream_to_file(url, gmt_path, timeout=120, progress=progress)


def load_gmt(
    organism: str = "hsapiens",
    cache_dir: str | Path | None = None,
    return_format: str = "pandas",
    force: bool = False,
) -> pd.DataFrame | dict[tuple[str, str], list[str]]:
    """Download + parse the gProfiler GMT into a long DataFrame (or dict).

    Parameters
    ----------
    organism : str, default ``"hsapiens"``
    cache_dir : str or Path, optional
    return_format : {"pandas", "dict"}
        See :func:`biodb.utils.read_gmt`.
    force : bool, default False
    """
    from biodb.utils import read_gmt

    path = download_gmt(organism=organism, cache_dir=cache_dir, force=force)
    return read_gmt(path, return_format=return_format)


def gost(
    query: list[str] | dict[str, list[str]],
    organism: str = "hsapiens",
    sources: list[str] | None = None,
    user_threshold: float = 0.05,
    significant: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """Run a live g:Profiler ``gost`` functional enrichment.

    Prefers the official ``gprofiler-official`` Python package when
    available (richer parameter handling, DataFrame return), falls back
    to a plain ``requests.post`` against :data:`GPROFILER_REST_API`.

    Parameters
    ----------
    query : list[str] or dict[str, list[str]]
        Gene IDs (single query) or ``{label: [ids]}`` (named multi-query).
    organism : str, default ``"hsapiens"``
    sources : list[str], optional
        Annotation source filter (``["GO:BP", "KEGG", "REAC", …]``).
    user_threshold : float, default 0.05
        Significance threshold.
    significant : bool, default True
        Only return significant hits.
    **kwargs
        Extra arguments forwarded to ``GProfiler.profile()`` (when
        ``gprofiler-official`` is installed) or merged into the REST
        request body otherwise.
    """
    try:
        from gprofiler import GProfiler

        gp = GProfiler(return_dataframe=True)
        # gprofiler-official's ``profile`` filters by ``user_threshold``
        # internally; it doesn't expose a separate ``significant`` flag.
        # We mirror the REST endpoint's semantics by post-filtering when
        # the caller asked for ``significant=True``.
        df = gp.profile(
            organism=organism,
            query=query,
            sources=sources,
            user_threshold=user_threshold,
            **kwargs,
        )
        if significant and "p_value" in df.columns:
            df = df[df["p_value"] < user_threshold].reset_index(drop=True)
        return df
    except ImportError:
        pass

    payload: dict[str, Any] = {
        "organism": organism,
        "query": query,
        "user_threshold": user_threshold,
        "significant": significant,
    }
    if sources is not None:
        payload["sources"] = sources
    payload.update(kwargs)
    response = requests.post(GPROFILER_REST_API, json=payload, timeout=60)
    response.raise_for_status()
    result = response.json().get("result", [])
    return pd.DataFrame(result)


__all__ = [
    "CACHE_DIR",
    "GPROFILER_GMT_URL_TEMPLATE",
    "GPROFILER_REST_API",
    "download_gmt",
    "gost",
    "load_gmt",
]
