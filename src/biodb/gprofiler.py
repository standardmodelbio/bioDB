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
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

GPROFILER_GMT_URL_TEMPLATE = (
    "https://biit.cs.ut.ee/gprofiler//static/gprofiler_full_{organism}.name.gmt.zip"
)
"""URL template for the per-organism g:Profiler gene-set library zip."""

GPROFILER_REST_API = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"
"""REST endpoint for live g:Profiler functional enrichment (``gost``)."""

CACHE_DIR = Path("~/.cache/biodb/gprofiler").expanduser()
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def download_gmt(
    organism: str = "hsapiens",
    cache_dir: str | Path | None = None,
    force: bool = False,
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
    zip_path = root / f"gprofiler_full_{organism}.name.gmt.zip"
    logger.info("Downloading %s", url)
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1 << 16):
            f.write(chunk)

    with zipfile.ZipFile(zip_path) as zf:
        # The archive contains a single .gmt; extract to the cache root.
        for name in zf.namelist():
            if name.endswith(".gmt"):
                with zf.open(name) as src, open(gmt_path, "wb") as dst:
                    dst.write(src.read())
                break
        else:
            raise RuntimeError(f"No .gmt entry found inside {zip_path}")

    return gmt_path


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
        return gp.profile(
            organism=organism,
            query=query,
            sources=sources,
            user_threshold=user_threshold,
            significant=significant,
            **kwargs,
        )
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
