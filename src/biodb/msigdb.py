"""MSigDB client — Broad Institute Molecular Signatures Database.

MSigDB ships its gene-set libraries as ``.gmt`` files at
``https://data.broadinstitute.org/gsea-msigdb/msigdb/release/<version>/``.
Each release contains multiple collections (``H`` hallmark,
``C1``–``C8``) and identifier variants (``.symbols.gmt``,
``.entrez.gmt``).

This module focuses on the **bulk-download path** because that's how
real pipelines consume MSigDB. The targeted REST mode is a 🚧 stub.

Examples
--------
>>> from biodb.msigdb import download_gmt, load_gmt
>>> path = download_gmt(collection="msigdb", version="2025.1.Hs")     # doctest: +SKIP
>>> df = load_gmt(collection="msigdb", version="2025.1.Hs")           # doctest: +SKIP
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

from biodb._downloads import stream_to_file

logger = logging.getLogger(__name__)

MSIGDB_BASE_URL = "https://data.broadinstitute.org/gsea-msigdb/msigdb/release"
"""MSigDB FTP release root."""

DEFAULT_VERSION = "2025.1.Hs"
"""Default MSigDB release (Human). Bump after testing against new release."""

KNOWN_COLLECTIONS = (
    "msigdb",  # combined (all collections)
    "h.all",  # H — hallmark gene sets
    "c1.all",  # C1 — positional gene sets
    "c2.all",  # C2 — curated gene sets
    "c2.cp",  # C2 — canonical pathways subset
    "c3.all",  # C3 — regulatory target gene sets
    "c4.all",  # C4 — computational gene sets
    "c5.all",  # C5 — ontology gene sets
    "c6.all",  # C6 — oncogenic signature gene sets
    "c7.all",  # C7 — immunologic signature gene sets
    "c8.all",  # C8 — cell-type signature gene sets
)
"""Common MSigDB collection slugs. Pass any of these as ``collection=`` to
:func:`download_gmt` / :func:`load_gmt`."""

CACHE_DIR = Path("~/.cache/biodb/msigdb").expanduser()
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _gmt_filename(collection: str, version: str, id_type: str) -> str:
    """Build the canonical GMT filename, e.g. ``msigdb.v2025.1.Hs.symbols.gmt``."""
    return f"{collection}.v{version}.{id_type}.gmt"


def _gmt_url(collection: str, version: str, id_type: str) -> str:
    return f"{MSIGDB_BASE_URL}/{version}/{_gmt_filename(collection, version, id_type)}"


def download_gmt(
    collection: str = "msigdb",
    version: str = DEFAULT_VERSION,
    id_type: str = "symbols",
    cache_dir: str | Path | None = None,
    force: bool = False,
    progress: bool = True,
) -> Path:
    """Download a single MSigDB GMT release file and return its local path.

    Parameters
    ----------
    collection : str, default ``"msigdb"``
        MSigDB collection slug (see :data:`KNOWN_COLLECTIONS`).
    version : str, default :data:`DEFAULT_VERSION`
        Release tag. Human releases look like ``"2025.1.Hs"``; mouse
        releases use the ``.Mm`` suffix.
    id_type : {"symbols", "entrez"}, default ``"symbols"``
        Identifier namespace baked into the GMT.
    cache_dir : str or Path, optional
        Cache root. Defaults to :data:`CACHE_DIR`.
    force : bool, default False
        Re-download even if cached.
    progress : bool, default True
        Show a tqdm download bar.
    """
    root = Path(cache_dir).expanduser() if cache_dir else CACHE_DIR
    root.mkdir(parents=True, exist_ok=True)
    dst = root / _gmt_filename(collection, version, id_type)
    if dst.exists() and not force:
        return dst

    url = _gmt_url(collection, version, id_type)
    logger.info("Downloading %s", url)
    return stream_to_file(url, dst, timeout=120, progress=progress)


def load_gmt(
    collection: str = "msigdb",
    version: str = DEFAULT_VERSION,
    id_type: str = "symbols",
    cache_dir: str | Path | None = None,
    return_format: str = "pandas",
    force: bool = False,
) -> pd.DataFrame | dict[tuple[str, str], list[str]]:
    """Download + parse one MSigDB GMT into a long DataFrame (or dict).

    Parameters mirror :func:`download_gmt`; ``return_format`` is forwarded
    to :func:`biodb.utils.read_gmt`.
    """
    from biodb.utils import read_gmt

    path = download_gmt(
        collection=collection,
        version=version,
        id_type=id_type,
        cache_dir=cache_dir,
        force=force,
    )
    return read_gmt(path, return_format=return_format)


# ─── Per-set targeted REST lookup ───────────────────────────────────────────
# MSigDB serves a per-set JSON endpoint at
# ``/gsea/msigdb/<organism>/geneset/<SET_NAME>.json``. Use this to pull
# one gene set with its full metadata (PMID, exact source, gene symbols)
# without downloading the entire collection GMT.

MSIGDB_GENESET_URL_TEMPLATE = (
    "https://www.gsea-msigdb.org/gsea/msigdb/{organism}/geneset/{set_name}.json"
)
"""URL template for the MSigDB per-set JSON endpoint."""


def query_gene_set(
    set_name: str,
    *,
    organism: str = "human",
    timeout: int = 30,
) -> dict:
    """Fetch one MSigDB gene set by name and return its full metadata.

    Parameters
    ----------
    set_name : str
        MSigDB set name (e.g. ``"HALLMARK_APOPTOSIS"``,
        ``"KEGG_MEDICUS_PATHWAY_OF_GENE_EXPRESSION_BY_TYPE_I_INTERFERON"``).
    organism : str, default ``"human"``
        One of ``"human"`` / ``"mouse"`` (case-sensitive in the URL).
    timeout : int, default 30

    Returns
    -------
    dict
        The unwrapped set record: keys include ``systematicName``,
        ``pmid``, ``exactSource``, ``geneSymbols`` (list of strings),
        ``description``.

    Raises
    ------
    KeyError
        If MSigDB returns a payload without the requested set.
    requests.HTTPError
        On non-2xx response (typically 404 for unknown set names).
    """
    url = MSIGDB_GENESET_URL_TEMPLATE.format(organism=organism, set_name=set_name)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    # MSigDB serves an HTML error page (with status 200) for unknown sets
    # — we have to sniff the response shape rather than trust the status.
    ctype = response.headers.get("content-type", "")
    if "application/json" not in ctype:
        raise KeyError(
            f"Set {set_name!r} not found — MSigDB returned a non-JSON page "
            f"(content-type={ctype!r})."
        )
    payload = response.json()
    # The MSigDB JSON endpoint nests the record under the set name.
    if set_name not in payload:
        # Some endpoints wrap in {"<systematicName>": {...}} instead;
        # surface whichever single entry is present.
        if len(payload) == 1:
            return next(iter(payload.values()))
        raise KeyError(f"Set {set_name!r} not found in MSigDB response (keys: {list(payload)[:5]})")
    return payload[set_name]


def query_genes(
    set_name: str,
    *,
    organism: str = "human",
    timeout: int = 30,
) -> list[str]:
    """Convenience: return just the gene-symbol list from a MSigDB set."""
    record = query_gene_set(set_name, organism=organism, timeout=timeout)
    return list(record.get("geneSymbols", []))


__all__ = [
    "CACHE_DIR",
    "DEFAULT_VERSION",
    "KNOWN_COLLECTIONS",
    "MSIGDB_BASE_URL",
    "MSIGDB_GENESET_URL_TEMPLATE",
    "download_gmt",
    "load_gmt",
    "query_gene_set",
    "query_genes",
]
