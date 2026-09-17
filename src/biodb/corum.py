"""CORUM client — manually curated mammalian protein complexes, as gene pairs.

`CORUM <https://mips.helmholtz-muenchen.de/corum/>`_ (the Comprehensive
Resource of Mammalian protein complexes, Helmholtz Munich) curates protein
complexes from the literature. Release 5.3 (April 2026) holds 7,812 human
complexes. This module reads the release files through CORUM's public API:

* :func:`list_releases` — every release version with its date.
* :func:`download_complexes` / :func:`load_complexes` — the tab-delimited
  complex table (``allComplexes`` for every organism, ``humanComplexes``
  for human), current or archived by version; one row per complex with
  semicolon-joined subunit UniProt ids and gene names, purification
  methods, GO functions and functional-category assignments.
* :func:`complex_pairs` — every unordered pair of subunit gene symbols
  within a complex, one row per (pair, complex): the co-complex gold
  standard for any gene-pair predictor.

Cached files live at ``~/.cache/biodb/corum/``. CORUM is licensed
CC BY-NC-SA 4.0: it is downloaded on demand and never redistributed, and
the licence is non-commercial.

Known upstream issue (September 2026): the CORUM host serves an incomplete
TLS certificate chain, so ``requests`` with certifi's bundle can fail with
``CERTIFICATE_VERIFY_FAILED`` while a system trust store (macOS Keychain,
``curl``) succeeds. Verification is *not* disabled here; point
``REQUESTS_CA_BUNDLE`` at a bundle that carries the intermediate, or set
``SSL_CERT_FILE``, if your environment hits it.

Examples
--------
>>> from biodb.corum import complex_pairs
>>> pairs = complex_pairs()                                # doctest: +SKIP
>>> pairs.filter(pl.col("gene_a") == "BCL6").head()        # doctest: +SKIP
"""

from __future__ import annotations

import csv
import logging
from itertools import combinations
from pathlib import Path

import polars as pl
import requests

from biodb._downloads import stream_to_file

logger = logging.getLogger(__name__)

CORUM_API = "https://mips.helmholtz-muenchen.de/fastapi-corum/public"
"""CORUM's public FastAPI root (the website is a client of it)."""

DEFAULT_VERSION = "current"
"""``"current"`` follows the latest release; pin a version such as ``"5.3"``
for reproducibility. The downloaded file's SHA-256 is what a record should
carry."""

DEFAULT_FILE = "human"
"""``"human"`` (humanComplexes) or ``"complete"`` (allComplexes)."""

CACHE_DIR = Path("~/.cache/biodb/corum").expanduser()
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _download_url(file_id: str, version: str, file_format: str = "txt") -> str:
    if version == "current":
        return f"{CORUM_API}/file/download_current_file?file_id={file_id}&file_format={file_format}"
    return (
        f"{CORUM_API}/file/download_archived_file?file_id={file_id}"
        f"&file_format={file_format}&version={version}"
    )


def list_releases(*, timeout: float = 30.0) -> pl.DataFrame:
    """Every CORUM release: ``version``, ``date``, ``is_current``."""
    response = requests.get(f"{CORUM_API}/releases", timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    return pl.DataFrame(
        payload, schema={"version": pl.Utf8, "date": pl.Utf8, "is_current": pl.Boolean}
    )


def download_complexes(
    *,
    file_id: str = DEFAULT_FILE,
    version: str = DEFAULT_VERSION,
    cache_dir: str | Path | None = None,
    force: bool = False,
    progress: bool = True,
) -> Path:
    """Download the tab-delimited complex table for ``file_id`` at ``version``.

    Returns the local ``.txt`` path (the API serves plain UTF-8 text).
    """
    if file_id not in ("human", "complete"):
        raise ValueError("file_id must be 'human' or 'complete'")
    cache = Path(cache_dir or CACHE_DIR).expanduser() / version
    cache.mkdir(parents=True, exist_ok=True)
    dst = cache / f"corum_{file_id}Complexes.txt"
    if dst.exists() and not force:
        return dst
    return stream_to_file(_download_url(file_id, version), dst, progress=progress, desc=dst.name)


def load_complexes(
    *,
    file_id: str = DEFAULT_FILE,
    version: str = DEFAULT_VERSION,
    cache_dir: str | Path | None = None,
    force: bool = False,
    progress: bool = True,
) -> pl.DataFrame:
    """The complex table as published, one row per complex.

    Returns
    -------
    polars.DataFrame
        The 28 published columns (``complex_id``, ``complex_name``,
        ``organism``, ``cell_line``, ``pmid``, ``subunits_uniprot_id``,
        ``subunits_gene_name``, ``purification_methods``, ``functions_go_id``,
        ``fcgs_name``, ...), all as strings except ``complex_id`` (int64);
        list-valued columns keep CORUM's ``;`` separator.
    """
    path = download_complexes(
        file_id=file_id, version=version, cache_dir=cache_dir, force=force, progress=progress
    )
    # Free-text columns (comments, drug annotations) carry newlines inside
    # quoted cells, so the file is a tab-separated CSV dialect rather than one
    # row per line; the standard csv reader is what parses it faithfully.
    # Short rows are padded and long rows truncated to the header.
    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", quotechar='"')
        header = next(reader)
        width = len(header)
        rows = []
        for fields in reader:
            if not fields:
                continue
            if len(fields) < width:
                fields = fields + [""] * (width - len(fields))
            rows.append(fields[:width])
    frame = pl.DataFrame(
        {name: [row[i] or None for row in rows] for i, name in enumerate(header)},
        schema={name: pl.Utf8 for name in header},
    )
    frame = frame.with_columns(pl.col("complex_id").cast(pl.Int64))
    logger.info("Loaded %d CORUM complexes from %s", frame.height, path.name)
    return frame


def complex_pairs(
    *,
    organism: str | None = "Human",
    minimum_subunits: int = 2,
    file_id: str = DEFAULT_FILE,
    version: str = DEFAULT_VERSION,
    cache_dir: str | Path | None = None,
    force: bool = False,
    progress: bool = True,
) -> pl.DataFrame:
    """Every unordered pair of subunit gene symbols within a complex.

    Parameters
    ----------
    organism : str, optional
        Keep only complexes annotated to this organism (default ``"Human"``;
        ``None`` keeps all).
    minimum_subunits : int
        Skip complexes with fewer named subunits than this.

    Returns
    -------
    polars.DataFrame
        ``gene_a``, ``gene_b`` (upper-cased symbols with ``gene_a < gene_b``),
        ``complex_id``, ``complex_name``, ``subunits`` (the complex's size).
        A pair that sits in several complexes appears once per complex; use
        ``.unique(["gene_a", "gene_b"])`` for the pair set.
    """
    frame = load_complexes(
        file_id=file_id, version=version, cache_dir=cache_dir, force=force, progress=progress
    )
    if organism is not None:
        frame = frame.filter(pl.col("organism") == organism)
    rows: list[dict[str, object]] = []
    for complex_id, name, genes in frame.select(
        "complex_id", "complex_name", "subunits_gene_name"
    ).iter_rows():
        if genes is None:
            continue
        symbols = sorted(
            {g.strip().upper() for g in str(genes).split(";") if g.strip() and g.strip() != "None"}
        )
        if len(symbols) < minimum_subunits:
            continue
        for a, b in combinations(symbols, 2):
            rows.append(
                {
                    "gene_a": a,
                    "gene_b": b,
                    "complex_id": int(complex_id),
                    "complex_name": name,
                    "subunits": len(symbols),
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "gene_a": pl.Utf8,
            "gene_b": pl.Utf8,
            "complex_id": pl.Int64,
            "complex_name": pl.Utf8,
            "subunits": pl.Int64,
        },
    ).sort(["gene_a", "gene_b", "complex_id"])
