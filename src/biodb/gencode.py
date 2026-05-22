"""GENCODE annotation fetcher.

Downloads (and caches) the GENCODE human annotation GTF for a given
release, then exposes :func:`fetch_mane_select` which returns one row
per protein-coding gene joined to its MANE Select transcript and
CDS-derived amino-acid length.

Cache location
--------------
``pooch``'s OS cache (typically
``~/Library/Caches/pooch/biodb_gencode/`` on macOS,
``~/.cache/pooch/biodb_gencode/`` on Linux), keyed by release.

Use
---

.. code-block:: python

    from biodb.gencode import fetch_mane_select
    df = fetch_mane_select(release="47")
    # df.columns -> gene_id, gene_symbol, chrom, start, end, strand,
    #               mane_select_tx, aa_length

History
-------
Originally lived in ``standardmodelbio/seqlab`` under
``src/seqlab/panel/gencode.py`` (and before that in
``smb-protopheno``); migrated here so every downstream consumer
queries one place for canonical bulk-annotation data.
"""

from __future__ import annotations

import gzip
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import polars as pl

_URL_FMT: str = (
    "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/"
    "release_{release}/gencode.v{release}.annotation.gtf.gz"
)
"""GENCODE annotation GTF URL format. ``{release}`` is the integer release
tag, e.g. ``"47"``. EBI also publishes a ``-mouse`` variant; only human
is wired here -- add a ``species`` arg if needed."""


def _pooch_cache(name: str) -> str:
    """Resolve ``name`` inside pooch's default OS cache directory.

    Lazy import so plain ``import biodb.gencode`` doesn't pull pooch
    into every consumer that just wants the helpers.
    """
    import os

    import pooch

    return os.path.join(pooch.os_cache("biodb_gencode"), name)


_ATTR_RE = re.compile(r'(\w+) "([^"]*)"')


def _parse_attrs(attr: str) -> dict[str, str]:
    """Parse a GTF attribute string into a dict.

    ``tag`` keys are joined with ``,`` if a record carries multiple
    (e.g. ``tag "MANE_Select"; tag "appris_principal_1";``).

    Examples
    --------
    >>> d = _parse_attrs('gene_id "ENSG00000012048.25"; gene_name "BRCA1"; tag "MANE_Select";')
    >>> d["gene_id"]
    'ENSG00000012048.25'
    >>> d["gene_name"]
    'BRCA1'
    >>> "MANE_Select" in d["tag"]
    True
    """
    out: dict[str, str] = {}
    for match in _ATTR_RE.finditer(attr):
        key, value = match.group(1), match.group(2)
        if key == "tag":
            prev = out.get("tag", "")
            out["tag"] = f"{prev},{value}" if prev else value
        else:
            out[key] = value
    return out


def _strip_version(ensembl_id: str) -> str:
    """Return ``ensembl_id`` without the trailing ``.N`` version suffix.

    Examples
    --------
    >>> _strip_version("ENSG00000012048.25")
    'ENSG00000012048'
    >>> _strip_version("ENSG00000012048")
    'ENSG00000012048'
    """
    return ensembl_id.split(".", 1)[0]


def download_gencode_gtf(release: str, cache_dir: str | Path | None = None) -> Path:
    """Download (or reuse cached) the GENCODE annotation GTF.

    Parameters
    ----------
    release : str
        GENCODE release, e.g. ``"47"``.
    cache_dir : str or Path, optional
        Override the default pooch cache directory.

    Returns
    -------
    pathlib.Path
        Local path to the cached ``.gtf.gz``.
    """
    import pooch

    url = _URL_FMT.format(release=release)
    fname = f"gencode.v{release}.annotation.gtf.gz"
    path = pooch.retrieve(
        url=url,
        known_hash=None,  # GENCODE doesn't publish checksums in a simple form; trust TLS
        fname=fname,
        path=str(cache_dir) if cache_dir is not None else _pooch_cache(""),
        progressbar=True,
    )
    return Path(path)


def _iter_gtf_rows(
    path: Path,
) -> Iterator[tuple[str, str, int, int, str, dict[str, str]]]:
    """Yield ``(chrom, feature, start, end, strand, attrs)`` tuples from a GTF.gz.

    Skips comment lines and feature types we don't need (``gene``,
    ``transcript``, ``CDS`` are kept). Coordinates are 1-based
    inclusive in GTF; we emit 0-based half-open here so downstream
    arithmetic matches the rest of biodb.

    Parameters
    ----------
    path : pathlib.Path
        Path to a ``.gtf.gz``.

    Yields
    ------
    tuple
        ``(chrom, feature, start, end, strand, attrs_dict)``.
    """
    with gzip.open(path, mode="rt", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            chrom, _source, feature, start, end, _score, strand, _frame, attr = parts
            if feature not in ("gene", "transcript", "CDS"):
                continue
            yield (
                chrom,
                feature,
                int(start) - 1,
                int(end),
                strand,
                _parse_attrs(attr),
            )


def fetch_mane_select(release: str, cache_dir: str | Path | None = None) -> pl.DataFrame:
    """Fetch GENCODE annotation and extract one row per protein-coding gene.

    Picks the MANE Select transcript (``tag "MANE_Select"``) for each
    gene and computes ``aa_length = (cds_length - 3) // 3`` (subtracts
    the stop codon, then divides by 3).

    Genes without a MANE Select transcript are dropped. (The seqlab
    original notes that a few releases ship a gene with only an
    ``Ensembl_canonical`` tag but no MANE; the join below silently
    drops them. If you need that fallback, filter the intermediate
    ``tx_rows`` differently before the join.)

    Parameters
    ----------
    release : str
        GENCODE release, e.g. ``"47"``.
    cache_dir : str or Path, optional
        Override the default pooch cache directory.

    Returns
    -------
    polars.DataFrame
        Columns:

        - ``gene_id`` (versionless Ensembl ID, ``ENSG...``)
        - ``gene_symbol``
        - ``chrom``, ``start``, ``end``, ``strand``
        - ``mane_select_tx`` (versioned, ``ENST...N``)
        - ``aa_length``

    Raises
    ------
    RuntimeError
        If the GTF parses no protein-coding genes (corrupt download)
        or no MANE Select transcripts (wrong release tag).
    """
    gtf_path = download_gencode_gtf(release, cache_dir=cache_dir)

    gene_rows: list[dict[str, Any]] = []
    tx_rows: list[dict[str, Any]] = []
    cds_sum: dict[str, int] = {}

    for chrom, feature, start, end, strand, attrs in _iter_gtf_rows(gtf_path):
        if feature == "gene":
            if attrs.get("gene_type") != "protein_coding":
                continue
            gene_rows.append(
                {
                    "gene_id": _strip_version(attrs["gene_id"]),
                    "gene_symbol": attrs.get("gene_name", ""),
                    "chrom": chrom,
                    "start": start,
                    "end": end,
                    "strand": strand,
                }
            )
        elif feature == "transcript":
            if attrs.get("transcript_type") != "protein_coding":
                continue
            tag = attrs.get("tag", "")
            if "MANE_Select" not in tag:
                continue
            tx_rows.append(
                {
                    "gene_id": _strip_version(attrs["gene_id"]),
                    "mane_select_tx": attrs["transcript_id"],
                }
            )
        elif feature == "CDS":
            tx_id = attrs.get("transcript_id")
            if tx_id is None:
                continue
            tag = attrs.get("tag", "")
            if "MANE_Select" not in tag:
                continue
            cds_sum[tx_id] = cds_sum.get(tx_id, 0) + (end - start)

    if not gene_rows:
        raise RuntimeError(f"GENCODE v{release}: no protein-coding genes parsed")
    if not tx_rows:
        raise RuntimeError(f"GENCODE v{release}: no MANE Select transcripts found")

    genes_df = pl.DataFrame(gene_rows)
    tx_df = pl.DataFrame(tx_rows)
    cds_df = pl.DataFrame(
        [
            {
                "mane_select_tx": tx_id,
                "aa_length": max((cds_len - 3) // 3, 0),
            }
            for tx_id, cds_len in cds_sum.items()
        ]
    )
    out = genes_df.join(tx_df, on="gene_id", how="inner")
    out = out.join(cds_df, on="mane_select_tx", how="inner")
    return out


__all__ = [
    "download_gencode_gtf",
    "fetch_mane_select",
]
