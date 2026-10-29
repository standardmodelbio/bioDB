"""Topologically associating domains from ENCODE Hi-C contact-domain calls.

A TAD is a stretch of a chromosome whose sequence contacts itself far more
often than it contacts the sequence outside it. The practical consequence is
regulatory: an enhancer usually acts on genes inside its own domain, and the
boundary between two domains is the wall it rarely reaches across. Any
analysis that pairs two genomic positions therefore has TAD co-membership as
a competing explanation for whatever it found — two elements in one domain
are in contact, co-regulated and co-conserved for reasons that have nothing
to do with the hypothesis under test — so this module exists mostly to be
used as a *control*.

The source is the ENCODE portal's Hi-C uniform processing pipeline, which
publishes Arrowhead ``contact domains`` (and ``nested contact domains``) as
bedpe per experiment, on GRCh38, with a stable accession per file:

* :func:`list_domain_files` — query the portal for the domain calls
  available, optionally for one biosample (``"K562"``, ``"GM12878"``, …).
* :func:`download_domains` / :func:`load_domains` — one file by accession,
  cached, as ``chrom``/``start``/``end`` (0-based half-open) with the
  Arrowhead corner columns kept.
* :func:`boundaries` — the domain edges, each widened by ``flank`` bases,
  which is what a "does this pair cross a boundary?" test needs.
* :func:`same_domain` — for pairs of positions, whether some one domain
  contains both. This is the covariate to adjust for.

The 3D Genome Browser, which earlier drafts of this module targeted, now
serves a JavaScript shell rather than its ``hg38.TADs.zip``; ENCODE is used
instead because it is versioned, accession-addressable and openly licensed.

Cached files live at ``~/.cache/biodb/tads/``.

Examples
--------
>>> from biodb.tads import list_domain_files, load_domains, same_domain
>>> files = list_domain_files(biosample="K562")          # doctest: +SKIP
>>> domains = load_domains(files["accession"][0])        # doctest: +SKIP
>>> same_domain(pairs, domains)                          # doctest: +SKIP
"""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import Any

import polars as pl
import requests

from biodb._downloads import stream_to_file

logger = logging.getLogger(__name__)

ENCODE_BASE = "https://www.encodeproject.org"
"""ENCODE portal root; the search API and the file downloads both hang off it."""

DEFAULT_ASSEMBLY = "GRCh38"
"""Only GRCh38 is exposed: this package does not lift over between builds."""

DOMAIN_OUTPUT_TYPES = ("contact domains", "nested contact domains")
"""ENCODE ``output_type`` values that carry Arrowhead domain calls."""

CACHE_DIR = Path("~/.cache/biodb/tads").expanduser()
"""Where downloaded bedpe files are kept."""

_TIMEOUT = 60


def _search(params: dict[str, Any], *, session: requests.Session | None = None) -> list[dict]:
    """One ENCODE search request, returned as the ``@graph`` list."""

    get = (session or requests).get
    response = get(
        f"{ENCODE_BASE}/search/",
        params={**params, "format": "json"},
        headers={"Accept": "application/json"},
        timeout=_TIMEOUT,
    )
    if response.status_code == 404:
        return []  # the portal answers an empty search with 404 rather than an empty graph
    response.raise_for_status()
    payload = response.json()
    graph = payload.get("@graph")
    return list(graph) if isinstance(graph, list) else []


def _biosamples(item: dict[str, Any]) -> list[str]:
    """Every biosample the portal attaches to a file, as a sorted list.

    ENCODE gives a File's ``biosample_ontology`` as a *list* when its dataset
    covers several biosamples, and the search filter
    ``biosample_ontology.term_name=X`` matches a file whose **dataset**
    mentions X rather than a file *of* X. A pooled call set therefore comes
    back from a single-cell-type query, which is how someone ends up using
    HepG2 domains as K562 domains. Both shapes are handled and the full list
    is returned so the caller can see when the answer is ambiguous.
    """

    value = item.get("biosample_ontology")
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return []
    names = {
        str(entry.get("term_name"))
        for entry in value
        if isinstance(entry, dict) and entry.get("term_name")
    }
    return sorted(names)


def list_domain_files(
    *,
    biosample: str | None = None,
    assembly: str = DEFAULT_ASSEMBLY,
    output_type: str = "contact domains",
    limit: int = 200,
    session: requests.Session | None = None,
) -> pl.DataFrame:
    """The ENCODE domain-call files available, one row each.

    Parameters
    ----------
    biosample
        An ENCODE ``biosample_ontology.term_name`` such as ``"K562"``. ``None``
        returns every biosample, which is the right default for surveying and
        the wrong one for an analysis: domains are cell-type specific, so pick
        the biosample your other data came from.
    output_type
        ``"contact domains"`` (the Arrowhead calls) or ``"nested contact
        domains"``.

    Returns
    -------
    polars.DataFrame
        ``accession``, ``biosamples`` (every biosample the file's dataset
        covers), ``biosample_count``, ``file_format``, ``href``, ``dataset``
        and ``status``, sorted by accession so the order is reproducible.

    Warnings
    --------
    The portal's biosample filter matches the **dataset**, not the file, so a
    query for one cell type can return pooled call sets covering several. Rows
    with ``biosample_count > 1`` are counted in a warning; filter on
    ``biosample_count == 1`` when you need calls specific to one cell type.
    """

    if output_type not in DOMAIN_OUTPUT_TYPES:
        raise ValueError(f"output_type must be one of {DOMAIN_OUTPUT_TYPES}, got {output_type!r}")
    params: dict[str, Any] = {
        "type": "File",
        "assembly": assembly,
        "output_type": output_type,
        "status": "released",
        "limit": limit,
        "field": [
            "accession",
            "biosample_ontology.term_name",
            "file_format",
            "href",
            "dataset",
            "status",
        ],
    }
    if biosample is not None:
        params["biosample_ontology.term_name"] = biosample
    rows = [
        {
            "accession": item.get("accession"),
            "biosamples": _biosamples(item),
            "biosample_count": len(_biosamples(item)),
            "file_format": item.get("file_format"),
            "href": item.get("href"),
            "dataset": item.get("dataset"),
            "status": item.get("status"),
        }
        for item in _search(params, session=session)
    ]
    if not rows:
        return pl.DataFrame(
            schema={
                "accession": pl.Utf8,
                "biosamples": pl.List(pl.Utf8),
                "biosample_count": pl.Int64,
                "file_format": pl.Utf8,
                "href": pl.Utf8,
                "dataset": pl.Utf8,
                "status": pl.Utf8,
            }
        )
    frame = pl.DataFrame(rows).sort("accession")
    if biosample is not None:
        pooled = frame.filter(pl.col("biosample_count") > 1).height
        if pooled:
            logger.warning(
                "%d of %d files matching biosample=%r come from datasets covering several "
                "biosamples; the portal filter matches the dataset, not the file. Check "
                "'biosamples' before using one as %s-specific domains.",
                pooled,
                frame.height,
                biosample,
                biosample,
            )
    logger.info("Found %d %s files on %s", len(rows), output_type, assembly)
    return frame


def download_domains(
    accession: str,
    *,
    cache_dir: Path | str | None = None,
    force: bool = False,
    progress: bool = True,
) -> Path:
    """Fetch one ENCODE domain file by accession; returns the cached path."""

    if not accession.startswith("ENCFF"):
        raise ValueError(f"expected an ENCODE file accession (ENCFF…), got {accession!r}")
    directory = Path(cache_dir).expanduser() if cache_dir is not None else CACHE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{accession}.bedpe.gz"
    if destination.exists() and not force:
        return destination
    url = f"{ENCODE_BASE}/files/{accession}/@@download/{accession}.bedpe.gz"
    return stream_to_file(url, destination, progress=progress, desc=accession)


def load_domains(
    accession: str,
    *,
    cache_dir: Path | str | None = None,
    force: bool = False,
    progress: bool = True,
) -> pl.DataFrame:
    """One domain file as ``chrom``/``start``/``end``, 0-based half-open.

    Arrowhead writes each domain as the two corners of a square on the contact
    map: ``x1 x2`` and ``y1 y2`` on the same chromosome. The domain occupies
    ``x1`` to ``y2``, which is what ``start``/``end`` carry; the corners are
    kept as ``corner_start``/``corner_end`` for anyone who needs them. Rows
    whose two corners are on different chromosomes are not domains and are
    dropped.
    """

    path = download_domains(accession, cache_dir=cache_dir, force=force, progress=progress)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[operator]
        lines = [line for line in handle if not line.startswith(("#", "track", "browser"))]
    rows = []
    for line in lines:
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 6:
            continue
        chrom_a, start_a, end_a, chrom_b, start_b, end_b = fields[:6]
        if chrom_a != chrom_b:
            continue
        try:
            x1, x2, y1, y2 = int(start_a), int(end_a), int(start_b), int(end_b)
        except ValueError:
            continue
        rows.append(
            {
                "chrom": chrom_a if chrom_a.startswith("chr") else f"chr{chrom_a}",
                "start": min(x1, y1),
                "end": max(x2, y2),
                "corner_start": x1,
                "corner_end": y2,
                "accession": accession,
            }
        )
    if not rows:
        raise ValueError(f"{path} contained no intra-chromosomal domains")
    frame = pl.DataFrame(rows).sort(["chrom", "start", "end"])
    logger.info("Loaded %d domains from %s", frame.height, accession)
    return frame


def boundaries(domains: pl.DataFrame, *, flank: int = 0) -> pl.DataFrame:
    """The domain edges as intervals, each widened by ``flank`` bases either side.

    Two intervals per domain (its start edge and its end edge), clipped at
    zero, sorted and with overlaps merged so a boundary shared by adjacent
    domains is one row rather than two.
    """

    if flank < 0:
        raise ValueError("flank cannot be negative")
    edges = pl.concat(
        [
            domains.select(
                pl.col("chrom"),
                (pl.col("start") - flank).clip(lower_bound=0).alias("start"),
                (pl.col("start") + flank + 1).alias("end"),
            ),
            domains.select(
                pl.col("chrom"),
                (pl.col("end") - flank).clip(lower_bound=0).alias("start"),
                (pl.col("end") + flank + 1).alias("end"),
            ),
        ]
    ).sort(["chrom", "start", "end"])
    merged: list[dict[str, Any]] = []
    for row in edges.iter_rows(named=True):
        if merged and merged[-1]["chrom"] == row["chrom"] and row["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], row["end"])
            continue
        merged.append(dict(row))
    return pl.DataFrame(merged, schema={"chrom": pl.Utf8, "start": pl.Int64, "end": pl.Int64})


def same_domain(
    pairs: pl.DataFrame,
    domains: pl.DataFrame,
    *,
    left: str = "left",
    right: str = "right",
    chrom: str = "chrom",
) -> pl.DataFrame:
    """For each pair of positions, whether one domain contains both.

    `pairs` carries a chromosome column and two position columns (0-based).
    The result adds ``same_domain`` (bool) and ``shared_domains`` (how many
    domains contain both, since Arrowhead calls nest). A pair whose positions
    fall in no common domain — including one on a chromosome with no calls —
    is ``False`` with a count of zero, never null: "not observed together" is
    the answer the covariate needs.

    This is the control an element-pair analysis owes its reader. Two
    positions in one TAD are in physical contact and share a regulatory
    environment, so they will look related under almost any measure; a result
    that survives conditioning on it is about something else.
    """

    for column in (chrom, left, right):
        if column not in pairs.columns:
            raise ValueError(f"pairs lacks column {column!r}")
    by_chrom: dict[str, list[tuple[int, int]]] = {}
    for row in domains.iter_rows(named=True):
        by_chrom.setdefault(row["chrom"], []).append((int(row["start"]), int(row["end"])))
    counts = []
    for row in pairs.iter_rows(named=True):
        spans = by_chrom.get(row[chrom], ())
        low, high = sorted((int(row[left]), int(row[right])))
        counts.append(sum(1 for start, end in spans if start <= low and high < end))
    return pairs.with_columns(
        pl.Series("shared_domains", counts, dtype=pl.Int64),
        pl.Series("same_domain", [c > 0 for c in counts], dtype=pl.Boolean),
    )


def domain_metadata(accession: str, *, session: requests.Session | None = None) -> dict[str, Any]:
    """The portal's record for one domain file: biosample, dataset, pipeline, md5."""

    get = (session or requests).get
    response = get(
        f"{ENCODE_BASE}/files/{accession}/",
        params={"format": "json"},
        headers={"Accept": "application/json"},
        timeout=_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, str):  # a mocked body handed back as text
        payload = json.loads(payload)
    return {
        "accession": payload.get("accession"),
        "assembly": payload.get("assembly"),
        "output_type": payload.get("output_type"),
        "biosamples": _biosamples(payload),
        "dataset": payload.get("dataset"),
        "md5sum": payload.get("md5sum"),
        "date_created": payload.get("date_created"),
    }
