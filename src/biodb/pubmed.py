"""PubMed client — NCBI E-utilities (API) + bulk XML.gz downloads.

PubMed comprises 40M+ citations from MEDLINE, life-science journals, and
online books. ``bioDB`` exposes both modes:

* **API mode** hits NCBI's E-utilities (``esearch`` + ``esummary`` +
  ``efetch`` on ``db=pubmed``) for one-record-at-a-time work.
  Rate-limit-friendly: each call sleeps ~340 ms to stay under the
  3 req/sec un-keyed cap. Pass ``api_key`` to lift to 10 req/sec.

* **Bulk mode** wraps the canonical PubMed Annual Baseline + Daily
  Update files at ``https://ftp.ncbi.nlm.nih.gov/pubmed/``. Baseline
  releases happen each December as a fixed set of ``pubmed{YY}n####.xml.gz``
  shards (~20 MB each, ~1,300 shards in the 2026 release covering
  the full corpus). Daily update files continue with new / revised /
  deleted citations.

Both paths share a small XML parser that yields normalized records
(``pmid``, ``title``, ``abstract``, ``authors``, ``journal``,
``pub_year``, ``doi``, ``mesh_terms``).

Examples
--------
>>> from biodb import pubmed
>>> hits = pubmed.search("BRCA1", retmax=5)
>>> df = pubmed.query_summaries(hits["pmids"])
>>> record = pubmed.query_abstract("30700171")
>>> # Bulk:
>>> files = pubmed.list_baseline_files()
>>> path = pubmed.download_baseline_file(files[0])  # doctest: +SKIP
>>> articles = pubmed.parse_pubmed_xml(path)        # doctest: +SKIP

References
----------
* PubMed download landing page: https://pubmed.ncbi.nlm.nih.gov/download/
* E-utilities docs: https://www.ncbi.nlm.nih.gov/books/NBK25501/
* PubMed DTD: https://dtd.nlm.nih.gov/ncbi/pubmed/out/pubmed_250101.dtd
"""

from __future__ import annotations

import gzip
import hashlib
import logging
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path
from typing import IO

import pandas as pd
import requests

from biodb._downloads import stream_to_file

logger = logging.getLogger(__name__)

NCBI_EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
"""NCBI E-utilities root. ``esearch`` / ``esummary`` / ``efetch`` live below."""

PUBMED_FTP_BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pubmed"
"""PubMed bulk FTP root (HTTPS-served)."""

DEFAULT_BASELINE_VERSION = "26"
"""Default PubMed baseline release tag. The 2026 baseline files are
named ``pubmed26n####.xml.gz``; bump this string after the December
release each year."""

CACHE_DIR = Path("~/.cache/biodb/pubmed").expanduser()
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_RATE_LIMIT_SLEEP_S = 0.34
"""Per-request sleep that keeps us under the un-keyed 3 req/sec cap."""

_USER_AGENT = "biodb/0.1 (+https://github.com/bschilder/bioDB)"


# ─── NCBI E-utilities transport ────────────────────────────────────────────


def _eutils_get(
    path: str,
    params: dict,
    *,
    timeout: int = 30,
) -> requests.Response:
    """GET an E-utilities endpoint with polite-rate-limit handling.

    Sleeps ~340 ms between calls and retries once on 429. Same pattern
    used by :mod:`biodb.clinvar` for the ClinVar E-utils path.
    """
    url = f"{NCBI_EUTILS_BASE_URL}/{path.lstrip('/')}"
    for attempt in range(2):
        response = requests.get(
            url, params=params, timeout=timeout, headers={"User-Agent": _USER_AGENT}
        )
        if response.status_code != 429:
            response.raise_for_status()
            time.sleep(_RATE_LIMIT_SLEEP_S)
            return response
        time.sleep(1.0 + attempt)
    response.raise_for_status()
    return response  # pragma: no cover  -- unreachable; raise_for_status raises


def search(
    term: str,
    *,
    retmax: int = 20,
    retstart: int = 0,
    api_key: str | None = None,
    timeout: int = 30,
) -> dict:
    """Run an ``esearch`` against ``db=pubmed`` and return PMIDs + metadata.

    Parameters
    ----------
    term : str
        PubMed query string (e.g. ``"BRCA1"``, ``"BRCA1 AND breast cancer"``,
        ``"Schilder BM[au]"``). NCBI's query translator is permissive — it
        expands free-text into MeSH-aware boolean queries automatically.
    retmax : int, default 20
        Max PMIDs to return.
    retstart : int, default 0
        Offset for pagination.
    api_key : str, optional
        NCBI E-utilities API key.
    timeout : int

    Returns
    -------
    dict
        ``{"pmids": [...], "total": int, "query_translation": str}``.
        ``total`` is the full match count even when ``retmax`` truncates.
    """
    params = {
        "db": "pubmed",
        "term": term,
        "retmax": retmax,
        "retstart": retstart,
        "retmode": "json",
    }
    if api_key:
        params["api_key"] = api_key
    payload = _eutils_get("esearch.fcgi", params, timeout=timeout).json()
    result = payload.get("esearchresult", {})
    return {
        "pmids": result.get("idlist", []),
        "total": int(result.get("count", 0)),
        "query_translation": result.get("querytranslation", ""),
    }


def query_summaries(
    pmids: str | int | list[str | int],
    *,
    api_key: str | None = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch ``esummary`` records for one or more PMIDs as a DataFrame.

    Columns: ``pmid``, ``title``, ``authors`` (comma-joined), ``source``
    (ISO journal abbreviation), ``pubdate``, ``epubdate``, ``doi``.
    Unknown PMIDs are silently dropped (E-utilities returns no record).
    """
    pmid_list = [str(pmids)] if isinstance(pmids, (str, int)) else [str(p) for p in pmids]
    if not pmid_list:
        return pd.DataFrame(
            columns=["pmid", "title", "authors", "source", "pubdate", "epubdate", "doi"]
        )
    params: dict = {"db": "pubmed", "id": ",".join(pmid_list), "retmode": "json"}
    if api_key:
        params["api_key"] = api_key
    payload = _eutils_get("esummary.fcgi", params, timeout=timeout).json()
    result = payload.get("result", {})
    rows = []
    for uid in result.get("uids", []):
        rec = result.get(uid, {})
        doi = ""
        for art_id in rec.get("articleids", []) or []:
            if art_id.get("idtype") == "doi":
                doi = art_id.get("value", "")
                break
        rows.append(
            {
                "pmid": uid,
                "title": rec.get("title", ""),
                "authors": ", ".join(a.get("name", "") for a in rec.get("authors", [])),
                "source": rec.get("source", ""),
                "pubdate": rec.get("pubdate", ""),
                "epubdate": rec.get("epubdate", ""),
                "doi": doi,
            }
        )
    return pd.DataFrame(rows)


def query_pmid(
    pmid: str | int,
    *,
    api_key: str | None = None,
    timeout: int = 30,
) -> dict:
    """Convenience: fetch the ``esummary`` record for a single PMID as a
    plain ``dict`` (one row from :func:`query_summaries`).
    """
    df = query_summaries(pmid, api_key=api_key, timeout=timeout)
    if df.empty:
        raise KeyError(f"PMID {pmid!r} not found in PubMed")
    return df.iloc[0].to_dict()


def query_abstract(
    pmid: str | int,
    *,
    api_key: str | None = None,
    timeout: int = 30,
) -> dict:
    """Fetch the full PubMed XML record for ``pmid`` and parse it.

    Returns
    -------
    dict
        ``pmid``, ``title``, ``abstract``, ``authors``, ``journal``,
        ``pub_year``, ``doi``, ``mesh_terms``. Labeled abstract sections
        (``AIM`` / ``METHODS`` / ``RESULTS`` / ``CONCLUSIONS``) are
        concatenated with their section labels.

    Raises
    ------
    KeyError
        If no record is returned for ``pmid``.
    """
    params: dict = {"db": "pubmed", "id": str(pmid), "rettype": "abstract", "retmode": "xml"}
    if api_key:
        params["api_key"] = api_key
    response = _eutils_get("efetch.fcgi", params, timeout=timeout)
    records = list(_iter_articles_from_xml(response.content))
    if not records:
        raise KeyError(f"PMID {pmid!r} not found in PubMed")
    return records[0]


# ─── Bulk FTP — baseline + daily update files ──────────────────────────────


_HREF_RE = re.compile(r'href="([^"]+\.xml\.gz)"')


def _list_directory(directory: str, *, timeout: int = 30) -> list[str]:
    """Scrape an Apache-style directory index for ``.xml.gz`` filenames."""
    response = requests.get(
        f"{PUBMED_FTP_BASE_URL}/{directory}/",
        timeout=timeout,
        headers={"User-Agent": _USER_AGENT},
    )
    response.raise_for_status()
    return _HREF_RE.findall(response.text)


def list_baseline_files(*, timeout: int = 30) -> list[str]:
    """List every ``.xml.gz`` filename in the PubMed Annual Baseline.

    The full baseline is ~1,300 files (~20 MB each, ~30 GB total
    compressed). Each filename has the form ``pubmed{YY}n####.xml.gz``.
    """
    return _list_directory("baseline", timeout=timeout)


def list_update_files(*, timeout: int = 30) -> list[str]:
    """List every ``.xml.gz`` filename in the PubMed Daily Update directory.

    These continue numerically from the baseline (e.g. baseline ends at
    ``pubmed26n1334.xml.gz``; the first update file is
    ``pubmed26n1335.xml.gz``).
    """
    return _list_directory("updatefiles", timeout=timeout)


def _download_xml_gz(
    directory: str,
    filename: str,
    *,
    cache_dir: str | Path | None,
    force: bool,
    verify_md5: bool,
    timeout: int,
    progress: bool = True,
) -> Path:
    if not filename.endswith(".xml.gz"):
        raise ValueError(f"{filename!r} doesn't look like a PubMed XML gzip filename")
    root = Path(cache_dir).expanduser() if cache_dir else CACHE_DIR / directory
    root.mkdir(parents=True, exist_ok=True)
    dst = root / filename
    if dst.exists() and not force:
        return dst
    url = f"{PUBMED_FTP_BASE_URL}/{directory}/{filename}"
    logger.info("Downloading PubMed %s/%s -> %s", directory, filename, dst)
    stream_to_file(
        url,
        dst,
        headers={"User-Agent": _USER_AGENT},
        timeout=timeout,
        progress=progress,
    )
    if verify_md5:
        _verify_md5(dst, f"{url}.md5", timeout=timeout)
    return dst


def download_baseline_file(
    filename: str,
    *,
    cache_dir: str | Path | None = None,
    force: bool = False,
    verify_md5: bool = True,
    timeout: int = 600,
    progress: bool = True,
) -> Path:
    """Download one baseline ``pubmed{YY}n####.xml.gz`` shard.

    Cached at ``~/.cache/biodb/pubmed/baseline/<filename>`` by default.
    The companion ``.md5`` file is fetched and checked unless
    ``verify_md5=False``. ``progress=False`` silences the tqdm bar.
    """
    return _download_xml_gz(
        "baseline",
        filename,
        cache_dir=cache_dir,
        force=force,
        verify_md5=verify_md5,
        timeout=timeout,
        progress=progress,
    )


def download_update_file(
    filename: str,
    *,
    cache_dir: str | Path | None = None,
    force: bool = False,
    verify_md5: bool = True,
    timeout: int = 600,
    progress: bool = True,
) -> Path:
    """Download one daily-update ``pubmed{YY}n####.xml.gz`` shard."""
    return _download_xml_gz(
        "updatefiles",
        filename,
        cache_dir=cache_dir,
        force=force,
        verify_md5=verify_md5,
        timeout=timeout,
        progress=progress,
    )


# ─── MD5 verification ──────────────────────────────────────────────────────


_MD5_HASH_RE = re.compile(r"\b([a-fA-F0-9]{32})\b")


def _verify_md5(local_path: Path, md5_url: str, *, timeout: int) -> None:
    """Download the sibling ``.md5`` file and check the hash matches.

    NCBI ships ``.md5`` files in the format ``MD5(<filename>)= <hash>``.
    We just extract the first 32-hex run — robust to format tweaks.
    """
    response = requests.get(md5_url, timeout=timeout, headers={"User-Agent": _USER_AGENT})
    response.raise_for_status()
    match = _MD5_HASH_RE.search(response.text)
    if not match:
        raise RuntimeError(f"Could not parse an MD5 hash from {md5_url}: {response.text[:120]!r}")
    expected = match.group(1).lower()
    digest = hashlib.md5()  # noqa: S324  -- not cryptographic; matches NCBI's published checksums
    with open(local_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    actual = digest.hexdigest().lower()
    if actual != expected:
        raise RuntimeError(
            f"MD5 mismatch for {local_path.name}: expected {expected}, got {actual} — "
            f"download is likely corrupt; re-run with force=True."
        )


# ─── PubMed XML parser ─────────────────────────────────────────────────────


def parse_pubmed_xml(xml_path: str | Path) -> pd.DataFrame:
    """Parse a PubMed ``.xml`` or ``.xml.gz`` file into a long DataFrame.

    Each row is one article — see :func:`_article_to_dict` for the column
    list. The parser uses iterative ``ElementTree.iterparse`` and clears
    each element after read, so memory use stays roughly flat even on
    multi-GB baseline shards.
    """
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(xml_path)
    opener = gzip.open if xml_path.suffix == ".gz" else open
    with opener(xml_path, "rb") as f:
        return pd.DataFrame(_iter_articles_from_stream(f))


def _iter_articles_from_xml(xml_bytes: bytes) -> Iterator[dict]:
    """Stream ``PubmedArticle`` elements out of in-memory XML bytes."""
    from io import BytesIO

    yield from _iter_articles_from_stream(BytesIO(xml_bytes))


def _iter_articles_from_stream(stream: IO[bytes]) -> Iterator[dict]:
    """Stream ``PubmedArticle`` elements out of a binary stream."""
    for _, elem in ET.iterparse(stream, events=("end",)):
        if elem.tag == "PubmedArticle":
            yield _article_to_dict(elem)
            elem.clear()


def _article_to_dict(article: ET.Element) -> dict:
    """Pull the load-bearing fields out of one ``<PubmedArticle>`` element."""
    pmid = article.findtext("MedlineCitation/PMID", default="").strip()
    title = (article.findtext("MedlineCitation/Article/ArticleTitle", default="") or "").strip()

    # Labeled abstracts: concatenate "<LABEL>: <text>" sections.
    abstract_chunks = []
    for ab in article.findall("MedlineCitation/Article/Abstract/AbstractText"):
        label = (ab.attrib.get("Label") or "").strip()
        text = "".join(ab.itertext()).strip()
        abstract_chunks.append(f"{label}: {text}" if label else text)
    abstract = "\n\n".join(c for c in abstract_chunks if c)

    authors = []
    for au in article.findall("MedlineCitation/Article/AuthorList/Author"):
        last = au.findtext("LastName", default="").strip()
        initials = au.findtext("Initials", default="").strip()
        collective = au.findtext("CollectiveName", default="").strip()
        if last or initials:
            authors.append(f"{last} {initials}".strip())
        elif collective:
            authors.append(collective)

    journal = article.findtext(
        "MedlineCitation/Article/Journal/ISOAbbreviation", default=""
    ).strip()
    pub_year = article.findtext(
        "MedlineCitation/Article/Journal/JournalIssue/PubDate/Year", default=""
    ).strip()

    doi = ""
    for art_id in article.findall("PubmedData/ArticleIdList/ArticleId"):
        if art_id.attrib.get("IdType") == "doi":
            doi = (art_id.text or "").strip()
            break

    mesh_terms = [
        (mh.findtext("DescriptorName", default="") or "").strip()
        for mh in article.findall("MedlineCitation/MeshHeadingList/MeshHeading")
    ]
    mesh_terms = [m for m in mesh_terms if m]

    return {
        "pmid": pmid,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "journal": journal,
        "pub_year": pub_year,
        "doi": doi,
        "mesh_terms": mesh_terms,
    }


__all__ = [
    "CACHE_DIR",
    "DEFAULT_BASELINE_VERSION",
    "NCBI_EUTILS_BASE_URL",
    "PUBMED_FTP_BASE_URL",
    "download_baseline_file",
    "download_update_file",
    "list_baseline_files",
    "list_update_files",
    "parse_pubmed_xml",
    "query_abstract",
    "query_pmid",
    "query_summaries",
    "search",
]
