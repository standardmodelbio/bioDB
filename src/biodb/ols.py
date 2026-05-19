"""EBI OLS4 (Ontology Lookup Service) REST client.

The **targeted-query mode** for any OBO Foundry ontology — Mondo, HPO,
EFO, SO, GO, and ~280 others hosted at the EMBL-EBI's `OLS4
<https://www.ebi.ac.uk/ols4/>`_.

Where :mod:`biodb.ontology` requires a local OWL file to walk
(via ``owlready2``), this module hits the OLS REST API and lets you
look up terms, descendants, ancestors, and full-text-search across
ontologies *without* any local data. The natural fit when you only
need a handful of terms, or in environments where you can't or don't
want to download a multi-hundred-MB OWL file.

Docs: https://www.ebi.ac.uk/ols4/api-docs

Examples
--------
>>> from biodb.ols import get_term, get_descendants, search
>>> ad = get_term("mondo", "MONDO:0004975")            # doctest: +SKIP
>>> ad["label"]                                         # doctest: +SKIP
'Alzheimer disease'
>>> kids = get_descendants("mondo", "MONDO:0004975")    # doctest: +SKIP
>>> hits = search("alzheimer", ontology="mondo")        # doctest: +SKIP
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests

logger = logging.getLogger(__name__)

OLS_API_BASE_URL = "https://www.ebi.ac.uk/ols4/api"
"""OLS4 REST API root."""

DEFAULT_CACHE_DIR = Path("~/.cache/biodb/ols").expanduser()
"""Default on-disk cache root for :func:`list_terms`. One subdir per
ontology id, one parquet per OLS-reported version inside it. Lets a
:func:`list_terms` call survive process / kernel restarts without
re-walking the OLS pagination, and lets you keep a paper-trail of
every ontology release you've ever queried."""

OBO_PURL_BASE = "http://purl.obolibrary.org/obo"
"""PURL root for OBO Foundry ontologies. Used to expand CURIEs."""

NON_OBO_PREFIXES: dict[str, str] = {
    # Vocabularies whose IRIs *don't* follow ``http://purl.obolibrary.org/obo/{prefix}_{local}``.
    # OLS indexes them anyway; expand their CURIEs to the right IRI scheme so
    # ``get_term`` / ``get_descendants`` / etc. work without callers having to
    # know the per-vocabulary URI grammar.
    "SNOMED": "http://snomed.info/id/{local}",
    "SCTID": "http://snomed.info/id/{local}",  # synonym sometimes used in MEDLINE
    "EFO": "http://www.ebi.ac.uk/efo/EFO_{local}",
    "ORPHA": "http://www.orpha.net/ORDO/Orphanet_{local}",
    "ORPHANET": "http://www.orpha.net/ORDO/Orphanet_{local}",
}
"""Per-prefix IRI templates for vocabularies that don't use the OBO PURL pattern."""

_DEFAULT_PAGE_SIZE = 500
"""OLS paginates everything. 500 is the largest size that still keeps
each response well under a megabyte."""

_TERM_COLUMNS = ("obo_id", "label", "iri", "description", "synonyms", "is_obsolete")
"""Columns surfaced from OLS term records in the DataFrame return path."""


def curie_to_iri(curie_or_iri: str) -> str:
    """Expand a CURIE (e.g. ``"MONDO:0004975"``) to its full IRI.

    Most OBO Foundry vocabularies follow the standard PURL pattern
    ``http://purl.obolibrary.org/obo/{prefix}_{local}``. A small set
    of high-traffic non-OBO vocabularies (SNOMED, EFO, ORDO) use their
    own URI schemes — see :data:`NON_OBO_PREFIXES` for the mapping.

    Anything that already looks like an IRI (starts with ``http://`` or
    ``https://``) is returned unchanged.
    """
    if curie_or_iri.startswith(("http://", "https://")):
        return curie_or_iri
    if ":" not in curie_or_iri:
        raise ValueError(
            f"{curie_or_iri!r} is neither an IRI nor a CURIE (expected ``PREFIX:local``)."
        )
    prefix, local = curie_or_iri.split(":", 1)
    template = NON_OBO_PREFIXES.get(prefix.upper())
    if template is not None:
        return template.format(local=local)
    return f"{OBO_PURL_BASE}/{prefix}_{local}"


def _double_quote_iri(iri: str) -> str:
    """OLS expects the IRI path segment to be **double**-URL-encoded.

    The first ``quote`` turns ``/`` and ``:`` into ``%2F`` / ``%3A``; the
    second turns those ``%`` characters into ``%25``. Without the second
    pass the OLS Spring router rejects the request as a malformed path.
    """
    return quote(quote(iri, safe=""), safe="")


def _get(
    url: str,
    params: dict | None = None,
    timeout: int = 30,
    *,
    max_retries: int = 5,
    backoff_s: float = 1.0,
) -> dict:
    """GET ``url`` with exponential-backoff retries on transient failures.

    OLS occasionally drops connections mid-stream or 5xx's under load --
    the per-request error rate is low (sub-percent) but adds up across
    long paginated walks: at 753 pages even a 99% success rate gives
    only a ~0.05% chance of completing without retries. We retry on
    ``requests.RequestException`` (covers timeouts + connection resets
    + 5xx) with exponential backoff; 4xx errors short-circuit immediately.

    Parameters
    ----------
    url : str
    params : dict, optional
    timeout : int, default 30
        Per-attempt timeout (not cumulative).
    max_retries : int, default 5
        Total attempts including the first. ``1`` disables retry (the
        legacy behavior).
    backoff_s : float, default 1.0
        Initial backoff; doubles on every retry.

    Returns
    -------
    dict
        The parsed JSON body.

    Raises
    ------
    requests.HTTPError
        On 4xx (immediate) or 5xx after retries are exhausted.
    requests.RequestException
        On connection-level errors after retries are exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            # 4xx aren't worth retrying -- the request is wrong, not flaky.
            if 400 <= response.status_code < 500:
                response.raise_for_status()
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_exc = exc
            is_4xx = (
                isinstance(exc, requests.HTTPError)
                and exc.response is not None
                and 400 <= exc.response.status_code < 500
            )
            if is_4xx or attempt == max_retries - 1:
                raise
            sleep_s = backoff_s * (2**attempt)
            logger.warning(
                "OLS GET failed (attempt %d/%d): %s; retrying in %.1fs",
                attempt + 1,
                max_retries,
                exc,
                sleep_s,
            )
            time.sleep(sleep_s)
    assert last_exc is not None
    raise last_exc


def _paginate(
    url: str,
    params: dict | None = None,
    timeout: int = 30,
    *,
    progress: bool = False,
    desc: str | None = None,
) -> Iterator[dict]:
    """Walk OLS's HAL-style paginated endpoints (yielding each term).

    Parameters
    ----------
    url, params, timeout
        Forwarded to :func:`_get`.
    progress : bool, default False
        When True, show a tqdm progress bar keyed on ``page.totalPages``
        from the first response. Caller-facing wrappers
        (:func:`list_terms`, :func:`iter_terms`) opt in by default
        because their walks can take many minutes; the per-term
        relationship wrappers (descendants / ancestors / children /
        parents) leave it off because they're usually fast and the
        bar would be noise.
    desc : str or None
        Description shown on the tqdm bar. Ignored when ``progress=False``.
    """
    from tqdm.auto import tqdm

    next_url: str | None = url
    next_params: dict | None = params
    pbar: Any = None
    try:
        while next_url is not None:
            payload = _get(next_url, params=next_params, timeout=timeout)
            if progress and pbar is None:
                total_pages = (payload.get("page") or {}).get("totalPages")
                pbar = tqdm(total=total_pages, desc=desc or "OLS pages", unit="page")
            yield from (payload.get("_embedded") or {}).get("terms") or []
            if pbar is not None:
                pbar.update(1)
            next_url = (payload.get("_links") or {}).get("next", {}).get("href")
            # The `next` href already carries pagination params; don't duplicate.
            next_params = None
    finally:
        if pbar is not None:
            pbar.close()


def _terms_to_dataframe(terms: list[dict]) -> pd.DataFrame:
    rows = [{col: t.get(col) for col in _TERM_COLUMNS} for t in terms]
    return pd.DataFrame(rows, columns=list(_TERM_COLUMNS))


def get_ontology(ontology_id: str, *, timeout: int = 30) -> dict:
    """Fetch ontology metadata — version, term count, homepage, etc.

    Parameters
    ----------
    ontology_id : str
        Lowercase OLS slug, e.g. ``"mondo"``, ``"hp"``, ``"efo"``,
        ``"go"``, ``"so"``, ``"chebi"``.

    Returns
    -------
    dict
        Top-level keys include ``ontologyId``, ``version``,
        ``numberOfTerms``, ``numberOfProperties``, ``loaded``,
        ``updated``, plus the embedded ``config`` block (homepage,
        title, description, …).
    """
    return _get(f"{OLS_API_BASE_URL}/ontologies/{ontology_id}", timeout=timeout)


def get_term(ontology_id: str, term: str, *, timeout: int = 30) -> dict:
    """Fetch one term by CURIE (``"MONDO:0004975"``) or IRI.

    Returns the OLS term record verbatim — ``label``, ``description``,
    ``synonyms``, ``annotation``, ``is_obsolete``, ``has_children``,
    ``is_root``, ``term_replaced_by``, plus the IRI / OBO ID.
    """
    iri = curie_to_iri(term)
    encoded = _double_quote_iri(iri)
    return _get(
        f"{OLS_API_BASE_URL}/ontologies/{ontology_id}/terms/{encoded}",
        timeout=timeout,
    )


def _walk_relationship(
    ontology_id: str,
    term: str,
    relationship: str,
    *,
    size: int = _DEFAULT_PAGE_SIZE,
    timeout: int = 30,
) -> pd.DataFrame:
    iri = curie_to_iri(term)
    encoded = _double_quote_iri(iri)
    url = f"{OLS_API_BASE_URL}/ontologies/{ontology_id}/terms/{encoded}/{relationship}"
    terms = list(_paginate(url, params={"size": size}, timeout=timeout))
    return _terms_to_dataframe(terms)


def iter_terms(
    ontology_id: str,
    *,
    size: int = _DEFAULT_PAGE_SIZE,
    timeout: int = 60,
    progress: bool = True,
) -> Iterator[dict]:
    """Yield every term in ``ontology_id`` one row at a time.

    Walks OLS4's ``GET /ontologies/{ontology_id}/terms`` paginated
    endpoint. Each yielded dict carries the full per-term metadata OLS
    returns -- ``iri``, ``label``, ``description``, ``synonyms``,
    ``obo_id``, ``is_obsolete``, ``has_children``, ``is_root``, plus
    the HATEOAS ``_links`` block for fetching parents / children /
    descendants on demand. Generator-shaped so the caller can stream
    very large ontologies (SNOMED-CT has ~376k terms across ~750
    pages of 500) without materialising the whole list in memory.

    A tqdm page-counter is shown by default so multi-minute walks
    don't look like the process is hung; pass ``progress=False`` to
    suppress (e.g. inside pipelines that already have their own
    progress reporting).

    Use :func:`list_terms` instead when you want a DataFrame keyed on
    the canonical biodb columns (``obo_id``, ``label``, ``iri``,
    ``description``, ``synonyms``, ``is_obsolete``).

    Parameters
    ----------
    ontology_id : str
        Lowercase OLS slug, e.g. ``"snomed"``, ``"mondo"``, ``"hp"``.
    size : int, default 500
        Per-page size sent to OLS. Larger values cut round-trip count
        but each page is heavier; 500 is the OLS-recommended max.
    timeout : int, default 60
        Per-request timeout in seconds.
    progress : bool, default True
        Show a tqdm page-counter on stderr while walking. Set False
        to silence (caller has its own progress UX, or you're piping
        to a log file where the bar is noise).

    Yields
    ------
    dict
        One raw OLS term payload per yielded record.

    Examples
    --------
    >>> from biodb.ols import iter_terms
    >>> for t in iter_terms("mondo"):  # doctest: +SKIP
    ...     print(t["obo_id"], t["label"])
    """
    url = f"{OLS_API_BASE_URL}/ontologies/{ontology_id}/terms"
    yield from _paginate(
        url,
        params={"size": size},
        timeout=timeout,
        progress=progress,
        desc=f"OLS {ontology_id} terms",
    )


_VERSION_TOKEN_RE = re.compile(r"[^0-9A-Za-z._-]+")


def _ontology_version_token(ontology_id: str, *, timeout: int = 30) -> tuple[str, dict]:
    """Return ``(version_token, raw_metadata_dict)`` for ``ontology_id``.

    The token is a filesystem-safe slug derived from whichever of
    ``config.versionIri`` / ``config.version`` / ``updated`` /
    ``fileHash`` OLS exposes for the ontology -- prefer the most
    semantic one available so a SNOMED CT release tag survives in
    the cache filename, but always degrade to a hash if nothing
    structured is present. Never raises on missing fields: the
    fallback to a hash of the raw payload guarantees a token.
    """
    meta = get_ontology(ontology_id, timeout=timeout)
    cfg = meta.get("config") or {}
    candidates: list[str] = []
    if cfg.get("versionIri"):
        # SNOMED CT et al. -- this carries the upstream release id, the
        # most semantically meaningful version signal OLS exposes.
        candidates.append(str(cfg["versionIri"]))
    if cfg.get("version"):
        candidates.append(str(cfg["version"]))
    if meta.get("version"):
        candidates.append(str(meta["version"]))
    if meta.get("updated"):
        # OLS's own load timestamp -- a fair last-resort signal that
        # the indexed payload changed.
        candidates.append(str(meta["updated"]))
    if meta.get("fileHash"):
        candidates.append(str(meta["fileHash"]))

    if candidates:
        raw_token = candidates[0]
        # Filesystem-safe slug: replace anything that isn't [A-Za-z0-9._-]
        # with a dash, and cap length so very long IRIs don't blow up the
        # filename. Append an 8-char hash of the full raw_token to keep
        # the slug uniquely identifying even after truncation.
        slug = _VERSION_TOKEN_RE.sub("-", raw_token).strip("-")[:80]
        digest = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()[:8]
        return f"{slug}.{digest}", meta
    # No version signal at all -- hash the whole payload so we still
    # cache, but bust the cache the moment OLS changes anything.
    fallback = hashlib.sha256(repr(sorted(meta.items())).encode("utf-8")).hexdigest()[:16]
    return f"unknown-{fallback}", meta


def list_terms(
    ontology_id: str,
    *,
    size: int = _DEFAULT_PAGE_SIZE,
    timeout: int = 60,
    include_obsolete: bool = False,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
    progress: bool = True,
) -> pd.DataFrame:
    """Materialise every term in ``ontology_id`` as a DataFrame, with
    version-aware on-disk caching.

    The OMOP-CONCEPT.csv alternative for any OLS-indexed ontology:
    paginates ``GET /ontologies/{ontology}/terms`` and collects every
    page into a single DataFrame keyed on the canonical biodb columns
    (``obo_id``, ``label``, ``iri``, ``description``, ``synonyms``,
    ``is_obsolete``).

    **Caching is automatic** -- on first call the result is written to
    ``{cache_dir}/{ontology_id}/{version_token}.parquet`` (default
    ``cache_dir = ~/.cache/biodb/ols``). Subsequent calls in the same
    or another Python session re-read the parquet without hitting OLS,
    until OLS reports a new ontology version -- detected via
    ``config.versionIri`` / ``config.version`` / ``updated`` /
    ``fileHash`` (in priority order). A new version writes alongside
    the old one in the cache dir, so callers can rebuild from any
    historical release if they need to (e.g. for reproducibility on
    a paper).

    Parameters
    ----------
    ontology_id : str
        Lowercase OLS slug (e.g. ``"snomed"``, ``"mondo"``, ``"hp"``).
    size : int, default 500
        Per-page size sent to OLS. Larger values cut round-trip count
        but each page is heavier; 500 is the OLS-recommended max.
    timeout : int, default 30
    include_obsolete : bool, default False
        When False, deprecated / obsolete terms are dropped from the
        returned DataFrame. Flip this if you specifically need
        lifecycle bookkeeping.
    cache_dir : str or Path or None
        Cache root. ``None`` (default) uses :data:`DEFAULT_CACHE_DIR`
        (``~/.cache/biodb/ols``). Pass an explicit path to keep the
        cache somewhere project-local, or use a tmpdir in tests.
    refresh : bool, default False
        Force a fresh OLS walk even when a current-version parquet
        is already on disk. Useful for debugging OLS pagination
        regressions; never needed in normal operation because the
        version-token logic already busts the cache on upstream
        releases.
    progress : bool, default True
        Show a tqdm page-counter while walking. Suppressed
        automatically on cache hits (no walk happens).

    Returns
    -------
    pd.DataFrame
        One row per term, columns ``obo_id``, ``label``, ``iri``,
        ``description``, ``synonyms``, ``is_obsolete``.

    Examples
    --------
    >>> from biodb.ols import list_terms
    >>> mondo = list_terms("mondo")        # first call: walks OLS  # doctest: +SKIP
    >>> mondo_again = list_terms("mondo")  # second call: ~ms from parquet  # doctest: +SKIP
    >>> snomed = list_terms("snomed")      # ~5-10 min first run, then cached  # doctest: +SKIP
    """
    cache_root = Path(cache_dir).expanduser() if cache_dir is not None else DEFAULT_CACHE_DIR
    ontology_dir = cache_root / ontology_id
    version_token, _ = _ontology_version_token(ontology_id, timeout=timeout)
    # Encode include_obsolete in the filename so the two variants don't
    # collide (the underlying OLS data is the same, but the served
    # DataFrame differs).
    suffix = "-with-obsolete" if include_obsolete else ""
    cache_path = ontology_dir / f"{version_token}{suffix}.parquet"

    if cache_path.exists() and not refresh:
        logger.info("Loading cached %s terms from %s", ontology_id, cache_path)
        return pd.read_parquet(cache_path)

    logger.info(
        "Walking OLS4 /ontologies/%s/terms (version=%s, size=%d)",
        ontology_id,
        version_token,
        size,
    )
    terms = list(iter_terms(ontology_id, size=size, timeout=timeout, progress=progress))
    df = _terms_to_dataframe(terms)
    if not include_obsolete and "is_obsolete" in df.columns:
        df = df[~df["is_obsolete"].fillna(False)].reset_index(drop=True)

    ontology_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    logger.info("Cached %d %s terms -> %s", len(df), ontology_id, cache_path)
    return df


def get_descendants(
    ontology_id: str,
    term: str,
    *,
    size: int = _DEFAULT_PAGE_SIZE,
    timeout: int = 30,
) -> pd.DataFrame:
    """Return every transitive descendant of ``term`` as a DataFrame.

    Pagination is handled internally; for very large term subtrees
    (e.g. the GO root has tens of thousands of descendants) this will
    make multiple round-trips. Each row carries ``obo_id``, ``label``,
    ``iri``, ``description``, ``synonyms``, ``is_obsolete``.
    """
    return _walk_relationship(ontology_id, term, "descendants", size=size, timeout=timeout)


def get_ancestors(
    ontology_id: str,
    term: str,
    *,
    size: int = _DEFAULT_PAGE_SIZE,
    timeout: int = 30,
) -> pd.DataFrame:
    """Return every transitive ancestor of ``term`` as a DataFrame."""
    return _walk_relationship(ontology_id, term, "ancestors", size=size, timeout=timeout)


def get_children(
    ontology_id: str,
    term: str,
    *,
    size: int = _DEFAULT_PAGE_SIZE,
    timeout: int = 30,
) -> pd.DataFrame:
    """Return the **direct** (one-hop) children of ``term``."""
    return _walk_relationship(ontology_id, term, "children", size=size, timeout=timeout)


def get_parents(
    ontology_id: str,
    term: str,
    *,
    size: int = _DEFAULT_PAGE_SIZE,
    timeout: int = 30,
) -> pd.DataFrame:
    """Return the **direct** (one-hop) parents of ``term``."""
    return _walk_relationship(ontology_id, term, "parents", size=size, timeout=timeout)


def search(
    query: str,
    *,
    ontology: str | None = None,
    exact: bool = False,
    rows: int = 20,
    timeout: int = 30,
    **extra: Any,
) -> pd.DataFrame:
    """Solr-backed full-text search across one (or all) OLS ontologies.

    Parameters
    ----------
    query : str
        Free-text search string (label, synonym, definition, …).
    ontology : str, optional
        Restrict to one OLS slug (``"mondo"``, ``"hp"``, …). If omitted,
        searches every ontology OLS hosts.
    exact : bool, default False
        Match the query string exactly (skip Solr fuzziness).
    rows : int, default 20
        Max rows to return. The OLS hard cap is 1000.
    **extra
        Forwarded as query parameters — handy escape hatch for less-used
        OLS knobs like ``fieldList=``, ``childrenOf=``, ``allChildrenOf=``.

    Returns
    -------
    pandas.DataFrame
        Columns: ``obo_id``, ``label``, ``iri``, ``description``,
        ``synonyms``, ``ontology_name``.
    """
    params: dict[str, Any] = {"q": query, "rows": rows, "exact": str(exact).lower()}
    if ontology is not None:
        params["ontology"] = ontology
    params.update(extra)
    payload = _get(f"{OLS_API_BASE_URL}/search", params=params, timeout=timeout)
    docs = (payload.get("response") or {}).get("docs") or []
    cols = (*_TERM_COLUMNS, "ontology_name")
    rows_out = [{col: doc.get(col) for col in cols} for doc in docs]
    return pd.DataFrame(rows_out, columns=list(cols))


__all__ = [
    "OBO_PURL_BASE",
    "OLS_API_BASE_URL",
    "curie_to_iri",
    "get_ancestors",
    "get_children",
    "get_descendants",
    "get_ontology",
    "get_parents",
    "get_term",
    "iter_terms",
    "list_terms",
    "search",
]
