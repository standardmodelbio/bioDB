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

import logging
from collections.abc import Iterator
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests

logger = logging.getLogger(__name__)

OLS_API_BASE_URL = "https://www.ebi.ac.uk/ols4/api"
"""OLS4 REST API root."""

OBO_PURL_BASE = "http://purl.obolibrary.org/obo"
"""PURL root for OBO Foundry ontologies. Used to expand CURIEs."""

_DEFAULT_PAGE_SIZE = 500
"""OLS paginates everything. 500 is the largest size that still keeps
each response well under a megabyte."""

_TERM_COLUMNS = ("obo_id", "label", "iri", "description", "synonyms", "is_obsolete")
"""Columns surfaced from OLS term records in the DataFrame return path."""


def curie_to_iri(curie_or_iri: str) -> str:
    """Expand an OBO CURIE (e.g. ``"MONDO:0004975"``) to its PURL IRI.

    Anything that already looks like an IRI (starts with ``http://`` or
    ``https://``) is returned unchanged — useful for non-OBO ontologies
    like EFO whose IRIs don't follow the standard PURL pattern.
    """
    if curie_or_iri.startswith(("http://", "https://")):
        return curie_or_iri
    if ":" not in curie_or_iri:
        raise ValueError(
            f"{curie_or_iri!r} is neither an IRI nor a CURIE (expected ``PREFIX:local``)."
        )
    prefix, local = curie_or_iri.split(":", 1)
    return f"{OBO_PURL_BASE}/{prefix}_{local}"


def _double_quote_iri(iri: str) -> str:
    """OLS expects the IRI path segment to be **double**-URL-encoded.

    The first ``quote`` turns ``/`` and ``:`` into ``%2F`` / ``%3A``; the
    second turns those ``%`` characters into ``%25``. Without the second
    pass the OLS Spring router rejects the request as a malformed path.
    """
    return quote(quote(iri, safe=""), safe="")


def _get(url: str, params: dict | None = None, timeout: int = 30) -> dict:
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _paginate(url: str, params: dict | None = None, timeout: int = 30) -> Iterator[dict]:
    """Walk OLS's HAL-style paginated endpoints (yielding each term)."""
    next_url: str | None = url
    next_params: dict | None = params
    while next_url is not None:
        payload = _get(next_url, params=next_params, timeout=timeout)
        yield from (payload.get("_embedded") or {}).get("terms") or []
        next_url = (payload.get("_links") or {}).get("next", {}).get("href")
        # The `next` href already carries pagination params; don't duplicate.
        next_params = None


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
    "search",
]
