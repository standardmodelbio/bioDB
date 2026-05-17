"""Targeted OpenTargets Platform GraphQL queries (API mode).

Complements :mod:`biodb.opentargets`'s bulk-download (FTP-mode) helpers with
one-record-at-a-time GraphQL lookups. Use this when you need a fresh
single-target / single-disease / single-drug / single-variant payload and
don't want to download an entire Parquet release.

Endpoint: https://api.platform.opentargets.org/api/v4/graphql

Examples
--------
>>> from biodb.opentargets_graphql import query_target
>>> brca1 = query_target("ENSG00000012048")  # doctest: +SKIP
>>> brca1["approvedSymbol"]  # doctest: +SKIP
'BRCA1'
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

OT_GRAPHQL_API = "https://api.platform.opentargets.org/api/v4/graphql"
"""Open Targets Platform v4 GraphQL endpoint."""

_DEFAULT_TIMEOUT_S: float = 30.0
_DEFAULT_MAX_RETRIES: int = 3
_DEFAULT_BACKOFF_S: float = 1.0


def graphql_post(
    query: str,
    variables: dict[str, Any],
    *,
    endpoint: str = OT_GRAPHQL_API,
    client: httpx.Client | None = None,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    backoff_s: float = _DEFAULT_BACKOFF_S,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """POST a GraphQL query with exponential-backoff retries; return the ``data`` block.

    Parameters
    ----------
    query : str
        GraphQL query string.
    variables : dict
        GraphQL variables.
    endpoint : str
        GraphQL endpoint URL. Defaults to OT Platform v4.
    client : httpx.Client, optional
        Reusable HTTP client for connection pooling. If ``None``, one is
        created per call.
    max_retries : int, default 3
        Maximum retry attempts on transient failures.
    backoff_s : float, default 1.0
        Initial backoff in seconds; doubles each retry.
    timeout_s : float, default 30.0
        Per-request timeout.

    Returns
    -------
    dict
        The ``data`` field of the GraphQL response.

    Raises
    ------
    httpx.HTTPError
        If all retries fail.
    RuntimeError
        If the GraphQL response carries an ``errors`` block.

    Examples
    --------
    >>> graphql_post(  # doctest: +SKIP
    ...     "query Q($id: String!) { target(ensemblId: $id) { approvedSymbol } }",
    ...     {"id": "ENSG00000012048"},
    ... )
    {'target': {'approvedSymbol': 'BRCA1'}}
    """
    owned_client = client is None
    client = client or httpx.Client(timeout=timeout_s)
    last_err: Exception | None = None
    try:
        for attempt in range(max_retries):
            try:
                resp = client.post(
                    endpoint, json={"query": query, "variables": variables}, timeout=timeout_s
                )
                resp.raise_for_status()
                body = resp.json()
                if errors := body.get("errors"):
                    raise RuntimeError(f"OpenTargets GraphQL error: {errors}")
                return body["data"]
            except (httpx.HTTPError, RuntimeError) as exc:
                last_err = exc
                if attempt < max_retries - 1:
                    time.sleep(backoff_s * (2**attempt))
        assert last_err is not None
        raise last_err
    finally:
        if owned_client:
            client.close()


_TARGET_QUERY = """
query Target($ensemblId: String!) {
  target(ensemblId: $ensemblId) {
    id
    approvedSymbol
    approvedName
    biotype
    functionDescriptions
    genomicLocation { chromosome start end strand }
    proteinIds { id source }
    symbolSynonyms { label source }
    nameSynonyms { label source }
    subcellularLocations { location source }
    pathways { pathway pathwayId topLevelTerm }
  }
}
"""
# Note: the ``go { term aspect source }`` field was removed from
# Open Targets' Target GraphQL type sometime before 2026-05; pulling
# Gene Ontology annotations now requires a separate API. Dropping the
# field here keeps the query well-formed against the current schema.

_DISEASE_QUERY = """
query Disease($efoId: String!) {
  disease(efoId: $efoId) {
    id
    name
    description
    synonyms { terms relation }
    therapeuticAreas { id name }
  }
}
"""

_DRUG_QUERY = """
query Drug($chemblId: String!) {
  drug(chemblId: $chemblId) {
    id
    name
    tradeNames
    synonyms
    drugType
    isApproved
    maximumClinicalTrialPhase
    yearOfFirstApproval
    description
  }
}
"""

_VARIANT_QUERY = """
query Variant($variantId: String!) {
  variant(variantId: $variantId) {
    id
    chromosome
    position
    referenceAllele
    alternateAllele
    rsIds
    mostSevereConsequence
  }
}
"""


def query_target(
    ensembl_id: str,
    *,
    client: httpx.Client | None = None,
) -> dict[str, Any] | None:
    """Fetch a single target by Ensembl gene ID.

    Parameters
    ----------
    ensembl_id : str
        Ensembl stable gene ID (e.g. ``"ENSG00000012048"``).
    client : httpx.Client, optional
        Reusable HTTP client.

    Returns
    -------
    dict or None
        The ``target`` object from the GraphQL response, or ``None`` if not found.

    Examples
    --------
    >>> brca1 = query_target("ENSG00000012048")  # doctest: +SKIP
    >>> brca1["approvedSymbol"]  # doctest: +SKIP
    'BRCA1'
    """
    data = graphql_post(_TARGET_QUERY, {"ensemblId": ensembl_id}, client=client)
    return data.get("target")


def query_disease(
    efo_id: str,
    *,
    client: httpx.Client | None = None,
) -> dict[str, Any] | None:
    """Fetch a single disease by EFO/MONDO ID.

    Parameters
    ----------
    efo_id : str
        Disease ID in OT's internal underscored form (``"EFO_0000305"``,
        ``"MONDO_0007254"``, etc.). Colons are accepted and normalized.
    client : httpx.Client, optional

    Returns
    -------
    dict or None

    Examples
    --------
    >>> bc = query_disease("MONDO_0007254")  # doctest: +SKIP
    >>> bc["name"]  # doctest: +SKIP
    'breast carcinoma'
    """
    efo_id = efo_id.replace(":", "_")
    data = graphql_post(_DISEASE_QUERY, {"efoId": efo_id}, client=client)
    return data.get("disease")


def query_drug(
    chembl_id: str,
    *,
    client: httpx.Client | None = None,
) -> dict[str, Any] | None:
    """Fetch a single drug by ChEMBL ID.

    Parameters
    ----------
    chembl_id : str
        ChEMBL drug ID (``"CHEMBL25"``).
    client : httpx.Client, optional

    Returns
    -------
    dict or None

    Examples
    --------
    >>> aspirin = query_drug("CHEMBL25")  # doctest: +SKIP
    >>> aspirin["name"]  # doctest: +SKIP
    'ASPIRIN'
    """
    data = graphql_post(_DRUG_QUERY, {"chemblId": chembl_id}, client=client)
    return data.get("drug")


def query_variant(
    variant_id: str,
    *,
    client: httpx.Client | None = None,
) -> dict[str, Any] | None:
    """Fetch a single variant by OT variant ID.

    Parameters
    ----------
    variant_id : str
        OT variant ID (``"chr1_55039774_T_C"``).
    client : httpx.Client, optional

    Returns
    -------
    dict or None

    Examples
    --------
    >>> v = query_variant("chr1_55039774_T_C")  # doctest: +SKIP
    """
    data = graphql_post(_VARIANT_QUERY, {"variantId": variant_id}, client=client)
    return data.get("variant")
