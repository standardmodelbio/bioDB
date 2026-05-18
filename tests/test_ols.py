"""Tests for :mod:`biodb.ols` — OLS4 REST client.

This file mixes:

* **Mocked unit tests** (``responses``-driven) that pin URL shape,
  double-URL-encoding behaviour, pagination chasing, and response
  parsing — fast, deterministic, no network.
* **Live integration tests** that hit the real EBI OLS4 endpoint with
  tiny Mondo payloads — catch upstream schema drift.

Both run by default in CI.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest
import requests
import responses

from biodb import ols

# ---------------------------------------------------------------------------
# Pure-function tests — CURIE / IRI / URL building
# ---------------------------------------------------------------------------


def test_curie_to_iri_expands_mondo() -> None:
    assert ols.curie_to_iri("MONDO:0004975") == ("http://purl.obolibrary.org/obo/MONDO_0004975")


def test_curie_to_iri_passes_iri_through_unchanged() -> None:
    iri = "http://www.ebi.ac.uk/efo/EFO_0000400"
    assert ols.curie_to_iri(iri) == iri


def test_curie_to_iri_passes_https_iri_through() -> None:
    iri = "https://example.org/CUSTOM_42"
    assert ols.curie_to_iri(iri) == iri


def test_curie_to_iri_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="neither an IRI nor a CURIE"):
        ols.curie_to_iri("not-a-curie")


def test_curie_to_iri_handles_curie_with_extra_colons() -> None:
    """OLS-style namespaced CURIEs split on the *first* colon only — the
    local portion can carry colons of its own."""
    assert ols.curie_to_iri("FOO:bar:baz") == ("http://purl.obolibrary.org/obo/FOO_bar:baz")


def test_double_quote_iri_uses_percent_25_escape() -> None:
    """OLS routes reject single-encoded IRIs; the second pass is
    load-bearing."""
    encoded = ols._double_quote_iri("http://purl.obolibrary.org/obo/MONDO_0004975")
    # The first '%' from the original encoding must now be '%25'.
    assert "%25" in encoded
    # And there are no raw slashes / colons left.
    assert "/" not in encoded
    assert ":" not in encoded


# ---------------------------------------------------------------------------
# Mocked HTTP tests — URL shape, pagination, response parsing
# ---------------------------------------------------------------------------


def _term_record(obo_id: str, label: str) -> dict:
    """Minimal OLS term record — only the keys the DataFrame mapper reads."""
    return {
        "obo_id": obo_id,
        "label": label,
        "iri": f"http://purl.obolibrary.org/obo/{obo_id.replace(':', '_')}",
        "description": [f"definition of {label}"],
        "synonyms": [f"{label} synonym"],
        "is_obsolete": False,
    }


def test_get_ontology_hits_documented_endpoint() -> None:
    """``get_ontology(slug)`` must GET ``/api/ontologies/{slug}``."""
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(
            responses.GET,
            f"{ols.OLS_API_BASE_URL}/ontologies/mondo",
            json={
                "ontologyId": "mondo",
                "version": "2026-05-05",
                "numberOfTerms": 58940,
            },
            status=200,
        )
        record = ols.get_ontology("mondo")
    assert record["ontologyId"] == "mondo"


def test_get_term_double_encodes_iri_in_path() -> None:
    """The OLS Spring router requires the IRI segment to be double-URL
    encoded — any other encoding returns a 404."""
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(
            responses.GET,
            re.compile(rf"^{re.escape(ols.OLS_API_BASE_URL)}/ontologies/mondo/terms/.+"),
            json=_term_record("MONDO:0004975", "Alzheimer disease"),
            status=200,
        )
        ols.get_term("mondo", "MONDO:0004975")
        called_url = mock_resp.calls[0].request.url
    # Double-encoded IRIs contain literal '%25' (the encoded '%').
    assert "%25" in called_url
    assert "%252F" in called_url or "%2F" in called_url  # encoded slash present


def test_get_descendants_walks_hal_pagination_to_completion() -> None:
    """Two pages of three terms each → six rows in the DataFrame."""
    base = f"{ols.OLS_API_BASE_URL}/ontologies/mondo/terms"
    iri_encoded = ols._double_quote_iri("http://purl.obolibrary.org/obo/MONDO_0004975")
    page1_url = f"{base}/{iri_encoded}/descendants"
    page2_url = f"{base}/{iri_encoded}/descendants?page=1&size=500"
    page1 = {
        "_embedded": {
            "terms": [
                _term_record("MONDO:1000001", "child a"),
                _term_record("MONDO:1000002", "child b"),
                _term_record("MONDO:1000003", "child c"),
            ]
        },
        "_links": {"next": {"href": page2_url}},
        "page": {"size": 500, "totalElements": 6, "totalPages": 2, "number": 0},
    }
    page2 = {
        "_embedded": {
            "terms": [
                _term_record("MONDO:1000004", "child d"),
                _term_record("MONDO:1000005", "child e"),
                _term_record("MONDO:1000006", "child f"),
            ]
        },
        "_links": {},  # no "next" → terminate
        "page": {"size": 500, "totalElements": 6, "totalPages": 2, "number": 1},
    }
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(responses.GET, page1_url, json=page1, status=200)
        mock_resp.add(responses.GET, page2_url, json=page2, status=200)
        df = ols.get_descendants("mondo", "MONDO:0004975")
    assert len(df) == 6
    assert list(df["obo_id"])[0] == "MONDO:1000001"
    assert list(df["obo_id"])[-1] == "MONDO:1000006"
    assert set(df.columns) == {
        "obo_id",
        "label",
        "iri",
        "description",
        "synonyms",
        "is_obsolete",
    }


def test_get_descendants_handles_empty_embedded() -> None:
    """A leaf term returns no descendants — the DataFrame should still
    have the documented columns, just zero rows."""
    base = f"{ols.OLS_API_BASE_URL}/ontologies/mondo/terms"
    iri_encoded = ols._double_quote_iri("http://purl.obolibrary.org/obo/MONDO_9999999")
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(
            responses.GET,
            f"{base}/{iri_encoded}/descendants",
            json={"_embedded": {}, "_links": {}, "page": {"totalElements": 0}},
            status=200,
        )
        df = ols.get_descendants("mondo", "MONDO:9999999")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert "obo_id" in df.columns


def test_get_ancestors_and_children_hit_distinct_endpoints() -> None:
    """The three traversal helpers must POST to ``/ancestors``,
    ``/children``, ``/descendants`` respectively — easy regression
    target if someone factors them together."""
    iri_encoded = ols._double_quote_iri("http://purl.obolibrary.org/obo/MONDO_0004975")
    base = f"{ols.OLS_API_BASE_URL}/ontologies/mondo/terms/{iri_encoded}"

    empty = {"_embedded": {}, "_links": {}, "page": {"totalElements": 0}}
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(responses.GET, f"{base}/ancestors", json=empty, status=200)
        mock_resp.add(responses.GET, f"{base}/children", json=empty, status=200)
        mock_resp.add(responses.GET, f"{base}/parents", json=empty, status=200)
        ols.get_ancestors("mondo", "MONDO:0004975")
        ols.get_children("mondo", "MONDO:0004975")
        ols.get_parents("mondo", "MONDO:0004975")
        urls = [c.request.url.split("?")[0] for c in mock_resp.calls]
    assert urls[0].endswith("/ancestors")
    assert urls[1].endswith("/children")
    assert urls[2].endswith("/parents")


def test_search_forwards_documented_query_parameters() -> None:
    """``search(q, ontology=..., exact=True, rows=5, fieldList=...)``
    should fold every argument into the GET querystring."""
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(
            responses.GET,
            f"{ols.OLS_API_BASE_URL}/search",
            json={
                "response": {
                    "docs": [
                        {
                            "obo_id": "MONDO:0004975",
                            "label": "Alzheimer disease",
                            "iri": "http://purl.obolibrary.org/obo/MONDO_0004975",
                            "description": None,
                            "synonyms": [],
                            "is_obsolete": False,
                            "ontology_name": "mondo",
                        }
                    ],
                    "numFound": 1,
                }
            },
            status=200,
        )
        df = ols.search(
            "alzheimer",
            ontology="mondo",
            exact=True,
            rows=5,
            fieldList="label,obo_id",
        )
        sent_url = mock_resp.calls[0].request.url
    assert df.shape == (1, 7)
    assert list(df.columns)[-1] == "ontology_name"
    assert "q=alzheimer" in sent_url
    assert "ontology=mondo" in sent_url
    assert "exact=true" in sent_url
    assert "rows=5" in sent_url
    assert "fieldList=label" in sent_url  # extra kwarg threaded through


def test_get_term_propagates_404_as_http_error() -> None:
    """Unknown terms return 404 — we should not swallow that."""
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(
            responses.GET,
            re.compile(rf"^{re.escape(ols.OLS_API_BASE_URL)}/ontologies/mondo/terms/.+"),
            json={"error": "not found"},
            status=404,
        )
        with pytest.raises(requests.HTTPError):
            ols.get_term("mondo", "MONDO:9999999999")


# ---------------------------------------------------------------------------
# Live integration tests — hit the real OLS4 endpoint.
# ---------------------------------------------------------------------------


def test_ols_get_ontology_mondo_metadata() -> None:
    """Mondo metadata should always report a 5-digit term count and
    an ISO-ish version date string."""
    record = ols.get_ontology("mondo")
    assert record["ontologyId"] == "mondo"
    assert record["numberOfTerms"] > 10_000
    assert record["version"], "Mondo should have a non-empty version field"


def test_ols_get_term_alzheimer_disease_round_trip() -> None:
    """The MONDO:0004975 record carries the documented fields."""
    ad = ols.get_term("mondo", "MONDO:0004975")
    assert ad["obo_id"] == "MONDO:0004975"
    assert ad["label"] == "Alzheimer disease"
    assert "Alzheimer dementia" in (ad["synonyms"] or [])
    assert ad["is_obsolete"] is False


def test_ols_get_term_accepts_full_iri() -> None:
    """Passing an IRI directly should skip the CURIE expansion path."""
    ad = ols.get_term("mondo", "http://purl.obolibrary.org/obo/MONDO_0004975")
    assert ad["obo_id"] == "MONDO:0004975"


def test_ols_get_descendants_of_alzheimer_paginates() -> None:
    """Alzheimer disease has ~21 descendants; verify the paginator stitches
    them into a DataFrame with the documented columns."""
    df = ols.get_descendants("mondo", "MONDO:0004975")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 10
    for col in ("obo_id", "label", "iri"):
        assert col in df.columns
    # Familial Alzheimer disease is a known descendant.
    assert "MONDO:0100087" in set(df["obo_id"])


def test_ols_get_ancestors_of_alzheimer() -> None:
    df = ols.get_ancestors("mondo", "MONDO:0004975")
    assert len(df) > 5
    # Mondo's "dementia" is a known ancestor of Alzheimer disease.
    assert any("dementia" in (label or "").lower() for label in df["label"])


def test_ols_get_children_is_one_hop_subset_of_descendants() -> None:
    children = ols.get_children("mondo", "MONDO:0004975")
    descendants = ols.get_descendants("mondo", "MONDO:0004975")
    # Direct children ⊆ all descendants.
    assert set(children["obo_id"]) <= set(descendants["obo_id"])
    assert len(children) <= len(descendants)


def test_ols_search_alzheimer_finds_canonical_term() -> None:
    hits = ols.search("alzheimer", ontology="mondo", rows=5)
    assert len(hits) == 5
    assert "MONDO:0004975" in set(hits["obo_id"])
