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


def test_curie_to_iri_expands_snomed_via_non_obo_template() -> None:
    """SNOMED uses ``http://snomed.info/id/<local>``, not the OBO PURL."""
    assert ols.curie_to_iri("SNOMED:38341003") == "http://snomed.info/id/38341003"


def test_curie_to_iri_is_case_insensitive_on_prefix() -> None:
    """A lower-case ``snomed:`` prefix should still hit the SNOMED template."""
    assert ols.curie_to_iri("snomed:38341003") == "http://snomed.info/id/38341003"


def test_curie_to_iri_expands_efo_to_ebi_url() -> None:
    """EFO IRIs live under ``ebi.ac.uk/efo/EFO_<local>``."""
    assert ols.curie_to_iri("EFO:0000400") == "http://www.ebi.ac.uk/efo/EFO_0000400"


def test_curie_to_iri_expands_orpha_to_ordo_url() -> None:
    """Orphanet rare-disease ontology — IRI under orpha.net/ORDO."""
    assert ols.curie_to_iri("ORPHA:733") == "http://www.orpha.net/ORDO/Orphanet_733"


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


# ---------------------------------------------------------------------------
# iter_terms + list_terms (the "every term in this ontology" path)
# ---------------------------------------------------------------------------


def test_iter_terms_walks_pagination_to_completion() -> None:
    """Two pages of two terms each → four yielded dicts."""
    base = f"{ols.OLS_API_BASE_URL}/ontologies/mondo/terms"
    page2_url = f"{base}?page=1&size=500"
    page1 = {
        "_embedded": {
            "terms": [
                _term_record("MONDO:1000001", "alpha"),
                _term_record("MONDO:1000002", "beta"),
            ]
        },
        "_links": {"next": {"href": page2_url}},
        "page": {"size": 500, "totalElements": 4, "totalPages": 2, "number": 0},
    }
    page2 = {
        "_embedded": {
            "terms": [
                _term_record("MONDO:1000003", "gamma"),
                _term_record("MONDO:1000004", "delta"),
            ]
        },
        "_links": {},
        "page": {"size": 500, "totalElements": 4, "totalPages": 2, "number": 1},
    }
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(responses.GET, base, json=page1, status=200)
        mock_resp.add(responses.GET, page2_url, json=page2, status=200)
        terms = list(ols.iter_terms("mondo"))
    assert [t["obo_id"] for t in terms] == [
        "MONDO:1000001",
        "MONDO:1000002",
        "MONDO:1000003",
        "MONDO:1000004",
    ]


def test_list_terms_caches_to_versioned_parquet(tmp_path) -> None:
    """A first call walks OLS + writes a parquet; a second call must
    re-read the parquet without hitting OLS at all."""
    base = f"{ols.OLS_API_BASE_URL}/ontologies/mondo/terms"
    onto_meta = {
        "ontologyId": "mondo",
        "config": {
            "versionIri": "http://purl.obolibrary.org/obo/mondo/releases/2026-05-05/mondo.owl",
            "version": None,
        },
        "updated": "2026-05-05T00:00:00",
        "fileHash": "abc123",
        "numberOfTerms": 2,
    }
    page1 = {
        "_embedded": {
            "terms": [
                _term_record("MONDO:1", "first"),
                _term_record("MONDO:2", "second"),
            ]
        },
        "_links": {},
        "page": {"size": 500, "totalElements": 2, "totalPages": 1, "number": 0},
    }
    with responses.RequestsMock(assert_all_requests_are_fired=False) as mock_resp:
        mock_resp.add(
            responses.GET, f"{ols.OLS_API_BASE_URL}/ontologies/mondo", json=onto_meta, status=200
        )
        mock_resp.add(responses.GET, base, json=page1, status=200)
        df1 = ols.list_terms("mondo", cache_dir=tmp_path)
        first_call_count = len(mock_resp.calls)
        # The second call should NOT consume the still-registered mocks
        # (i.e. should not hit OLS at all -- pure parquet read).
        df2 = ols.list_terms("mondo", cache_dir=tmp_path)
        # The ontology-metadata endpoint is still consulted (cheap;
        # needed to compute the version token), but the /terms walk
        # must NOT re-run. So total calls is exactly first_call_count + 1.
        assert len(mock_resp.calls) == first_call_count + 1, (
            "second list_terms call re-walked /terms; cache not honored"
        )
    assert len(df1) == 2
    pd.testing.assert_frame_equal(df1, df2)

    # Cache file lands under {cache_dir}/{ontology}/{version_token}.parquet.
    files = list((tmp_path / "mondo").glob("*.parquet"))
    assert len(files) == 1
    # The 2026-05-05 release tag must end up in the filename so a
    # human inspecting the cache dir can see which version they have.
    assert "2026-05-05" in files[0].name


def test_list_terms_busts_cache_on_new_ontology_version(tmp_path) -> None:
    """Two list_terms calls with different OLS-reported versionIri
    must end up with two parquets in the cache dir -- the old version
    is preserved (paper-trail / reproducibility) and the new release
    triggers a fresh walk."""
    base = f"{ols.OLS_API_BASE_URL}/ontologies/mondo/terms"
    onto_v1 = {
        "ontologyId": "mondo",
        "config": {
            "versionIri": "http://purl.obolibrary.org/obo/mondo/releases/2026-05-05/mondo.owl",
        },
    }
    onto_v2 = {
        "ontologyId": "mondo",
        "config": {
            "versionIri": "http://purl.obolibrary.org/obo/mondo/releases/2026-06-01/mondo.owl",
        },
    }
    page = {
        "_embedded": {"terms": [_term_record("MONDO:1", "x")]},
        "_links": {},
        "page": {"size": 500, "totalElements": 1, "totalPages": 1, "number": 0},
    }
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(
            responses.GET, f"{ols.OLS_API_BASE_URL}/ontologies/mondo", json=onto_v1, status=200
        )
        mock_resp.add(responses.GET, base, json=page, status=200)
        ols.list_terms("mondo", cache_dir=tmp_path)
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(
            responses.GET, f"{ols.OLS_API_BASE_URL}/ontologies/mondo", json=onto_v2, status=200
        )
        mock_resp.add(responses.GET, base, json=page, status=200)
        ols.list_terms("mondo", cache_dir=tmp_path)

    files = sorted(p.name for p in (tmp_path / "mondo").glob("*.parquet"))
    assert len(files) == 2
    assert any("2026-05-05" in n for n in files)
    assert any("2026-06-01" in n for n in files)


def test_list_terms_refresh_forces_walk_even_with_cache(tmp_path) -> None:
    """``refresh=True`` should hit OLS again even when a current-version
    parquet is already on disk -- escape hatch for debugging pagination
    regressions."""
    base = f"{ols.OLS_API_BASE_URL}/ontologies/mondo/terms"
    onto = {"ontologyId": "mondo", "config": {"versionIri": "v1"}}
    page = {
        "_embedded": {"terms": [_term_record("MONDO:1", "x")]},
        "_links": {},
        "page": {"size": 500, "totalElements": 1, "totalPages": 1, "number": 0},
    }
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(
            responses.GET, f"{ols.OLS_API_BASE_URL}/ontologies/mondo", json=onto, status=200
        )
        mock_resp.add(responses.GET, base, json=page, status=200)
        ols.list_terms("mondo", cache_dir=tmp_path)

    # With refresh=True we must call BOTH endpoints again.
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(
            responses.GET, f"{ols.OLS_API_BASE_URL}/ontologies/mondo", json=onto, status=200
        )
        mock_resp.add(responses.GET, base, json=page, status=200)
        ols.list_terms("mondo", cache_dir=tmp_path, refresh=True)
        # Both calls landed -- responses raises on any unconsumed mock.
        assert len(mock_resp.calls) == 2


def test_iter_terms_progress_emits_tqdm_bar(monkeypatch) -> None:
    """When ``progress=True``, ``iter_terms`` must hand the page count
    to tqdm and call ``.update(1)`` per page -- otherwise long walks
    look hung to the user. Stub tqdm so the test doesn't depend on
    the bar's rendering."""
    base = f"{ols.OLS_API_BASE_URL}/ontologies/mondo/terms"
    page2_url = f"{base}?page=1&size=500"
    page1 = {
        "_embedded": {"terms": [_term_record("MONDO:1", "x")]},
        "_links": {"next": {"href": page2_url}},
        "page": {"size": 500, "totalElements": 2, "totalPages": 2, "number": 0},
    }
    page2 = {
        "_embedded": {"terms": [_term_record("MONDO:2", "y")]},
        "_links": {},
        "page": {"size": 500, "totalElements": 2, "totalPages": 2, "number": 1},
    }

    captured: dict = {"total": None, "updates": 0, "closed": False, "desc": None}

    class _StubTqdm:
        def __init__(self, *, total, desc, unit):
            captured["total"] = total
            captured["desc"] = desc

        def update(self, n: int = 1) -> None:
            captured["updates"] += n

        def close(self) -> None:
            captured["closed"] = True

    import tqdm.auto

    monkeypatch.setattr(tqdm.auto, "tqdm", _StubTqdm)

    with responses.RequestsMock() as mock_resp:
        mock_resp.add(responses.GET, base, json=page1, status=200)
        mock_resp.add(responses.GET, page2_url, json=page2, status=200)
        terms = list(ols.iter_terms("mondo", progress=True))

    assert len(terms) == 2
    assert captured["total"] == 2  # page.totalPages from the first response
    assert captured["updates"] == 2  # one update per page
    assert captured["closed"] is True
    assert captured["desc"] == "OLS mondo terms"


def test_iter_terms_progress_off_does_not_construct_tqdm(monkeypatch) -> None:
    """``progress=False`` must skip tqdm entirely -- important for
    pipelines that pipe to log files where the bar would be noise."""
    base = f"{ols.OLS_API_BASE_URL}/ontologies/mondo/terms"
    page = {
        "_embedded": {"terms": [_term_record("MONDO:1", "x")]},
        "_links": {},
        "page": {"size": 500, "totalElements": 1, "totalPages": 1, "number": 0},
    }

    constructed = {"count": 0}

    class _BoomTqdm:
        def __init__(self, *a, **kw):
            constructed["count"] += 1

    import tqdm.auto

    monkeypatch.setattr(tqdm.auto, "tqdm", _BoomTqdm)

    with responses.RequestsMock() as mock_resp:
        mock_resp.add(responses.GET, base, json=page, status=200)
        list(ols.iter_terms("mondo", progress=False))

    assert constructed["count"] == 0


def test_get_retries_on_transient_5xx(monkeypatch) -> None:
    """``_get`` must retry on transient 5xx / connection errors -- a
    long paginated walk hits enough flakes that no retries means the
    walk effectively never completes."""
    import time as _time

    monkeypatch.setattr(_time, "sleep", lambda _s: None)
    base = f"{ols.OLS_API_BASE_URL}/ontologies/mondo"
    with responses.RequestsMock() as mock_resp:
        # First two attempts: 503; third: 200.
        mock_resp.add(responses.GET, base, status=503)
        mock_resp.add(responses.GET, base, status=503)
        mock_resp.add(responses.GET, base, json={"ontologyId": "mondo"}, status=200)
        out = ols._get(base, max_retries=5, backoff_s=0)
    assert out == {"ontologyId": "mondo"}


def test_get_does_not_retry_on_4xx(monkeypatch) -> None:
    """A 4xx is a request error, not flakiness -- retrying just burns
    network. The first 4xx must raise immediately."""
    import time as _time

    calls = {"n": 0}

    def _fake_sleep(_s):
        calls["n"] += 1

    monkeypatch.setattr(_time, "sleep", _fake_sleep)
    base = f"{ols.OLS_API_BASE_URL}/ontologies/notreal"
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(responses.GET, base, status=404)
        with pytest.raises(requests.HTTPError):
            ols._get(base, max_retries=5, backoff_s=0)
    # ``time.sleep`` never called -- no backoff happened.
    assert calls["n"] == 0


def test_list_terms_drops_obsolete_terms_by_default(tmp_path) -> None:
    base = f"{ols.OLS_API_BASE_URL}/ontologies/mondo/terms"
    onto = {"ontologyId": "mondo", "config": {"versionIri": "vobsolete"}}
    obsolete = _term_record("MONDO:DEAD", "deprecated")
    obsolete["is_obsolete"] = True
    page = {
        "_embedded": {
            "terms": [_term_record("MONDO:1", "alive"), obsolete],
        },
        "_links": {},
        "page": {"size": 500, "totalElements": 2, "totalPages": 1, "number": 0},
    }
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(
            responses.GET, f"{ols.OLS_API_BASE_URL}/ontologies/mondo", json=onto, status=200
        )
        mock_resp.add(responses.GET, base, json=page, status=200)
        df = ols.list_terms("mondo", cache_dir=tmp_path)
    assert set(df["obo_id"]) == {"MONDO:1"}

    # ``include_obsolete=True`` keeps both rows -- and writes to a
    # separate cache file so the two variants don't collide.
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(
            responses.GET, f"{ols.OLS_API_BASE_URL}/ontologies/mondo", json=onto, status=200
        )
        mock_resp.add(responses.GET, base, json=page, status=200)
        df_all = ols.list_terms("mondo", cache_dir=tmp_path, include_obsolete=True)
    assert set(df_all["obo_id"]) == {"MONDO:1", "MONDO:DEAD"}
    # Two distinct parquets on disk -- active-only vs with-obsolete.
    assert len(list((tmp_path / "mondo").glob("*.parquet"))) == 2


# ---------------------------------------------------------------------------
# find_terms / find_term — ranked OLS lookup
# ---------------------------------------------------------------------------


def _search_payload(*rows: dict) -> dict:
    """Build a fake OLS /search response with the supplied docs."""
    return {"response": {"docs": list(rows), "numFound": len(rows)}}


def test_find_terms_promotes_exact_label_above_solr_rank() -> None:
    """Solr ranked "Breast carcinoma in situ" first, but the exact-label
    "breast carcinoma" hit should win after re-ranking."""
    docs = [
        {
            "obo_id": "EFO:0000999",
            "label": "Breast carcinoma in situ",
            "iri": "http://www.ebi.ac.uk/efo/EFO_0000999",
            "description": None,
            "synonyms": [],
            "is_obsolete": False,
            "ontology_name": "efo",
        },
        {
            "obo_id": "EFO:0000305",
            "label": "breast carcinoma",
            "iri": "http://www.ebi.ac.uk/efo/EFO_0000305",
            "description": None,
            "synonyms": ["mammary carcinoma"],
            "is_obsolete": False,
            "ontology_name": "efo",
        },
    ]
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(
            responses.GET,
            f"{ols.OLS_API_BASE_URL}/search",
            json=_search_payload(*docs),
            status=200,
        )
        df = ols.find_terms("breast carcinoma", ontology="efo", top_k=5)
    assert df.iloc[0]["obo_id"] == "EFO:0000305"
    assert df.iloc[0]["match_quality"] == 4  # exact label
    # "Breast carcinoma in situ" starts with the query → prefix match (2).
    assert df.iloc[1]["match_quality"] == 2


def test_find_terms_synonym_match_beats_substring() -> None:
    """An exact synonym hit (quality 3) should outrank a substring
    hit on label (quality 1)."""
    docs = [
        {
            "obo_id": "EFO:1",
            "label": "Tumor of breast — late stage",
            "iri": "x",
            "description": None,
            "synonyms": [],
            "is_obsolete": False,
            "ontology_name": "efo",
        },
        {
            "obo_id": "EFO:2",
            "label": "Some other label",
            "iri": "y",
            "description": None,
            "synonyms": ["breast tumor", "neoplasm of breast"],
            "is_obsolete": False,
            "ontology_name": "efo",
        },
    ]
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(
            responses.GET,
            f"{ols.OLS_API_BASE_URL}/search",
            json=_search_payload(*docs),
            status=200,
        )
        df = ols.find_terms("breast tumor", ontology="efo", top_k=5)
    assert df.iloc[0]["obo_id"] == "EFO:2"
    assert df.iloc[0]["match_quality"] == 3


def test_find_terms_empty_when_ols_returns_nothing() -> None:
    """An empty OLS response yields an empty DataFrame with the
    match_quality column still present (typed)."""
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(
            responses.GET,
            f"{ols.OLS_API_BASE_URL}/search",
            json=_search_payload(),
            status=200,
        )
        df = ols.find_terms("nonsense xyzzy", top_k=5)
    assert df.empty
    assert "match_quality" in df.columns


def test_find_term_returns_best_hit_or_none() -> None:
    docs = [
        {
            "obo_id": "EFO:0000305",
            "label": "breast carcinoma",
            "iri": "x",
            "description": None,
            "synonyms": [],
            "is_obsolete": False,
            "ontology_name": "efo",
        },
    ]
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(
            responses.GET,
            f"{ols.OLS_API_BASE_URL}/search",
            json=_search_payload(*docs),
            status=200,
        )
        hit = ols.find_term("breast carcinoma", ontology="efo")
    assert hit is not None
    assert hit["obo_id"] == "EFO:0000305"

    with responses.RequestsMock() as mock_resp:
        mock_resp.add(
            responses.GET,
            f"{ols.OLS_API_BASE_URL}/search",
            json=_search_payload(),
            status=200,
        )
        assert ols.find_term("nonsense xyzzy") is None


# ---------------------------------------------------------------------------
# ontology_id_from_curie — namespace → OLS slug
# ---------------------------------------------------------------------------


def test_ontology_id_from_curie_efo() -> None:
    assert ols.ontology_id_from_curie("EFO:0000311") == "efo"
    assert ols.ontology_id_from_curie("EFO_0000311") == "efo"


def test_ontology_id_from_curie_mondo() -> None:
    assert ols.ontology_id_from_curie("MONDO:0007254") == "mondo"


def test_ontology_id_from_curie_snomed_alias() -> None:
    """SCTID is the alternate prefix MEDLINE / UMLS use for SNOMED CT."""
    assert ols.ontology_id_from_curie("SCTID:38341003") == "snomed"
    assert ols.ontology_id_from_curie("SNOMED:38341003") == "snomed"


def test_ontology_id_from_curie_orphanet_to_ordo() -> None:
    """Orphanet IDs live under the ORDO slug at OLS."""
    assert ols.ontology_id_from_curie("ORPHA:733") == "ordo"


def test_ontology_id_from_curie_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="not a CURIE"):
        ols.ontology_id_from_curie("not-a-curie")
