"""Tests for :mod:`biodb.opentargets_graphql`.

Uses :class:`httpx.MockTransport` to intercept GraphQL POSTs so the
tests never touch the live Open Targets API. The two ``network``-marked
smoke tests at the bottom hit the real endpoint on demand.
"""

from __future__ import annotations

import time

import httpx
import pytest

from biodb import opentargets_graphql as gql

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _client_returning(payload: dict | list, status_code: int = 200) -> httpx.Client:
    """Build an httpx.Client wired to a MockTransport that returns ``payload``."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_module_imports() -> None:
    assert gql.__name__ == "biodb.opentargets_graphql"


def test_endpoint_constant() -> None:
    assert gql.OT_GRAPHQL_API.endswith("/graphql")
    assert "platform.opentargets.org" in gql.OT_GRAPHQL_API


def test_public_api_signatures_stable() -> None:
    for name in ("graphql_post", "query_target", "query_disease", "query_drug", "query_variant"):
        assert hasattr(gql, name)


# ---------------------------------------------------------------------------
# graphql_post — happy path, retries, errors-block
# ---------------------------------------------------------------------------


def test_graphql_post_returns_data_block() -> None:
    client = _client_returning({"data": {"target": {"approvedSymbol": "BRCA1"}}})
    out = gql.graphql_post("query Q { x }", {}, client=client)
    assert out == {"target": {"approvedSymbol": "BRCA1"}}


def test_graphql_post_raises_on_errors_block() -> None:
    """GraphQL responses with non-empty ``errors`` must surface as RuntimeError."""
    client = _client_returning({"errors": [{"message": "Unknown field"}], "data": None})
    with pytest.raises((RuntimeError, httpx.HTTPError)):
        gql.graphql_post("bad query", {}, client=client, max_retries=1, backoff_s=0)


def test_graphql_post_retries_then_succeeds(monkeypatch) -> None:
    """Transient 5xx errors should be retried up to ``max_retries`` times."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, json={})
        return httpx.Response(200, json={"data": {"ok": True}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(time, "sleep", lambda _: None)  # don't actually wait

    out = gql.graphql_post("Q", {}, client=client, max_retries=4, backoff_s=0)
    assert calls["n"] == 3
    assert out == {"ok": True}


def test_graphql_post_exhausts_retries(monkeypatch) -> None:
    """When all attempts fail, the last exception is re-raised."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, json={})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(time, "sleep", lambda _: None)

    with pytest.raises(httpx.HTTPError):
        gql.graphql_post("Q", {}, client=client, max_retries=3, backoff_s=0)
    assert calls["n"] == 3


def test_graphql_post_owns_client_when_none_provided(monkeypatch) -> None:
    """When the caller passes ``client=None`` we own the client lifetime and
    must close it on exit."""
    constructed: list[httpx.Client] = []

    real_client_cls = httpx.Client

    def fake_client(*args, **kwargs):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": {"ok": True}})

        kwargs["transport"] = httpx.MockTransport(handler)
        c = real_client_cls(**kwargs)
        constructed.append(c)
        return c

    monkeypatch.setattr(httpx, "Client", fake_client)
    out = gql.graphql_post("Q", {})
    assert out == {"ok": True}
    assert len(constructed) == 1
    assert constructed[0].is_closed


# ---------------------------------------------------------------------------
# query_target / query_disease / query_drug / query_variant
# ---------------------------------------------------------------------------


def test_query_target_returns_target_object() -> None:
    client = _client_returning(
        {"data": {"target": {"approvedSymbol": "BRCA1", "id": "ENSG00000012048"}}}
    )
    out = gql.query_target("ENSG00000012048", client=client)
    assert out is not None
    assert out["approvedSymbol"] == "BRCA1"


def test_query_target_returns_none_when_not_found() -> None:
    """OT returns ``data: {target: null}`` for unknown Ensembl IDs."""
    client = _client_returning({"data": {"target": None}})
    out = gql.query_target("ENSG_FAKE", client=client)
    assert out is None


def test_query_disease_normalises_colons() -> None:
    """``MONDO:0007254`` and ``MONDO_0007254`` must both work — colon is
    normalised to underscore before sending to OT."""
    captured: dict[str, dict] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = httpx.Request.read(request)
        import json

        captured["body"] = json.loads(body)
        return httpx.Response(200, json={"data": {"disease": {"name": "ex"}}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    gql.query_disease("MONDO:0007254", client=client)
    assert captured["body"]["variables"] == {"efoId": "MONDO_0007254"}


def test_query_drug_returns_drug_object() -> None:
    client = _client_returning({"data": {"drug": {"id": "CHEMBL25", "name": "ASPIRIN"}}})
    out = gql.query_drug("CHEMBL25", client=client)
    assert out["name"] == "ASPIRIN"


def test_query_variant_returns_variant_object() -> None:
    client = _client_returning(
        {"data": {"variant": {"id": "chr1_55039774_T_C", "chromosome": "1"}}}
    )
    out = gql.query_variant("chr1_55039774_T_C", client=client)
    assert out["chromosome"] == "1"


def test_query_returns_none_when_data_key_missing() -> None:
    """``data.get("target")`` handles servers that omit the key entirely."""
    client = _client_returning({"data": {}})
    assert gql.query_target("X", client=client) is None
    assert gql.query_drug("X", client=client) is None
    assert gql.query_variant("X", client=client) is None
    assert gql.query_disease("X", client=client) is None


# ---------------------------------------------------------------------------
# Live network smoke tests (CI skips)
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_query_target_live() -> None:
    brca1 = gql.query_target("ENSG00000012048")
    assert brca1 is not None
    assert brca1["approvedSymbol"] == "BRCA1"


@pytest.mark.network
def test_query_disease_live() -> None:
    bc = gql.query_disease("MONDO_0007254")
    assert bc is not None
    assert "carcinoma" in bc["name"].lower() or "cancer" in bc["name"].lower()
