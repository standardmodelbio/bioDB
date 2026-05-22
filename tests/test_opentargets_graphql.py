"""Tests for :mod:`biodb.opentargets_graphql`.

Uses :class:`httpx.MockTransport` to intercept GraphQL POSTs so the
tests never touch the live Open Targets API. The two ``network``-marked
smoke tests at the bottom hit the real endpoint on demand.
"""

from __future__ import annotations

import time

import httpx
import polars as pl
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
    for name in (
        "graphql_post",
        "query_target",
        "query_disease",
        "query_drug",
        "query_variant",
        "map_symbols_to_ensembl",
        "target_associated_diseases",
        "fetch_panel_scores",
        "fetch_aggregated_panel_scores",
    ):
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
    # Only the EFO id is normalized -- other variables (paging sizes for
    # nested fields) are passed through with their current default values.
    assert captured["body"]["variables"]["efoId"] == "MONDO_0007254"


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
# Live integration tests — RUN BY DEFAULT in CI.
#
# The GraphQL endpoint is fast (sub-second) and the response is tiny, so
# we can afford to hit it on every CI run. These tests are the only thing
# that would catch upstream schema changes (e.g. fields being removed
# from the ``Target`` type — has happened before).
# ---------------------------------------------------------------------------


def test_query_target_live() -> None:
    """Fetch the real BRCA1 record and verify the documented fields. Catches
    schema drift like the 2026-05 removal of the ``go`` field."""
    brca1 = gql.query_target("ENSG00000012048")
    assert brca1 is not None, "query_target returned None for BRCA1 — query rejected?"
    assert brca1["approvedSymbol"] == "BRCA1"
    # A few other documented fields that bioDB users depend on.
    assert brca1["biotype"] == "protein_coding"
    assert isinstance(brca1.get("pathways"), list)


def test_query_disease_live() -> None:
    """Fetch the real ``MONDO_0007254`` (breast cancer) record.

    Also exercises the nested fields the deep query now pulls back so we
    catch upstream schema drift on any of them (every page reload of OT
    Platform that breaks one of these fields would fail this test).
    """
    bc = gql.query_disease("MONDO_0007254", assoc_size=5, pheno_size=5)
    assert bc is not None
    name = (bc.get("name") or "").lower()
    assert "carcinoma" in name or "cancer" in name, (
        f"Got unexpected disease name {bc.get('name')!r} for MONDO_0007254."
    )
    # The deep query must return populated containers (the wrappers always
    # exist; their ``rows`` lists hold real data for any well-studied disease).
    assert isinstance(bc.get("synonyms"), list)
    assert isinstance(bc.get("therapeuticAreas"), list)
    at = bc.get("associatedTargets") or {}
    assert isinstance(at.get("rows"), list)
    assert at.get("rows"), "associatedTargets rows empty -- field shape changed?"
    first = at["rows"][0]
    assert "score" in first
    assert first.get("target", {}).get("approvedSymbol")


def test_query_drug_live() -> None:
    """Fetch the real ``CHEMBL25`` (aspirin) record + verify deep fields."""
    aspirin = gql.query_drug("CHEMBL25", ae_size=5)
    assert aspirin is not None
    assert (aspirin.get("name") or "").upper() == "ASPIRIN"
    moa = aspirin.get("mechanismsOfAction") or {}
    assert isinstance(moa.get("rows"), list)
    assert moa.get("rows"), "MOA rows empty -- field shape changed?"
    indications = aspirin.get("indications") or {}
    assert isinstance(indications.get("rows"), list)


def test_query_variant_live() -> None:
    """Fetch a real LDLR variant + verify the nested fields the deep query exposes."""
    v = gql.query_variant("19_11100252_C_T")  # rs121908024
    assert v is not None
    assert v.get("rsIds") == ["rs121908024"]
    assert isinstance(v.get("alleleFrequencies"), list)
    assert isinstance(v.get("transcriptConsequences"), list)
    assert isinstance(v.get("variantEffect"), list)


# ---------------------------------------------------------------------------
# map_symbols_to_ensembl
# ---------------------------------------------------------------------------


def test_map_symbols_to_ensembl_takes_first_hit_per_symbol() -> None:
    """A symbol can resolve to multiple OT records; ``map_symbols_to_ensembl``
    takes the first hit (OT's relevance-ranked default) so the caller
    gets a deterministic 1-to-1 dict."""
    client = _client_returning(
        {
            "data": {
                "mapIds": {
                    "mappings": [
                        {
                            "term": "BRCA1",
                            "hits": [
                                {"id": "ENSG00000012048", "name": "BRCA1", "entity": "target"},
                                {"id": "ENSG_OTHER", "name": "BRCA1-AS1", "entity": "target"},
                            ],
                        },
                        {
                            "term": "TP53",
                            "hits": [
                                {"id": "ENSG00000141510", "name": "TP53", "entity": "target"},
                            ],
                        },
                    ],
                }
            }
        }
    )
    out = gql.map_symbols_to_ensembl(["BRCA1", "TP53"], client=client)
    assert out == {"BRCA1": "ENSG00000012048", "TP53": "ENSG00000141510"}


def test_map_symbols_to_ensembl_omits_unresolved() -> None:
    """A symbol with no hits is dropped from the dict -- callers check
    ``in`` membership rather than getting a ``None`` sentinel."""
    client = _client_returning(
        {
            "data": {
                "mapIds": {
                    "mappings": [
                        {
                            "term": "BRCA1",
                            "hits": [
                                {"id": "ENSG00000012048", "name": "BRCA1", "entity": "target"}
                            ],
                        },
                        {"term": "MADEUPSYM", "hits": []},
                    ]
                }
            }
        }
    )
    out = gql.map_symbols_to_ensembl(["BRCA1", "MADEUPSYM"], client=client)
    assert "MADEUPSYM" not in out
    assert out["BRCA1"] == "ENSG00000012048"


def test_map_symbols_to_ensembl_empty_input_returns_empty_dict() -> None:
    """Edge case: ``terms=[]`` -> OT returns ``mappings=[]`` -> empty dict."""
    client = _client_returning({"data": {"mapIds": {"mappings": []}}})
    assert gql.map_symbols_to_ensembl([], client=client) == {}


# ---------------------------------------------------------------------------
# target_associated_diseases
# ---------------------------------------------------------------------------


def test_target_associated_diseases_returns_target_envelope() -> None:
    """Helper returns the ``target`` envelope (id + approvedSymbol +
    associatedDiseases) so the caller can filter rows by disease ID
    without re-fetching the symbol."""
    client = _client_returning(
        {
            "data": {
                "target": {
                    "id": "ENSG00000012048",
                    "approvedSymbol": "BRCA1",
                    "associatedDiseases": {
                        "count": 2,
                        "rows": [
                            {
                                "score": 0.95,
                                "datatypeScores": [{"id": "literature", "score": 0.95}],
                                "disease": {
                                    "id": "EFO_0000305",
                                    "name": "breast cancer",
                                    "therapeuticAreas": [],
                                },
                            },
                            {
                                "score": 0.42,
                                "datatypeScores": [{"id": "rna_expression", "score": 0.42}],
                                "disease": {
                                    "id": "EFO_0001075",
                                    "name": "ovarian cancer",
                                    "therapeuticAreas": [],
                                },
                            },
                        ],
                    },
                }
            }
        }
    )
    target = gql.target_associated_diseases("ENSG00000012048", client=client)
    assert target is not None
    assert target["approvedSymbol"] == "BRCA1"
    assert target["associatedDiseases"]["count"] == 2
    disease_ids = [r["disease"]["id"] for r in target["associatedDiseases"]["rows"]]
    assert disease_ids == ["EFO_0000305", "EFO_0001075"]


def test_target_associated_diseases_returns_none_when_target_missing() -> None:
    """``target: null`` -> ``None``. Callers shouldn't have to repeat
    the get-with-default boilerplate."""
    client = _client_returning({"data": {"target": None}})
    assert gql.target_associated_diseases("ENSG_NONEXISTENT", client=client) is None


def test_target_associated_diseases_passes_size_param() -> None:
    """``size=`` must reach the GraphQL ``$size`` variable -- a default
    of 200 would silently truncate panels for genes with >200 disease
    associations."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        body = json.loads(request.content)
        captured["variables"] = body["variables"]
        return httpx.Response(
            200,
            json={
                "data": {
                    "target": {
                        "id": "X",
                        "approvedSymbol": "X",
                        "associatedDiseases": {"count": 0, "rows": []},
                    }
                }
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    gql.target_associated_diseases("ENSG_X", size=42, client=client)
    assert captured["variables"] == {"ensemblId": "ENSG_X", "size": 42}


# ---------------------------------------------------------------------------
# _normalise_id (private but small enough to nail down behaviour)
# ---------------------------------------------------------------------------


def test_normalise_id_replaces_efo_colon() -> None:
    assert gql._normalise_id("EFO:0000305") == "EFO_0000305"


def test_normalise_id_passes_through_underscore_form() -> None:
    assert gql._normalise_id("EFO_0000305") == "EFO_0000305"


def test_normalise_id_handles_mondo() -> None:
    assert gql._normalise_id("MONDO:0004975") == "MONDO_0004975"


# ---------------------------------------------------------------------------
# fetch_panel_scores / fetch_aggregated_panel_scores
#
# These compose ``map_symbols_to_ensembl`` and ``target_associated_diseases``
# end-to-end against a single httpx.MockTransport that routes by the GraphQL
# query string -- so the tests cover the per-gene loop, the disease-id
# filter, the polars schema, and the group-by aggregation together.
# ---------------------------------------------------------------------------


def _assoc_target(rows: list[dict], approved_symbol: str = "X", ensembl_id: str = "ENSG_X") -> dict:
    """Build a ``target`` envelope for the ``associatedDiseases`` mock response."""
    return {
        "id": ensembl_id,
        "approvedSymbol": approved_symbol,
        "associatedDiseases": {"count": len(rows), "rows": rows},
    }


def _panel_transport(
    *,
    symbol_map: dict[str, str],
    targets: dict[str, dict | None],
    assoc_error: Exception | None = None,
) -> httpx.MockTransport:
    """Build a MockTransport that routes mapIds vs target queries by query body.

    Parameters
    ----------
    symbol_map : dict[str, str]
        Symbols that will resolve via ``map_symbols_to_ensembl`` -- emitted
        as the ``mapIds.mappings`` payload.
    targets : dict[str, dict | None]
        Per-Ensembl-id ``target`` envelope to return from
        ``target_associated_diseases``. A ``None`` value is returned as
        ``{"data": {"target": null}}`` so the helper yields ``None``.
    assoc_error : Exception, optional
        If set, the ``target`` query raises by emitting a 503 (which the
        retrying ``graphql_post`` bubbles as ``httpx.HTTPError``) -- exercises
        the per-gene swallow path.
    """
    import json

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        query = body["query"]
        if "mapIds" in query:
            mappings = [
                {
                    "term": sym,
                    "hits": [{"id": ens, "name": sym, "entity": "target"}],
                }
                for sym, ens in symbol_map.items()
            ]
            return httpx.Response(200, json={"data": {"mapIds": {"mappings": mappings}}})
        if "TargetAssociatedDiseases" in query or "associatedDiseases" in query:
            if assoc_error is not None:
                return httpx.Response(503, json={})
            ensembl_id = body["variables"]["ensemblId"]
            target = targets.get(ensembl_id)
            return httpx.Response(200, json={"data": {"target": target}})
        raise AssertionError(f"unexpected query: {query!r}")

    return httpx.MockTransport(handler)


def test_fetch_panel_scores_filters_to_wanted_diseases() -> None:
    """Only rows whose ``disease.id`` is in the requested set survive --
    the test also confirms colon-form input is normalised to underscore."""
    target = _assoc_target(
        [
            {"disease": {"id": "EFO_0000305", "name": "breast cancer"}, "score": 0.9},
            {"disease": {"id": "EFO_9999999", "name": "unrelated"}, "score": 0.1},
        ]
    )
    client = httpx.Client(
        transport=_panel_transport(
            symbol_map={"X": "ENSG_X"},
            targets={"ENSG_X": target},
        )
    )
    df = gql.fetch_panel_scores(["X"], ["EFO:0000305"], client=client)
    assert df.height == 1
    assert df["cancer_type"].to_list() == ["EFO_0000305"]
    assert df["ot_score"].to_list() == [pytest.approx(0.9)]
    assert df["ot_evidence"].to_list() == ["breast cancer"]


def test_fetch_panel_scores_empty_when_no_mapping() -> None:
    """If no symbol resolves to an Ensembl id, the associated-diseases
    query is never issued and the result carries the full empty schema."""
    client = httpx.Client(
        transport=_panel_transport(symbol_map={}, targets={}),
    )
    df = gql.fetch_panel_scores(["UNKNOWN"], ["EFO_0000305"], client=client)
    assert df.height == 0
    assert df.columns == ["gene_id", "gene_symbol", "cancer_type", "ot_score", "ot_evidence"]


def test_fetch_panel_scores_logs_warning_on_per_gene_failure(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A per-gene associated-disease HTTP failure is logged as WARNING and
    the gene is skipped (matches seqlab's prior behaviour, but via logging
    instead of stderr)."""
    monkeypatch.setattr(time, "sleep", lambda _: None)  # zip past graphql_post backoff
    client = httpx.Client(
        transport=_panel_transport(
            symbol_map={"X": "ENSG_X"},
            targets={"ENSG_X": None},
            assoc_error=httpx.HTTPError("nope"),
        )
    )
    with caplog.at_level("WARNING", logger="biodb.opentargets_graphql"):
        df = gql.fetch_panel_scores(["X"], ["EFO_0000305"], client=client)
    assert df.height == 0
    assert any("associated-diseases fetch failed" in rec.message for rec in caplog.records)


def test_fetch_panel_scores_skips_null_target() -> None:
    """``target: null`` from OT (unknown Ensembl id post-resolve) returns
    ``None`` from :func:`target_associated_diseases` and is skipped without
    emitting a warning."""
    client = httpx.Client(
        transport=_panel_transport(
            symbol_map={"X": "ENSG_X"},
            targets={"ENSG_X": None},
        )
    )
    df = gql.fetch_panel_scores(["X"], ["EFO_0000305"], client=client)
    assert df.height == 0


def test_fetch_panel_scores_uses_approved_symbol_when_present() -> None:
    """``gene_symbol`` should be the OT-canonical ``approvedSymbol``, not
    the input HGNC symbol (which may differ in case or be an alias)."""
    target = _assoc_target(
        [{"disease": {"id": "EFO_0000305", "name": "breast cancer"}, "score": 0.9}],
        approved_symbol="BRCA1",
        ensembl_id="ENSG_X",
    )
    client = httpx.Client(
        transport=_panel_transport(
            symbol_map={"brca1": "ENSG_X"},
            targets={"ENSG_X": target},
        )
    )
    df = gql.fetch_panel_scores(["brca1"], ["EFO_0000305"], client=client)
    assert df.height == 1
    assert df["gene_symbol"].to_list() == ["BRCA1"]


def test_fetch_panel_scores_returns_typed_empty_schema_on_no_matches() -> None:
    """When mappings + targets succeed but no row matches the disease set,
    the empty DataFrame still has the documented schema."""
    target = _assoc_target([{"disease": {"id": "EFO_9999999", "name": "unrelated"}, "score": 0.5}])
    client = httpx.Client(
        transport=_panel_transport(
            symbol_map={"X": "ENSG_X"},
            targets={"ENSG_X": target},
        )
    )
    df = gql.fetch_panel_scores(["X"], ["EFO_0000305"], client=client)
    assert df.height == 0
    assert df.schema["gene_id"] == pl.Utf8
    assert df.schema["ot_score"] == pl.Float64


def test_fetch_aggregated_panel_scores_empty_input() -> None:
    """No mappings -> empty aggregated DataFrame with the full schema
    (List[Utf8] for the list columns, not Null)."""
    client = httpx.Client(
        transport=_panel_transport(symbol_map={}, targets={}),
    )
    df = gql.fetch_aggregated_panel_scores(["UNKNOWN"], ["EFO_0000305"], client=client)
    assert df.height == 0
    assert df.columns == ["gene_id", "gene_symbol", "ot_score", "cancer_types", "diseases"]
    assert df.schema["cancer_types"] == pl.List(pl.Utf8)
    assert df.schema["diseases"] == pl.List(pl.Utf8)


def test_fetch_aggregated_panel_scores_groups_per_gene() -> None:
    """Two matched diseases for one gene collapse into one row with
    ``ot_score`` = max, plus list-columns for the matched ids/names."""
    target = _assoc_target(
        [
            {"disease": {"id": "EFO_0000305", "name": "breast cancer"}, "score": 0.9},
            {"disease": {"id": "EFO_0001075", "name": "ovarian cancer"}, "score": 0.7},
        ]
    )
    client = httpx.Client(
        transport=_panel_transport(
            symbol_map={"X": "ENSG_X"},
            targets={"ENSG_X": target},
        )
    )
    df = gql.fetch_aggregated_panel_scores(["X"], ["EFO_0000305", "EFO_0001075"], client=client)
    assert df.height == 1
    row = df.to_dicts()[0]
    assert row["ot_score"] == pytest.approx(0.9)
    assert sorted(row["cancer_types"]) == ["EFO_0000305", "EFO_0001075"]
    assert sorted(row["diseases"]) == ["breast cancer", "ovarian cancer"]


def test_fetch_aggregated_panel_scores_reuses_caller_client() -> None:
    """The ``client=`` kwarg is forwarded to :func:`fetch_panel_scores` so
    callers can share connection pools across the two phases."""
    target = _assoc_target(
        [{"disease": {"id": "EFO_0000305", "name": "breast cancer"}, "score": 0.9}],
    )
    transport = _panel_transport(
        symbol_map={"X": "ENSG_X"},
        targets={"ENSG_X": target},
    )
    client = httpx.Client(transport=transport)
    df = gql.fetch_aggregated_panel_scores(["X"], ["EFO_0000305"], client=client)
    assert df.height == 1
    assert not client.is_closed  # caller-owned -> stays open
