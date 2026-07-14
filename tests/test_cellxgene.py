"""Tests for :mod:`biodb.cellxgene` (WMG REST client).

Offline tests mock the three WMG endpoints (``primary_filter_dimensions``,
``filters``, ``markers``) with the ``responses`` library.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import responses

import biodb
from biodb import _celltype, cellxgene
from tests.conftest import is_upstream_outage

BASE = cellxgene.WMG_API_BASE_URL

_PFD = {
    "snapshot_id": "test-snap",
    "organism_terms": [{"NCBITaxon:9606": "Homo sapiens"}],
    "tissue_terms": {"NCBITaxon:9606": [{"UBERON:0002106": "spleen"}]},
    "gene_terms": {
        "NCBITaxon:9606": [
            {"ENSG00000105369": "CD79A"},
            {"ENSG00000156738": "MS4A1"},
            {"ENSG00000000003": "TSPAN6"},
        ]
    },
}
_FILTERS = {
    "snapshot_id": "test-snap",
    "filter_dims": {
        "cell_type_terms": [{"CL:0000236": "B cell"}, {"CL:0000084": "T cell"}],
        "disease_terms": [
            {"PATO:0000461": "normal"},
            {"MONDO:0015925": "interstitial lung disease"},
        ],
    },
}
_MARKERS = {
    "snapshot_id": "test-snap",
    "marker_genes": [
        {"gene_ontology_term_id": "ENSG00000105369", "marker_score": 2.3, "specificity": 0.98},
        {"gene_ontology_term_id": "ENSG00000156738", "marker_score": 2.2, "specificity": 0.98},
    ],
}
_DE = {
    "snapshot_id": "test-snap",
    "successCode": 0,
    "n_overlap": 0,
    "differentialExpressionResults": [
        {
            "gene_ontology_term_id": "ENSG00000019582",
            "gene_symbol": "CD74",
            "effect_size": 1.4,
            "log_fold_change": 1.9,
            "adjusted_p_value": 0.0,
        },
        {
            "gene_ontology_term_id": "ENSG00000204287",
            "gene_symbol": "HLA-DRA",
            "effect_size": 0.9,
            "log_fold_change": 0.8,
            "adjusted_p_value": 0.0,
        },
    ],
}


@pytest.fixture(autouse=True)
def _clear_pfd_cache() -> None:
    cellxgene._primary_filter_dimensions.cache_clear()


def _register(rsps: responses.RequestsMock) -> None:
    rsps.add(responses.GET, f"{BASE}/primary_filter_dimensions", json=_PFD)
    rsps.add(responses.POST, f"{BASE}/filters", json=_FILTERS)
    rsps.add(responses.POST, f"{BASE}/markers", json=_MARKERS)
    rsps.add(responses.POST, f"{cellxgene.DE_API_BASE_URL}/differentialExpression", json=_DE)


# ── module surface ────────────────────────────────────────────────────────────


def test_module_imports_offline() -> None:
    assert cellxgene.__name__ == "biodb.cellxgene"
    assert cellxgene.WMG_API_BASE_URL.endswith("/wmg/v2")
    assert cellxgene.DEFAULT_ORGANISM == "Homo sapiens"
    assert cellxgene.CACHE_DIR.exists()


# ── discovery ─────────────────────────────────────────────────────────────────


@responses.activate
def test_list_tissues() -> None:
    _register(responses)
    assert cellxgene.list_tissues() == ["spleen"]


@responses.activate
def test_list_cell_types() -> None:
    _register(responses)
    df = cellxgene.list_cell_types("spleen")
    assert set(df["cell_ontology_id"]) == {"CL:0000236", "CL:0000084"}
    mapping = dict(zip(df["cell_ontology_id"], df["cell_type_name"], strict=False))
    assert mapping["CL:0000236"] == "B cell"


def test_resolve_organism_and_tissue_by_id_or_label() -> None:
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        _register(rsps)
        assert cellxgene._resolve_organism("Homo sapiens") == ("NCBITaxon:9606", "Homo sapiens")
        assert cellxgene._resolve_organism("NCBITaxon:9606")[1] == "Homo sapiens"
        assert cellxgene._resolve_tissue("spleen", "NCBITaxon:9606")[0] == "UBERON:0002106"


def test_resolve_organism_unknown() -> None:
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        _register(rsps)
        with pytest.raises(ValueError, match="Unknown organism"):
            cellxgene._resolve_organism("Klingon")


# ── query markers ─────────────────────────────────────────────────────────────


@responses.activate
def test_query_markers_maps_ensembl_to_symbols_and_ranks() -> None:
    _register(responses)
    m = cellxgene.query_markers("CL:0000236", tissue="spleen", n_top=15)
    assert list(m.columns) == _celltype.NORMALIZED_COLUMNS
    assert (m["source"] == "cellxgene").all()
    assert (m["cell_ontology_id"] == "CL:0000236").all()
    assert m.iloc[0]["gene_symbol"] == "CD79A"  # highest marker_score
    assert m.iloc[0]["gene_id"] == "ENSG00000105369"
    assert m.iloc[0]["rank"] == 1
    assert m.iloc[0]["cell_type_name"] == "B cell"
    assert float(m.iloc[0]["score"]) == pytest.approx(2.3)


@responses.activate
def test_query_markers_resolves_name_via_ols(monkeypatch: pytest.MonkeyPatch) -> None:
    from biodb import ols

    monkeypatch.setattr(ols, "find_term", lambda *a, **k: {"obo_id": "CL:0000236"})
    _register(responses)
    m = cellxgene.query_markers("B cell", tissue="spleen")
    assert (m["cell_ontology_id"] == "CL:0000236").all()


def test_resolve_cl_id_unresolvable(monkeypatch: pytest.MonkeyPatch) -> None:
    from biodb import ols

    monkeypatch.setattr(ols, "find_term", lambda *a, **k: None)
    with pytest.raises(ValueError, match="Could not resolve"):
        cellxgene._resolve_cl_id("not a real cell type")


# ── bulk + GMT ────────────────────────────────────────────────────────────────


@responses.activate
def test_get_tissue_markers_and_gmt(tmp_path: Path) -> None:
    from biodb.utils import read_gmt

    _register(responses)
    table = cellxgene.get_tissue_markers("spleen", cache_dir=tmp_path)
    assert list(table.columns) == _celltype.NORMALIZED_COLUMNS
    assert set(table["cell_ontology_id"]) == {"CL:0000236", "CL:0000084"}

    out = tmp_path / "cx.gmt"
    cellxgene.to_gmt(out, tissue="spleen", cache_dir=tmp_path)
    sets = read_gmt(out, return_format="dict")
    assert any(k[0] == "CL:0000236" for k in sets)


@responses.activate
def test_get_all_markers_explicit_tissues(tmp_path: Path) -> None:
    _register(responses)
    table = cellxgene.get_all_markers(tissues=["spleen"], cache_dir=tmp_path, progress=False)
    assert list(table.columns) == _celltype.NORMALIZED_COLUMNS
    assert set(table["cell_ontology_id"]) == {"CL:0000236", "CL:0000084"}


# ── differential expression (disease vs normal) ───────────────────────────────


@responses.activate
def test_list_diseases_excludes_normal() -> None:
    _register(responses)
    df = cellxgene.list_diseases("spleen")
    assert list(df["disease_ontology_term_id"]) == ["MONDO:0015925"]
    assert "normal" not in set(df["disease"])


@responses.activate
def test_disease_vs_normal() -> None:
    _register(responses)
    de = cellxgene.disease_vs_normal(
        "CL:0000236", tissue="spleen", disease="interstitial lung disease"
    )
    assert list(de.columns) == cellxgene.DE_COLUMNS
    assert de.iloc[0]["gene_symbol"] == "CD74"  # highest effect_size
    assert de.iloc[0]["rank"] == 1
    assert (de["disease"] == "interstitial lung disease").all()
    assert (de["cell_ontology_id"] == "CL:0000236").all()
    assert (de["source"] == "cellxgene").all()


@responses.activate
def test_disease_vs_normal_bad_disease() -> None:
    _register(responses)
    with pytest.raises(ValueError, match="not found"):
        cellxgene.disease_vs_normal("CL:0000236", tissue="spleen", disease="dragon pox")


# ── top-level re-export ───────────────────────────────────────────────────────


def test_reexports() -> None:
    assert biodb.cellxgene is cellxgene
    assert biodb.cellxgene_query_markers is cellxgene.query_markers
    assert biodb.cellxgene_get_tissue_markers is cellxgene.get_tissue_markers
    assert biodb.cellxgene_get_all_markers is cellxgene.get_all_markers
    assert biodb.cellxgene_disease_vs_normal is cellxgene.disease_vs_normal


# ── live network smoke ────────────────────────────────────────────────────────


@pytest.mark.network
def test_query_markers_live() -> None:
    try:
        m = cellxgene.query_markers("CL:0000236", tissue="spleen", n_top=10)
    except Exception as exc:  # noqa: BLE001
        if is_upstream_outage(exc):
            pytest.skip(f"CELLxGENE WMG upstream outage: {exc}")
        raise
    assert not m.empty
    assert (m["cell_ontology_id"] == "CL:0000236").all()
    assert (m["score"] > 0).all()
    assert m["gene_symbol"].notna().all()


@pytest.mark.network
@pytest.mark.slow
def test_disease_vs_normal_live() -> None:
    try:
        de = cellxgene.disease_vs_normal(
            "CL:0000082",  # epithelial cell of lung
            tissue="lung",
            disease="interstitial lung disease",
            n_top=20,
        )
    except Exception as exc:  # noqa: BLE001
        if is_upstream_outage(exc):
            pytest.skip(f"CELLxGENE DE upstream outage: {exc}")
        raise
    assert list(de.columns) == cellxgene.DE_COLUMNS
    assert not de.empty
    assert de["gene_symbol"].notna().all()
    assert de.iloc[0]["rank"] == 1
