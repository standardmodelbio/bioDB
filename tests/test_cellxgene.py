"""Tests for :mod:`biodb.cellxgene`.

The Census pull needs the heavy ``[cellxgene]`` extra, but the marker-scoring
maths is exercised offline against a synthetic AnnData-like object (scipy only).
"""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import biodb
from biodb import _celltype, cellxgene


def _fake_adata() -> types.SimpleNamespace:
    """A minimal AnnData stand-in: two cell types with a clear marker each."""
    rng = np.random.default_rng(0)
    n_per = 20
    # 3 genes; GENE0 marks type A, GENE1 marks type B, GENE2 is background.
    a = np.column_stack([rng.poisson(50, n_per), rng.poisson(1, n_per), rng.poisson(10, n_per)])
    b = np.column_stack([rng.poisson(1, n_per), rng.poisson(50, n_per), rng.poisson(10, n_per)])
    x = np.vstack([a, b]).astype(np.float64)
    var = pd.DataFrame(
        {"feature_name": ["GENE0", "GENE1", "GENE2"], "feature_id": ["ENSG0", "ENSG1", "ENSG2"]}
    )
    obs = pd.DataFrame(
        {
            "cell_type_ontology_term_id": ["CL:0000540"] * n_per + ["CL:0000127"] * n_per,
            "cell_type": ["Neuron"] * n_per + ["Astrocyte"] * n_per,
        }
    )
    return types.SimpleNamespace(X=x, var=var, obs=obs)


# ── module surface ────────────────────────────────────────────────────────────


def test_module_imports_offline() -> None:
    # Importing must NOT require cellxgene-census.
    assert cellxgene.__name__ == "biodb.cellxgene"
    assert cellxgene.DEFAULT_CENSUS_VERSION
    assert cellxgene.DEFAULT_ORGANISM == "Homo sapiens"
    assert cellxgene.CACHE_DIR.exists()


def test_slug() -> None:
    assert cellxgene._slug("Homo sapiens") == "homo-sapiens"
    assert cellxgene._slug("brain / cortex") == "brain-cortex"


def test_require_census_guard() -> None:
    try:
        import cellxgene_census  # noqa: F401

        pytest.skip("cellxgene-census installed; import-guard path not exercised")
    except ImportError:
        pass
    with pytest.raises(ImportError, match="cellxgene-census"):
        cellxgene._require_census()


# ── marker scoring maths (offline) ────────────────────────────────────────────


def test_marker_table_ranks_expected_markers() -> None:
    table = cellxgene._marker_table(
        _fake_adata(), tissue="brain", organism="Homo sapiens", top_n_per_type=3
    )
    assert list(table.columns) == _celltype.NORMALIZED_COLUMNS
    assert (table["source"] == "cellxgene").all()

    neuron_top = table[table["cell_ontology_id"] == "CL:0000540"].sort_values("rank").iloc[0]
    astro_top = table[table["cell_ontology_id"] == "CL:0000127"].sort_values("rank").iloc[0]
    assert neuron_top["gene_symbol"] == "GENE0"
    assert astro_top["gene_symbol"] == "GENE1"
    assert neuron_top["score"] > 0
    assert neuron_top["rank"] == 1


def test_subsample_joinids_caps_per_type() -> None:
    obs = pd.DataFrame(
        {
            "soma_joinid": list(range(100)),
            "cell_type_ontology_term_id": ["CL:1"] * 60 + ["CL:2"] * 40,
        }
    )
    ids = cellxgene._subsample_joinids(obs, max_per_type=10)
    assert len(ids) == 20  # 10 from each type
    assert ids == sorted(ids)
    assert all(isinstance(i, int) for i in ids)


# ── CL resolution ─────────────────────────────────────────────────────────────


def test_resolve_cl_id_passthrough() -> None:
    assert cellxgene._resolve_cl_id("CL:0000540") == "CL:0000540"
    assert cellxgene._resolve_cl_id("CL_0000540") == "CL:0000540"


def test_resolve_cl_id_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    from biodb import ols

    monkeypatch.setattr(ols, "find_term", lambda *a, **k: {"obo_id": "CL:0000540"})
    assert cellxgene._resolve_cl_id("neuron") == "CL:0000540"


def test_resolve_cl_id_unresolvable(monkeypatch: pytest.MonkeyPatch) -> None:
    from biodb import ols

    monkeypatch.setattr(ols, "find_term", lambda *a, **k: None)
    with pytest.raises(ValueError, match="Could not resolve"):
        cellxgene._resolve_cl_id("not a real cell type")


# ── top-level re-export ───────────────────────────────────────────────────────


def test_reexports() -> None:
    assert biodb.cellxgene is cellxgene
    assert biodb.cellxgene_query_markers is cellxgene.query_markers
    assert biodb.cellxgene_compute_tissue_markers is cellxgene.compute_tissue_markers


# ── live network smoke ────────────────────────────────────────────────────────


@pytest.mark.network
@pytest.mark.slow
def test_query_markers_live(tmp_path: Path) -> None:
    pytest.importorskip("cellxgene_census")
    markers = cellxgene.query_markers(
        "CL:0000540", tissue="brain", n_top=10, max_cells_per_type=200, cache_dir=tmp_path
    )
    assert not markers.empty
    assert set(markers["cell_ontology_id"]) == {"CL:0000540"}
    assert (markers["score"] > 0).all()
