"""Tests for :mod:`biodb.cellmarker`."""

from __future__ import annotations

from pathlib import Path

import pytest

import biodb
from biodb import _celltype, cellmarker
from tests.conftest import is_upstream_outage

FIXTURES = Path(__file__).parent / "fixtures" / "cellmarker"


@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    cellmarker._MARKERS_CACHE = {}


# ── module surface ────────────────────────────────────────────────────────────


def test_module_imports_offline() -> None:
    assert cellmarker.__name__ == "biodb.cellmarker"


def test_constants_present() -> None:
    assert cellmarker.DOWNLOAD_BASE_URL.endswith("/file")
    assert cellmarker.FILES["all"] == "Cell_marker_All.xlsx"
    assert cellmarker.SOURCE_NAME == "cellmarker"
    assert cellmarker.CACHE_DIR.exists()


def test_download_bad_which() -> None:
    with pytest.raises(ValueError, match="which must be"):
        cellmarker.download("bogus")


# ── normalized loading (offline, from .xlsx fixture) ──────────────────────────


def test_get_markers_normalizes_cl_underscore() -> None:
    df = cellmarker.get_markers(which="all", cache_dir=FIXTURES)
    assert list(df.columns) == _celltype.NORMALIZED_COLUMNS
    # CellMarker stores CL_0000540; we normalize to CL:0000540.
    assert "CL:0000540" in set(df["cell_ontology_id"])
    assert not any(str(v).startswith("CL_") for v in df["cell_ontology_id"])


def test_get_markers_ranking_from_pmid_counts() -> None:
    df = cellmarker.get_markers(which="all", cache_dir=FIXTURES)
    neuron = df[(df["cell_ontology_id"] == "CL:0000540") & (df["species"] == "Human")]
    by_gene = neuron.set_index("gene_symbol")
    # SNAP25 supported by 2 PMIDs, RBFOX3 by 1.
    assert by_gene.loc["SNAP25", "score"] == 2
    assert by_gene.loc["SNAP25", "rank"] == 1
    assert by_gene.loc["RBFOX3", "rank"] == 2


# ── targeted query ────────────────────────────────────────────────────────────


def test_query_markers_by_cl() -> None:
    hits = cellmarker.query_markers("CL:0000540", cache_dir=FIXTURES)
    assert set(hits["cell_ontology_id"]) == {"CL:0000540"}


def test_query_markers_species_filter() -> None:
    hits = cellmarker.query_markers("CL:0000540", species="Mouse", cache_dir=FIXTURES)
    assert set(hits["species"]) == {"Mouse"}
    assert "Rbfox3" in set(hits["gene_symbol"])


# ── GMT export ────────────────────────────────────────────────────────────────


def test_to_gmt_roundtrips(tmp_path: Path) -> None:
    from biodb.utils import read_gmt

    out = tmp_path / "cm.gmt"
    cellmarker.to_gmt(out, which="all", cache_dir=FIXTURES)
    sets = read_gmt(out, return_format="dict")
    assert any(k[0] == "CL:0000540" for k in sets)


# ── top-level re-export ───────────────────────────────────────────────────────


def test_reexports() -> None:
    assert biodb.cellmarker is cellmarker
    assert biodb.cellmarker_get_markers is cellmarker.get_markers
    assert biodb.cellmarker_query_markers is cellmarker.query_markers


# ── live network smoke ────────────────────────────────────────────────────────


@pytest.mark.network
@pytest.mark.slow
def test_download_live(tmp_path: Path) -> None:
    try:
        path = cellmarker.download("human", cache_dir=tmp_path)
    except Exception as exc:  # noqa: BLE001
        if is_upstream_outage(exc):
            pytest.skip(f"CellMarker upstream outage: {exc}")
        raise
    assert path.exists() and path.stat().st_size > 100_000
