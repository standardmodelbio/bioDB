"""Tests for :mod:`biodb.markers` — the multi-source aggregator."""

from __future__ import annotations

import pandas as pd
import pytest

import biodb
from biodb import markers


def _celltax(**_kw):
    return pd.DataFrame(
        {
            "species": ["Homo sapiens", "Mus musculus"],
            "tissue": ["brain", "brain"],
            "tissue_ontology_id": ["UBERON:0000955", "UBERON:0000955"],
            "cell_type_name": ["neuron", "neuron"],
            "cell_ontology_id": ["CL:0000540", "CL:0000540"],
            "gene_symbol": ["RBFOX3", "Rbfox3"],
            "gene_id": ["146713", "52897"],
            "score": [3, 1],
            "rank": [1, 1],
            "source": ["celltaxonomy", "celltaxonomy"],
        }
    )


def _cellmarker(**_kw):
    return pd.DataFrame(
        {
            "species": ["Human"],
            "tissue": ["blood"],
            "tissue_ontology_id": ["UBERON:0000178"],
            "cell_type_name": ["T cell"],
            "cell_ontology_id": ["CL:0000084"],
            "gene_symbol": ["CD3D"],
            "gene_id": ["915"],
            "score": [2],
            "rank": [1],
            "source": ["cellmarker"],
        }
    )


def _cellguide(**_kw):
    return pd.DataFrame(
        {
            "species": ["Homo sapiens", None],
            "tissue": ["spleen", "spleen"],
            "tissue_ontology_id": ["UBERON:0002106", "UBERON:0002106"],
            "cell_type_name": ["B cell", "B cell"],
            "cell_ontology_id": ["CL:0000236", "CL:0000236"],
            "gene_symbol": ["CD79A", "MS4A1"],
            "gene_id": ["ENSG00000105369", None],
            "marker_type": ["computational", "canonical"],
            "marker_score": [2.3, None],
            "rank": [1, None],
            "source": ["cellxgene", "cellxgene"],
        }
    )


@pytest.fixture(autouse=True)
def _patch_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(markers.celltaxonomy, "get_markers", _celltax)
    monkeypatch.setattr(markers.cellmarker, "get_markers", _cellmarker)
    monkeypatch.setattr(markers.cellxgene, "get_all_cellguide_markers", _cellguide)


def test_null_means_all() -> None:
    df = markers.load_markers()
    assert list(df.columns) == markers.UNIFIED_COLUMNS
    assert set(df["source"]) == {"celltaxonomy", "cellmarker", "cellxgene"}
    assert set(df["marker_type"]) == {"canonical", "computational"}
    # cellxgene computational marker_score mapped into unified `score`
    assert 2.3 in set(df["score"].dropna())


def test_filter_by_source() -> None:
    df = markers.load_markers(source="cellxgene")
    assert set(df["source"]) == {"cellxgene"}


def test_filter_species_and_celltype() -> None:
    df = markers.load_markers(species="Homo sapiens", cell_type="CL:0000540")
    assert (df["species"] == "Homo sapiens").all()
    assert (df["cell_ontology_id"] == "CL:0000540").all()


def test_cell_type_matches_name_or_id() -> None:
    by_name = markers.load_markers(cell_type="b cell")  # case-insensitive name
    assert set(by_name["cell_ontology_id"]) == {"CL:0000236"}


def test_tissue_matches_label_or_uberon() -> None:
    by_uberon = markers.load_markers(tissue="UBERON:0000178")
    assert set(by_uberon["tissue"]) == {"blood"}


def test_bad_source() -> None:
    with pytest.raises(ValueError, match="Unknown source"):
        markers.load_markers(source="nope")


def test_reexport() -> None:
    assert biodb.load_markers is markers.load_markers
