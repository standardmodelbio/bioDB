"""Tests for :mod:`biodb.celltaxonomy`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import biodb
from biodb import _celltype, celltaxonomy
from tests.conftest import is_upstream_outage

FIXTURES = Path(__file__).parent / "fixtures" / "celltaxonomy"


@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    """Clear the module-level query cache between tests."""
    celltaxonomy._MARKERS_CACHE = None


# ── module surface ────────────────────────────────────────────────────────────


def test_module_imports_offline() -> None:
    assert celltaxonomy.__name__ == "biodb.celltaxonomy"


def test_constants_present() -> None:
    assert celltaxonomy.DOWNLOAD_BASE_URL.endswith("/celltaxonomy")
    assert celltaxonomy.RESOURCE_FILE == "Cell_Taxonomy_resource.txt"
    assert celltaxonomy.SOURCE_NAME == "celltaxonomy"
    assert celltaxonomy.CACHE_DIR.exists()


# ── raw + normalized loading (offline, from fixture) ──────────────────────────


def test_load_resource_reads_fixture() -> None:
    raw = celltaxonomy.load_resource(cache_dir=FIXTURES)
    assert "Specific_Cell_Ontology_ID" in raw.columns
    assert "Cell_Marker" in raw.columns
    assert (raw["Species"] == "Homo sapiens").any()


def test_get_markers_schema_and_ranking() -> None:
    df = celltaxonomy.get_markers(cache_dir=FIXTURES)
    assert list(df.columns) == _celltype.NORMALIZED_COLUMNS
    assert (df["source"] == "celltaxonomy").all()

    neuron = df[(df["cell_ontology_id"] == "CL:0000540") & (df["species"] == "Homo sapiens")]
    # GENEA (3 PMIDs) outranks GENEB (2) outranks GENEC (1).
    by_gene = neuron.set_index("gene_symbol")
    assert by_gene.loc["GENEA", "score"] == 3
    assert by_gene.loc["GENEB", "score"] == 2
    assert by_gene.loc["GENEA", "rank"] == 1
    assert by_gene.loc["GENEC", "rank"] == 3


def test_get_markers_species_filter() -> None:
    df = celltaxonomy.get_markers(species="Mus musculus", cache_dir=FIXTURES)
    assert set(df["species"]) == {"Mus musculus"}
    assert "Rbfox3" in set(df["gene_symbol"])


def test_get_markers_map_to_cl_fills_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    # The "Mystery cell" row has no native CL id; map_to_cl resolves it via OLS.
    def fake_resolve_cl(labels, **_kwargs):
        return pd.DataFrame(
            {"label": list(labels), "cell_ontology_id": ["CL:9999999"] * len(list(labels))}
        )

    monkeypatch.setattr(_celltype, "resolve_cl", fake_resolve_cl)
    df = celltaxonomy.get_markers(map_to_cl=True, cache_dir=FIXTURES)
    mystery = df[df["cell_type_name"] == "Mystery cell"]
    assert (mystery["cell_ontology_id"] == "CL:9999999").all()


# ── targeted query ────────────────────────────────────────────────────────────


def test_query_markers_by_cl() -> None:
    hits = celltaxonomy.query_markers("CL:0000540", cache_dir=FIXTURES)
    assert not hits.empty
    assert set(hits["cell_ontology_id"]) == {"CL:0000540"}
    # sorted by rank ascending (strongest first)
    assert hits.iloc[0]["rank"] == 1


def test_query_markers_by_name() -> None:
    hits = celltaxonomy.query_markers("astrocyte", by="name", cache_dir=FIXTURES)
    assert set(hits["cell_type_name"]) == {"Astrocyte"}


def test_query_markers_bad_by() -> None:
    with pytest.raises(ValueError, match="by must be"):
        celltaxonomy.query_markers("CL:0000540", by="nope", cache_dir=FIXTURES)


# ── GMT export ────────────────────────────────────────────────────────────────


def test_to_gmt_roundtrips(tmp_path: Path) -> None:
    from biodb.utils import read_gmt

    out = tmp_path / "ct.gmt"
    celltaxonomy.to_gmt(out, cache_dir=FIXTURES)
    sets = read_gmt(out, return_format="dict")
    keys = {k[0] for k in sets}
    assert "CL:0000540" in keys
    # neuron gene set is ordered by rank: GENEA first.
    neuron_genes = next(v for k, v in sets.items() if k[0] == "CL:0000540" and "GENEA" in v)
    assert neuron_genes[0] == "GENEA"


# ── top-level re-export ───────────────────────────────────────────────────────


def test_reexports() -> None:
    assert biodb.celltaxonomy is celltaxonomy
    assert biodb.celltaxonomy_get_markers is celltaxonomy.get_markers
    assert biodb.celltaxonomy_query_markers is celltaxonomy.query_markers


# ── live network smoke ────────────────────────────────────────────────────────


@pytest.mark.network
@pytest.mark.slow
def test_download_live(tmp_path: Path) -> None:
    try:
        path = celltaxonomy.download(cache_dir=tmp_path)
    except Exception as exc:  # noqa: BLE001
        if is_upstream_outage(exc):
            pytest.skip(f"Cell Taxonomy upstream outage: {exc}")
        raise
    assert path.exists() and path.stat().st_size > 1_000_000
