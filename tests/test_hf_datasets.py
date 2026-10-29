"""Tests for :mod:`biodb.hf_datasets` (curated-source dataset builders)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from biodb import hf_datasets as hd

FIXTURES = Path(__file__).parent / "fixtures"


def test_build_celltaxonomy_dataset_per_species(tmp_path: Path) -> None:
    out = hd.build_celltaxonomy_dataset(tmp_path / "ct", cache_dir=FIXTURES / "celltaxonomy")
    # one parquet per species, species encoded in the file name (no species col)
    files = sorted((out / "markers").glob("*.parquet"))
    assert files, "expected per-species parquet files"
    assert (out / "markers" / "homo_sapiens.parquet").exists()
    df = pd.read_parquet(out / "markers" / "homo_sapiens.parquet")
    assert "species" not in df.columns and "source" not in df.columns
    assert {"tissue_ontology_id", "cell_ontology_id", "gene_symbol", "score", "rank"} <= set(
        df.columns
    )
    card = (out / "README.md").read_text()
    assert "biodb_celltaxonomy" in card
    assert "config_name: homo_sapiens" in card
    assert card.startswith("---")


def test_build_cellmarker_dataset_per_species(tmp_path: Path) -> None:
    out = hd.build_cellmarker_dataset(
        tmp_path / "cm", which="all", cache_dir=FIXTURES / "cellmarker"
    )
    df = pd.read_parquet(out / "markers" / "human.parquet")
    assert "species" not in df.columns
    assert "CL:0000540" in set(df["cell_ontology_id"])
    assert "UBERON:0000955" in set(df["tissue_ontology_id"].dropna())
    card = (out / "README.md").read_text()
    assert "config_name: human" in card


def test_species_slug() -> None:
    assert hd._species_slug("Macaca fascicularis") == "macaca_fascicularis"
    assert hd._species_slug("Homo sapiens") == "homo_sapiens"
