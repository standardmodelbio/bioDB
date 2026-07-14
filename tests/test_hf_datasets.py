"""Tests for :mod:`biodb.hf_datasets` (curated-source dataset builders)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from biodb import hf_datasets as hd

FIXTURES = Path(__file__).parent / "fixtures"


def test_build_celltaxonomy_dataset(tmp_path: Path) -> None:
    out = hd.build_celltaxonomy_dataset(tmp_path / "ct", cache_dir=FIXTURES / "celltaxonomy")
    assert (out / "markers.parquet").exists()
    df = pd.read_parquet(out / "markers.parquet")
    assert {"species", "cell_ontology_id", "gene_symbol", "score", "rank"} <= set(df.columns)
    card = (out / "README.md").read_text()
    assert "biodb_celltaxonomy" in card
    assert "Species coverage" in card
    assert card.startswith("---")  # YAML front matter for HF


def test_build_cellmarker_dataset(tmp_path: Path) -> None:
    out = hd.build_cellmarker_dataset(
        tmp_path / "cm", which="all", cache_dir=FIXTURES / "cellmarker"
    )
    assert (out / "markers.parquet").exists()
    df = pd.read_parquet(out / "markers.parquet")
    assert not df.empty
    assert "CL:0000540" in set(df["cell_ontology_id"])  # normalized from CL_0000540
    card = (out / "README.md").read_text()
    assert "biodb_cellmarker" in card
    assert "config_name: default" in card
