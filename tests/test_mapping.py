"""Smoke tests for :mod:`biodb.mapping`."""

from __future__ import annotations

import inspect

import pandas as pd
import pytest

from biodb import mapping


def test_module_imports_offline() -> None:
    """Importing must not require gprofiler — it's lazy."""
    assert mapping.__name__ == "biodb.mapping"
    assert hasattr(mapping, "map_gene_ids")


def test_map_gene_ids_signature() -> None:
    sig = inspect.signature(mapping.map_gene_ids)
    params = sig.parameters
    assert "df" in params
    assert params["target_id_col"].default == "targetId"
    assert params["target_namespace"].default == "HGNC"
    assert params["organism"].default == "hsapiens"


def test_map_gene_ids_rejects_missing_column() -> None:
    df = pd.DataFrame({"foo": ["a", "b"]})
    with pytest.raises(ValueError, match="not found in DataFrame"):
        mapping.map_gene_ids(df, target_id_col="missing", verbose=False)


def test_map_gene_ids_empty_df_no_network() -> None:
    """Empty input short-circuits before touching gprofiler."""
    out = mapping.map_gene_ids(pd.DataFrame({"targetId": []}), verbose=False)
    assert out.empty
    assert "targetId" in out.columns


@pytest.mark.network
def test_map_gene_ids_ensembl_to_hgnc() -> None:
    pytest.importorskip("gprofiler")
    df = pd.DataFrame({"targetId": ["ENSG00000012048", "ENSG00000141510"], "score": [0.5, 0.7]})
    out = mapping.map_gene_ids(df, target_id_col="targetId", target_namespace="HGNC", verbose=False)
    assert "HGNC" in out.columns
    assert len(out) == 2
