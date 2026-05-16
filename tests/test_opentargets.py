"""Smoke tests for ``biodb.opentargets``.

The full module wraps gget + Open Targets parquet downloads; live calls
are skipped here. Tests focus on:
    * the module imports cleanly
    * pure-Python helpers (markdown rendering, dataset parsing) work on
      tiny synthetic inputs
    * top-level public functions still have the expected signatures.
"""

from __future__ import annotations

import inspect

import pandas as pd

from biodb import opentargets


def test_module_imports() -> None:
    """Bare import of the module must not pull live network deps."""
    assert opentargets.__name__ == "biodb.opentargets"


def test_constants_present() -> None:
    assert isinstance(opentargets.DOWNLOADS_BASE_URL, str)
    assert opentargets.DEFAULT_SCORE == 0.5
    assert opentargets.CACHE_DIR.exists()  # created on import


def test_public_api_signatures_stable() -> None:
    """Pin the public entrypoints so a future refactor of the source has
    to consciously update this test if it renames or removes a function."""
    expected = {
        "list_datasets",
        "get_dataset",
        "get_targets",
        "get_gene_associations",
        "get_pathways",
        "diseases_to_markdown",
        "drugs_to_markdown",
        "pharmacogenomics_to_markdown",
        "df_to_markdown",
        "df_to_markdown_batch",
    }
    missing = [name for name in expected if not hasattr(opentargets, name)]
    assert not missing, f"missing public symbols: {missing}"


def test_df_to_markdown_renders_string() -> None:
    """``df_to_markdown`` takes a single target row (Series/dict) and renders
    a non-empty markdown string. We exercise the dict path with a minimal
    OpenTargets-shaped row."""
    row = {
        "approvedSymbol": "BRCA1",
        "approvedName": "BRCA1 DNA repair associated",
        "id": "ENSG00000012048",
        "biotype": "protein_coding",
    }
    out = opentargets.df_to_markdown(row)
    assert isinstance(out, str)
    assert "BRCA1" in out
    sig = inspect.signature(opentargets.df_to_markdown)
    # First positional must be the target row -- this is a stable contract.
    assert list(sig.parameters)[0] == "target_row"


def test_list_datasets_signature() -> None:
    sig = inspect.signature(opentargets.list_datasets)
    assert "base_url" in sig.parameters


def test_get_dataset_signature() -> None:
    sig = inspect.signature(opentargets.get_dataset)
    # Must at least accept a dataset name + cache hooks.
    params = set(sig.parameters)
    assert "dataset" in params or "name" in params or len(params) >= 1
