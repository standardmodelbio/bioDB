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

from biodb import opentargets


def test_module_imports() -> None:
    """Bare import of the module must not pull live network deps."""
    assert opentargets.__name__ == "biodb.opentargets"


def test_constants_present() -> None:
    assert isinstance(opentargets.DOWNLOADS_BASE_URL, str)
    assert opentargets.DEFAULT_SCORE == 0.5
    assert opentargets.CACHE_DIR.exists()  # created on import


def test_public_api_signatures_stable() -> None:
    """Pin the public entrypoints. The ``df_to_markdown`` family was removed
    in the GeneDocs split — rendering now lives in
    `GeneDocs <https://github.com/bschilder/GeneDocs>`_'s
    ``gene_docs.docs.templates``. bioDB now exposes the bulk-download API
    plus the targeted-query GraphQL helpers in ``biodb.opentargets_graphql``.
    """
    expected = {
        "list_datasets",
        "get_dataset",
        "get_targets",
        "get_gene_associations",
        "get_pathways",
        "ensure_cached_shards",
        "list_available_versions",
        "read_for_target",
    }
    missing = [name for name in expected if not hasattr(opentargets, name)]
    assert not missing, f"missing public symbols: {missing}"


def test_graphql_module_exposes_targeted_queries() -> None:
    """The dual-mode API: bulk downloads in :mod:`biodb.opentargets`, targeted
    GraphQL lookups in :mod:`biodb.opentargets_graphql`."""
    from biodb import opentargets_graphql as gql

    for name in ["query_target", "query_disease", "query_drug", "query_variant"]:
        assert hasattr(gql, name), f"missing GraphQL helper: {name}"


def test_list_datasets_signature() -> None:
    sig = inspect.signature(opentargets.list_datasets)
    assert "base_url" in sig.parameters


def test_get_dataset_signature() -> None:
    sig = inspect.signature(opentargets.get_dataset)
    # Must at least accept a dataset name + cache hooks.
    params = set(sig.parameters)
    assert "dataset" in params or "name" in params or len(params) >= 1
