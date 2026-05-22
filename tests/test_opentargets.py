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
import pytest

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
        "variants_for_target",
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


# ---------------------------------------------------------------------------
# Live integration tests — RUN BY DEFAULT in CI.
#
# bioDB advertises Open Targets bulk Parquet downloads as a core feature.
# The previous test file had ZERO real-data coverage — every assertion
# checked signatures and constants, none verified the downloader actually
# works against the upstream FTP at ftp.ebi.ac.uk.
#
# These tests use ``limit_files=1`` to download a single Parquet shard
# from one of the smaller datasets, keeping the per-CI cost bounded
# while still proving the end-to-end download + parse path.
# ---------------------------------------------------------------------------


def test_list_datasets_against_live_server() -> None:
    """The Open Targets FTP root really exposes the documented datasets
    (``target``, ``association_overall_direct``, ``drug``, …).

    Catches both URL rot (release directory moved) and content changes
    (a documented dataset disappearing)."""
    datasets = opentargets.list_datasets()
    assert isinstance(datasets, dict)
    assert len(datasets) > 5, (
        f"Got only {len(datasets)} datasets — Open Targets release "
        f"normally has dozens. Probable upstream change."
    )
    for required in ("target", "association_overall_direct"):
        assert required in datasets, f"Documented dataset {required!r} missing from FTP listing."
    # Each URL must be on the EBI FTP server.
    for url in datasets.values():
        assert "ebi.ac.uk" in url


def test_get_dataset_target_downloads_one_real_parquet_shard(tmp_path) -> None:
    """Download ONE real Parquet shard from the ``target`` dataset and
    verify it parses with the documented schema.

    ``target`` is the OT gene-metadata dataset (no association evidence),
    so per-shard size is bounded. ``limit_files=1`` is the key knob that
    keeps the per-CI cost bounded — without it, the full ``target``
    dataset would be 100+ MB across many shards.
    """
    df = opentargets.get_dataset(
        "target",
        version=opentargets.DEFAULT_VERSION,
        cache_dir=tmp_path,
        limit_files=1,
        verbose=0,
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 100, (
        f"Got only {len(df)} rows from the target shard — looks like "
        "an empty response or error page."
    )

    # The ``target`` dataset is documented to carry these columns; if they
    # disappear, every downstream pipeline breaks.
    columns = set(df.columns)
    for required in ("id", "approvedSymbol", "biotype"):
        assert required in columns, (
            f"Required column {required!r} missing from target shard. Got: {sorted(columns)[:20]}"
        )

    # Sanity: every row should have an Ensembl gene id starting with ``ENSG``.
    assert df["id"].astype(str).str.startswith("ENSG").all(), (
        "Some target rows don't have an ENSG-prefixed id — schema drift?"
    )


def test_get_dataset_caches_locally(tmp_path) -> None:
    """Second call with the same ``cache_dir`` reuses the local file
    instead of redownloading."""
    import time

    t0 = time.monotonic()
    df1 = opentargets.get_dataset(
        "target",
        version=opentargets.DEFAULT_VERSION,
        cache_dir=tmp_path,
        limit_files=1,
        verbose=0,
    )
    dt_download = time.monotonic() - t0

    t0 = time.monotonic()
    df2 = opentargets.get_dataset(
        "target",
        version=opentargets.DEFAULT_VERSION,
        cache_dir=tmp_path,
        limit_files=1,
        verbose=0,
    )
    dt_cached = time.monotonic() - t0

    assert len(df1) == len(df2)
    assert set(df1.columns) == set(df2.columns)
    # Cached re-read should be at least 2x faster than the original
    # download (and never slower than 5 s on commodity hardware).
    assert dt_cached < max(dt_download / 2, 5.0), (
        f"Cached read took {dt_cached:.1f}s vs initial {dt_download:.1f}s — "
        "the local cache layer probably re-downloaded."
    )


def _write_synthetic_variant_shards(cache_root, version="TESTVER"):
    """Build a 2-shard variant parquet at the layout ``ensure_cached_shards``
    expects. Each row is a real OT-25.12-shaped variant record with a
    nested ``transcriptConsequences`` list-of-struct so the helper has
    something to filter on."""
    import polars as pl

    dataset_dir = cache_root / version / "variant"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    shard0 = pl.DataFrame(
        {
            "variantId": ["1_100_A_T", "1_200_C_G", "1_300_G_A"],
            "chromosome": ["1", "1", "1"],
            "transcriptConsequences": [
                [{"transcriptId": "ENST1", "targetId": "ENSG_A"}],
                [{"transcriptId": "ENST2", "targetId": "ENSG_B"}],
                # multi-transcript variant with both targets
                [
                    {"transcriptId": "ENST3a", "targetId": "ENSG_A"},
                    {"transcriptId": "ENST3b", "targetId": "ENSG_C"},
                ],
            ],
        }
    )
    shard1 = pl.DataFrame(
        {
            "variantId": ["2_100_T_C"],
            "chromosome": ["2"],
            "transcriptConsequences": [
                [{"transcriptId": "ENST4", "targetId": "ENSG_A"}],
            ],
        }
    )
    shard0.write_parquet(dataset_dir / "part-00000.parquet")
    shard1.write_parquet(dataset_dir / "part-00001.parquet")
    return dataset_dir


def test_variants_for_target_filters_nested_targetid(tmp_path) -> None:
    """The helper's load-bearing guarantee: filter on
    ``transcriptConsequences[*].targetId`` even though that's nested
    inside a list-of-struct that pyarrow's pushdown can't reach."""
    import polars as pl

    _write_synthetic_variant_shards(tmp_path)
    df = opentargets.variants_for_target(
        "ENSG_A",
        version="TESTVER",
        cache_dir=tmp_path,
    )
    # ENSG_A appears in shard 0 rows 0 and 2, and shard 1 row 0 -> 3 total.
    assert isinstance(df, pl.DataFrame)
    assert sorted(df["variantId"].to_list()) == sorted(["1_100_A_T", "1_300_G_A", "2_100_T_C"])


def test_variants_for_target_unknown_gene_returns_empty(tmp_path) -> None:
    """Gene with no matching ``targetId`` in any transcript consequence
    returns an empty DataFrame rather than raising."""
    import polars as pl

    _write_synthetic_variant_shards(tmp_path)
    df = opentargets.variants_for_target(
        "ENSG_DOES_NOT_EXIST",
        version="TESTVER",
        cache_dir=tmp_path,
    )
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 0


def test_variants_for_target_column_projection(tmp_path) -> None:
    """``columns=`` restricts the returned schema -- useful for big
    parquet shards where you only need variantId + position."""
    _write_synthetic_variant_shards(tmp_path)
    df = opentargets.variants_for_target(
        "ENSG_A",
        version="TESTVER",
        cache_dir=tmp_path,
        columns=["variantId", "chromosome"],
    )
    assert set(df.columns) == {"variantId", "chromosome"}


# ---------------------------------------------------------------------------
# Configurable default datasets (DEFAULT_GENE_ASSOCIATION_DATASETS +
# BIODB_OT_GENE_ASSOC_DATASETS env var + validation)
# ---------------------------------------------------------------------------


def test_supported_gene_association_datasets_is_authoritative_allowlist() -> None:
    """The supported set names every dataset ``get_gene_associations`` has
    a ``_prepare_*`` handler for. Drift here means silently-dropped rows
    or runtime crashes -- pin it explicitly."""
    assert (
        frozenset(
            {
                "disease-to-gene",
                "known_drug",
                "pharmacogenomics",
                "mouse_phenotype",
                "target_essentiality",
                "expression",
            }
        )
        == opentargets.SUPPORTED_GENE_ASSOCIATION_DATASETS
    )


def test_default_gene_association_datasets_subset_of_supported() -> None:
    """The default list MUST be a subset of the allow-list; otherwise
    calling ``get_gene_associations()`` with no args would raise."""
    assert set(opentargets.DEFAULT_GENE_ASSOCIATION_DATASETS) <= (
        opentargets.SUPPORTED_GENE_ASSOCIATION_DATASETS
    )


def test_resolve_gene_association_datasets_caller_kwarg_wins(monkeypatch) -> None:
    """Explicit kwarg overrides both the env var and the module default."""
    monkeypatch.setenv("BIODB_OT_GENE_ASSOC_DATASETS", "known_drug")
    out = opentargets._resolve_gene_association_datasets(["expression"])
    assert out == ["expression"]


def test_resolve_gene_association_datasets_env_var_wins_over_default(monkeypatch) -> None:
    """When the caller passes ``None``, ``BIODB_OT_GENE_ASSOC_DATASETS``
    overrides the module-level constant."""
    monkeypatch.setenv("BIODB_OT_GENE_ASSOC_DATASETS", "known_drug, expression")
    out = opentargets._resolve_gene_association_datasets(None)
    assert out == ["known_drug", "expression"]


def test_resolve_gene_association_datasets_falls_back_to_module_default(monkeypatch) -> None:
    """No kwarg + no env var -> the module-level
    ``DEFAULT_GENE_ASSOCIATION_DATASETS``."""
    monkeypatch.delenv("BIODB_OT_GENE_ASSOC_DATASETS", raising=False)
    out = opentargets._resolve_gene_association_datasets(None)
    assert out == opentargets.DEFAULT_GENE_ASSOCIATION_DATASETS


def test_resolve_gene_association_datasets_module_override(monkeypatch) -> None:
    """Users can override ``DEFAULT_GENE_ASSOCIATION_DATASETS`` at module
    level for a session-wide change -- the resolver picks it up because
    it reads the attribute live, not at function-definition time."""
    monkeypatch.delenv("BIODB_OT_GENE_ASSOC_DATASETS", raising=False)
    monkeypatch.setattr(opentargets, "DEFAULT_GENE_ASSOCIATION_DATASETS", ["known_drug"])
    assert opentargets._resolve_gene_association_datasets(None) == ["known_drug"]


def test_resolve_gene_association_datasets_raises_on_unknown(monkeypatch) -> None:
    """Unknown dataset names previously silently produced no rows (the
    function looped through ``if "X" in datasets:`` checks; nothing
    matched, no error). Now they raise with a list of supported names."""
    monkeypatch.delenv("BIODB_OT_GENE_ASSOC_DATASETS", raising=False)
    with pytest.raises(ValueError, match="Unknown OT gene-association dataset"):
        opentargets._resolve_gene_association_datasets(["known_drug", "made_up_dataset"])
