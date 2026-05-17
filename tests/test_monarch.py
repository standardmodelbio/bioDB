"""Smoke + live integration tests for ``biodb.monarch``.

The signature/import smoke tests at the top are the fast-feedback layer.
The live integration test at the bottom downloads a real Monarch TSV
from ``data.monarchinitiative.org`` and proves the downloader still
works against the upstream knowledge graph.
"""

from __future__ import annotations

import inspect

import pandas as pd

from biodb import monarch


def test_module_imports() -> None:
    assert monarch.__name__ == "biodb.monarch"


def test_constants_present() -> None:
    assert monarch.ASSOCIATIONS_BASE_URL.startswith("https://data.monarchinitiative.org/")
    assert monarch.CAUSAL_GENE_TO_DISEASE_URL.endswith(".tsv.gz")
    assert monarch.CACHE_DIR.exists()


def test_public_api_signatures_stable() -> None:
    expected = {
        "list_datasets",
        "get_dataset",
        "read_causal_gene_to_disease_association",
        "get_gene_associations",
    }
    missing = [name for name in expected if not hasattr(monarch, name)]
    assert not missing, f"missing public symbols: {missing}"


def test_get_gene_associations_signature() -> None:
    sig = inspect.signature(monarch.get_gene_associations)
    # We expect a `force` and a `verbose` knob like every Monarch helper.
    params = set(sig.parameters)
    assert "verbose" in params


def test_read_causal_gene_to_disease_association_signature() -> None:
    sig = inspect.signature(monarch.read_causal_gene_to_disease_association)
    params = set(sig.parameters)
    # The fn at minimum takes a save_path / url / verbose-ish knob.
    assert len(params) >= 1


# ---------------------------------------------------------------------------
# Live integration test — RUN BY DEFAULT in CI.
#
# Downloads the real causal-gene-to-disease association TSV (~few MB
# gzipped) from data.monarchinitiative.org and verifies the schema bioDB
# advertises. This is the test that actually proves "can we download
# Monarch data?" — the previous test file had ZERO real-data coverage.
# ---------------------------------------------------------------------------


def test_read_causal_gene_to_disease_association_from_live_server(tmp_path) -> None:
    """Download + parse the real causal-gene-to-disease TSV.

    Monarch publishes this at
    ``data.monarchinitiative.org/.../causal_gene_to_disease_association.all.tsv.gz``.
    Schema is documented in the Monarch KG README; this test pins the
    columns + a sanity row count.
    """
    df = monarch.read_causal_gene_to_disease_association(
        cache_dir=str(tmp_path), output_format="pandas", verbose=0
    )
    assert isinstance(df, pd.DataFrame)
    # Monarch's causal-gene-to-disease catalog has thousands of rows.
    assert len(df) > 1000, (
        f"Got only {len(df)} rows — Monarch download likely returned an error page."
    )

    # Documented columns from the Monarch KG TSV schema. These power
    # downstream gene-association matrix construction.
    columns = set(df.columns)
    for required in ("subject", "predicate", "object"):
        assert required in columns, f"Required column {required!r} missing from {sorted(columns)}"

    # The "subject" column is gene IDs in CURIE form (e.g. 'HGNC:1100').
    sample_subject = df["subject"].dropna().iloc[0]
    assert ":" in sample_subject, f"Sample subject {sample_subject!r} doesn't look like a CURIE."
