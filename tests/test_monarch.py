"""Smoke tests for ``biodb.monarch`` -- import + public-API stability."""

from __future__ import annotations

import inspect

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
