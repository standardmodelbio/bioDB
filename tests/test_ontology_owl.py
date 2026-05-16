"""Smoke tests for :mod:`biodb.ontology_owl`.

Live OWL loads (Sequence Ontology, MONDO, …) are network-bound and
marked ``@pytest.mark.network`` so CI skips them. The default suite
exercises only the import surface + signature shape, matching the
convention in ``test_opentargets.py``.
"""

from __future__ import annotations

import inspect

import pytest

from biodb import ontology_owl


def test_module_imports() -> None:
    """Importing the module must not require ``owlready2`` (it's lazy)."""
    assert ontology_owl.__name__ == "biodb.ontology_owl"


def test_constants_present() -> None:
    assert ontology_owl.SEQUENCE_ONTOLOGY_URL.endswith("/so.owl")
    assert ontology_owl.HPO_URL.endswith("/hp.owl")
    assert ontology_owl.MONDO_URL.endswith("/mondo.owl")


def test_public_api_signatures_stable() -> None:
    expected = {
        "get_ontology",
        "get_sequence_ontology",
        "get_descendants",
        "get_ancestors",
        "get_labels",
        "get_ids",
        "get_id_map",
        "is_label_or_id",
        "map_terms",
        "get_mrca",
        "get_mrca_counts",
    }
    missing = [name for name in expected if not hasattr(ontology_owl, name)]
    assert not missing, f"missing public symbols: {missing}"


def test_get_descendants_default_return_format() -> None:
    """``return_as`` defaults to ``"label"`` (string default, not list sentinel)."""
    sig = inspect.signature(ontology_owl.get_descendants)
    assert sig.parameters["return_as"].default == "label"


def test_multi_handler_pure() -> None:
    """``_multi_handler`` doesn't touch owlready2 and is exercised here."""
    assert ontology_owl._multi_handler(["a", "b", "c"], "first") == "a"
    assert ontology_owl._multi_handler(["a", "b", "c"], "join", sep=".") == "a.b.c"
    assert ontology_owl._multi_handler(["a", "b", "c"], "all") == ["a", "b", "c"]
    with pytest.raises(ValueError, match="Invalid multi"):
        ontology_owl._multi_handler(["a"], "bogus")  # type: ignore[arg-type]


@pytest.mark.network
def test_sequence_ontology_loads() -> None:
    """Smoke check that the SO URL still resolves."""
    ont = ontology_owl.get_sequence_ontology()
    assert ont is not None
    labels = ontology_owl.get_labels(ont)
    assert "coding_sequence_variant" in labels or len(labels) > 100
