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
    """``_multi_handler`` doesn't touch owlready2 and is exercised here.

    Note: the helper moved into :mod:`biodb.ontology` when ``ontology_owl``
    was merged in. We pull it from the canonical home; the shim doesn't
    re-export private helpers.
    """
    from biodb.ontology import _multi_handler

    assert _multi_handler(["a", "b", "c"], "first") == "a"
    assert _multi_handler(["a", "b", "c"], "join", sep=".") == "a.b.c"
    assert _multi_handler(["a", "b", "c"], "all") == ["a", "b", "c"]
    with pytest.raises(ValueError, match="Invalid multi"):
        _multi_handler(["a"], "bogus")  # type: ignore[arg-type]


# Live integration test — RUN BY DEFAULT. SO is the smallest OBO ontology
# (~3 MB) so the cost-per-run is acceptable. Proves the generic OWL
# loader pulls + parses real upstream OBO Foundry data.


def test_sequence_ontology_loads_real_data() -> None:
    """Download + parse the real Sequence Ontology and verify a known
    SO class exists.

    Pinning a specific well-known SO term (``coding_sequence_variant``,
    used throughout VEP) means this catches both URL rot AND content
    changes that would silently remove the term from the public release.
    """
    pytest.importorskip("owlready2")
    ont = ontology_owl.get_sequence_ontology()
    assert ont is not None

    labels = ontology_owl.get_labels(ont)
    # SO has ~2000 classes; anything < 100 is an error page.
    assert len(labels) > 100, (
        f"Got only {len(labels)} labels — SO download probably returned an error page."
    )
    assert "coding_sequence_variant" in labels, (
        "Canonical SO class missing — upstream content drifted."
    )

    # The descendants walk is the load-bearing API for downstream users.
    desc = ontology_owl.get_descendants("coding_sequence_variant", ont=ont, return_as="label")
    assert isinstance(desc, list)
    assert "missense_variant" in desc, (
        "missense_variant should be a descendant of coding_sequence_variant in SO."
    )
