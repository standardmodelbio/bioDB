"""Smoke tests for ``biodb.ontology``.

Covers the pure-Python helpers (keyword expansion, flatten/hierarchical
helpers) on a tiny synthetic ontology. Heavy paths (Mondo OWL loading,
attention analysis on real models) are not exercised here.
"""

from __future__ import annotations

import inspect

import pytest

from biodb import ontology


def test_module_imports() -> None:
    assert ontology.__name__ == "biodb.ontology"


def test_lazy_optional_deps_have_flags() -> None:
    # The vendored module wraps each optional dep in a *_AVAILABLE flag.
    assert hasattr(ontology, "NETWORKX_AVAILABLE")
    assert hasattr(ontology, "OWLREADY2_AVAILABLE")
    assert hasattr(ontology, "SCIPY_AVAILABLE")
    assert hasattr(ontology, "NUMPY_AVAILABLE")
    assert hasattr(ontology, "MATPLOTLIB_AVAILABLE")
    assert hasattr(ontology, "DATASHADER_AVAILABLE")


def test_expand_keyword_sets_from_dict(tiny_ontology_dict) -> None:
    """N-hop expansion: 1-hop from `dementia` should grab its direct children."""
    pytest.importorskip("networkx")
    seeds = {"dementia": ["dementia"]}
    expanded = ontology.expand_keyword_sets_from_ontology(
        seed_keywords=seeds,
        ontology_dict=tiny_ontology_dict,
        n_hops=1,
        include_seeds=True,
    )
    assert "dementia" in expanded
    out = set(expanded["dementia"])
    assert "dementia" in out
    assert "alzheimer's disease" in out
    assert "vascular dementia" in out


def test_expand_keyword_sets_two_hop(tiny_ontology_dict) -> None:
    """2-hop expansion should also pick up grandchild ``early onset alzheimer's``."""
    pytest.importorskip("networkx")
    seeds = {"dementia": ["dementia"]}
    expanded = ontology.expand_keyword_sets_from_ontology(
        seed_keywords=seeds,
        ontology_dict=tiny_ontology_dict,
        n_hops=2,
        include_seeds=True,
    )
    out = set(expanded["dementia"])
    assert "early onset alzheimer's" in out


def test_public_api_signatures_stable() -> None:
    expected = {
        "expand_keyword_sets_from_ontology",
        "create_hierarchical_keyword_sets",
        "flatten_hierarchical_sets",
        "random_seed_keyword_sets",
        "extract_relationships_for_keyword_set",
        "format_relationship_as_text",
        "generate_keyword_sets_from_ontology",
        "load_mondo_ontology",
        "extract_graph_from_owl",
        "list_ontology_relationship_types",
        "count_ontology_relationship_types",
        "get_ontology_terms",
        "get_ontology_synonyms",
        "mondo_to_dict",
        "aggregate_keyword_set_embeddings",
        "compute_event_concept_similarity_matrix",
        "get_event_concept_attention_weights",
        "analyze_attention_weights",
        "get_attention_weight_for_concept",
        "ontology_to_gene_phenotype_matrix",
        "get_ontology_id_label_mapping",
        "compute_pairwise_ontological_similarity",
    }
    missing = [name for name in expected if not hasattr(ontology, name)]
    assert not missing, f"missing public symbols: {missing}"


def test_expand_keyword_sets_signature() -> None:
    sig = inspect.signature(ontology.expand_keyword_sets_from_ontology)
    params = set(sig.parameters)
    assert {"seed_keywords", "n_hops"}.issubset(params)


# ---------------------------------------------------------------------------
# Live integration test — RUN BY DEFAULT in CI.
#
# Mondo (the Monarch Disease Ontology) is the canonical disease ontology
# used throughout the AoU pipeline. It's ~25 MB OWL, the load takes a few
# seconds, and the file changes monthly — exactly the kind of upstream
# that needs real verification on every CI run, not a stub against
# synthetic dicts.
# ---------------------------------------------------------------------------


def test_load_mondo_ontology_from_live_server(tmp_path) -> None:
    """Download + parse the real MONDO OWL and verify the documented
    return shape + a canonical disease term exists.

    Catches both URL rot (purl.obolibrary.org redirects breaking) and
    Mondo content changes (a canonical disease class disappearing).
    """
    pytest.importorskip("owlready2")
    pytest.importorskip("networkx")

    graph = ontology.load_mondo_ontology(cache_dir=str(tmp_path))
    # Mondo has tens of thousands of disease classes.
    assert graph.number_of_nodes() > 5000, (
        f"Mondo graph has only {graph.number_of_nodes()} nodes — "
        "probable corrupt download or upstream change."
    )

    # ``Alzheimer disease`` is one of the most-cited MONDO classes
    # (MONDO:0004975); it's a load-bearing concept for the AoU
    # phenome pipeline. If it vanishes, downstream gene-knowledge
    # selection breaks.
    node_names = {str(n).lower() for n in graph.nodes}
    assert any("alzheimer" in name for name in node_names), (
        "No MONDO node contains 'alzheimer' — Mondo download likely broken."
    )
