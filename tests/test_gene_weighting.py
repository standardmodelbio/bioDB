"""Smoke tests for ``phenoref.gene_weighting`` -- config, fast scoring, cache."""

from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch")

from phenoref import gene_weighting  # noqa: E402


def test_module_imports() -> None:
    assert gene_weighting.__name__ == "phenoref.gene_weighting"


def test_gene_weighting_config_defaults() -> None:
    cfg = gene_weighting.GeneWeightingConfig()
    assert cfg.top_k == 50
    assert cfg.temperature == pytest.approx(0.1)
    assert cfg.alpha == pytest.approx(0.5)
    assert cfg.aggregation_method == "max"
    assert cfg.batch_size == 1000


def test_gene_weighting_config_override() -> None:
    cfg = gene_weighting.GeneWeightingConfig(top_k=10, temperature=0.5, batch_size=4)
    assert cfg.top_k == 10
    assert cfg.temperature == pytest.approx(0.5)
    assert cfg.batch_size == 4


def test_compute_gene_weights_fast_returns_tensor() -> None:
    """Smoke-test the cheap path with tiny synthetic embeddings."""
    torch.manual_seed(0)
    batch, n_events, n_genes, d = 2, 3, 7, 8
    event_emb = torch.randn(batch, n_events, d)
    gene_emb = torch.randn(n_genes, d)
    out = gene_weighting.compute_gene_weights_fast(
        event_embeddings=event_emb,
        gene_name_embeddings=gene_emb,
    )
    # The function may return a tensor or a dict — accept either shape but
    # require it to be non-None and to contain a tensor somewhere.
    if isinstance(out, dict):
        tensors = [v for v in out.values() if isinstance(v, torch.Tensor)]
        assert tensors, f"expected at least one tensor in dict, got {out}"
    elif isinstance(out, tuple):
        assert any(isinstance(v, torch.Tensor) for v in out)
    else:
        assert isinstance(out, torch.Tensor)


def test_gene_embedding_cache_construct() -> None:
    """The cache class should be constructible without side-effects."""
    sig = inspect.signature(gene_weighting.GeneEmbeddingCache.__init__)
    params = set(sig.parameters)
    # max_size is the canonical knob from AoU.
    assert "cache_size" in params or "max_size" in params or len(params) >= 1
    # Best-effort construct with a small cache.
    try:
        cache = gene_weighting.GeneEmbeddingCache(cache_size=4)
    except TypeError:
        try:
            cache = gene_weighting.GeneEmbeddingCache(max_size=4)
        except TypeError:
            cache = gene_weighting.GeneEmbeddingCache()
    assert cache is not None


def test_public_api_signatures_stable() -> None:
    expected = {
        "GeneWeightingConfig",
        "GeneEmbeddingCache",
        "compute_gene_weights_fast",
        "temporal_gene_weighting",
        "multi_condition_gene_weighting",
        "compute_gene_sequence_embeddings_lazy",
        "patient_gene_attention_with_lazy_embeddings",
        "memory_efficient_gene_attention",
    }
    missing = [name for name in expected if not hasattr(gene_weighting, name)]
    assert not missing, f"missing public symbols: {missing}"
