"""Smoke tests for ``biodb.utils`` -- random seeding, similarity helpers,
``filter_adaptive`` and ``create_gene_association_matrix``."""

from __future__ import annotations

import numpy as np
import pytest

from biodb import utils


def test_random_seed_constant() -> None:
    assert utils.RANDOM_SEED == 42


def test_set_random_seed_repeatable() -> None:
    utils.set_random_seed(123)
    a = np.random.randint(0, 10_000, size=5)
    utils.set_random_seed(123)
    b = np.random.randint(0, 10_000, size=5)
    assert (a == b).all()


def test_set_random_seed_default_seed_does_not_raise() -> None:
    utils.set_random_seed(None)


def test_l2_normalize_unit_rows() -> None:
    x = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    n = utils.l2_normalize(x)
    assert n.shape == x.shape
    assert pytest.approx(np.linalg.norm(n[0]), rel=1e-4) == 1.0
    assert pytest.approx(np.linalg.norm(n[1]), rel=1e-4) == 1.0
    # zero rows must stay zero (no NaN)
    assert np.linalg.norm(n[2]) == 0.0


def test_cosine_similarity_shape_and_self() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=(4, 6)).astype(np.float32)
    a = utils.l2_normalize(a)
    sim = utils.cosine_similarity(a, a)
    assert sim.shape == (4, 4)
    assert np.allclose(np.diag(sim), 1.0, atol=1e-4)


def test_euclidean_similarity_negative_distance() -> None:
    a = np.array([[0.0, 0.0]], dtype=np.float32)
    b = np.array([[3.0, 4.0]], dtype=np.float32)
    sim = utils.euclidean_similarity(a, b)
    # ||a-b||^2 = 25, similarity is -||..||^2 with this implementation
    assert pytest.approx(sim[0, 0], abs=1e-4) == -25.0


def test_dot_product_similarity_shape() -> None:
    a = np.eye(3, dtype=np.float32)
    b = np.eye(3, dtype=np.float32)
    sim = utils.dot_product_similarity(a, b)
    assert sim.shape == (3, 3)
    assert np.allclose(sim, np.eye(3), atol=1e-6)


def test_count_tokens_approximate() -> None:
    # Approximate path avoids the tiktoken dependency entirely.
    n = utils.count_tokens("hello world", approximate=True)
    assert n == len("hello world") // 4


def test_count_tokens_list_input_approximate() -> None:
    """The verbatim AoU port returns ``len(text) // 4`` before joining lists,
    so a 2-element list maps to ``2 // 4 == 0`` on the approximate path."""
    n = utils.count_tokens(["hello", "world"], approximate=True, sep=" ")
    assert n == 0


def test_filter_adaptive_keeps_top_percentile(small_associations) -> None:
    df = utils.filter_adaptive(
        small_associations,
        source_id_col="sourceId",
        target_id_col="HGNC",
        percentile=0.5,
        sort_by="score",
        verbose=False,
    )
    # Each source had 4 genes; keeping top 50% -> 2 per source -> 6 rows total.
    counts = df.groupby("sourceId")["HGNC"].nunique()
    assert (counts <= 2).all()
    assert len(df) <= len(small_associations)


def test_create_gene_association_matrix_dense(small_associations) -> None:
    X, meta = utils.create_gene_association_matrix(
        small_associations,
        source_id_col="sourceId",
        target_id_col="HGNC",
        score_col="score",
        convert_to_dense=True,
        verbose=False,
    )
    assert isinstance(X, np.ndarray)
    assert X.shape == (3, 4)
    assert X.dtype == np.float32
    assert "obs" in meta and "var" in meta and "metadata" in meta
    assert list(meta["obs"]["sourceId"]) == ["DIS:1", "DIS:2", "DIS:3"]


def test_create_gene_association_matrix_sparse(small_associations) -> None:
    from scipy.sparse import issparse

    X, meta = utils.create_gene_association_matrix(
        small_associations,
        source_id_col="sourceId",
        target_id_col="HGNC",
        score_col="score",
        convert_to_dense=False,
        verbose=False,
    )
    assert issparse(X)
    assert X.shape == (3, 4)
    assert meta["metadata"]["matrix_shape"] == (3, 4)


def test_create_gene_association_matrix_missing_columns_raises(small_associations) -> None:
    with pytest.raises(ValueError, match="Missing required columns"):
        utils.create_gene_association_matrix(
            small_associations.drop(columns=["score"]),
            verbose=False,
        )


def test_create_gene_association_matrix_requires_data_when_no_cache() -> None:
    with pytest.raises(ValueError, match="associations is required"):
        utils.create_gene_association_matrix(associations=None, verbose=False)
