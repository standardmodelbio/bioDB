"""Smoke tests for :mod:`biodb.transform`."""

from __future__ import annotations

import numpy as np

from biodb import transform, utils


def test_module_imports() -> None:
    assert transform.__name__ == "biodb.transform"
    assert hasattr(transform, "create_gene_association_matrix")


def test_transform_re_exports_utils_impl() -> None:
    """``biodb.transform`` is a re-export — same callable as ``biodb.utils``."""
    assert transform.create_gene_association_matrix is utils.create_gene_association_matrix


def test_create_gene_association_matrix_dense_via_transform(small_associations) -> None:
    """End-to-end: the matrix builder is callable via the new module path."""
    X, meta = transform.create_gene_association_matrix(
        small_associations,
        source_id_col="sourceId",
        target_id_col="HGNC",
        score_col="score",
        convert_to_dense=True,
        verbose=False,
    )
    assert isinstance(X, np.ndarray)
    assert X.shape == (3, 4)
    assert {"obs", "var", "metadata"}.issubset(meta)
