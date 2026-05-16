"""Smoke tests for ``biodb.utils`` -- random seeding, similarity helpers,
``filter_adaptive`` and ``create_gene_association_matrix``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.sparse import issparse

from biodb import utils

# --------------------------------------------------------------------------
# Random seed
# --------------------------------------------------------------------------


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


def test_set_random_seed_seeds_torch_too() -> None:
    """Both numpy and torch must come out identical for a given seed."""
    utils.set_random_seed(7)
    a_np = np.random.rand(3)
    a_th = torch.rand(3)
    utils.set_random_seed(7)
    b_np = np.random.rand(3)
    b_th = torch.rand(3)
    assert np.allclose(a_np, b_np)
    assert torch.allclose(a_th, b_th)


# --------------------------------------------------------------------------
# Similarity helpers
# --------------------------------------------------------------------------


def test_l2_normalize_numpy_unit_rows() -> None:
    x = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    n = utils.l2_normalize(x)
    assert n.shape == x.shape
    assert pytest.approx(np.linalg.norm(n[0]), rel=1e-4) == 1.0
    assert pytest.approx(np.linalg.norm(n[1]), rel=1e-4) == 1.0
    assert np.linalg.norm(n[2]) == 0.0


def test_l2_normalize_handles_readonly_numpy_input() -> None:
    """Read-only buffers must be copied internally."""
    x = np.array([[3.0, 4.0]], dtype=np.float32)
    x.setflags(write=False)
    n = utils.l2_normalize(x)
    assert pytest.approx(np.linalg.norm(n[0]), rel=1e-4) == 1.0


def test_l2_normalize_torch_tensor_passthrough() -> None:
    t = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    n = utils.l2_normalize(t)
    assert isinstance(n, torch.Tensor)
    assert pytest.approx(float(torch.linalg.norm(n[0])), rel=1e-4) == 1.0


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
    assert pytest.approx(sim[0, 0], abs=1e-4) == -25.0


def test_dot_product_similarity_shape() -> None:
    a = np.eye(3, dtype=np.float32)
    b = np.eye(3, dtype=np.float32)
    sim = utils.dot_product_similarity(a, b)
    assert sim.shape == (3, 3)
    assert np.allclose(sim, np.eye(3), atol=1e-6)


# --------------------------------------------------------------------------
# count_tokens
# --------------------------------------------------------------------------


def test_count_tokens_approximate() -> None:
    n = utils.count_tokens("hello world", approximate=True)
    assert n == len("hello world") // 4


def test_count_tokens_list_input_approximate() -> None:
    """The verbatim AoU port returns ``len(text) // 4`` before joining lists,
    so a 2-element list maps to ``2 // 4 == 0`` on the approximate path."""
    n = utils.count_tokens(["hello", "world"], approximate=True, sep=" ")
    assert n == 0


def test_count_tokens_real_tiktoken() -> None:
    """Drive the tiktoken path (not approximate) for string + list inputs."""
    pytest.importorskip("tiktoken")
    n_str = utils.count_tokens("hello world", approximate=False)
    n_list = utils.count_tokens(["hello", "world"], approximate=False, sep=" ")
    assert n_str > 0
    assert n_list == n_str


# --------------------------------------------------------------------------
# filter_adaptive
# --------------------------------------------------------------------------


def test_filter_adaptive_keeps_top_percentile(small_associations) -> None:
    df = utils.filter_adaptive(
        small_associations,
        source_id_col="sourceId",
        target_id_col="HGNC",
        percentile=0.5,
        sort_by="score",
        verbose=False,
    )
    counts = df.groupby("sourceId")["HGNC"].nunique()
    assert (counts <= 2).all()
    assert len(df) <= len(small_associations)
    # sourceId column must survive the filter (regression: pandas-2.2 apply
    # behavior used to drop the groupby key).
    assert "sourceId" in df.columns


def test_filter_adaptive_target_id_alias_added(small_associations) -> None:
    """When target_id_col != 'targetId', the function adds a 'targetId' alias."""
    df = utils.filter_adaptive(
        small_associations,
        source_id_col="sourceId",
        target_id_col="HGNC",
        percentile=0.5,
        verbose=False,
    )
    assert "targetId" in df.columns
    assert (df["targetId"] == df["HGNC"]).all()


def test_filter_adaptive_verbose_prints(small_associations, capsys) -> None:
    utils.filter_adaptive(
        small_associations,
        source_id_col="sourceId",
        target_id_col="HGNC",
        percentile=0.5,
        verbose=True,
    )
    captured = capsys.readouterr().out
    assert "Initial:" in captured
    assert "Before filtering" in captured
    assert "After percentile filtering" in captured


def test_filter_adaptive_pval_threshold(small_associations) -> None:
    df = utils.filter_adaptive(
        small_associations,
        source_id_col="sourceId",
        target_id_col="HGNC",
        pval_threshold=0.005,
        verbose=False,
    )
    assert (df["pval"] <= 0.005).all()


def test_filter_adaptive_pval_threshold_missing_column_raises(
    small_associations,
) -> None:
    with pytest.raises(ValueError, match="pval_threshold"):
        utils.filter_adaptive(
            small_associations.drop(columns=["pval"]),
            source_id_col="sourceId",
            target_id_col="HGNC",
            pval_threshold=0.005,
            verbose=False,
        )


def test_filter_adaptive_fdr_threshold(small_associations) -> None:
    df = small_associations.copy()
    df["fdr"] = df["pval"]
    out = utils.filter_adaptive(
        df,
        source_id_col="sourceId",
        target_id_col="HGNC",
        fdr_threshold=0.005,
        verbose=False,
    )
    assert (out["fdr"] <= 0.005).all()


def test_filter_adaptive_fdr_threshold_missing_column_raises(
    small_associations,
) -> None:
    with pytest.raises(ValueError, match="fdr_threshold"):
        utils.filter_adaptive(
            small_associations,
            source_id_col="sourceId",
            target_id_col="HGNC",
            fdr_threshold=0.005,
            verbose=False,
        )


def test_filter_adaptive_score_auto_from_pval(small_associations) -> None:
    """If score_col is missing but pval is present, the function derives it."""
    df = small_associations.drop(columns=["score"])
    out = utils.filter_adaptive(
        df,
        source_id_col="sourceId",
        target_id_col="HGNC",
        percentile=0.5,
        sort_by="score",
        verbose=False,
    )
    assert "score" in out.columns


def test_filter_adaptive_score_missing_no_pval_raises(small_associations) -> None:
    df = small_associations.drop(columns=["score", "pval"])
    with pytest.raises(ValueError, match="neither"):
        utils.filter_adaptive(
            df,
            source_id_col="sourceId",
            target_id_col="HGNC",
            sort_by="score",
            verbose=False,
        )


def test_filter_adaptive_sort_by_pval(small_associations) -> None:
    out = utils.filter_adaptive(
        small_associations,
        source_id_col="sourceId",
        target_id_col="HGNC",
        percentile=0.5,
        sort_by="pval",
        verbose=False,
    )
    assert len(out) > 0


def test_filter_adaptive_sort_by_pval_missing_column_raises(
    small_associations,
) -> None:
    with pytest.raises(ValueError, match="pval"):
        utils.filter_adaptive(
            small_associations.drop(columns=["pval"]),
            source_id_col="sourceId",
            target_id_col="HGNC",
            sort_by="pval",
            verbose=False,
        )


def test_filter_adaptive_sort_by_arbitrary_column(small_associations) -> None:
    out = utils.filter_adaptive(
        small_associations,
        source_id_col="sourceId",
        target_id_col="HGNC",
        percentile=0.5,
        sort_by="label",
        verbose=False,
    )
    assert len(out) > 0


def test_filter_adaptive_sort_by_unknown_column_raises(small_associations) -> None:
    with pytest.raises(ValueError, match="not found"):
        utils.filter_adaptive(
            small_associations,
            source_id_col="sourceId",
            target_id_col="HGNC",
            sort_by="nope",
            verbose=False,
        )


def test_filter_adaptive_min_max_genes_clamps(small_associations) -> None:
    """min_genes forces at least N per source; max_genes caps it."""
    df = utils.filter_adaptive(
        small_associations,
        source_id_col="sourceId",
        target_id_col="HGNC",
        percentile=0.99,
        min_genes=3,
        max_genes=3,
        verbose=False,
    )
    counts = df.groupby("sourceId")["HGNC"].nunique()
    assert (counts == 3).all()


# --------------------------------------------------------------------------
# create_gene_association_matrix
# --------------------------------------------------------------------------


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


def test_create_gene_association_matrix_missing_columns_raises(
    small_associations,
) -> None:
    with pytest.raises(ValueError, match="Missing required columns"):
        utils.create_gene_association_matrix(
            small_associations.drop(columns=["score"]),
            verbose=False,
        )


def test_create_gene_association_matrix_requires_data_when_no_cache() -> None:
    with pytest.raises(ValueError, match="associations is required"):
        utils.create_gene_association_matrix(associations=None, verbose=False)


def test_create_gene_association_matrix_invalid_aggregation_raises(
    small_associations,
) -> None:
    with pytest.raises(ValueError, match="Invalid aggregation_method"):
        utils.create_gene_association_matrix(
            small_associations,
            source_id_col="sourceId",
            target_id_col="HGNC",
            aggregation_method="not_an_agg",
            verbose=False,
        )


def test_create_gene_association_matrix_verbose_prints(small_associations, capsys) -> None:
    utils.create_gene_association_matrix(
        small_associations,
        source_id_col="sourceId",
        target_id_col="HGNC",
        verbose=True,
    )
    captured = capsys.readouterr().out
    assert "Creating gene association matrix" in captured
    assert "Matrix shape" in captured
    assert "Sparsity" in captured


def test_create_gene_association_matrix_chunked_path(small_associations) -> None:
    """Force the chunked-aggregation branch by setting chunk_size very small."""
    X, _ = utils.create_gene_association_matrix(
        small_associations,
        source_id_col="sourceId",
        target_id_col="HGNC",
        chunk_size=2,
        verbose=False,
    )
    assert X.shape == (3, 4)


def test_create_gene_association_matrix_with_group_database_dataset_label(
    small_associations,
) -> None:
    """Group / database / dataset / label columns are mapped onto ``obs``."""
    df = small_associations.copy()
    df["database"] = "fake_db"
    df["dataset"] = "fake_ds"
    _, meta = utils.create_gene_association_matrix(
        df,
        source_id_col="sourceId",
        target_id_col="HGNC",
        group_col="group",
        label_col="label",
        verbose=False,
    )
    obs = meta["obs"]
    assert "group" in obs.columns
    assert "label" in obs.columns
    assert "database" in obs.columns
    assert "dataset" in obs.columns


def test_create_gene_association_matrix_fillna_with_constant() -> None:
    """fillna=<float> replaces NaN entries that survive aggregation.

    ``std`` of a single-row group is NaN, so the (DIS:1, BRCA1) cell
    enters the sparse buffer as NaN; fillna replaces it with -1.0.
    """
    df = pd.DataFrame(
        {
            "sourceId": ["DIS:1", "DIS:2", "DIS:2"],
            "HGNC": ["BRCA1", "TP53", "TP53"],
            "score": [1.0, 0.4, 0.6],
        }
    )
    X, _ = utils.create_gene_association_matrix(
        df,
        source_id_col="sourceId",
        target_id_col="HGNC",
        fillna=-1.0,
        convert_to_dense=True,
        aggregation_method="std",
        verbose=False,
    )
    assert np.any(X == -1.0)


def test_create_gene_association_matrix_fillna_mean() -> None:
    """fillna='mean' replaces NaN cells with per-column means."""
    df = pd.DataFrame(
        {
            "sourceId": ["DIS:1", "DIS:2", "DIS:2"],
            "HGNC": ["BRCA1", "BRCA1", "BRCA1"],
            "score": [1.0, 0.4, 0.6],
        }
    )
    X, _ = utils.create_gene_association_matrix(
        df,
        source_id_col="sourceId",
        target_id_col="HGNC",
        fillna="mean",
        convert_to_dense=True,
        aggregation_method="std",
        verbose=False,
    )
    # DIS:2 has std~0.14, DIS:1 std=NaN → filled with the column mean.
    assert not np.isnan(X).any()


def test_create_gene_association_matrix_save_and_reload_dense(
    small_associations, tmp_path: Path
) -> None:
    """Save -> reload roundtrip for dense matrix preserves shape + obs IDs."""
    save_path = tmp_path / "assoc"
    X1, meta1 = utils.create_gene_association_matrix(
        small_associations,
        source_id_col="sourceId",
        target_id_col="HGNC",
        save_path=str(save_path),
        convert_to_dense=True,
        verbose=False,
    )
    assert (tmp_path / "assoc.npy").exists()
    assert (tmp_path / "assoc.pkl").exists()

    X2, meta2 = utils.create_gene_association_matrix(
        associations=None,
        save_path=str(save_path),
        verbose=False,
    )
    assert X2.shape == X1.shape
    assert list(meta2["obs"]["sourceId"]) == list(meta1["obs"]["sourceId"])


def test_create_gene_association_matrix_save_and_reload_sparse(
    small_associations, tmp_path: Path
) -> None:
    """Save -> reload roundtrip for sparse matrix preserves shape."""
    save_path = tmp_path / "assoc_sparse"
    X1, _ = utils.create_gene_association_matrix(
        small_associations,
        source_id_col="sourceId",
        target_id_col="HGNC",
        save_path=str(save_path),
        convert_to_dense=False,
        verbose=False,
    )
    assert (tmp_path / "assoc_sparse.npz").exists()
    assert (tmp_path / "assoc_sparse.pkl").exists()

    X2, _ = utils.create_gene_association_matrix(
        associations=None,
        save_path=str(save_path),
        verbose=False,
    )
    assert X2.shape == X1.shape


def test_create_gene_association_matrix_save_and_reload_verbose(
    small_associations, tmp_path: Path, capsys
) -> None:
    """Verbose reload prints the load + shape + sparsity messages."""
    save_path = tmp_path / "verbose_assoc"
    utils.create_gene_association_matrix(
        small_associations,
        source_id_col="sourceId",
        target_id_col="HGNC",
        save_path=str(save_path),
        convert_to_dense=True,
        verbose=False,
    )
    _ = capsys.readouterr()
    utils.create_gene_association_matrix(
        associations=None,
        save_path=str(save_path),
        verbose=True,
    )
    captured = capsys.readouterr().out
    assert "Loading existing matrix from" in captured
    assert "Loaded matrix shape" in captured


def test_create_gene_association_matrix_reload_corrupt_cache_regenerates(
    small_associations, tmp_path: Path
) -> None:
    """If cache files exist but are corrupt, fall back to regeneration."""
    save_path = tmp_path / "corrupt"
    save_path.with_suffix(".npy").write_bytes(b"not a real .npy file")
    save_path.with_suffix(".pkl").write_bytes(b"not a real pickle")

    X, _ = utils.create_gene_association_matrix(
        small_associations,
        source_id_col="sourceId",
        target_id_col="HGNC",
        save_path=str(save_path),
        verbose=False,
    )
    assert X.shape == (3, 4)


def test_create_gene_association_matrix_force_regenerates(
    small_associations, tmp_path: Path
) -> None:
    """force=True ignores existing cache."""
    save_path = tmp_path / "force"
    utils.create_gene_association_matrix(
        small_associations,
        source_id_col="sourceId",
        target_id_col="HGNC",
        save_path=str(save_path),
        verbose=False,
    )
    X, _ = utils.create_gene_association_matrix(
        small_associations,
        source_id_col="sourceId",
        target_id_col="HGNC",
        save_path=str(save_path),
        force=True,
        verbose=False,
    )
    assert X.shape == (3, 4)


def test_create_gene_association_matrix_empty_after_dropna() -> None:
    """All-NaN score column produces an empty matrix (no exception)."""
    df = pd.DataFrame(
        {
            "sourceId": ["DIS:1", "DIS:2"],
            "HGNC": ["BRCA1", "TP53"],
            "score": [np.nan, np.nan],
        }
    )
    X, _ = utils.create_gene_association_matrix(
        df,
        source_id_col="sourceId",
        target_id_col="HGNC",
        verbose=False,
    )
    assert X.shape == (2, 0)


def test_create_gene_association_matrix_max_dense_elements_keeps_sparse(
    small_associations,
) -> None:
    """If matrix would exceed max_dense_elements, stay sparse."""
    X, _ = utils.create_gene_association_matrix(
        small_associations,
        source_id_col="sourceId",
        target_id_col="HGNC",
        convert_to_dense=True,
        max_dense_elements=1,
        verbose=False,
    )
    assert issparse(X)


# --------------------------------------------------------------------------
# Verbose / branch-coverage rounders
# --------------------------------------------------------------------------


def test_filter_adaptive_pval_threshold_verbose(small_associations, capsys) -> None:
    utils.filter_adaptive(
        small_associations,
        source_id_col="sourceId",
        target_id_col="HGNC",
        pval_threshold=0.005,
        verbose=True,
    )
    assert "After pval" in capsys.readouterr().out


def test_filter_adaptive_fdr_threshold_verbose(small_associations, capsys) -> None:
    df = small_associations.copy()
    df["fdr"] = df["pval"]
    utils.filter_adaptive(
        df,
        source_id_col="sourceId",
        target_id_col="HGNC",
        fdr_threshold=0.005,
        verbose=True,
    )
    assert "After fdr" in capsys.readouterr().out


def test_filter_adaptive_score_auto_from_pval_verbose(small_associations, capsys) -> None:
    """Hits the verbose 'calculating from pval' print branch."""
    df = small_associations.drop(columns=["score"])
    utils.filter_adaptive(
        df,
        source_id_col="sourceId",
        target_id_col="HGNC",
        percentile=0.5,
        sort_by="score",
        verbose=True,
    )
    assert "calculating from" in capsys.readouterr().out


def test_create_gene_association_matrix_group_database_dataset_label_verbose(
    small_associations, capsys
) -> None:
    """Verbose path of group/database/dataset/label printing."""
    df = small_associations.copy()
    df["database"] = "fake_db"
    df["dataset"] = "fake_ds"
    utils.create_gene_association_matrix(
        df,
        source_id_col="sourceId",
        target_id_col="HGNC",
        group_col="group",
        label_col="label",
        verbose=True,
    )
    captured = capsys.readouterr().out
    assert "Found 'group'" in captured
    assert "Found 'database'" in captured
    assert "Found 'dataset'" in captured
    assert "Found 'label'" in captured


def test_create_gene_association_matrix_fillna_constant_verbose(capsys) -> None:
    """fillna constant path with verbose=True prints the NaN-fill message."""
    df = pd.DataFrame(
        {
            "sourceId": ["DIS:1", "DIS:2", "DIS:2"],
            "HGNC": ["BRCA1", "TP53", "TP53"],
            "score": [1.0, 0.4, 0.6],
        }
    )
    utils.create_gene_association_matrix(
        df,
        source_id_col="sourceId",
        target_id_col="HGNC",
        fillna=-1.0,
        convert_to_dense=True,
        aggregation_method="std",
        verbose=True,
    )
    assert "Found" in capsys.readouterr().out


def test_create_gene_association_matrix_fillna_mean_verbose(capsys) -> None:
    df = pd.DataFrame(
        {
            "sourceId": ["DIS:1", "DIS:2", "DIS:2"],
            "HGNC": ["BRCA1", "BRCA1", "BRCA1"],
            "score": [1.0, 0.4, 0.6],
        }
    )
    utils.create_gene_association_matrix(
        df,
        source_id_col="sourceId",
        target_id_col="HGNC",
        fillna="mean",
        convert_to_dense=True,
        aggregation_method="std",
        verbose=True,
    )
    assert "per-gene means" in capsys.readouterr().out


def test_create_gene_association_matrix_no_nans_after_fillna_verbose(
    small_associations, capsys
) -> None:
    """If fillna requested but no NaN found, hits the 'no NaN' verbose branch."""
    utils.create_gene_association_matrix(
        small_associations,
        source_id_col="sourceId",
        target_id_col="HGNC",
        fillna=0.0,
        convert_to_dense=True,
        verbose=True,
    )
    assert "No NaN values" in capsys.readouterr().out


def test_create_gene_association_matrix_keep_sparse_verbose(small_associations, capsys) -> None:
    utils.create_gene_association_matrix(
        small_associations,
        source_id_col="sourceId",
        target_id_col="HGNC",
        convert_to_dense=False,
        verbose=True,
    )
    assert "Keeping sparse format" in capsys.readouterr().out


def test_create_gene_association_matrix_max_dense_elements_verbose(
    small_associations, capsys
) -> None:
    utils.create_gene_association_matrix(
        small_associations,
        source_id_col="sourceId",
        target_id_col="HGNC",
        convert_to_dense=True,
        max_dense_elements=1,
        verbose=True,
    )
    assert "too large for dense" in capsys.readouterr().out


def test_create_gene_association_matrix_save_verbose(
    small_associations, tmp_path: Path, capsys
) -> None:
    utils.create_gene_association_matrix(
        small_associations,
        source_id_col="sourceId",
        target_id_col="HGNC",
        save_path=str(tmp_path / "verbose_save"),
        convert_to_dense=True,
        verbose=True,
    )
    captured = capsys.readouterr().out
    assert "Saving dense matrix to" in captured
    assert "Metadata saved to" in captured


def test_create_gene_association_matrix_save_sparse_verbose(
    small_associations, tmp_path: Path, capsys
) -> None:
    utils.create_gene_association_matrix(
        small_associations,
        source_id_col="sourceId",
        target_id_col="HGNC",
        save_path=str(tmp_path / "verbose_save_sparse"),
        convert_to_dense=False,
        verbose=True,
    )
    assert "Saving sparse matrix to" in capsys.readouterr().out


def test_create_gene_association_matrix_chunked_with_verbose(small_associations, capsys) -> None:
    utils.create_gene_association_matrix(
        small_associations,
        source_id_col="sourceId",
        target_id_col="HGNC",
        chunk_size=2,
        verbose=True,
    )
    assert "chunked pandas" in capsys.readouterr().out


def test_create_gene_association_matrix_all_nan_after_dropna_chunked() -> None:
    """All-NaN score column + chunked path → empty matrix, no crash."""
    df = pd.DataFrame(
        {
            "sourceId": ["DIS:1", "DIS:2", "DIS:3", "DIS:4", "DIS:5"],
            "HGNC": ["BRCA1", "TP53", "EGFR", "MYC", "PTEN"],
            "score": [np.nan] * 5,
        }
    )
    X, _ = utils.create_gene_association_matrix(
        df,
        source_id_col="sourceId",
        target_id_col="HGNC",
        chunk_size=2,
        verbose=False,
    )
    assert X.shape == (5, 0)


def test_create_gene_association_matrix_partial_nan_warns_missing_sources(
    capsys,
) -> None:
    """sourceIds with no non-NaN data trigger a warning + zero-row in matrix."""
    df = pd.DataFrame(
        {
            "sourceId": ["DIS:1", "DIS:2", "DIS:3"],
            "HGNC": ["BRCA1", "TP53", "EGFR"],
            "score": [0.5, 0.7, np.nan],  # DIS:3 has only NaN
        }
    )
    X, meta = utils.create_gene_association_matrix(
        df,
        source_id_col="sourceId",
        target_id_col="HGNC",
        verbose=True,
    )
    captured = capsys.readouterr().out
    assert "had no valid" in captured
    assert "DIS:3" in list(meta["obs"]["sourceId"])
    # DIS:3 row is all zero
    di3_idx = list(meta["obs"]["sourceId"]).index("DIS:3")
    assert (X[di3_idx] == 0).all()


def test_create_gene_association_matrix_legacy_pickle_format(
    tmp_path: Path,
) -> None:
    """Old flat-schema cache (no obs/var) is upgraded to the new format on load.

    Builds the legacy cache by first running the function in modern format,
    then rewriting the metadata file with the legacy flat schema (via pandas'
    ``to_pickle``/``read_pickle``).
    """
    save_path = tmp_path / "legacy"
    X = np.array([[0.5, 0.0], [0.0, 0.7]], dtype=np.float32)
    np.save(save_path.with_suffix(".npy"), X)
    legacy_meta = {
        "unique_data_ids": np.array(["DIS:1", "DIS:2"]),
        "unique_target_ids": np.array(["BRCA1", "TP53"]),
        "matrix_shape": X.shape,
        "sparsity": 0.5,
    }
    pd.to_pickle(legacy_meta, save_path.with_suffix(".pkl"))

    X_loaded, meta_loaded = utils.create_gene_association_matrix(
        associations=None,
        save_path=str(save_path),
        verbose=True,
    )
    assert "obs" in meta_loaded and "var" in meta_loaded
    assert list(meta_loaded["obs"]["sourceId"]) == ["DIS:1", "DIS:2"]
    assert X_loaded.shape == X.shape


def test_create_gene_association_matrix_new_format_missing_nonzero(
    small_associations, tmp_path: Path
) -> None:
    """New-format cache missing nonzero_per_row + missing sourceId gets repaired."""
    save_path = tmp_path / "newish"
    utils.create_gene_association_matrix(
        small_associations,
        source_id_col="sourceId",
        target_id_col="HGNC",
        save_path=str(save_path),
        convert_to_dense=True,
        verbose=False,
    )

    # Strip nonzero_per_row, nonzero_per_col, and sourceId, then re-save.
    cached = pd.read_pickle(save_path.with_suffix(".pkl"))
    cached["obs"] = cached["obs"].drop(
        columns=[c for c in ("nonzero_per_row", "sourceId") if c in cached["obs"].columns]
    )
    cached["var"] = cached["var"].drop(
        columns=[c for c in ("nonzero_per_col",) if c in cached["var"].columns]
    )
    pd.to_pickle(cached, save_path.with_suffix(".pkl"))

    _, meta_loaded = utils.create_gene_association_matrix(
        associations=None,
        save_path=str(save_path),
        verbose=True,
    )
    assert "sourceId" in meta_loaded["obs"].columns
    assert "nonzero_per_row" in meta_loaded["obs"].columns
    assert "nonzero_per_col" in meta_loaded["var"].columns
