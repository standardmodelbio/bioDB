"""Shared helpers — verbatim ports from ``AoU.utils``. Re-vendored here so
``phenoref`` has no AoU dependency at runtime.

Public surface (mirrors the slice of ``AoU.utils`` actually called by the
four phenoref modules):

* :data:`RANDOM_SEED`, :func:`set_random_seed`
* :func:`count_tokens` -- tiktoken token counter (used by
  ``opentargets.diseases_to_markdown`` etc).
* :func:`l2_normalize`, :func:`cosine_similarity`,
  :func:`euclidean_similarity`, :func:`dot_product_similarity`
  (used by ``ontology.compute_pairwise_ontological_similarity``).
* :func:`create_gene_association_matrix` and :func:`filter_adaptive`
  (used by ``opentargets.get_gene_associations`` + downstream helpers).
"""

from __future__ import annotations

import pickle  # noqa: S403 -- verbatim port; needed for AoU-compat .pkl cache files
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, spmatrix
from tqdm import tqdm

try:
    from scipy.sparse import load_npz, save_npz
except ImportError:  # Fallback for older scipy versions
    import scipy.sparse as _sp

    save_npz = getattr(_sp, "save_npz", None)
    load_npz = getattr(_sp, "load_npz", None)


# Mirrors ``AoU.RANDOM_SEED`` (42). Provided for downstream callers that
# expect a single, project-wide deterministic seed.
RANDOM_SEED = 42


def set_random_seed(seed: int | None = None) -> None:
    """Seed every RNG the dataset code paths touch.

    Mirrors ``AoU.set_random_seed`` -- seeds Python's ``random``,
    NumPy, and (when installed) PyTorch (CPU + CUDA + cuDNN
    deterministic flag). Missing optional libs are silently tolerated.
    """
    if seed is None:
        seed = RANDOM_SEED
    try:
        import random as _r

        _r.seed(seed)
    except ImportError:
        pass
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def count_tokens(
    text, approximate: bool = False, model: str = "gpt-4o-mini", sep: str = " "
) -> int:
    """Count tokens in ``text`` using ``tiktoken`` (or a 4-char heuristic).

    Parameters
    ----------
    text : str or list[str]
        Input text. A list is joined with ``sep``.
    approximate : bool, default False
        If True, use ``len(text) // 4`` as a rough token count and
        skip the tiktoken dependency.
    model : str, default "gpt-4o-mini"
        Tiktoken model name (only consulted when ``approximate=False``).
    """
    if approximate:
        return len(text) // 4
    import tiktoken

    if isinstance(text, list):
        text = sep.join(text)
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


def l2_normalize(embeddings):
    """L2-normalize embedding rows to unit length.

    Accepts numpy arrays or torch tensors; returns the same type with
    rows scaled so their L2 norm is 1.0. Zero rows pass through
    unchanged (no NaN). Mirrors ``AoU.utils.l2_normalize``.
    """
    import torch
    import torch.nn.functional as F

    if isinstance(embeddings, torch.Tensor):
        return F.normalize(embeddings, p=2, dim=1)

    arr = np.asarray(embeddings, dtype=np.float32)
    if not arr.flags.writeable:
        arr = arr.copy()
    return F.normalize(torch.from_numpy(arr), p=2, dim=1).numpy()


def cosine_similarity(embeddings1, embeddings2):
    """Pairwise cosine similarity, assuming inputs are L2-normalized.

    For pre-normalized inputs cosine equals dot product, so this is
    just ``e1 @ e2.T``. If you haven't normalized, run
    :func:`l2_normalize` first or use :func:`dot_product_similarity`.

    Returns
    -------
    np.ndarray of shape ``(n1, n2)``.
    """
    e1 = np.ascontiguousarray(embeddings1, dtype=np.float32)
    e2 = np.ascontiguousarray(embeddings2, dtype=np.float32)
    return e1 @ e2.T


def euclidean_similarity(embeddings1, embeddings2):
    """Negative Euclidean distance as a similarity (higher = closer).

    Uses ``||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b`` to avoid materializing
    the full pairwise diff tensor.
    """
    e1 = np.ascontiguousarray(embeddings1, dtype=np.float32)
    e2 = np.ascontiguousarray(embeddings2, dtype=np.float32)
    n1_sq = np.sum(e1**2, axis=1, keepdims=True)
    n2_sq = np.sum(e2**2, axis=1, keepdims=True).T
    return -(n1_sq + n2_sq - 2 * (e1 @ e2.T))


def dot_product_similarity(embeddings1, embeddings2):
    """Raw dot-product similarity (no normalization). Shape ``(n1, n2)``."""
    e1 = np.ascontiguousarray(embeddings1, dtype=np.float32)
    e2 = np.ascontiguousarray(embeddings2, dtype=np.float32)
    return e1 @ e2.T


def _report_sparsity(df, source_id_col, target_id_col, step_name, verbose=True):
    """Helper used by :func:`filter_adaptive` to print sparsity stats."""
    if not verbose:
        return

    n_trait = df[source_id_col].nunique()
    n_gene = df[target_id_col].nunique()
    n_nz = len(df)
    fraction_nonzero = n_nz / (n_trait * n_gene) if (n_trait * n_gene) > 0 else 0
    sparsity = 1 - fraction_nonzero

    print(f"\n{step_name} Sparsity:")
    print(f"  Traits: {n_trait:,}, Genes: {n_gene:,}")
    print(f"  Nonzero associations: {n_nz:,}")
    print(f"  Theoretical full matrix shape: {n_trait:,} x {n_gene:,} = {n_trait*n_gene:,}")
    print(f"  Fraction nonzero: {fraction_nonzero:.6f} ({fraction_nonzero*100:.4f}%)")
    print(f"  Sparsity (fraction zero): {sparsity:.6f} ({sparsity*100:.4f}%)")


def filter_adaptive(
    df: pd.DataFrame,
    source_id_col: str = "sourceId",
    target_id_col: str = "targetId",
    percentile: float = 0.95,
    pval_threshold: Optional[float] = None,
    fdr_threshold: Optional[float] = None,
    pval_col: str = "pval",
    fdr_col: str = "fdr",
    score_col: str = "score",
    sort_by: str = "score",
    verbose: bool = True,
    min_genes: Optional[int] = None,
    max_genes: Optional[int] = None,
) -> pd.DataFrame:
    """Apply adaptive filtering: optional pvalue/FDR filter + per-sample top-percentile keep.

    See ``AoU.utils.filter_adaptive`` for the original docstring; verbatim port.
    """
    df = df.copy()
    initial_count = len(df)
    initial_samples = df[source_id_col].nunique()
    initial_genes = df[target_id_col].nunique()

    if verbose:
        print(
            f"Initial: {initial_count:,} rows, {initial_samples:,} samples, {initial_genes:,} genes"
        )

    _report_sparsity(df, source_id_col, target_id_col, "Before filtering", verbose)

    if pval_threshold is not None:
        if pval_col not in df.columns:
            raise ValueError(f"pval_threshold provided but '{pval_col}' column not found")
        df = df[df[pval_col] <= pval_threshold].copy()
        if verbose:
            print(f"\nAfter pval <= {pval_threshold}: {len(df):,} rows")
        _report_sparsity(df, source_id_col, target_id_col, "After pval filtering", verbose)
    elif fdr_threshold is not None:
        if fdr_col not in df.columns:
            raise ValueError(f"fdr_threshold provided but '{fdr_col}' column not found")
        df = df[df[fdr_col] <= fdr_threshold].copy()
        if verbose:
            print(f"\nAfter fdr <= {fdr_threshold}: {len(df):,} rows")
        _report_sparsity(df, source_id_col, target_id_col, "After fdr filtering", verbose)

    if sort_by == "score":
        if score_col not in df.columns:
            if pval_col in df.columns:
                if verbose:
                    print(f"  Note: '{score_col}' column not found, calculating from '{pval_col}'")
                pval_clamped = np.clip(df[pval_col], 1e-300, 1.0)
                df[score_col] = -np.log10(pval_clamped)
            else:
                raise ValueError(
                    f"sort_by='score' but neither '{score_col}' nor '{pval_col}' column found"
                )
        df = df.sort_values([source_id_col, score_col], ascending=[True, False])
    elif sort_by == "pval":
        if pval_col not in df.columns:
            raise ValueError(f"sort_by='pval' but '{pval_col}' column not found")
        df = df.sort_values([source_id_col, pval_col], ascending=[True, True])
    else:
        if sort_by not in df.columns:
            raise ValueError(f"sort_by='{sort_by}' but column '{sort_by}' not found in DataFrame")
        df = df.sort_values([source_id_col, sort_by], ascending=[True, True])

    def keep_top_percentile(group):
        n_keep = max(1, int(len(group) * (1 - percentile)))
        if min_genes is not None:
            n_keep = max(n_keep, min_genes)
        if max_genes is not None:
            n_keep = min(n_keep, max_genes)
        n_keep = min(n_keep, len(group))
        return group.head(n_keep)

    filtered = (
        df.groupby(source_id_col, group_keys=False)
        .apply(keep_top_percentile)
        .reset_index(drop=True)
    )

    if target_id_col != "targetId" and "targetId" not in filtered.columns:
        filtered["targetId"] = filtered[target_id_col]

    if verbose:
        final_samples = filtered[source_id_col].nunique()
        final_genes = filtered[target_id_col].nunique()
        final_count = len(filtered)
        genes_per_sample = filtered.groupby(source_id_col)[target_id_col].nunique()

        print(
            f"\nAfter percentile filtering (top {100*(1-percentile):.1f}%): "
            f"{final_count:,} rows, {final_samples:,} samples, {final_genes:,} genes"
        )
        print(
            f"  Genes per sample: min={genes_per_sample.min()}, "
            f"max={genes_per_sample.max()}, mean={genes_per_sample.mean():.1f}"
        )
        print(
            f"  Retained: {final_count/initial_count*100:.1f}% of rows, "
            f"{final_samples/initial_samples*100:.1f}% of samples"
        )
        _report_sparsity(
            filtered, source_id_col, target_id_col, "After percentile filtering", verbose
        )

    return filtered


def create_gene_association_matrix(
    associations: Optional[pd.DataFrame] = None,
    source_id_col: str = "sourceId",
    target_id_col: str = "HGNC",
    score_col: str = "score",
    group_col: Optional[str] = "group",
    label_col: Optional[str] = "label",
    chunk_size: int = 5_000_000,
    convert_to_dense: bool = True,
    max_dense_elements: int = 1_000_000_000,
    verbose: bool = True,
    save_path: Optional[str] = None,
    force: bool = False,
    fillna: Optional[Union[float, str]] = None,
    aggregation_method: str = "mean",
) -> Tuple[Union[np.ndarray, spmatrix], Dict[str, Any]]:
    """Create a gene-association matrix from a long DataFrame of (sample, gene, score).

    Verbatim port of ``AoU.utils.create_gene_association_matrix``. See the
    AoU upstream for the long-form docstring. Returns ``(X, metadata)``
    where ``X`` is either ``np.ndarray`` (float32) or a ``scipy.sparse.csr_matrix``
    and ``metadata`` is the AnnData-like dict with ``obs`` / ``var`` /
    ``metadata`` keys.
    """
    if save_path is not None and not force:
        save_path_obj = Path(save_path)
        metadata_path = save_path_obj.with_suffix(".pkl")
        npy_path = save_path_obj.with_suffix(".npy")
        npz_path = save_path_obj.with_suffix(".npz")

        if (npy_path.exists() or npz_path.exists()) and metadata_path.exists():
            if verbose:
                print(f"Loading existing matrix from: {save_path}")
            try:
                if npz_path.exists():
                    if load_npz is None:
                        raise ImportError(
                            "scipy.sparse.load_npz is not available. "
                            "Please upgrade scipy to version >= 0.19.0"
                        )
                    X = load_npz(npz_path)
                    if verbose:
                        print(f"  Loaded sparse matrix from: {npz_path}")
                else:
                    X = np.load(npy_path, mmap_mode="r")
                    if verbose:
                        print(f"  Loaded dense matrix from: {npy_path}")

                with open(metadata_path, "rb") as f:
                    loaded_metadata = pickle.load(f)  # noqa: S301 -- AoU-compat cache

                if "obs" not in loaded_metadata or "var" not in loaded_metadata:
                    if verbose:
                        print("  Converting old metadata format to new format...")
                    unique_data_ids = loaded_metadata.get(
                        "unique_data_ids", np.arange(X.shape[0])
                    )
                    unique_target_ids = loaded_metadata.get(
                        "unique_target_ids", np.arange(X.shape[1])
                    )

                    if (
                        "nonzero_per_row" not in loaded_metadata
                        or "nonzero_per_col" not in loaded_metadata
                    ):
                        if verbose:
                            print("  Computing nonzero counts (missing from cached metadata)...")
                        try:
                            from scipy import sparse

                            if sparse.issparse(X):
                                nonzero_per_row = X.getnnz(axis=1).astype(np.int32)
                                nonzero_per_col = X.getnnz(axis=0).astype(np.int32)
                            else:
                                nonzero_per_row = np.count_nonzero(X, axis=1).astype(np.int32)
                                nonzero_per_col = np.count_nonzero(X, axis=0).astype(np.int32)
                            loaded_metadata["nonzero_per_row"] = nonzero_per_row
                            loaded_metadata["nonzero_per_col"] = nonzero_per_col
                        except Exception as e:
                            if verbose:
                                print(f"  Warning: Could not compute nonzero counts ({e})")
                            if "nonzero_per_row" not in loaded_metadata:
                                loaded_metadata["nonzero_per_row"] = np.zeros(
                                    X.shape[0], dtype=np.int32
                                )
                            if "nonzero_per_col" not in loaded_metadata:
                                loaded_metadata["nonzero_per_col"] = np.zeros(
                                    X.shape[1], dtype=np.int32
                                )

                    obs_data = {
                        "nonzero_per_row": loaded_metadata["nonzero_per_row"],
                        "sourceId": unique_data_ids,
                    }
                    if group_col is not None and group_col in loaded_metadata:
                        obs_data[group_col] = loaded_metadata[group_col]
                    obs = pd.DataFrame(obs_data, index=unique_data_ids)

                    var = pd.DataFrame(
                        {"nonzero_per_col": loaded_metadata["nonzero_per_col"]},
                        index=unique_target_ids,
                    )

                    metadata_dict = {
                        "matrix_shape": loaded_metadata.get("matrix_shape", X.shape),
                        "sparsity": loaded_metadata.get("sparsity", "N/A"),
                        "obs_to_idx": loaded_metadata.get(
                            "obs_to_idx",
                            {id_: idx for idx, id_ in enumerate(unique_data_ids)},
                        ),
                        "var_to_idx": loaded_metadata.get(
                            "var_to_idx",
                            {id_: idx for idx, id_ in enumerate(unique_target_ids)},
                        ),
                    }

                    loaded_metadata = {
                        "obs": obs,
                        "var": var,
                        "metadata": metadata_dict,
                    }
                else:
                    if (
                        "nonzero_per_row" not in loaded_metadata["obs"].columns
                        or "nonzero_per_col" not in loaded_metadata["var"].columns
                    ):
                        if verbose:
                            print("  Computing nonzero counts (missing from cached metadata)...")
                        try:
                            from scipy import sparse

                            if sparse.issparse(X):
                                nonzero_per_row = X.getnnz(axis=1).astype(np.int32)
                                nonzero_per_col = X.getnnz(axis=0).astype(np.int32)
                            else:
                                nonzero_per_row = np.count_nonzero(X, axis=1).astype(np.int32)
                                nonzero_per_col = np.count_nonzero(X, axis=0).astype(np.int32)
                            loaded_metadata["obs"]["nonzero_per_row"] = nonzero_per_row
                            loaded_metadata["var"]["nonzero_per_col"] = nonzero_per_col
                        except Exception as e:
                            if verbose:
                                print(f"  Warning: Could not compute nonzero counts ({e})")

                    if "sourceId" not in loaded_metadata["obs"].columns:
                        if verbose:
                            print("  Adding 'sourceId' column to obs...")
                        loaded_metadata["obs"]["sourceId"] = loaded_metadata["obs"].index.values

                if verbose:
                    print(f"  Loaded matrix shape: {X.shape}")
                    sparsity_val = loaded_metadata["metadata"].get("sparsity", "N/A")
                    if isinstance(sparsity_val, (int, float)):
                        print(f"  Sparsity: {sparsity_val:.2%}")
                    else:
                        print(f"  Sparsity: {sparsity_val}")

                return X, loaded_metadata
            except Exception as e:
                if verbose:
                    print(f"  Warning: Error loading cached file ({e}), regenerating...")

    if associations is None:
        raise ValueError(
            "associations is required when creating a new matrix. "
            "It is only optional when loading from an existing file "
            "(save_path exists and force=False)."
        )

    if verbose:
        print("Creating gene association matrix (optimized)...")
        print(f"Data shape: {associations.shape}")

    required_cols = [source_id_col, target_id_col, score_col]
    missing_cols = [col for col in required_cols if col not in associations.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    all_unique_source_ids = np.array(associations[source_id_col].unique())
    if verbose:
        print(f"Total unique sourceIds in input: {len(all_unique_source_ids):,}")

    group_mapping = None
    if group_col is not None and group_col in associations.columns:
        group_df = associations[[source_id_col, group_col]].drop_duplicates(
            subset=[source_id_col], keep="first"
        )
        group_mapping = dict(zip(group_df[source_id_col], group_df[group_col]))
        if verbose:
            unique_groups = group_df[group_col].unique()
            print(
                f"Found '{group_col}' column with {len(unique_groups)} unique groups: {unique_groups}"
            )

    database_mapping = None
    dataset_mapping = None
    if "database" in associations.columns:
        database_df = associations[[source_id_col, "database"]].drop_duplicates(
            subset=[source_id_col], keep="first"
        )
        database_mapping = dict(zip(database_df[source_id_col], database_df["database"]))
        if verbose:
            unique_databases = database_df["database"].unique()
            print(
                f"Found 'database' column with {len(unique_databases)} unique values: {unique_databases}"
            )

    if "dataset" in associations.columns:
        dataset_df = associations[[source_id_col, "dataset"]].drop_duplicates(
            subset=[source_id_col], keep="first"
        )
        dataset_mapping = dict(zip(dataset_df[source_id_col], dataset_df["dataset"]))
        if verbose:
            unique_datasets = dataset_df["dataset"].unique()
            print(
                f"Found 'dataset' column with {len(unique_datasets)} unique values: {len(unique_datasets)} datasets"
            )

    label_mapping = None
    if label_col is not None and label_col in associations.columns:
        label_df = associations[[source_id_col, label_col]].drop_duplicates(
            subset=[source_id_col], keep="first"
        )
        label_mapping = dict(zip(label_df[source_id_col], label_df[label_col]))
        if verbose:
            unique_labels = label_df[label_col].unique()
            print(f"Found '{label_col}' column with {len(unique_labels)} unique values")

    valid_methods = [
        "mean", "max", "min", "sum", "median", "first", "last", "std", "var", "count",
    ]
    if aggregation_method not in valid_methods:
        try:
            test_grouped = pd.DataFrame({score_col: [1, 2, 3]}).groupby([0, 0, 0])[score_col]
            getattr(test_grouped, aggregation_method)
        except AttributeError as exc:
            raise ValueError(
                f"Invalid aggregation_method: '{aggregation_method}'. "
                f"Valid methods: {valid_methods}. "
                f"Or any valid pandas groupby aggregation function."
            ) from exc

    n_rows_total = len(associations)
    use_chunked_agg = n_rows_total > chunk_size * 2

    if use_chunked_agg:
        if verbose:
            print(
                f"Aggregating scores using chunked pandas groupby (method: {aggregation_method}, {n_rows_total:,} rows)..."
            )
        num_chunks = int(np.ceil(n_rows_total / chunk_size))
        aggregated_chunks = []
        iterator = tqdm(range(num_chunks), desc="Aggregating chunks", disable=not verbose)

        for i in iterator:
            start = i * chunk_size
            end = min((i + 1) * chunk_size, n_rows_total)
            chunk = associations.iloc[start:end]
            chunk_filtered = chunk.dropna(subset=[score_col])
            if len(chunk_filtered) > 0:
                grouped = chunk_filtered.groupby(
                    [source_id_col, target_id_col], sort=False
                )[score_col]
                chunk_agg = getattr(grouped, aggregation_method)().reset_index()
                aggregated_chunks.append(chunk_agg)

        if verbose:
            print("Combining and finalizing aggregation...")
        if aggregated_chunks:
            aggregated = pd.concat(aggregated_chunks, ignore_index=True)
            del aggregated_chunks
            aggregated = aggregated.dropna(subset=[score_col])
            if len(aggregated) > 0:
                grouped = aggregated.groupby(
                    [source_id_col, target_id_col], sort=False
                )[score_col]
                aggregated = getattr(grouped, aggregation_method)().reset_index()
            else:
                aggregated = pd.DataFrame(columns=[source_id_col, target_id_col, score_col])
        else:
            aggregated = pd.DataFrame(columns=[source_id_col, target_id_col, score_col])
    else:
        if verbose:
            print(
                f"Aggregating scores using pandas groupby (method: {aggregation_method}, skipping NaN values)..."
            )
        associations_filtered = associations.dropna(subset=[score_col])
        if len(associations_filtered) > 0:
            grouped = associations_filtered.groupby(
                [source_id_col, target_id_col], sort=False
            )[score_col]
            aggregated = getattr(grouped, aggregation_method)().reset_index()
        else:
            aggregated = pd.DataFrame(columns=[source_id_col, target_id_col, score_col])

    if len(aggregated) > 0:
        aggregated_source_ids = np.array(aggregated[source_id_col].unique())
        unique_target_ids = np.array(aggregated[target_id_col].unique())
        unique_data_ids = np.unique(
            np.concatenate([all_unique_source_ids, aggregated_source_ids])
        )
    else:
        unique_data_ids = all_unique_source_ids
        unique_target_ids = np.array([], dtype=object)

    if verbose:
        print(
            f"Unique data_ids in final matrix: {len(unique_data_ids):,} (includes all input sourceIds)"
        )
        print(f"Unique targetIds (genes): {len(unique_target_ids):,}")
        print(f"Aggregated pairs: {len(aggregated)}")

        if len(aggregated) > 0:
            aggregated_source_ids_set = set(aggregated[source_id_col].unique())
            missing_source_ids = [
                sid for sid in all_unique_source_ids if sid not in aggregated_source_ids_set
            ]
            if len(missing_source_ids) > 0:
                print(
                    f"  Warning: {len(missing_source_ids):,} sourceIds had no valid (non-NaN) "
                    "associations and will have all-zero rows"
                )

    n_rows = len(unique_data_ids)
    n_cols = len(unique_target_ids) if len(unique_target_ids) > 0 else 0

    if len(aggregated) > 0 and n_cols > 0:
        aggregated[source_id_col] = pd.Categorical(
            aggregated[source_id_col], categories=unique_data_ids
        )
        aggregated[target_id_col] = pd.Categorical(
            aggregated[target_id_col], categories=unique_target_ids
        )

        row_indices = aggregated[source_id_col].cat.codes.values.astype(np.int32)
        col_indices = aggregated[target_id_col].cat.codes.values.astype(np.int32)
        scores = aggregated[score_col].values.astype(np.float32)

        del aggregated

        if verbose:
            print("Building sparse matrix...")
        X_sparse = coo_matrix(
            (scores, (row_indices, col_indices)),
            shape=(n_rows, n_cols),
            dtype=np.float32,
        )
    else:
        if verbose:
            print("Building sparse matrix (no valid associations found)...")
        X_sparse = coo_matrix((n_rows, n_cols), dtype=np.float32)
        if "aggregated" in locals():
            del aggregated

    X_sparse = X_sparse.tocsr()

    total_elements = n_rows * n_cols
    sparsity = 1.0 - (X_sparse.nnz / total_elements) if total_elements > 0 else 1.0

    if verbose:
        print(f"Matrix shape: {X_sparse.shape}")
        print(f"Sparsity: {sparsity:.2%} ({(1-sparsity)*100:.2f}% non-zero)")

    if fillna is not None:
        nan_mask = np.isnan(X_sparse.data)
        if nan_mask.any():
            if verbose:
                print(f"  Found {nan_mask.sum()} NaN values in sparse matrix")
            if fillna == "mean":
                X_csc = X_sparse.tocsc()
                nan_mask_csc = np.isnan(X_csc.data)
                valid_data = X_csc.data[~nan_mask_csc]
                global_mean = valid_data.mean() if len(valid_data) > 0 else 0.0
                n_cols_csc = X_csc.shape[1]
                col_means = np.zeros(n_cols_csc, dtype=np.float32)
                for col_idx in range(n_cols_csc):
                    col_start = X_csc.indptr[col_idx]
                    col_end = X_csc.indptr[col_idx + 1]
                    col_values = X_csc.data[col_start:col_end]
                    valid_col_values = col_values[~np.isnan(col_values)]
                    if len(valid_col_values) > 0:
                        col_means[col_idx] = valid_col_values.mean()
                    else:
                        col_means[col_idx] = global_mean
                nan_indices_csc = np.where(nan_mask_csc)[0]
                for nan_idx in nan_indices_csc:
                    col_idx = np.searchsorted(X_csc.indptr, nan_idx + 1, side="right") - 1
                    X_csc.data[nan_idx] = col_means[col_idx]
                X_sparse = X_csc.tocsr()
                if verbose:
                    print(
                        f"Filling NaN values with per-gene means (global mean={global_mean:.6f} as fallback)..."
                    )
            else:
                fillna_value = fillna
                if verbose:
                    print(f"Filling NaN values with {fillna_value}...")
                X_sparse.data[nan_mask] = fillna_value
        else:
            if verbose:
                print("  No NaN values found in sparse matrix")

    if verbose:
        print("Computing nonzero counts per row and column...")
    nonzero_per_row = X_sparse.getnnz(axis=1).astype(np.int32)
    nonzero_per_col = X_sparse.getnnz(axis=0).astype(np.int32)

    if convert_to_dense and total_elements < max_dense_elements:
        if verbose:
            print("Converting to dense array...")
        X = X_sparse.toarray().astype(np.float32)
        del X_sparse
        if fillna is not None and np.isnan(X).any():
            if verbose:
                print(f"  Filling remaining NaN values with {fillna}...")
            X = np.nan_to_num(X, nan=fillna)
        if verbose:
            print(f"Dense array shape: {X.shape}, dtype: {X.dtype}")
    else:
        if verbose:
            if not convert_to_dense:
                print("Keeping sparse format (convert_to_dense=False)")
            else:
                print(
                    f"Matrix too large for dense conversion ({total_elements:,} elements > "
                    f"{max_dense_elements:,}). Using sparse format."
                )
        X = X_sparse

    obs_to_idx = {id_: idx for idx, id_ in enumerate(unique_data_ids)}
    var_to_idx = {id_: idx for idx, id_ in enumerate(unique_target_ids)}

    assert len(nonzero_per_row) == len(unique_data_ids)

    obs_data = {
        "nonzero_per_row": nonzero_per_row,
        "sourceId": unique_data_ids,
    }
    if database_mapping is not None:
        databases = np.array(
            [database_mapping.get(data_id, None) for data_id in unique_data_ids],
            dtype=object,
        )
        obs_data["database"] = databases
    if dataset_mapping is not None:
        datasets = np.array(
            [dataset_mapping.get(data_id, None) for data_id in unique_data_ids],
            dtype=object,
        )
        obs_data["dataset"] = datasets
    if group_mapping is not None:
        groups = np.array(
            [group_mapping.get(data_id, None) for data_id in unique_data_ids],
            dtype=object,
        )
        obs_data[group_col] = groups
    if label_mapping is not None:
        labels = np.array(
            [label_mapping.get(data_id, None) for data_id in unique_data_ids],
            dtype=object,
        )
        obs_data[label_col] = labels

    obs = pd.DataFrame(obs_data, index=unique_data_ids)
    var = pd.DataFrame({"nonzero_per_col": nonzero_per_col}, index=unique_target_ids)

    metadata = {
        "matrix_shape": X.shape if hasattr(X, "shape") else (n_rows, n_cols),
        "sparsity": sparsity,
        "obs_to_idx": obs_to_idx,
        "var_to_idx": var_to_idx,
    }

    result_metadata = {
        "obs": obs,
        "var": var,
        "metadata": metadata,
    }

    if verbose:
        print(f"\nFinal matrix: {result_metadata['metadata']['matrix_shape']}")
        try:
            if hasattr(X, "nbytes"):
                mem_gb = X.nbytes / 1024**3
                print(f"Memory usage: {mem_gb:.2f} GB")
            elif hasattr(X, "data"):
                mem_gb = (X.data.nbytes + X.indices.nbytes + X.indptr.nbytes) / 1024**3
                print(f"Memory usage (sparse): {mem_gb:.2f} GB")
        except Exception:
            pass

    if save_path is not None:
        save_path_obj = Path(save_path)
        metadata_path = save_path_obj.with_suffix(".pkl")
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(X, spmatrix):
            if save_npz is None:
                raise ImportError(
                    "scipy.sparse.save_npz is not available. "
                    "Please upgrade scipy to version >= 0.19.0"
                )
            npz_path = save_path_obj.with_suffix(".npz")
            if verbose:
                print(f"\nSaving sparse matrix to: {npz_path}")
            save_npz(npz_path, X)
        else:
            npy_path = save_path_obj.with_suffix(".npy")
            X_to_save = X
            if fillna is not None and np.isnan(X_to_save).any():
                X_to_save = np.nan_to_num(X_to_save, nan=fillna)
            if verbose:
                print(f"\nSaving dense matrix to: {npy_path}")
            np.save(npy_path, X_to_save)

        result_metadata["metadata"]["is_sparse"] = isinstance(X, spmatrix)
        result_metadata["metadata"]["matrix_format"] = (
            "sparse" if isinstance(X, spmatrix) else "dense"
        )

        with open(metadata_path, "wb") as f:
            pickle.dump(result_metadata, f)

        if verbose:
            print(f"  Metadata saved to: {metadata_path}")

    return X, result_metadata


__all__ = [
    "RANDOM_SEED",
    "cosine_similarity",
    "count_tokens",
    "create_gene_association_matrix",
    "dot_product_similarity",
    "euclidean_similarity",
    "filter_adaptive",
    "l2_normalize",
    "set_random_seed",
]
