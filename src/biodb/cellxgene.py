"""CZ CELLxGENE Discover / Census client — compute-on-demand marker genes.

`CZ CELLxGENE Discover <https://cellxgene.cziscience.com/>`_ aggregates
single-cell data across studies; its `Census
<https://chanzuckerberg.github.io/cellxgene-census/>`_ exposes the whole corpus
as a cloud-hosted TileDB-SOMA store, queryable by
``cell_type_ontology_term_id`` (native `Cell Ontology
<https://obofoundry.org/ontology/cl.html>`_ ids) and ``tissue_general``.

Unlike the curated sources (:mod:`biodb.celltaxonomy`, :mod:`biodb.cellmarker`),
the Census publishes **no bulk marker file** — its "Marker Score" is a web-UI
feature. This module therefore *computes* markers on demand from the Census
expression matrices, replicating the documented Marker Score approach in
simplified form: for a tissue, each cell type is compared one-vs-rest and every
gene is scored by its **effect size** (Cohen's d) on log-normalized expression.
This is not byte-identical to the CELLxGENE UI's Marker Score (the 10th
percentile of bootstrapped Welch's-t effect sizes) but ranks the same signal.

Both access modes map onto that one computation:

* **API mode** — :func:`query_markers` returns the top markers for one cell
  type in one tissue (resolving a plain name to a CL id via OLS if needed).
* **Bulk mode** — :func:`compute_tissue_markers` returns markers for *every*
  cell type in a tissue, in the normalized *(species, tissue, cell type, CL id,
  gene, score, rank)* schema shared across the cell-type sources.
  :func:`list_tissues` / :func:`list_cell_types` support discovery.

.. note::
   Requires the ``[cellxgene]`` extra (``cellxgene-census``, which pulls
   ``tiledbsoma`` + ``anndata``). The Census store lives in AWS ``us-west-2``
   (read-only). Computed marker tables are cached under
   ``~/.cache/biodb/cellxgene/<census_version>/``.

Examples
--------
>>> from biodb import cellxgene as cx
>>> tissues = cx.list_tissues()                                  # doctest: +SKIP
>>> markers = cx.query_markers("CL:0000540", tissue="brain")     # doctest: +SKIP
>>> table = cx.compute_tissue_markers("brain")                   # doctest: +SKIP
>>> cx.to_gmt("cellxgene_brain.gmt", tissue="brain")             # doctest: +SKIP

References
----------
* Census: https://chanzuckerberg.github.io/cellxgene-census/
* Marker Score docs:
  https://cellxgene.cziscience.com/docs/04__Analyze%20Public%20Data/4_2__Gene%20Expression%20Documentation/4_2_5__Find%20Marker%20Genes
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from biodb import _celltype
from biodb.utils import RANDOM_SEED

if TYPE_CHECKING:  # pragma: no cover - typing only
    from anndata import AnnData

logger = logging.getLogger(__name__)

DEFAULT_CENSUS_VERSION = "2025-11-08"
"""Pinned CELLxGENE Census LTS release for reproducibility. Pass
``census_version="stable"`` / ``"latest"`` to track the moving pointers."""

DEFAULT_ORGANISM = "Homo sapiens"
"""Census organism string (note the space + capitalisation the API expects)."""

SOURCE_NAME = "cellxgene"
"""Value written into the normalized ``source`` column."""

CACHE_DIR = Path("~/.cache/biodb/cellxgene").expanduser()
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_MIN_CELLS_PER_TYPE = 10
"""Cell types with fewer cells than this in the (subsampled) pull are skipped —
too few observations for a stable effect size."""


def _require_census() -> Any:
    """Import ``cellxgene_census`` or raise a helpful error."""
    try:
        import cellxgene_census
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "biodb.cellxgene requires the 'cellxgene-census' package. "
            "Install it with:  pip install 'biodb[cellxgene]'"
        ) from exc
    return cellxgene_census


def _slug(text: str) -> str:
    """Filesystem-safe token for a cache filename."""
    return re.sub(r"[^0-9A-Za-z]+", "-", text.strip().lower()).strip("-")


# ─── Discovery ───────────────────────────────────────────────────────────────


def list_tissues(
    *,
    organism: str = DEFAULT_ORGANISM,
    census_version: str = DEFAULT_CENSUS_VERSION,
) -> list[str]:
    """List the distinct ``tissue_general`` values available for an organism.

    Parameters
    ----------
    organism
        Census organism string (default :data:`DEFAULT_ORGANISM`).
    census_version
        Census release (default :data:`DEFAULT_CENSUS_VERSION`).

    Returns
    -------
    list[str]
        Sorted unique tissue names.
    """
    census = _require_census()
    with census.open_soma(census_version=census_version) as store:
        obs = census.get_obs(
            store, organism, value_filter="is_primary_data == True", column_names=["tissue_general"]
        )
    return sorted(obs["tissue_general"].dropna().unique().tolist())


def list_cell_types(
    *,
    tissue: str | None = None,
    organism: str = DEFAULT_ORGANISM,
    census_version: str = DEFAULT_CENSUS_VERSION,
) -> pd.DataFrame:
    """List cell types (and their CL ids + cell counts) for an organism/tissue.

    Parameters
    ----------
    tissue
        Optional ``tissue_general`` filter.
    organism, census_version
        See :func:`list_tissues`.

    Returns
    -------
    pandas.DataFrame
        Columns ``cell_ontology_id``, ``cell_type_name``, ``n_cells``, sorted
        by ``n_cells`` descending.
    """
    census = _require_census()
    value_filter = "is_primary_data == True"
    if tissue is not None:
        value_filter += f" and tissue_general == '{tissue}'"
    with census.open_soma(census_version=census_version) as store:
        obs = census.get_obs(
            store,
            organism,
            value_filter=value_filter,
            column_names=["cell_type", "cell_type_ontology_term_id"],
        )
    counts = (
        obs.groupby(["cell_type_ontology_term_id", "cell_type"], dropna=False)
        .size()
        .reset_index(name="n_cells")
        .rename(
            columns={
                "cell_type_ontology_term_id": "cell_ontology_id",
                "cell_type": "cell_type_name",
            }
        )
    )
    return counts.sort_values("n_cells", ascending=False).reset_index(drop=True)


# ─── Bulk mode — compute markers for a whole tissue ──────────────────────────


def compute_tissue_markers(
    tissue: str,
    *,
    organism: str = DEFAULT_ORGANISM,
    census_version: str = DEFAULT_CENSUS_VERSION,
    max_cells_per_type: int = 2000,
    top_n_per_type: int = 100,
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Compute one-vs-rest marker genes for every cell type in a tissue.

    Pulls a (subsampled) AnnData for the tissue once, log-normalizes it, and
    scores every gene per cell type by Cohen's d effect size. The result is
    cached to parquet so repeated calls are free.

    Parameters
    ----------
    tissue
        A ``tissue_general`` value (see :func:`list_tissues`).
    organism, census_version
        See :func:`list_tissues`.
    max_cells_per_type
        Cap on cells sampled per cell type (deterministic, seeded) to bound the
        pull. Default 2000.
    top_n_per_type
        Keep this many top-scoring genes per cell type in the cached table.
    cache_dir, force
        Cache location override / bypass.

    Returns
    -------
    pandas.DataFrame
        Columns per :data:`biodb._celltype.NORMALIZED_COLUMNS`.
    """
    root = Path(cache_dir).expanduser() if cache_dir else CACHE_DIR / census_version
    root.mkdir(parents=True, exist_ok=True)
    cache_path = root / f"{_slug(organism)}__{_slug(tissue)}.parquet"
    if cache_path.exists() and not force:
        return pd.read_parquet(cache_path)

    adata = _pull_tissue_adata(
        tissue,
        organism=organism,
        census_version=census_version,
        max_cells_per_type=max_cells_per_type,
    )
    table = _marker_table(adata, tissue=tissue, organism=organism, top_n_per_type=top_n_per_type)
    table.to_parquet(cache_path, index=False)
    return table


def _pull_tissue_adata(
    tissue: str,
    *,
    organism: str,
    census_version: str,
    max_cells_per_type: int,
) -> AnnData:
    """Fetch a subsampled AnnData of primary cells for one tissue."""
    census = _require_census()
    value_filter = f"is_primary_data == True and tissue_general == '{tissue}'"
    with census.open_soma(census_version=census_version) as store:
        obs = census.get_obs(
            store,
            organism,
            value_filter=value_filter,
            column_names=["soma_joinid", "cell_type_ontology_term_id"],
        )
        coords = _subsample_joinids(obs, max_cells_per_type)
        if not coords:
            raise ValueError(f"No primary cells found for tissue {tissue!r} in {organism}.")
        adata = census.get_anndata(
            store,
            organism=organism,
            obs_coords=coords,
            obs_column_names=["cell_type_ontology_term_id", "cell_type", "tissue_general"],
            X_name="raw",
        )
    return adata


def _subsample_joinids(obs: pd.DataFrame, max_per_type: int) -> list[int]:
    """Deterministically cap cells per cell type; return sorted soma_joinids."""
    rng = np.random.default_rng(RANDOM_SEED)
    keep: list[np.ndarray] = []
    for _, grp in obs.groupby("cell_type_ontology_term_id", dropna=False):
        ids = grp["soma_joinid"].to_numpy()
        if len(ids) > max_per_type:
            ids = rng.choice(ids, size=max_per_type, replace=False)
        keep.append(np.asarray(ids))
    if not keep:
        return []
    return sorted(int(i) for i in np.concatenate(keep))


def _marker_table(
    adata: AnnData,
    *,
    tissue: str,
    organism: str,
    top_n_per_type: int,
) -> pd.DataFrame:
    """Score every gene per cell type by one-vs-rest Cohen's d effect size."""
    from scipy.sparse import csr_matrix, diags, issparse

    x = adata.X
    x = csr_matrix(x) if not issparse(x) else x.tocsr().astype(np.float64)
    # Library-size normalize to 1e4 then log1p (sparsity-preserving).
    lib = np.asarray(x.sum(axis=1)).ravel()
    lib[lib == 0] = 1.0
    xn = diags(1e4 / lib) @ x
    xn.data = np.log1p(xn.data)

    total_n = xn.shape[0]
    g_sum = np.asarray(xn.sum(axis=0)).ravel()
    g_sq = np.asarray(xn.multiply(xn).sum(axis=0)).ravel()

    var = adata.var
    gene_symbols = var["feature_name"].to_numpy() if "feature_name" in var else var.index.to_numpy()
    gene_ids = var["feature_id"].to_numpy() if "feature_id" in var else var.index.to_numpy()

    labels = adata.obs["cell_type_ontology_term_id"].to_numpy()
    names = adata.obs["cell_type"].to_numpy() if "cell_type" in adata.obs else labels

    frames: list[pd.DataFrame] = []
    for cl_id in pd.unique(labels):
        mask = labels == cl_id
        n1 = int(mask.sum())
        if n1 < _MIN_CELLS_PER_TYPE or n1 == total_n:
            continue
        sub = xn[mask]
        s1 = np.asarray(sub.sum(axis=0)).ravel()
        sq1 = np.asarray(sub.multiply(sub).sum(axis=0)).ravel()
        n2 = total_n - n1
        m1, m2 = s1 / n1, (g_sum - s1) / n2
        v1 = np.clip(sq1 / n1 - m1**2, 0, None)
        v2 = np.clip((g_sq - sq1) / n2 - m2**2, 0, None)
        pooled = np.sqrt((v1 + v2) / 2.0) + 1e-8
        cohens_d = (m1 - m2) / pooled

        order = np.argsort(cohens_d)[::-1][:top_n_per_type]
        order = order[cohens_d[order] > 0]  # keep up-regulated markers only
        if order.size == 0:
            continue
        cell_name = names[mask][0]
        frames.append(
            pd.DataFrame(
                {
                    "species": organism,
                    "tissue": tissue,
                    "cell_type_name": cell_name,
                    "cell_ontology_id": _celltype.normalize_cl_id(cl_id) or cl_id,
                    "gene_symbol": gene_symbols[order],
                    "gene_id": gene_ids[order],
                    "score": cohens_d[order],
                    "source": SOURCE_NAME,
                }
            )
        )

    if not frames:
        return pd.DataFrame(columns=_celltype.NORMALIZED_COLUMNS)
    table = pd.concat(frames, ignore_index=True)
    table["rank"] = _celltype.rank_within_group(
        table, group_cols=["cell_ontology_id"], score_col="score"
    )
    return table[_celltype.NORMALIZED_COLUMNS].reset_index(drop=True)


# ─── API mode — one cell type ────────────────────────────────────────────────


def query_markers(
    cell_type: str,
    *,
    tissue: str,
    organism: str = DEFAULT_ORGANISM,
    census_version: str = DEFAULT_CENSUS_VERSION,
    n_top: int = 25,
    max_cells_per_type: int = 2000,
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Return the top computed marker genes for one cell type in one tissue.

    Parameters
    ----------
    cell_type
        A CL id (``"CL:0000540"``) or a plain cell-type name; a name is resolved
        to a CL id via :func:`biodb.ols.find_term` (scoped to ``cl``).
    tissue
        The ``tissue_general`` context to compute markers within (required —
        the Marker Score is tissue-specific).
    organism, census_version, max_cells_per_type, cache_dir, force
        Forwarded to :func:`compute_tissue_markers` (the per-tissue table is
        computed once and cached, then filtered here).
    n_top
        Number of top-ranked genes to return.

    Returns
    -------
    pandas.DataFrame
        Normalized rows for the requested cell type, sorted by ``rank``.
    """
    cl_id = _resolve_cl_id(cell_type)
    table = compute_tissue_markers(
        tissue,
        organism=organism,
        census_version=census_version,
        max_cells_per_type=max_cells_per_type,
        cache_dir=cache_dir,
        force=force,
    )
    hits = table[table["cell_ontology_id"] == cl_id]
    return hits.sort_values("rank", na_position="last").head(n_top).reset_index(drop=True)


def _resolve_cl_id(cell_type: str) -> str:
    """Return a canonical CL id, resolving a plain name via OLS if needed."""
    normalized = _celltype.normalize_cl_id(cell_type)
    if normalized is not None:
        return normalized
    from biodb import ols

    hit = ols.find_term(cell_type, ontology="cl")
    if not hit or not hit.get("obo_id"):
        raise ValueError(f"Could not resolve {cell_type!r} to a Cell Ontology id via OLS.")
    return hit["obo_id"]


def to_gmt(
    path: str | Path,
    *,
    tissue: str,
    organism: str = DEFAULT_ORGANISM,
    census_version: str = DEFAULT_CENSUS_VERSION,
    by: str = "cell_ontology_id",
    cache_dir: str | Path | None = None,
) -> Path:
    """Export computed cell-type marker gene sets for a tissue to a GMT file.

    Parameters
    ----------
    path
        Destination ``.gmt`` path.
    tissue, organism, census_version, cache_dir
        Forwarded to :func:`compute_tissue_markers`.
    by
        Column used as the GMT set id.

    Returns
    -------
    pathlib.Path
        The written path.
    """
    table = compute_tissue_markers(
        tissue, organism=organism, census_version=census_version, cache_dir=cache_dir
    )
    return _celltype.celltype_to_gmt(table, path, by=by)


__all__ = [
    "DEFAULT_CENSUS_VERSION",
    "DEFAULT_ORGANISM",
    "SOURCE_NAME",
    "CACHE_DIR",
    "list_tissues",
    "list_cell_types",
    "compute_tissue_markers",
    "query_markers",
    "to_gmt",
]
