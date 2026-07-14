"""Cell Taxonomy (CNCB-NGDC) client — bulk flat file + in-memory query.

`Cell Taxonomy <https://ngdc.cncb.ac.cn/celltaxonomy/>`_ is a curated
cross-species catalog of cell types, each mapped to a `Cell Ontology
<https://obofoundry.org/ontology/cl.html>`_ (CL) term and annotated with
literature-supported marker genes (~26k markers, ~3.1k cell types, 34 species).
Unlike a single-cell atlas it carries **native CL IDs**, which makes it the
controlled-vocabulary backbone for bioDB's cell-type marker views.

``bioDB`` exposes both access modes:

* **Bulk mode** — :func:`download` pulls the single tab-delimited
  ``Cell_Taxonomy_resource.txt`` (~80 MB); :func:`load_resource` reads it raw,
  and :func:`get_markers` returns the normalized *(species, tissue, cell type,
  CL id, gene, score, rank)* schema shared with :mod:`biodb.cellmarker` /
  :mod:`biodb.cellxgene`.
* **API mode** — :func:`query_markers` filters the (lazily cached) normalized
  frame by CL id or cell-type name, one cell type at a time.

Cell Taxonomy has no per-gene numeric score, so the ``score`` column is the
**literature-support count** (distinct supporting PMIDs, falling back to the
record count) and ``rank`` is the dense within-cell-type rank of that count.
:func:`to_gmt` exports one ranked gene set per cell type.

Cached files live at ``~/.cache/biodb/celltaxonomy/``.

Examples
--------
>>> from biodb import celltaxonomy as ct
>>> path = ct.download()                                   # doctest: +SKIP
>>> markers = ct.get_markers(species="Homo sapiens")       # doctest: +SKIP
>>> neuron = ct.query_markers("CL:0000540")                # doctest: +SKIP
>>> ct.to_gmt("celltaxonomy.gmt")                          # doctest: +SKIP

References
----------
* Home: https://ngdc.cncb.ac.cn/celltaxonomy/
* Download: https://ngdc.cncb.ac.cn/celltaxonomy/download
* Paper: Jiang et al., *Nucleic Acids Research* 2023 (D853–D860).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from biodb import _celltype
from biodb._downloads import stream_to_file

logger = logging.getLogger(__name__)

DOWNLOAD_BASE_URL = "https://download.cncb.ac.cn/celltaxonomy"
"""Cell Taxonomy bulk-download root (HTTPS-served)."""

RESOURCE_FILE = "Cell_Taxonomy_resource.txt"
"""The single curated tab-delimited resource file (all species)."""

SOURCE_NAME = "celltaxonomy"
"""Value written into the normalized ``source`` column."""

CACHE_DIR = Path("~/.cache/biodb/celltaxonomy").expanduser()
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Raw column names in Cell_Taxonomy_resource.txt → normalized-schema fields.
_SPECIES_COL = "Species"
_TISSUE_COL = "Tissue_standard"
_TISSUE_ID_COL = "Tissue_UberonOntology_ID"
_CELL_NAME_COL = "Cell_standard"
_CL_COL = "Specific_Cell_Ontology_ID"
_GENE_SYMBOL_COL = "Cell_Marker"
_GENE_ID_COL = "Gene_ENTREZID"
_PMID_COL = "PMID"

# Lazy in-memory cache of the fully normalized (all-species) frame, backing
# ``query_markers`` — mirrors the pattern in ``biodb.gwas_atlas``.
_MARKERS_CACHE: pd.DataFrame | None = None


# ─── Bulk mode — download + readers ──────────────────────────────────────────


def download(
    cache_dir: str | Path | None = None,
    *,
    force: bool = False,
    progress: bool = True,
    timeout: int = 600,
) -> Path:
    """Download ``Cell_Taxonomy_resource.txt`` to the cache.

    Parameters
    ----------
    cache_dir
        Override the cache location. Defaults to :data:`CACHE_DIR`.
    force
        Re-download even if the file is already cached.
    progress
        Show a tqdm progress bar.
    timeout
        Per-request timeout (seconds). The file is ~80 MB.

    Returns
    -------
    pathlib.Path
        Path to the cached resource file.
    """
    root = Path(cache_dir).expanduser() if cache_dir else CACHE_DIR
    root.mkdir(parents=True, exist_ok=True)
    dst = root / RESOURCE_FILE
    if dst.exists() and not force:
        return dst
    url = f"{DOWNLOAD_BASE_URL}/{RESOURCE_FILE}"
    logger.info("Downloading Cell Taxonomy resource from %s", url)
    return stream_to_file(url, dst, progress=progress, desc=RESOURCE_FILE, timeout=timeout)


def load_resource(
    cache_dir: str | Path | None = None,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """Load the raw Cell Taxonomy resource file as a DataFrame.

    Downloads it first if not cached. The literal string ``"NA"`` is treated
    as missing.

    Parameters
    ----------
    cache_dir, force
        Forwarded to :func:`download`.

    Returns
    -------
    pandas.DataFrame
        The file's columns verbatim (``Species``, ``Tissue_standard``,
        ``Cell_standard``, ``Specific_Cell_Ontology_ID``, ``Cell_Marker``,
        ``Gene_ENTREZID``, ``PMID``, …).
    """
    path = download(cache_dir=cache_dir, force=force)
    return pd.read_csv(path, sep="\t", dtype=str, na_values=["NA"], keep_default_na=True)


def get_markers(
    *,
    species: str | None = None,
    map_to_cl: bool = False,
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Return normalized, ranked cell-type marker genes.

    Collapses the raw evidence rows to one row per *(species, cell type, gene)*,
    scoring each by its literature-support count (distinct PMIDs, falling back
    to record count) and adding a dense within-cell-type ``rank``.

    Parameters
    ----------
    species
        Case-insensitive exact species filter (e.g. ``"Homo sapiens"``). If
        omitted, all species are returned.
    map_to_cl
        If True, fill blank CL ids by resolving the cell-type name via
        :func:`biodb._celltype.resolve_cl` (OLS-backed, cached). Off by default
        (network cost).
    cache_dir, force
        Forwarded to :func:`load_resource`.

    Returns
    -------
    pandas.DataFrame
        Columns per :data:`biodb._celltype.NORMALIZED_COLUMNS`.
    """
    raw = load_resource(cache_dir=cache_dir, force=force)
    if species is not None:
        raw = raw[raw[_SPECIES_COL].str.lower() == species.lower()]

    df = pd.DataFrame(
        {
            "species": raw[_SPECIES_COL],
            "tissue": raw[_TISSUE_COL],
            "tissue_ontology_id": raw[_TISSUE_ID_COL].map(_celltype.normalize_uberon_id),
            "cell_type_name": raw[_CELL_NAME_COL],
            "cell_ontology_id": raw[_CL_COL].map(_celltype.normalize_cl_id),
            "gene_symbol": raw[_GENE_SYMBOL_COL],
            "gene_id": raw[_GENE_ID_COL],
            "pmid": raw[_PMID_COL],
        }
    )
    df = df[df["gene_symbol"].notna() & df["cell_type_name"].notna()]

    # One row per gene within a cell type; score = literature-support count.
    grouped = (
        df.groupby(["species", "cell_type_name", "cell_ontology_id", "gene_symbol"], dropna=False)
        .agg(
            tissue=("tissue", "first"),
            tissue_ontology_id=("tissue_ontology_id", "first"),
            gene_id=("gene_id", "first"),
            n_pmid=("pmid", "nunique"),
            n_records=("pmid", "size"),
        )
        .reset_index()
    )
    grouped["score"] = grouped["n_pmid"].where(grouped["n_pmid"] > 0, grouped["n_records"])
    grouped["source"] = SOURCE_NAME

    if map_to_cl:
        grouped = _fill_missing_cl(grouped, cache_dir=cache_dir)

    grouped["rank"] = _celltype.rank_within_group(
        grouped,
        group_cols=["species", "cell_type_name", "cell_ontology_id"],
        score_col="score",
    )
    return grouped[_celltype.NORMALIZED_COLUMNS].reset_index(drop=True)


def _fill_missing_cl(df: pd.DataFrame, *, cache_dir: str | Path | None) -> pd.DataFrame:
    """Resolve blank ``cell_ontology_id`` values from ``cell_type_name`` via OLS."""
    missing = df["cell_ontology_id"].isna()
    if not missing.any():
        return df
    resolution = _celltype.resolve_cl(
        df.loc[missing, "cell_type_name"].unique(), cache_dir=cache_dir
    )
    mapping = dict(zip(resolution["label"], resolution["cell_ontology_id"], strict=False))
    df = df.copy()
    df.loc[missing, "cell_ontology_id"] = df.loc[missing, "cell_type_name"].map(mapping)
    return df


# ─── API mode — targeted query ───────────────────────────────────────────────


def query_markers(
    cell_type: str,
    *,
    by: str = "cl",
    species: str | None = None,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Return ranked marker genes for a single cell type.

    Filters the normalized, all-species frame (built once and cached in memory)
    to one cell type.

    Parameters
    ----------
    cell_type
        A CL id (``"CL:0000540"``) when ``by="cl"``, or a cell-type name
        (case-insensitive) when ``by="name"``.
    by
        ``"cl"`` (default) or ``"name"``.
    species
        Optional case-insensitive species filter applied after lookup.
    cache_dir
        Forwarded to :func:`get_markers` on first (cache-populating) call.

    Returns
    -------
    pandas.DataFrame
        Normalized rows for the requested cell type, sorted by ``rank``.
    """
    global _MARKERS_CACHE
    if _MARKERS_CACHE is None:
        _MARKERS_CACHE = get_markers(cache_dir=cache_dir)

    df = _MARKERS_CACHE
    if by == "cl":
        target = _celltype.normalize_cl_id(cell_type)
        hits = df[df["cell_ontology_id"] == target]
    elif by == "name":
        hits = df[df["cell_type_name"].str.lower() == cell_type.lower()]
    else:
        raise ValueError(f"by must be 'cl' or 'name', got {by!r}")

    if species is not None:
        hits = hits[hits["species"].str.lower() == species.lower()]
    return hits.sort_values("rank", na_position="last").reset_index(drop=True)


def to_gmt(
    path: str | Path,
    *,
    by: str = "cell_ontology_id",
    species: str | None = None,
    map_to_cl: bool = False,
    cache_dir: str | Path | None = None,
) -> Path:
    """Export cell-type marker gene sets to a GMT file.

    Parameters
    ----------
    path
        Destination ``.gmt`` path.
    by
        Column used as the GMT set id (``"cell_ontology_id"`` or
        ``"cell_type_name"``).
    species, map_to_cl, cache_dir
        Forwarded to :func:`get_markers`.

    Returns
    -------
    pathlib.Path
        The written path.
    """
    markers = get_markers(species=species, map_to_cl=map_to_cl, cache_dir=cache_dir)
    return _celltype.celltype_to_gmt(markers, path, by=by)


__all__ = [
    "DOWNLOAD_BASE_URL",
    "RESOURCE_FILE",
    "SOURCE_NAME",
    "CACHE_DIR",
    "download",
    "load_resource",
    "get_markers",
    "query_markers",
    "to_gmt",
]
