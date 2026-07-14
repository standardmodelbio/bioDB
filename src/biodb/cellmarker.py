"""CellMarker 2.0 client — bulk XLSX + in-memory query.

`CellMarker 2.0 <http://bio-bigdata.hrbmu.edu.cn/CellMarker/>`_ (Harbin Medical
University) is a manually curated database of cell-type marker genes for human
and mouse (~83k tissue–cell-type–marker entries), each row carrying a native
`Cell Ontology <https://obofoundry.org/ontology/cl.html>`_ id
(``cellontology_id``, stored in ``CL_0000000`` form).

``bioDB`` exposes both access modes:

* **Bulk mode** — :func:`download` pulls a per-scope XLSX
  (``all`` / ``human`` / ``mouse`` / ``seq``); :func:`load` reads it raw, and
  :func:`get_markers` returns the normalized *(species, tissue, cell type, CL
  id, gene, score, rank)* schema shared across the cell-type sources.
* **API mode** — :func:`query_markers` filters the (lazily cached) normalized
  frame by CL id or cell-type name.

CellMarker carries no per-gene numeric score, so ``score`` is the
**literature-support count** (distinct supporting PMIDs, falling back to record
count) and ``rank`` is the dense within-cell-type rank of that count.

.. note::
   Reading the XLSX requires the ``openpyxl`` package (the ``[celltype]``
   extra). The canonical host (``bio-bigdata.hrbmu.edu.cn``) is intermittently
   unreachable; :func:`download` falls back to the published mirror
   automatically.

Cached files live at ``~/.cache/biodb/cellmarker/``.

Examples
--------
>>> from biodb import cellmarker as cm
>>> path = cm.download("human")                       # doctest: +SKIP
>>> markers = cm.get_markers(which="human")           # doctest: +SKIP
>>> neuron = cm.query_markers("CL:0000540")           # doctest: +SKIP
>>> cm.to_gmt("cellmarker.gmt", which="human")        # doctest: +SKIP

References
----------
* Home: http://bio-bigdata.hrbmu.edu.cn/CellMarker/
* Paper: Hu et al., *Nucleic Acids Research* 2023 (D870–D876).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

from biodb import _celltype
from biodb._downloads import stream_to_file

logger = logging.getLogger(__name__)

DOWNLOAD_BASE_URL = "http://bio-bigdata.hrbmu.edu.cn/CellMarker/CellMarker_download_files/file"
"""Canonical CellMarker 2.0 bulk-file root."""

MIRROR_BASE_URL = "http://117.50.127.228/CellMarker/CellMarker_download_files/file"
"""Published mirror, used when the canonical host is unreachable."""

FILES: dict[str, str] = {
    "all": "Cell_marker_All.xlsx",
    "human": "Cell_marker_Human.xlsx",
    "mouse": "Cell_marker_Mouse.xlsx",
    "seq": "Cell_marker_Seq.xlsx",
}
"""Bulk-file scopes → published XLSX filenames."""

SOURCE_NAME = "cellmarker"
"""Value written into the normalized ``source`` column."""

CACHE_DIR = Path("~/.cache/biodb/cellmarker").expanduser()
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Raw XLSX column names → normalized-schema fields.
_SPECIES_COL = "species"
_TISSUE_COL = "tissue_type"
_CELL_NAME_COL = "cell_name"
_CL_COL = "cellontology_id"
_GENE_SYMBOL_COL = "Symbol"
_GENE_ID_COL = "GeneID"
_PMID_COL = "PMID"

# Lazy in-memory cache of the normalized frame, keyed by ``which``.
_MARKERS_CACHE: dict[str, pd.DataFrame] = {}


# ─── Bulk mode — download + readers ──────────────────────────────────────────


def download(
    which: str = "all",
    cache_dir: str | Path | None = None,
    *,
    force: bool = False,
    progress: bool = True,
    timeout: int = 600,
) -> Path:
    """Download a CellMarker 2.0 XLSX to the cache.

    Tries the canonical host first, then the mirror.

    Parameters
    ----------
    which
        One of :data:`FILES` (``"all"``, ``"human"``, ``"mouse"``, ``"seq"``).
    cache_dir
        Override the cache location. Defaults to :data:`CACHE_DIR`.
    force
        Re-download even if cached.
    progress
        Show a tqdm progress bar.
    timeout
        Per-request timeout (seconds).

    Returns
    -------
    pathlib.Path
        Path to the cached XLSX.

    Raises
    ------
    ValueError
        If ``which`` is not a known scope.
    """
    if which not in FILES:
        raise ValueError(f"which must be one of {sorted(FILES)}, got {which!r}")
    root = Path(cache_dir).expanduser() if cache_dir else CACHE_DIR
    root.mkdir(parents=True, exist_ok=True)
    filename = FILES[which]
    dst = root / filename
    if dst.exists() and not force:
        return dst

    last_error: Exception | None = None
    for base in (DOWNLOAD_BASE_URL, MIRROR_BASE_URL):
        url = f"{base}/{filename}"
        try:
            logger.info("Downloading CellMarker %s from %s", which, url)
            return stream_to_file(url, dst, progress=progress, desc=filename, timeout=timeout)
        except requests.exceptions.RequestException as exc:  # noqa: PERF203
            logger.warning("CellMarker download from %s failed: %s", url, exc)
            last_error = exc
    raise RuntimeError(
        f"Could not download CellMarker {which!r} from canonical host or mirror"
    ) from last_error


def load(
    which: str = "all",
    cache_dir: str | Path | None = None,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """Load a raw CellMarker 2.0 XLSX as a DataFrame.

    Downloads it first if not cached. Requires ``openpyxl`` (the ``[celltype]``
    extra).

    Parameters
    ----------
    which, cache_dir, force
        Forwarded to :func:`download`.

    Returns
    -------
    pandas.DataFrame
        The sheet's columns verbatim (``species``, ``tissue_type``,
        ``cell_name``, ``cellontology_id``, ``Symbol``, ``GeneID``, ``PMID``, …).
    """
    path = download(which, cache_dir=cache_dir, force=force)
    return pd.read_excel(path, dtype=str, engine="openpyxl")


def get_markers(
    *,
    which: str = "all",
    map_to_cl: bool = False,
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Return normalized, ranked cell-type marker genes.

    Collapses the raw rows to one row per *(species, cell type, gene)*, scoring
    each by its literature-support count (distinct PMIDs, falling back to record
    count) and adding a dense within-cell-type ``rank``.

    Parameters
    ----------
    which
        Which bulk file to normalize (see :data:`FILES`).
    map_to_cl
        If True, fill blank CL ids by resolving the cell-type name via
        :func:`biodb._celltype.resolve_cl` (OLS-backed, cached).
    cache_dir, force
        Forwarded to :func:`load`.

    Returns
    -------
    pandas.DataFrame
        Columns per :data:`biodb._celltype.NORMALIZED_COLUMNS`.
    """
    raw = load(which, cache_dir=cache_dir, force=force)

    df = pd.DataFrame(
        {
            "species": raw[_SPECIES_COL],
            "tissue": raw[_TISSUE_COL],
            "cell_type_name": raw[_CELL_NAME_COL],
            "cell_ontology_id": raw[_CL_COL].map(_celltype.normalize_cl_id),
            "gene_symbol": raw[_GENE_SYMBOL_COL],
            "gene_id": raw[_GENE_ID_COL],
            "pmid": raw[_PMID_COL],
        }
    )
    df = df[df["gene_symbol"].notna() & df["cell_type_name"].notna()]

    grouped = (
        df.groupby(["species", "cell_type_name", "cell_ontology_id", "gene_symbol"], dropna=False)
        .agg(
            tissue=("tissue", "first"),
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
    which: str = "all",
    species: str | None = None,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Return ranked marker genes for a single cell type.

    Parameters
    ----------
    cell_type
        A CL id (``"CL:0000540"``) when ``by="cl"``, or a cell-type name
        (case-insensitive) when ``by="name"``.
    by
        ``"cl"`` (default) or ``"name"``.
    which
        Which bulk file to build the in-memory index from.
    species
        Optional case-insensitive species filter.
    cache_dir
        Forwarded to :func:`get_markers` on first (cache-populating) call.

    Returns
    -------
    pandas.DataFrame
        Normalized rows for the requested cell type, sorted by ``rank``.
    """
    if which not in _MARKERS_CACHE:
        _MARKERS_CACHE[which] = get_markers(which=which, cache_dir=cache_dir)

    df = _MARKERS_CACHE[which]
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
    which: str = "all",
    by: str = "cell_ontology_id",
    map_to_cl: bool = False,
    cache_dir: str | Path | None = None,
) -> Path:
    """Export cell-type marker gene sets to a GMT file.

    Parameters
    ----------
    path
        Destination ``.gmt`` path.
    which
        Which bulk file to export.
    by
        Column used as the GMT set id.
    map_to_cl, cache_dir
        Forwarded to :func:`get_markers`.

    Returns
    -------
    pathlib.Path
        The written path.
    """
    markers = get_markers(which=which, map_to_cl=map_to_cl, cache_dir=cache_dir)
    return _celltype.celltype_to_gmt(markers, path, by=by)


__all__ = [
    "DOWNLOAD_BASE_URL",
    "MIRROR_BASE_URL",
    "FILES",
    "SOURCE_NAME",
    "CACHE_DIR",
    "download",
    "load",
    "get_markers",
    "query_markers",
    "to_gmt",
]
