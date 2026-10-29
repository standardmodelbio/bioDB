"""Flexible multi-species, multi-source cell-type marker aggregator.

One entry point — :func:`load_markers` — to pull cell-type marker genes across
all bioDB cell-type sources (CELLxGENE CellGuide, Cell Taxonomy, CellMarker 2.0)
and species into a single unified table. Every filter is **null-means-all**:
omit it (or pass ``None``) to include everything, or pass a value / list to
restrict.

The unified schema re-introduces a ``source`` column (redundant *within* a
single source's dataset, but the essential provenance key once sources are
combined) and a ``marker_type`` (``"computational"`` vs ``"canonical"``):

    source, marker_type, species, tissue, tissue_ontology_id,
    cell_type_name, cell_ontology_id, gene_symbol, gene_id, score, rank

``score`` is each source's ranking metric (CELLxGENE Marker Score for
computational rows; literature-support count for the curated sources; null for
canonical reference markers). Data is read via the source modules' own
``get_markers`` helpers (locally cached).

Examples
--------
>>> from biodb import load_markers
>>> load_markers()                                    # everything  # doctest: +SKIP
>>> load_markers(source="cellxgene", species="Homo sapiens")        # doctest: +SKIP
>>> load_markers(cell_type="CL:0000236", tissue="spleen")           # doctest: +SKIP
>>> load_markers(source=["celltaxonomy", "cellmarker"],             # doctest: +SKIP
...              cell_type="neuron")
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from biodb import cellmarker, celltaxonomy, cellxgene

logger = logging.getLogger(__name__)

SOURCES = ("cellxgene", "celltaxonomy", "cellmarker")
"""All cell-type marker sources ``load_markers`` can aggregate."""

UNIFIED_COLUMNS = [
    "source",
    "marker_type",
    "species",
    "tissue",
    "tissue_ontology_id",
    "cell_type_name",
    "cell_ontology_id",
    "gene_symbol",
    "gene_id",
    "score",
    "rank",
]
"""Common schema returned by :func:`load_markers`."""


def _as_list(value: str | Iterable[str] | None) -> list[str] | None:
    """Normalize a null-means-all filter to a list (or ``None`` = all)."""
    if value is None:
        return None
    return [value] if isinstance(value, str) else list(value)


def _match(series: pd.Series, values: list[str]) -> pd.Series:
    """Case-insensitive membership test against a list of allowed values."""
    wanted = {v.lower() for v in values}
    return series.astype("string").str.lower().isin(wanted)


def load_markers(
    *,
    source: str | Iterable[str] | None = None,
    species: str | Iterable[str] | None = None,
    cell_type: str | Iterable[str] | None = None,
    tissue: str | Iterable[str] | None = None,
    marker_type: str | Iterable[str] | None = None,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Aggregate cell-type markers across sources + species. Null = all.

    Parameters
    ----------
    source
        Source(s): any of ``"cellxgene"``, ``"celltaxonomy"``, ``"cellmarker"``.
        ``None`` = all three.
    species
        Species label(s) (e.g. ``"Homo sapiens"``). ``None`` = all. (Canonical
        CELLxGENE markers have no species and are kept unless ``species`` is set.)
    cell_type
        CL id(s) (``"CL:0000236"``) **or** cell-type name(s) — matched against
        either ``cell_ontology_id`` or ``cell_type_name``. ``None`` = all.
    tissue
        Tissue label(s) (``"spleen"``) **or** UBERON id(s) — matched against
        either ``tissue`` or ``tissue_ontology_id``. ``None`` = all.
    marker_type
        ``"computational"`` and/or ``"canonical"``. ``None`` = all.
    cache_dir
        Optional cache override forwarded to the source loaders.

    Returns
    -------
    pandas.DataFrame
        Columns per :data:`UNIFIED_COLUMNS`.
    """
    sources = _as_list(source) or list(SOURCES)
    unknown = set(sources) - set(SOURCES)
    if unknown:
        raise ValueError(f"Unknown source(s) {sorted(unknown)}; choose from {list(SOURCES)}.")

    frames: list[pd.DataFrame] = []
    for src in sources:
        frames.append(_load_source(src, cache_dir=cache_dir))
    out = pd.concat(frames, ignore_index=True)[UNIFIED_COLUMNS]

    species_f = _as_list(species)
    if species_f is not None:
        out = out[_match(out["species"], species_f)]
    if (mt := _as_list(marker_type)) is not None:
        out = out[_match(out["marker_type"], mt)]
    if (ct := _as_list(cell_type)) is not None:
        out = out[_match(out["cell_ontology_id"], ct) | _match(out["cell_type_name"], ct)]
    if (tis := _as_list(tissue)) is not None:
        out = out[_match(out["tissue"], tis) | _match(out["tissue_ontology_id"], tis)]

    return out.reset_index(drop=True)


def _load_source(src: str, *, cache_dir: str | Path | None) -> pd.DataFrame:
    """Load one source's markers, coerced to :data:`UNIFIED_COLUMNS`."""
    if src == "celltaxonomy":
        df = celltaxonomy.get_markers(cache_dir=cache_dir).assign(marker_type="canonical")
    elif src == "cellmarker":
        df = cellmarker.get_markers(which="all", cache_dir=cache_dir).assign(
            marker_type="canonical"
        )
    elif src == "cellxgene":
        df = cellxgene.get_all_cellguide_markers(kind="both", cache_dir=cache_dir, progress=False)
        df = df.rename(columns={"marker_score": "score"})
    else:  # pragma: no cover - guarded by caller
        raise ValueError(src)
    df = df.copy()
    df["source"] = src
    for col in UNIFIED_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[UNIFIED_COLUMNS]


__all__ = ["SOURCES", "UNIFIED_COLUMNS", "load_markers"]
