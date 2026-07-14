"""Shared internals for the cell-type marker-gene sources.

Internal module (underscore-prefixed), the cell-type analogue of
:mod:`biodb._downloads`. :mod:`biodb.celltaxonomy`, :mod:`biodb.cellmarker`,
and :mod:`biodb.cellxgene` each expose a *(cell type → ranked gene list)*
view keyed to the `Cell Ontology <https://obofoundry.org/ontology/cl.html>`_
(CL). They differ only in *where* the data comes from; the normalized output
schema, the CL-resolution step, the per-cell-type ranking, and the GMT export
are identical, so they live here.

What this centralises
---------------------
* :data:`NORMALIZED_COLUMNS` — the long-format schema every source emits, so a
  caller can ``pd.concat`` frames from different sources without reconciling
  column names.
* :func:`normalize_cl_id` — coerce a CL identifier to canonical ``CL:0000000``
  form (CellMarker ships them as ``CL_0000000``; Cell Taxonomy uses the colon).
* :func:`resolve_cl` — a batched, on-disk-cached wrapper over
  :func:`biodb.ols.find_terms` that maps free-text cell-type labels to CL IDs.
  Used by every source's opt-in ``map_to_cl=True`` path so a label only ever
  hits OLS once across sessions.
* :func:`rank_within_group` — dense per-cell-type rank from a score column, so
  ``rank`` means the same thing (1 = strongest marker) in every source.
* :func:`celltype_to_gmt` — one GMT line per cell type, genes ordered by rank;
  round-trips through :func:`biodb.utils.read_gmt`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

NORMALIZED_COLUMNS: list[str] = [
    "species",
    "tissue",
    "tissue_ontology_id",
    "cell_type_name",
    "cell_ontology_id",
    "gene_symbol",
    "gene_id",
    "score",
    "rank",
    "source",
]
"""The long-format schema every cell-type source emits.

One row per *(cell type, gene)* observation. ``score`` is the source's ranking
metric (support/PMID count for the curated sources; an effect size for
CELLxGENE). ``rank`` is the dense within-cell-type rank of that score
(1 = strongest). ``source`` names the originating database."""

CL_RESOLUTION_CACHE = Path("~/.cache/biodb/celltype").expanduser()
"""On-disk cache root for :func:`resolve_cl`. One shared parquet keyed by
free-text label, so a label→CL lookup survives process restarts and is never
re-fetched from OLS."""

_CL_RESOLUTION_COLUMNS = ["label", "cell_ontology_id", "cl_label", "match_quality"]


def normalize_obo_id(value: object, prefix: str) -> str | None:
    """Coerce an OBO identifier to canonical ``PREFIX:0000000`` form.

    Sources vary between ``PREFIX_0000000`` (underscore) and ``PREFIX:0000000``
    (colon); this returns the colon form, or ``None`` for missing / non-matching
    values (``NaN``, ``"NA"``, ``""``, wrong prefix).

    Parameters
    ----------
    value
        A candidate identifier, or any non-string / missing value.
    prefix
        Expected OBO prefix (e.g. ``"CL"``, ``"UBERON"``).
    """
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or text.upper() in {"NA", "NAN", "NONE"}:
        return None
    text = text.replace("_", ":")
    if not text.upper().startswith(f"{prefix.upper()}:"):
        return None
    return f"{prefix}:{text.split(':', 1)[1]}"


def normalize_cl_id(value: object) -> str | None:
    """Coerce a Cell Ontology identifier to canonical ``CL:0000000`` form.

    CellMarker stores them as ``CL_0000449``; Cell Taxonomy uses ``CL:0000007``.
    Everything downstream (and :func:`biodb.ols.get_term`) expects the colon form.
    """
    return normalize_obo_id(value, "CL")


def normalize_uberon_id(value: object) -> str | None:
    """Coerce a UBERON tissue identifier to canonical ``UBERON:0000000`` form.

    CellMarker stores them as ``UBERON_0000916``; Cell Taxonomy uses the colon
    form; CELLxGENE/WMG uses the colon form.
    """
    return normalize_obo_id(value, "UBERON")


def resolve_cl(
    labels: Iterable[str],
    *,
    cache_dir: str | Path | None = None,
    timeout: int = 30,
    min_quality: int = 1,
) -> pd.DataFrame:
    """Map free-text cell-type labels to Cell Ontology IDs via OLS.

    Thin, cached wrapper over :func:`biodb.ols.find_terms` (scoped to the
    ``cl`` ontology). Only labels not already in the on-disk cache hit the
    network; results — including "no confident match" — are persisted so the
    same label is never looked up twice.

    Parameters
    ----------
    labels
        Free-text cell-type names (e.g. ``"CD8-positive, alpha-beta T cell"``).
        Deduplicated and stripped before lookup.
    cache_dir
        Override the parquet cache location. Defaults to
        :data:`CL_RESOLUTION_CACHE`.
    timeout
        Per-request OLS timeout, forwarded to :func:`biodb.ols.find_terms`.
    min_quality
        Minimum :func:`biodb.ols.find_terms` ``match_quality`` (0–4) to accept
        a hit. Default 1 keeps substring matches; pass 4 to demand an exact
        label match.

    Returns
    -------
    pandas.DataFrame
        Columns ``label``, ``cell_ontology_id`` (``None`` when unresolved),
        ``cl_label``, ``match_quality`` — one row per distinct input label.
    """
    from biodb import ols

    wanted = sorted({s.strip() for s in labels if isinstance(s, str) and s.strip()})
    cache = Path(cache_dir).expanduser() if cache_dir else CL_RESOLUTION_CACHE
    cache.mkdir(parents=True, exist_ok=True)
    cache_path = cache / "cl_resolution.parquet"

    known = pd.DataFrame(columns=_CL_RESOLUTION_COLUMNS)
    if cache_path.exists():
        known = pd.read_parquet(cache_path)

    todo = sorted(set(wanted) - set(known["label"].tolist()))
    new_rows: list[dict[str, object]] = []
    for label in todo:
        hit = ols.find_term(label, ontology="cl", timeout=timeout)
        if hit and int(hit.get("match_quality") or 0) >= min_quality:
            new_rows.append(
                {
                    "label": label,
                    "cell_ontology_id": hit.get("obo_id"),
                    "cl_label": hit.get("label"),
                    "match_quality": int(hit.get("match_quality") or 0),
                }
            )
        else:
            new_rows.append(
                {"label": label, "cell_ontology_id": None, "cl_label": None, "match_quality": 0}
            )

    if new_rows:
        known = pd.concat([known, pd.DataFrame(new_rows)], ignore_index=True)
        known.to_parquet(cache_path, index=False)

    return known[known["label"].isin(wanted)].reset_index(drop=True)


def rank_within_group(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    score_col: str,
    ascending: bool = False,
) -> pd.Series:
    """Dense rank of ``score_col`` within each ``group_cols`` group.

    ``method="dense"`` so ties share a rank and no rank values are skipped —
    rank 1 is the strongest marker for that cell type (with the default
    ``ascending=False``). Returns a nullable-integer Series aligned to ``df``.

    Parameters
    ----------
    df
        Long-format marker frame.
    group_cols
        Columns identifying one cell type (e.g. ``["species", "cell_ontology_id"]``).
    score_col
        The ranking metric column.
    ascending
        Rank direction. Default ``False`` → higher score = better (rank 1).

    Returns
    -------
    pandas.Series
        ``Int64`` ranks aligned to ``df.index``.
    """
    return (
        df.groupby(group_cols, dropna=False)[score_col]
        .rank(method="dense", ascending=ascending)
        .astype("Int64")
    )


def celltype_to_gmt(
    df: pd.DataFrame,
    path: str | Path,
    *,
    by: str = "cell_ontology_id",
    name_col: str = "cell_type_name",
    gene_col: str = "gene_symbol",
    rank_col: str = "rank",
) -> Path:
    """Write a normalized marker frame to a GMT file — one line per cell type.

    Each line is ``<set id>\\t<description>\\t<gene1>\\t<gene2>…`` (the standard
    GMT layout read by :func:`biodb.utils.read_gmt`). Genes are ordered by
    ``rank_col`` ascending (strongest marker first) and de-duplicated within a
    set, preserving that order. Rows whose ``by`` key or ``gene_col`` is missing
    are skipped.

    Parameters
    ----------
    df
        A frame in the :data:`NORMALIZED_COLUMNS` schema.
    path
        Destination ``.gmt`` path; parent dirs are created.
    by
        Column whose value becomes the GMT set id (default the CL id).
    name_col
        Column used for the GMT description field.
    gene_col
        Column holding the gene symbols.
    rank_col
        Column used to order genes within a set. Ignored if absent.

    Returns
    -------
    pathlib.Path
        The written path.
    """
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    frame = df[df[by].notna() & df[gene_col].notna()].copy()
    lines: list[str] = []
    for set_id, grp in frame.groupby(by, sort=True):
        if rank_col in grp.columns:
            grp = grp.sort_values(rank_col, na_position="last")
        description = ""
        if name_col in grp.columns:
            names = [n for n in grp[name_col].tolist() if isinstance(n, str) and n]
            if names:
                description = names[0]
        genes: list[str] = []
        seen: set[str] = set()
        for gene in grp[gene_col].tolist():
            if isinstance(gene, str) and gene and gene not in seen:
                seen.add(gene)
                genes.append(gene)
        if genes:
            lines.append("\t".join([str(set_id), description, *genes]))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote %d cell-type gene sets to %s", len(lines), path)
    return path


__all__ = [
    "NORMALIZED_COLUMNS",
    "CL_RESOLUTION_CACHE",
    "normalize_obo_id",
    "normalize_cl_id",
    "normalize_uberon_id",
    "resolve_cl",
    "rank_within_group",
    "celltype_to_gmt",
]
