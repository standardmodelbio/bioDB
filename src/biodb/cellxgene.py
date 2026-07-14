"""CZ CELLxGENE Discover client — precomputed marker genes via the WMG API.

`CZ CELLxGENE Discover <https://cellxgene.cziscience.com/>`_ aggregates
single-cell data across studies and — through its **WMG ("Where's My Gene")**
backend — serves *precomputed* marker genes for every cell type in every
tissue, keyed to the `Cell Ontology <https://obofoundry.org/ontology/cl.html>`_
(CL) and UBERON. This is the same data behind the Discover UI's "Find Marker
Genes" tool, exposed as a public REST API, so ``bioDB`` fetches CZI's own
**Marker Score** rather than recomputing anything.

The Marker Score is CZI's effect-size metric (roughly the 10th percentile of
per-comparison Cohen's d from Welch's t-tests of the target cell type against
the other cell types in the tissue); each gene also carries a **specificity**
(fraction of the tissue's other cell types the gene distinguishes it from).

``bioDB`` exposes both access modes over the same API:

* **API mode** — :func:`query_markers` returns the ranked markers for one cell
  type in one tissue (resolving a plain cell-type name to a CL id via OLS if
  needed).
* **Bulk mode** — :func:`get_tissue_markers` returns markers for *every* cell
  type in a tissue, in the normalized *(species, tissue, cell type, CL id, gene,
  score, rank)* schema shared across the cell-type sources.
  :func:`list_tissues` / :func:`list_cell_types` support discovery.

No heavy dependencies — this is a plain ``requests`` REST client (unlike the
bulk single-cell ``cellxgene-census`` stack). Cached tables live at
``~/.cache/biodb/cellxgene/``.

Examples
--------
>>> from biodb import cellxgene as cx
>>> cx.list_tissues()[:3]                                       # doctest: +SKIP
['adipose tissue', 'adrenal gland', 'blood']
>>> markers = cx.query_markers("CL:0000236", tissue="spleen")   # doctest: +SKIP
>>> markers.iloc[0]["gene_symbol"]                              # doctest: +SKIP
'CD79A'
>>> table = cx.get_tissue_markers("spleen")                     # doctest: +SKIP
>>> cx.to_gmt("cellxgene_spleen.gmt", tissue="spleen")          # doctest: +SKIP

References
----------
* WMG API base: https://api.cellxgene.cziscience.com/wmg/v2
* Find Marker Genes docs:
  https://cellxgene.cziscience.com/docs/04__Analyze%20Public%20Data/4_2__Gene%20Expression%20Documentation/4_2_5__Find%20Marker%20Genes
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm

from biodb import _celltype

logger = logging.getLogger(__name__)

WMG_API_BASE_URL = "https://api.cellxgene.cziscience.com/wmg/v2"
"""CZ CELLxGENE WMG ("Where's My Gene") REST API root."""

DE_API_BASE_URL = "https://api.cellxgene.cziscience.com/de/v1"
"""CZ CELLxGENE Differential Expression REST API root."""

CELLGUIDE_CDN_BASE = "https://cellguide.cellxgene.cziscience.com"
"""CZ CELLxGENE CellGuide CDN root. Serves per-cell-type snapshot JSON with
both computational (Marker Score) and canonical (curated) marker genes."""

CELLGUIDE_COLUMNS = [
    "species",
    "tissue",
    "cell_type_name",
    "cell_ontology_id",
    "gene_symbol",
    "gene_id",
    "marker_type",
    "marker_score",
    "specificity",
    "mean_expression",
    "pct_expressing",
    "publication",
    "rank",
    "source",
]
"""Schema for CellGuide markers (:func:`cellguide_markers`). ``marker_type`` is
``"computational"`` (Marker Score, per species × tissue) or ``"canonical"``
(curated, cross-species — ``species`` is left null)."""

DEFAULT_ORGANISM = "Homo sapiens"
"""Default organism (accepts the label or an ``NCBITaxon:`` id)."""

DEFAULT_TEST = "ttest"
"""Statistical test backing the Marker Score (the only value WMG exposes)."""

NORMAL_DISEASE_ID = "PATO:0000461"
"""CELLxGENE's ontology id for "normal" (non-diseased) cells."""

SOURCE_NAME = "cellxgene"
"""Value written into the normalized ``source`` column."""

CACHE_DIR = Path("~/.cache/biodb/cellxgene").expanduser()
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_USER_AGENT = "biodb/0.1 (+https://github.com/bschilder/bioDB)"


# ─── HTTP + reference-dimension helpers ──────────────────────────────────────


def _get(path: str, *, timeout: int = 60) -> dict[str, Any]:
    """GET a WMG endpoint and return parsed JSON."""
    resp = requests.get(
        f"{WMG_API_BASE_URL}/{path.lstrip('/')}",
        headers={"User-Agent": _USER_AGENT},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def _post(
    path: str, payload: dict, *, base: str = WMG_API_BASE_URL, timeout: int = 120
) -> dict[str, Any]:
    """POST JSON to a WMG/DE endpoint and return parsed JSON."""
    resp = requests.post(
        f"{base}/{path.lstrip('/')}",
        json=payload,
        headers={"User-Agent": _USER_AGENT, "Content-Type": "application/json"},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


@lru_cache(maxsize=1)
def _primary_filter_dimensions() -> dict[str, Any]:
    """Fetch (and memoize) WMG's primary filter dimensions.

    Carries the organism list, per-organism tissue list, per-organism
    gene-id→symbol map, and the current ``snapshot_id``.
    """
    return _get("primary_filter_dimensions")


def _flatten_terms(terms: list[dict[str, str]]) -> dict[str, str]:
    """Flatten WMG's ``[{"id": "label"}, ...]`` term lists into one dict."""
    out: dict[str, str] = {}
    for item in terms:
        out.update(item)
    return out


def _resolve_organism(organism: str) -> tuple[str, str]:
    """Return ``(NCBITaxon id, label)`` for an organism label or id."""
    terms = _flatten_terms(_primary_filter_dimensions()["organism_terms"])
    if organism in terms:  # already an id
        return organism, terms[organism]
    for oid, label in terms.items():
        if label.lower() == organism.lower():
            return oid, label
    raise ValueError(f"Unknown organism {organism!r}. Available: {sorted(terms.values())}")


def _resolve_tissue(tissue: str, organism_id: str) -> tuple[str, str]:
    """Return ``(UBERON id, label)`` for a tissue label or id in an organism."""
    terms = _flatten_terms(_primary_filter_dimensions()["tissue_terms"][organism_id])
    if tissue in terms:  # already an id
        return tissue, terms[tissue]
    for tid, label in terms.items():
        if label.lower() == tissue.lower():
            return tid, label
    raise ValueError(f"Unknown tissue {tissue!r} for {organism_id}.")


def _gene_symbol_map(organism_id: str) -> dict[str, str]:
    """Return the Ensembl-id → gene-symbol map for an organism."""
    return _flatten_terms(_primary_filter_dimensions()["gene_terms"][organism_id])


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


# ─── Discovery ───────────────────────────────────────────────────────────────


def list_tissues(*, organism: str = DEFAULT_ORGANISM) -> list[str]:
    """List the tissues WMG has marker data for, for an organism.

    Parameters
    ----------
    organism
        Organism label (``"Homo sapiens"``) or ``NCBITaxon:`` id.

    Returns
    -------
    list[str]
        Sorted tissue labels.
    """
    organism_id, _ = _resolve_organism(organism)
    terms = _flatten_terms(_primary_filter_dimensions()["tissue_terms"][organism_id])
    return sorted(terms.values())


def list_cell_types(tissue: str, *, organism: str = DEFAULT_ORGANISM) -> pd.DataFrame:
    """List the cell types present in a tissue (with their CL ids).

    Parameters
    ----------
    tissue
        Tissue label (``"spleen"``) or ``UBERON:`` id.
    organism
        Organism label or ``NCBITaxon:`` id.

    Returns
    -------
    pandas.DataFrame
        Columns ``cell_ontology_id``, ``cell_type_name``.
    """
    cell_map = _cell_type_map(tissue, organism)
    return pd.DataFrame(
        {"cell_ontology_id": list(cell_map), "cell_type_name": list(cell_map.values())}
    )


def _filter_dims(tissue: str, organism: str) -> dict[str, Any]:
    """Return WMG ``filter_dims`` for a tissue (cell types, diseases, sex, …)."""
    organism_id, _ = _resolve_organism(organism)
    tissue_id, _ = _resolve_tissue(tissue, organism_id)
    payload = {
        "filter": {
            "organism_ontology_term_id": organism_id,
            "tissue_ontology_term_ids": [tissue_id],
        }
    }
    return _post("filters", payload)["filter_dims"]


def _cell_type_map(tissue: str, organism: str) -> dict[str, str]:
    """CL-id → label map for a tissue, via the WMG ``filters`` endpoint."""
    return _flatten_terms(_filter_dims(tissue, organism)["cell_type_terms"])


def list_diseases(tissue: str, *, organism: str = DEFAULT_ORGANISM) -> pd.DataFrame:
    """List the diseases with data in a tissue (excluding ``normal``).

    Parameters
    ----------
    tissue
        Tissue label or ``UBERON:`` id.
    organism
        Organism label or ``NCBITaxon:`` id.

    Returns
    -------
    pandas.DataFrame
        Columns ``disease_ontology_term_id``, ``disease``. Empty if the tissue
        has only normal cells.
    """
    terms = _flatten_terms(_filter_dims(tissue, organism)["disease_terms"])
    rows = [
        {"disease_ontology_term_id": did, "disease": label}
        for did, label in terms.items()
        if did != NORMAL_DISEASE_ID
    ]
    return pd.DataFrame(rows, columns=["disease_ontology_term_id", "disease"])


# ─── API mode — one cell type ────────────────────────────────────────────────


def query_markers(
    cell_type: str,
    *,
    tissue: str,
    organism: str = DEFAULT_ORGANISM,
    n_top: int = 25,
    test: str = DEFAULT_TEST,
) -> pd.DataFrame:
    """Return the top precomputed marker genes for one cell type in one tissue.

    Parameters
    ----------
    cell_type
        A CL id (``"CL:0000236"``) or a plain cell-type name (resolved to a CL
        id via :func:`biodb.ols.find_term`).
    tissue
        Tissue label (``"spleen"``) or ``UBERON:`` id (required — the Marker
        Score is tissue-specific).
    organism
        Organism label or ``NCBITaxon:`` id.
    n_top
        Number of top-ranked markers to request.
    test
        WMG statistical test (only ``"ttest"`` is exposed).

    Returns
    -------
    pandas.DataFrame
        Columns per :data:`biodb._celltype.NORMALIZED_COLUMNS`, sorted by
        ``rank`` (``score`` is WMG's Marker Score).
    """
    organism_id, organism_label = _resolve_organism(organism)
    tissue_id, tissue_label = _resolve_tissue(tissue, organism_id)
    cl_id = _resolve_cl_id(cell_type)

    payload = {
        "celltype": cl_id,
        "tissue": tissue_id,
        "organism": organism_id,
        "n_markers": n_top,
        "test": test,
    }
    genes = _post("markers", payload).get("marker_genes", [])
    cell_name = _cell_type_map(tissue, organism).get(cl_id)
    symbols = _gene_symbol_map(organism_id)

    df = _normalize(genes, organism_label, tissue_label, cell_name, cl_id, symbols)
    return df.sort_values("rank", na_position="last").reset_index(drop=True)


def _normalize(
    marker_genes: list[dict],
    organism_label: str,
    tissue_label: str,
    cell_name: str | None,
    cl_id: str,
    symbols: dict[str, str],
) -> pd.DataFrame:
    """Build a normalized-schema frame from WMG ``marker_genes`` records."""
    rows = [
        {
            "species": organism_label,
            "tissue": tissue_label,
            "cell_type_name": cell_name,
            "cell_ontology_id": cl_id,
            "gene_symbol": symbols.get(g["gene_ontology_term_id"], g["gene_ontology_term_id"]),
            "gene_id": g["gene_ontology_term_id"],
            "score": g.get("marker_score"),
            "source": SOURCE_NAME,
        }
        for g in marker_genes
    ]
    df = pd.DataFrame(rows, columns=[c for c in _celltype.NORMALIZED_COLUMNS if c != "rank"])
    if df.empty:
        df["rank"] = pd.Series(dtype="Int64")
        return df[_celltype.NORMALIZED_COLUMNS]
    df["rank"] = _celltype.rank_within_group(df, group_cols=["cell_ontology_id"], score_col="score")
    return df[_celltype.NORMALIZED_COLUMNS]


# ─── Bulk mode — every cell type in a tissue ─────────────────────────────────


def get_tissue_markers(
    tissue: str,
    *,
    organism: str = DEFAULT_ORGANISM,
    n_top_per_type: int = 25,
    test: str = DEFAULT_TEST,
    max_workers: int = 8,
    cache_dir: str | Path | None = None,
    force: bool = False,
    progress: bool = False,
) -> pd.DataFrame:
    """Fetch marker genes for every cell type in a tissue.

    Iterates the tissue's cell types (:func:`list_cell_types`) and pulls each
    one's markers concurrently, concatenating into the normalized schema. The
    assembled table is cached to parquet keyed by
    ``(organism, tissue, snapshot, test, n)``.

    Parameters
    ----------
    tissue
        Tissue label or ``UBERON:`` id.
    organism
        Organism label or ``NCBITaxon:`` id.
    n_top_per_type
        Markers to request per cell type.
    test
        WMG statistical test.
    max_workers
        Concurrent ``markers`` requests. Kept modest by default — WMG is a
        free public service.
    cache_dir, force
        Cache location override / bypass.
    progress
        Show a tqdm bar over the tissue's cell types.

    Returns
    -------
    pandas.DataFrame
        Columns per :data:`biodb._celltype.NORMALIZED_COLUMNS`.
    """
    organism_id, organism_label = _resolve_organism(organism)
    tissue_id, tissue_label = _resolve_tissue(tissue, organism_id)
    snapshot = _primary_filter_dimensions().get("snapshot_id", "na")

    root = Path(cache_dir).expanduser() if cache_dir else CACHE_DIR
    root.mkdir(parents=True, exist_ok=True)
    slug = f"{organism_id}__{tissue_id}__{snapshot}__{test}__n{n_top_per_type}".replace(":", "-")
    cache_path = root / f"{slug}.parquet"
    if cache_path.exists() and not force:
        return pd.read_parquet(cache_path)

    cell_map = _cell_type_map(tissue, organism)
    symbols = _gene_symbol_map(organism_id)

    def _fetch(item: tuple[str, str]) -> pd.DataFrame | None:
        cl_id, cell_name = item
        payload = {
            "celltype": cl_id,
            "tissue": tissue_id,
            "organism": organism_id,
            "n_markers": n_top_per_type,
            "test": test,
        }
        genes = _post("markers", payload).get("marker_genes", [])
        if not genes:
            return None
        return _normalize(genes, organism_label, tissue_label, cell_name, cl_id, symbols)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = pool.map(_fetch, cell_map.items())
        if progress:
            results = tqdm(
                results, total=len(cell_map), desc=f"markers:{tissue_label}", leave=False
            )
        frames = [df for df in results if df is not None]

    table = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=_celltype.NORMALIZED_COLUMNS)
    )
    table.to_parquet(cache_path, index=False)
    return table


def get_all_markers(
    *,
    organism: str = DEFAULT_ORGANISM,
    tissues: list[str] | None = None,
    n_top_per_type: int = 25,
    test: str = DEFAULT_TEST,
    max_workers: int = 8,
    cache_dir: str | Path | None = None,
    force: bool = False,
    progress: bool = True,
) -> pd.DataFrame:
    """Fetch markers for **every cell type in every tissue** for one organism.

    The corpus-wide dump: enumerates all tissues (:func:`list_tissues`) and, for
    each, every cell type WMG serves — which already spans multiple levels of
    the Cell Ontology, since CZI rolls each cell up its CL lineage. Delegates to
    :func:`get_tissue_markers` per tissue, so the per-tissue parquet cache makes
    the whole run resumable: an interrupted dump picks up where it left off.

    Run this **once per organism** — human and mouse are separate corpora (and
    the cache keys already namespace by organism id, so they never mix).

    Parameters
    ----------
    organism
        Organism label (``"Homo sapiens"``) or ``NCBITaxon:`` id.
    tissues
        Optional explicit tissue subset (labels or ``UBERON:`` ids). Defaults to
        every tissue the organism has.
    n_top_per_type, test, max_workers, cache_dir, force
        Forwarded to :func:`get_tissue_markers`.
    progress
        Show a tqdm bar over tissues.

    Returns
    -------
    pandas.DataFrame
        The full *(species, tissue, cell type, CL id, gene, score, rank)* table
        for the organism.
    """
    tissue_list = tissues if tissues is not None else list_tissues(organism=organism)
    iterator = tqdm(tissue_list, desc=f"tissues:{organism}") if progress else tissue_list
    frames: list[pd.DataFrame] = []
    for tissue in iterator:
        frames.append(
            get_tissue_markers(
                tissue,
                organism=organism,
                n_top_per_type=n_top_per_type,
                test=test,
                max_workers=max_workers,
                cache_dir=cache_dir,
                force=force,
            )
        )
    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=_celltype.NORMALIZED_COLUMNS)
    )


# ─── Differential expression — disease vs normal, condition contrasts ────────

DE_COLUMNS = [
    "species",
    "tissue",
    "cell_type_name",
    "cell_ontology_id",
    "disease",
    "gene_symbol",
    "gene_id",
    "effect_size",
    "log_fold_change",
    "adjusted_p_value",
    "rank",
    "source",
]
"""Schema for :func:`disease_vs_normal` / :func:`differential_expression`.

A *differential* product (not the Marker Score): ``effect_size`` and
``log_fold_change`` describe the group-1-vs-group-2 contrast, ``rank`` is by
descending ``effect_size``."""


def differential_expression(
    group1_filters: dict[str, Any],
    group2_filters: dict[str, Any],
    *,
    exclude_overlapping_cells: str = "retainBoth",
    timeout: int = 300,
) -> pd.DataFrame:
    """Genome-wide differential expression between two cell populations.

    Thin client over CZI's served Differential Expression API — the same engine
    behind the Discover UI's "Differential Expression" tool. Each group is
    defined by a WMG-style filter dict (``organism_ontology_term_id`` plus any of
    ``tissue_ontology_term_ids`` / ``cell_type_ontology_term_ids`` /
    ``disease_ontology_term_ids`` / ``sex_ontology_term_ids`` /
    ``development_stage_ontology_term_ids`` /
    ``self_reported_ethnicity_ontology_term_ids`` / ``dataset_ids``).

    Parameters
    ----------
    group1_filters, group2_filters
        The two populations to contrast (group 1 vs group 2).
    exclude_overlapping_cells
        How to handle cells matching both groups: ``"retainBoth"`` (default),
        ``"excludeOne"``, or ``"excludeTwo"``.
    timeout
        Per-request timeout (seconds).

    Returns
    -------
    pandas.DataFrame
        Columns ``gene_symbol``, ``gene_id``, ``effect_size``,
        ``log_fold_change``, ``adjusted_p_value``, sorted by descending
        ``effect_size``.
    """
    payload = {
        "exclude_overlapping_cells": exclude_overlapping_cells,
        "queryGroup1Filters": group1_filters,
        "queryGroup2Filters": group2_filters,
    }
    results = _post("differentialExpression", payload, base=DE_API_BASE_URL, timeout=timeout)
    rows = [
        {
            "gene_symbol": r.get("gene_symbol"),
            "gene_id": r.get("gene_ontology_term_id"),
            "effect_size": r.get("effect_size"),
            "log_fold_change": r.get("log_fold_change"),
            "adjusted_p_value": r.get("adjusted_p_value"),
        }
        for r in results.get("differentialExpressionResults", [])
    ]
    df = pd.DataFrame(
        rows,
        columns=["gene_symbol", "gene_id", "effect_size", "log_fold_change", "adjusted_p_value"],
    )
    return df.sort_values("effect_size", ascending=False, na_position="last").reset_index(drop=True)


def disease_vs_normal(
    cell_type: str,
    *,
    tissue: str,
    disease: str,
    organism: str = DEFAULT_ORGANISM,
    exclude_overlapping_cells: str = "retainBoth",
    n_top: int | None = None,
) -> pd.DataFrame:
    """Disease-vs-normal differential expression for one cell type in a tissue.

    Contrasts the same cell type between a disease condition and ``normal`` via
    :func:`differential_expression`, so the result is CZI's *served* effect
    size — no local recomputation. Complements :func:`query_markers` (which is
    healthy-only cell-type specificity): this is a *within-cell-type,
    across-condition* contrast.

    Parameters
    ----------
    cell_type
        CL id or plain name (resolved via OLS).
    tissue
        Tissue label or ``UBERON:`` id.
    disease
        A disease id (``"MONDO:0015925"``) or label (``"interstitial lung
        disease"``) available in the tissue (see :func:`list_diseases`).
    organism
        Organism label or ``NCBITaxon:`` id.
    exclude_overlapping_cells
        Forwarded to :func:`differential_expression`.
    n_top
        If given, keep only the top-``n_top`` genes by effect size.

    Returns
    -------
    pandas.DataFrame
        Columns per :data:`DE_COLUMNS` (positive ``effect_size`` = up in
        disease).
    """
    organism_id, organism_label = _resolve_organism(organism)
    tissue_id, tissue_label = _resolve_tissue(tissue, organism_id)
    cl_id = _resolve_cl_id(cell_type)

    diseases = _flatten_terms(_filter_dims(tissue, organism)["disease_terms"])
    disease_id = disease if disease in diseases else None
    if disease_id is None:
        for did, label in diseases.items():
            if label.lower() == disease.lower():
                disease_id = did
                break
    if disease_id is None:
        raise ValueError(f"Disease {disease!r} not found in {tissue_label}. See list_diseases().")
    disease_label = diseases.get(disease_id, disease_id)

    base = {
        "organism_ontology_term_id": organism_id,
        "tissue_ontology_term_ids": [tissue_id],
        "cell_type_ontology_term_ids": [cl_id],
    }
    de = differential_expression(
        {**base, "disease_ontology_term_ids": [disease_id]},
        {**base, "disease_ontology_term_ids": [NORMAL_DISEASE_ID]},
        exclude_overlapping_cells=exclude_overlapping_cells,
    )
    cell_name = _cell_type_map(tissue, organism).get(cl_id)
    de.insert(0, "species", organism_label)
    de.insert(1, "tissue", tissue_label)
    de.insert(2, "cell_type_name", cell_name)
    de.insert(3, "cell_ontology_id", cl_id)
    de.insert(4, "disease", disease_label)
    de["rank"] = _celltype.rank_within_group(
        de, group_cols=["cell_ontology_id"], score_col="effect_size"
    )
    de["source"] = SOURCE_NAME
    if n_top is not None:
        de = de.head(n_top)
    return de[DE_COLUMNS].reset_index(drop=True)


# ─── CellGuide — precomputed computational + canonical markers (CDN) ─────────


@lru_cache(maxsize=1)
def _cellguide_snapshot() -> str:
    """Latest CellGuide snapshot id (its data is versioned by this token)."""
    resp = requests.get(
        f"{CELLGUIDE_CDN_BASE}/latest_snapshot_identifier",
        headers={"User-Agent": _USER_AGENT},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.text.strip()


def _cellguide_get(path: str, *, timeout: int = 120) -> Any:
    """GET a CellGuide snapshot JSON file (raises on 404 for missing cell types)."""
    snapshot = _cellguide_snapshot()
    resp = requests.get(
        f"{CELLGUIDE_CDN_BASE}/{snapshot}/{path}",
        headers={"User-Agent": _USER_AGENT},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def _cellguide_records(path: str) -> list[dict] | None:
    """GET a per-cell-type CellGuide file, or ``None`` if it's absent/unparseable.

    CellGuide serves a file per cell type; a cell type without markers of a given
    kind returns 404, and the CDN occasionally serves an empty/invalid body — both
    mean "no data" here rather than a hard error, so the bulk build can skip them.
    """
    try:
        return _cellguide_get(path)
    except requests.exceptions.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return None
        raise
    except requests.exceptions.JSONDecodeError:
        logger.warning("CellGuide file %s returned a non-JSON body; skipping.", path)
        return None


@lru_cache(maxsize=1)
def list_cellguide_cell_types() -> pd.DataFrame:
    """List every cell type in CellGuide's metadata (CL id + name).

    Returns
    -------
    pandas.DataFrame
        Columns ``cell_ontology_id``, ``cell_type_name``. Not every entry has
        marker files — :func:`cellguide_markers` returns empty for those.
    """
    meta = _cellguide_get("celltype_metadata.json")
    rows = [{"cell_ontology_id": cid, "cell_type_name": v.get("name")} for cid, v in meta.items()]
    return pd.DataFrame(rows, columns=["cell_ontology_id", "cell_type_name"])


def cellguide_markers(cell_type: str, *, kind: str = "both") -> pd.DataFrame:
    """Precomputed CellGuide markers for a cell type (all tissues / species).

    Fetches CZI's per-cell-type CellGuide snapshot files — the source behind the
    Discover "Find Marker Genes" / CellGuide UI. Unlike :func:`query_markers`
    (one tissue, live WMG), one call returns **every tissue and organism** for
    that cell type, plus the curated canonical markers.

    Parameters
    ----------
    cell_type
        CL id or plain name (resolved via OLS).
    kind
        ``"computational"`` (Marker Score, per species × tissue),
        ``"canonical"`` (curated, cross-species), or ``"both"`` (default).

    Returns
    -------
    pandas.DataFrame
        Columns per :data:`CELLGUIDE_COLUMNS`.
    """
    cl_id = _resolve_cl_id(cell_type)
    tag = cl_id.replace(":", "_")
    name = _cellguide_name(cl_id)
    frames: list[pd.DataFrame] = []
    if kind in ("computational", "both"):
        frames.append(_cellguide_computational(tag, cl_id, name))
    if kind in ("canonical", "both"):
        frames.append(_cellguide_canonical(tag, cl_id, name))
    if kind not in ("computational", "canonical", "both"):
        raise ValueError(f"kind must be computational/canonical/both, got {kind!r}")
    out = (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=CELLGUIDE_COLUMNS)
    )
    return out.reset_index(drop=True)


def _cellguide_name(cl_id: str) -> str | None:
    meta = list_cellguide_cell_types()
    hit = meta[meta["cell_ontology_id"] == cl_id]
    return None if hit.empty else hit.iloc[0]["cell_type_name"]


def _cellguide_computational(tag: str, cl_id: str, name: str | None) -> pd.DataFrame:
    records = _cellguide_records(f"computational_marker_genes/{tag}.json")
    if not records:
        return pd.DataFrame(columns=CELLGUIDE_COLUMNS)
    rows = [
        {
            "species": r.get("groupby_dims", {}).get("organism_ontology_term_label"),
            "tissue": r.get("groupby_dims", {}).get("tissue_ontology_term_label") or "All Tissues",
            "cell_type_name": name,
            "cell_ontology_id": cl_id,
            "gene_symbol": r.get("symbol"),
            "gene_id": r.get("gene_ontology_term_id"),
            "marker_type": "computational",
            "marker_score": r.get("marker_score"),
            "specificity": r.get("specificity"),
            "mean_expression": r.get("me"),
            "pct_expressing": r.get("pc"),
            "publication": None,
            "source": SOURCE_NAME,
        }
        for r in records
    ]
    df = pd.DataFrame(rows, columns=[c for c in CELLGUIDE_COLUMNS if c != "rank"])
    if df.empty:
        df["rank"] = pd.Series(dtype="Int64")
        return df[CELLGUIDE_COLUMNS]
    df["rank"] = _celltype.rank_within_group(
        df, group_cols=["species", "tissue", "cell_ontology_id"], score_col="marker_score"
    )
    return df[CELLGUIDE_COLUMNS]


def _cellguide_canonical(tag: str, cl_id: str, name: str | None) -> pd.DataFrame:
    records = _cellguide_records(f"canonical_marker_genes/{tag}.json")
    if not records:
        return pd.DataFrame(columns=CELLGUIDE_COLUMNS)
    rows = [
        {
            "species": None,  # canonical markers are curated + cross-species
            "tissue": r.get("tissue"),
            "cell_type_name": name,
            "cell_ontology_id": cl_id,
            "gene_symbol": r.get("symbol"),
            "gene_id": None,
            "marker_type": "canonical",
            "marker_score": None,
            "specificity": None,
            "mean_expression": None,
            "pct_expressing": None,
            "publication": r.get("publication") or None,
            "rank": None,
            "source": SOURCE_NAME,
        }
        for r in records
    ]
    return pd.DataFrame(rows, columns=CELLGUIDE_COLUMNS)


def get_all_cellguide_markers(
    *,
    kind: str = "both",
    max_workers: int = 16,
    cache_dir: str | Path | None = None,
    force: bool = False,
    progress: bool = True,
) -> pd.DataFrame:
    """Fetch CellGuide markers for **every** cell type (all tissues + species).

    Iterates :func:`list_cellguide_cell_types` and pulls each cell type's
    CellGuide files concurrently. The assembled table is cached to parquet
    (keyed by snapshot + kind) so re-runs are instant.

    Parameters
    ----------
    kind
        ``"computational"``, ``"canonical"``, or ``"both"``.
    max_workers
        Concurrent CDN fetches.
    cache_dir, force
        Cache location override / bypass.
    progress
        Show a tqdm bar over cell types.

    Returns
    -------
    pandas.DataFrame
        Columns per :data:`CELLGUIDE_COLUMNS`.
    """
    snapshot = _cellguide_snapshot()
    root = Path(cache_dir).expanduser() if cache_dir else CACHE_DIR / "cellguide"
    root.mkdir(parents=True, exist_ok=True)
    cache_path = root / f"cellguide_markers__{snapshot}__{kind}.parquet"
    if cache_path.exists() and not force:
        return pd.read_parquet(cache_path)

    cl_ids = list_cellguide_cell_types()["cell_ontology_id"].tolist()

    def _fetch(cl_id: str) -> pd.DataFrame:
        try:
            return cellguide_markers(cl_id, kind=kind)
        except Exception as exc:  # noqa: BLE001 - one bad cell type shouldn't abort the dump
            logger.warning("CellGuide markers skip %s: %s", cl_id, exc)
            return pd.DataFrame(columns=CELLGUIDE_COLUMNS)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = pool.map(_fetch, cl_ids)
        if progress:
            results = tqdm(results, total=len(cl_ids), desc=f"cellguide:{kind}")
        frames = [df for df in results if not df.empty]

    table = (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=CELLGUIDE_COLUMNS)
    )
    table.to_parquet(cache_path, index=False)
    return table


def to_gmt(
    path: str | Path,
    *,
    tissue: str,
    organism: str = DEFAULT_ORGANISM,
    by: str = "cell_ontology_id",
    cache_dir: str | Path | None = None,
) -> Path:
    """Export a tissue's cell-type marker gene sets to a GMT file.

    Parameters
    ----------
    path
        Destination ``.gmt`` path.
    tissue, organism, cache_dir
        Forwarded to :func:`get_tissue_markers`.
    by
        Column used as the GMT set id.

    Returns
    -------
    pathlib.Path
        The written path.
    """
    table = get_tissue_markers(tissue, organism=organism, cache_dir=cache_dir)
    return _celltype.celltype_to_gmt(table, path, by=by)


__all__ = [
    "WMG_API_BASE_URL",
    "DE_API_BASE_URL",
    "CELLGUIDE_CDN_BASE",
    "DEFAULT_ORGANISM",
    "DEFAULT_TEST",
    "NORMAL_DISEASE_ID",
    "SOURCE_NAME",
    "CACHE_DIR",
    "DE_COLUMNS",
    "CELLGUIDE_COLUMNS",
    "list_tissues",
    "list_cell_types",
    "list_diseases",
    "query_markers",
    "get_tissue_markers",
    "get_all_markers",
    "differential_expression",
    "disease_vs_normal",
    "list_cellguide_cell_types",
    "cellguide_markers",
    "get_all_cellguide_markers",
    "to_gmt",
]
