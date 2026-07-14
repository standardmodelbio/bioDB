"""Build the ``biodb_cellxgene`` HuggingFace dataset from CZ CELLxGENE.

Reproducible assembly of a cell-type knowledge dataset entirely from CZI's
served CELLxGENE data (no local recomputation), built on :mod:`biodb.cellxgene`:

1. **Markers** — computational (WMG Marker Score) + canonical (curated) marker
   genes for every cell type × tissue, split **per species** (human, mouse).
   Canonical markers are curated + cross-species, so they live in one shared
   table. Source: CellGuide snapshot (:func:`biodb.cellxgene.get_all_cellguide_markers`).
2. **Disease DEG** — disease-vs-normal differential-expression vectors for each
   *(disease × tissue × cell type)*, per species. Source: CELLxGENE's served
   Differential Expression API (:func:`biodb.cellxgene.disease_vs_normal`).

Layout written by :func:`build_dataset` (species encoded in file/dir names, so
no ``species`` column; no ``source``/``marker_type`` columns either)::

    <out_dir>/
      README.md                                    # dataset card (with configs)
      markers/computational_homo_sapiens.parquet
      markers/computational_mus_musculus.parquet
      markers/canonical.parquet                    # cross-species (no species)
      disease_deg/homo_sapiens/part_*.parquet      # optional; sharded per pair
      disease_deg/mus_musculus/part_*.parquet

:func:`push_to_hub` uploads the directory to the Hub (requires ``huggingface_hub``).

Examples
--------
>>> from biodb import cellxgene_dataset as cxd
>>> cxd.build_dataset("~/biodb_cellxgene", include_deg=False)          # doctest: +SKIP
>>> cxd.push_to_hub("~/biodb_cellxgene",                               # doctest: +SKIP
...                 "standardmodelbio/biodb_cellxgene", private=True)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from biodb import cellxgene
from biodb.cellxgene import _flatten_terms, _post, _resolve_organism, _resolve_tissue

logger = logging.getLogger(__name__)

DATASET_ORGANISMS = ["Homo sapiens", "Mus musculus"]
"""Species with CELLxGENE computational markers + disease DEG (WMG has no others)."""


def _slug(text: str) -> str:
    return text.strip().lower().replace(" ", "_")


# ─── Markers (CellGuide: computational + canonical) ──────────────────────────


def build_markers(
    *,
    cache_dir: str | Path | None = None,
    force: bool = False,
    progress: bool = True,
) -> dict[str, pd.DataFrame]:
    """Assemble the marker tables, split per species + a shared canonical table.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Keys: each organism label (computational markers for that species) plus
        ``"canonical"`` (curated, cross-species). Columns per
        :data:`biodb.cellxgene.CELLGUIDE_COLUMNS`.
    """
    allm = cellxgene.get_all_cellguide_markers(
        kind="both", cache_dir=cache_dir, force=force, progress=progress
    )
    comp = allm[allm["marker_type"] == "computational"]
    canonical = allm[allm["marker_type"] == "canonical"].reset_index(drop=True)
    out: dict[str, pd.DataFrame] = {"canonical": canonical}
    for species in sorted(comp["species"].dropna().unique()):
        out[species] = comp[comp["species"] == species].reset_index(drop=True)
    return out


# ─── Disease DEG (served Differential Expression) ────────────────────────────


def _cell_types_under_disease(tissue_id: str, disease_id: str, organism_id: str) -> dict[str, str]:
    """CL-id → label map for cell types present in a (tissue, disease)."""
    payload = {
        "filter": {
            "organism_ontology_term_id": organism_id,
            "tissue_ontology_term_ids": [tissue_id],
            "disease_ontology_term_ids": [disease_id],
        }
    }
    dims = _post("filters", payload)["filter_dims"]
    return _flatten_terms(dims["cell_type_terms"])


def _iter_disease_deg(
    organism: str,
    *,
    tissues: list[str] | None = None,
    max_workers: int = 8,
    cache_dir: str | Path | None = None,
    force: bool = False,
    progress: bool = True,
):
    """Yield one DEG DataFrame per (tissue × disease) pair — memory-safe.

    Computes + caches each pair's genome-wide DE to its own parquet (resumable),
    yielding one pair at a time so the full (multi-GB) corpus is never held in
    memory at once. Backfills ``disease_ontology_id`` / ``tissue_ontology_id``
    into parquets cached before those columns existed.
    """
    from concurrent.futures import ThreadPoolExecutor

    from tqdm import tqdm

    organism_id, _ = _resolve_organism(organism)
    root = Path(cache_dir).expanduser() if cache_dir else cellxgene.CACHE_DIR / "disease_deg"
    root.mkdir(parents=True, exist_ok=True)

    tissue_list = tissues if tissues is not None else cellxgene.list_tissues(organism=organism)
    pairs: list[tuple[str, str, str]] = []
    for tissue in tissue_list:
        for _, row in cellxgene.list_diseases(tissue, organism=organism).iterrows():
            pairs.append((tissue, row["disease_ontology_term_id"], row["disease"]))

    iterator = tqdm(pairs, desc=f"deg:{organism}") if progress else pairs
    for tissue, disease_id, disease_label in iterator:
        cache_path = root / f"{organism_id}__{_slug(tissue)}__{disease_id}.parquet".replace(
            ":", "-"
        )
        tissue_id, _ = _resolve_tissue(tissue, organism_id)
        if cache_path.exists() and not force:
            cached = pd.read_parquet(cache_path)
            # Backfill ontology-id columns for parquets cached before they existed.
            if not cached.empty and "tissue_ontology_id" not in cached.columns:
                cached.insert(2, "tissue_ontology_id", tissue_id)
            if not cached.empty and "disease_ontology_id" not in cached.columns:
                pos = cached.columns.get_loc("disease") + 1 if "disease" in cached.columns else 6
                cached.insert(pos, "disease_ontology_id", disease_id)
            yield cached
            continue
        cell_types = _cell_types_under_disease(tissue_id, disease_id, organism_id)

        def _fetch(cl_id: str, *, _t=tissue, _d=disease_label) -> pd.DataFrame | None:
            try:
                return cellxgene.disease_vs_normal(cl_id, tissue=_t, disease=_d, organism=organism)
            except Exception as exc:  # noqa: BLE001 - skip pairs the DE API can't compute
                logger.warning("DEG skip %s / %s / %s: %s", _t, _d, cl_id, exc)
                return None

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            got = [df for df in pool.map(_fetch, cell_types) if df is not None and not df.empty]
        pair_df = (
            pd.concat(got, ignore_index=True) if got else pd.DataFrame(columns=cellxgene.DE_COLUMNS)
        )
        pair_df.to_parquet(cache_path, index=False)
        yield pair_df


def build_disease_deg(organism: str, **kwargs) -> pd.DataFrame:
    """Concatenate all (tissue × disease) DEG for an organism into one frame.

    Thin wrapper over :func:`_iter_disease_deg`. **Memory-heavy** for the full
    genome-wide corpus (many GB); the dataset builder writes sharded parquet via
    the iterator instead. Prefer :func:`_iter_disease_deg` for large runs.
    """
    frames = [df for df in _iter_disease_deg(organism, **kwargs) if not df.empty]
    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=cellxgene.DE_COLUMNS)
    )


def _mondo_graph():
    """Load the MONDO ontology graph (child→parent edges) via obonet."""
    import obonet

    return obonet.read_obo("http://purl.obolibrary.org/obo/mondo/mondo-base.obo")


def _iter_disease_deg_rollup(
    organism: str,
    *,
    tissues: list[str] | None = None,
    min_pool: int = 2,
    max_workers: int = 8,
    cache_dir: str | Path | None = None,
    force: bool = False,
    progress: bool = True,
):
    """Yield **MONDO-category** DEG frames — descendant diseases pooled vs normal.

    For each tissue, walks the MONDO ancestors of the diseases present and, for
    every ancestor node that pools ``>= min_pool`` present diseases, contrasts
    the union of those diseases against ``normal`` (per cell type). This creates
    better-powered *disease-category* vectors (e.g. all carcinoma subtypes vs
    normal) — computed entirely via the served DE API (grouped disease filters),
    no recomputation. One parquet per (tissue × MONDO node), resumable.
    """
    from concurrent.futures import ThreadPoolExecutor

    import networkx as nx
    from tqdm import tqdm

    organism_id, organism_label = _resolve_organism(organism)
    root = Path(cache_dir).expanduser() if cache_dir else cellxgene.CACHE_DIR / "disease_deg_rollup"
    root.mkdir(parents=True, exist_ok=True)
    graph = _mondo_graph()

    tissue_list = tissues if tissues is not None else cellxgene.list_tissues(organism=organism)
    tasks: list[tuple[str, str, str, list[str]]] = []  # (tissue, node, node_label, pooled ids)
    for tissue in tissue_list:
        present = {
            row["disease_ontology_term_id"]: row["disease"]
            for _, row in cellxgene.list_diseases(tissue, organism=organism).iterrows()
        }
        node_pool: dict[str, set[str]] = {}
        for d in present:
            if d not in graph:
                continue
            for node in {d, *nx.descendants(graph, d)}:  # d + its MONDO ancestors
                node_pool.setdefault(node, set()).add(d)
        for node, pooled in node_pool.items():
            if len(pooled) < min_pool:
                continue  # only genuine multi-disease aggregations
            label = graph.nodes.get(node, {}).get("name", node)
            tasks.append((tissue, node, label, sorted(pooled)))

    iterator = tqdm(tasks, desc=f"deg-rollup:{organism}") if progress else tasks
    for tissue, node, node_label, pooled in iterator:
        cache_path = root / f"{organism_id}__{_slug(tissue)}__{node}.parquet".replace(":", "-")
        tissue_id, tissue_label = _resolve_tissue(tissue, organism_id)
        if cache_path.exists() and not force:
            yield pd.read_parquet(cache_path)
            continue
        cell_map: dict[str, str] = {}
        for d in pooled:
            cell_map.update(_cell_types_under_disease(tissue_id, d, organism_id))

        def _fetch(
            item: tuple[str, str],
            *,
            _p=pooled,
            _tid=tissue_id,
            _tl=tissue_label,
            _node=node,
            _nl=node_label,
        ) -> pd.DataFrame | None:
            cl_id, cell_name = item
            base = {
                "organism_ontology_term_id": organism_id,
                "tissue_ontology_term_ids": [_tid],
                "cell_type_ontology_term_ids": [cl_id],
            }
            try:
                de = cellxgene.differential_expression(
                    {**base, "disease_ontology_term_ids": _p},
                    {**base, "disease_ontology_term_ids": [cellxgene.NORMAL_DISEASE_ID]},
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("rollup skip %s/%s/%s: %s", _tl, _node, cl_id, exc)
                return None
            if de.empty:
                return None
            de.insert(0, "species", organism_label)
            de.insert(1, "tissue", _tl)
            de.insert(2, "tissue_ontology_id", _tid)
            de.insert(3, "cell_type_name", cell_name)
            de.insert(4, "cell_ontology_id", cl_id)
            de.insert(5, "disease", _nl)
            de.insert(6, "disease_ontology_id", _node)
            de["rank"] = de["effect_size"].rank(ascending=False, method="dense").astype("Int64")
            de["source"] = cellxgene.SOURCE_NAME
            return de.reindex(columns=cellxgene.DE_COLUMNS)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            got = [d for d in pool.map(_fetch, cell_map.items()) if d is not None and not d.empty]
        pair_df = (
            pd.concat(got, ignore_index=True) if got else pd.DataFrame(columns=cellxgene.DE_COLUMNS)
        )
        pair_df.to_parquet(cache_path, index=False)
        yield pair_df


# ─── Dataset assembly + push ─────────────────────────────────────────────────


def build_dataset(
    out_dir: str | Path,
    *,
    organisms: list[str] | None = None,
    include_deg: bool = False,
    include_deg_rollup: bool = False,
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Write the ``biodb_cellxgene`` dataset (parquet files + card) to ``out_dir``.

    Parameters
    ----------
    out_dir
        Destination directory (created if missing).
    organisms
        Species to include (default :data:`DATASET_ORGANISMS`).
    include_deg
        Also build + write the leaf disease-DEG tables (the long step).
    include_deg_rollup
        Also build + write the MONDO **disease-category** DEG (descendant
        diseases pooled vs normal) under ``disease_deg_rollup/<species>/``.
    cache_dir, force
        Forwarded to the builders.

    Returns
    -------
    pathlib.Path
        ``out_dir``.
    """
    organisms = organisms or DATASET_ORGANISMS
    out = Path(out_dir).expanduser()
    (out / "markers").mkdir(parents=True, exist_ok=True)

    # ``species`` (encoded in the file name), ``source`` (always "cellxgene"), and
    # ``marker_type`` (implied by the file) are dropped — see the dataset card.
    drop = ["species", "source", "marker_type"]
    markers = build_markers(cache_dir=cache_dir, force=force)
    canonical = markers["canonical"].dropna(axis=1, how="all").drop(columns=drop, errors="ignore")
    canonical.to_parquet(out / "markers" / "canonical.parquet", index=False)
    written = {"canonical": len(canonical)}
    for species in organisms:
        if species in markers:
            # per-species file (species inferred from the name); also drop the
            # all-null `publication` column (a canonical-only field).
            comp = markers[species].drop(columns=drop, errors="ignore").dropna(axis=1, how="all")
            comp.to_parquet(
                out / "markers" / f"computational_{_slug(species)}.parquet", index=False
            )
            written[species] = len(comp)

    deg_written: dict[str, int] = {}
    if include_deg:
        for species in organisms:
            sp_dir = out / "disease_deg" / _slug(species)
            sp_dir.mkdir(parents=True, exist_ok=True)
            n_shards = n_rows = 0
            for shard in _iter_disease_deg(species, cache_dir=cache_dir, force=force):
                if shard.empty:
                    continue
                shard = shard.reindex(columns=cellxgene.DE_COLUMNS).drop(
                    columns=drop, errors="ignore"
                )
                shard.to_parquet(sp_dir / f"part_{n_shards:04d}.parquet", index=False)
                n_shards += 1
                n_rows += len(shard)
            deg_written[species] = n_rows

    rollup_written: dict[str, int] = {}
    if include_deg_rollup:
        for species in organisms:
            sp_dir = out / "disease_deg_rollup" / _slug(species)
            sp_dir.mkdir(parents=True, exist_ok=True)
            n_shards = n_rows = 0
            for shard in _iter_disease_deg_rollup(species, cache_dir=cache_dir, force=force):
                if shard.empty:
                    continue
                shard = shard.reindex(columns=cellxgene.DE_COLUMNS).drop(
                    columns=drop, errors="ignore"
                )
                shard.to_parquet(sp_dir / f"part_{n_shards:04d}.parquet", index=False)
                n_shards += 1
                n_rows += len(shard)
            rollup_written[species] = n_rows

    (out / "README.md").write_text(
        _dataset_card(organisms, written, deg_written, rollup_written, include_deg),
        encoding="utf-8",
    )
    logger.info("Built biodb_cellxgene dataset at %s (%s)", out, written)
    return out


def push_to_hub(
    local_dir: str | Path,
    repo_id: str,
    *,
    private: bool = True,
    token: str | None = None,
) -> str:
    """Upload a built dataset directory to the HuggingFace Hub.

    Requires ``huggingface_hub`` and a valid token (cached login or ``token``).

    Parameters
    ----------
    local_dir
        Directory produced by :func:`build_dataset`.
    repo_id
        e.g. ``"standardmodelbio/biodb_cellxgene"``.
    private
        Create/keep the dataset repo private.
    token
        HF token; falls back to the cached login.

    Returns
    -------
    str
        The dataset repo URL.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(Path(local_dir).expanduser()), repo_id=repo_id, repo_type="dataset"
    )
    return f"https://huggingface.co/datasets/{repo_id}"


def _dataset_card(
    organisms: list[str],
    markers_written: dict[str, int],
    deg_written: dict[str, int],
    rollup_written: dict[str, int],
    include_deg: bool,
) -> str:
    """Render the dataset-card README (YAML front matter + docs)."""
    configs = [
        "  - config_name: canonical_markers\n    data_files: markers/canonical.parquet",
    ]
    for species in organisms:
        if species in markers_written:
            configs.append(
                f"  - config_name: computational_markers_{_slug(species)}\n"
                f"    data_files: markers/computational_{_slug(species)}.parquet"
            )
    for species in organisms:
        if species in deg_written:
            configs.append(
                f"  - config_name: disease_deg_{_slug(species)}\n"
                f"    data_files: disease_deg/{_slug(species)}/*.parquet"
            )
    for species in organisms:
        if species in rollup_written:
            configs.append(
                f"  - config_name: disease_deg_rollup_{_slug(species)}\n"
                f"    data_files: disease_deg_rollup/{_slug(species)}/*.parquet"
            )
    counts = "\n".join(
        f"- **{k}**: {v:,} rows"
        for k, v in {
            **markers_written,
            **{f"DEG {k}": v for k, v in deg_written.items()},
            **{f"DEG-rollup {k}": v for k, v in rollup_written.items()},
        }.items()
    )
    return f"""---
license: cc-by-4.0
language:
  - en
tags:
  - biology
  - single-cell
  - cell-ontology
  - marker-genes
  - differential-expression
  - cellxgene
pretty_name: bioDB CELLxGENE cell-type markers & disease DEG
configs:
{chr(10).join(configs)}
---

# biodb_cellxgene

Cell-type knowledge from [CZ CELLxGENE Discover](https://cellxgene.cziscience.com/),
assembled by [`bioDB`](https://github.com/bschilder/bioDB) entirely from CZI's
**served** data (no recomputation), keyed to the **Cell Ontology (CL)** and UBERON.

> **Species is encoded in the file/directory name, not a column.** Every file is
> single-species (e.g. `computational_homo_sapiens.parquet`,
> `disease_deg/mus_musculus/`), so a `species` column would be constant and is
> omitted. Likewise there is no `source` column (all rows are CELLxGENE) or
> `marker_type` column (implied by the file: computational vs canonical).

## Contents

### Marker genes (`markers/`)
Per-**species** computational markers + a shared **canonical** table, from the
CellGuide snapshot.

- `computational_<species>.parquet` — CZI's **Marker Score** (effect size) per
  gene × cell type × tissue, with `specificity`, `mean_expression`,
  `pct_expressing`, `rank`, `tissue`/`tissue_ontology_id`, `cell_type_name`/
  `cell_ontology_id`, `gene_symbol`/`gene_id`. One file per species (human, mouse).
- `canonical.parquet` — curated reference marker genes (HuBMAP ASCT+B + literature),
  organized **by tissue, not organism** (cross-species; CZI attaches no organism):
  `cell_ontology_id`, `cell_type_name`, `tissue`, `tissue_ontology_id`,
  `gene_symbol`, `publication`.

### Disease DEG (`disease_deg/<species>/`){"" if include_deg else " — *not yet built (follow-up run)*"}
Disease-vs-normal differential expression per *(disease × tissue × cell type)*,
from CELLxGENE's Differential Expression API. **Sharded** as one parquet per
(tissue × disease) pair under a per-species directory. Columns: `tissue`/
`tissue_ontology_id`, `cell_type_name`/`cell_ontology_id`, `disease`/
`disease_ontology_id`, `gene_symbol`/`gene_id`, `effect_size`, `log_fold_change`,
`adjusted_p_value`, `rank` (positive effect = up in disease). Genome-wide (all
tested genes per contrast).

### Disease-category DEG (`disease_deg_rollup/<species>/`)
The same disease-vs-normal contrast, but with diseases **rolled up the MONDO
ontology**: for each MONDO node that pools ≥2 present diseases, all descendant
diseases are combined into one case group vs normal (per cell type). This gives
better-powered *category* vectors (e.g. all carcinoma subtypes vs normal). Same
columns; `disease`/`disease_ontology_id` name the MONDO **category** node.
Computed via grouped disease filters on the served DE API — no recomputation.

## Row counts

{counts}

## Species coverage

CZ CELLxGENE's **served** marker/DEG products (WMG Marker Score, CellGuide, the
Differential Expression API) cover **human and mouse only** — this dataset
includes exactly those. The underlying Census (release `2025-11-08`) also
contains three primates — *Macaca mulatta*, *Callithrix jacchus*, *Pan
troglodytes* — but they have **no served Marker Score and no disease DEG**, and
only 1–2 tissues each, so they are intentionally **not** included here (adding
them would require locally computed, lower-coverage markers of a different
provenance). Cross-species curated markers for a longer tail of organisms are
available separately via `biodb.celltaxonomy`.

## Provenance & license

All data derives from CZ CELLxGENE Discover (CellGuide + WMG + DE APIs) under
**CC-BY 4.0**. Built with `biodb.cellxgene_dataset`; reproduce with:

```python
from biodb import cellxgene_dataset as cxd
cxd.build_dataset("biodb_cellxgene", include_deg=True)
```
"""


__all__ = [
    "DATASET_ORGANISMS",
    "build_markers",
    "build_disease_deg",
    "build_dataset",
    "push_to_hub",
]
