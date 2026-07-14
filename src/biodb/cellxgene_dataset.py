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

Layout written by :func:`build_dataset`::

    <out_dir>/
      README.md                                  # dataset card (with configs)
      markers/computational_homo_sapiens.parquet
      markers/computational_mus_musculus.parquet
      markers/canonical.parquet
      disease_deg/homo_sapiens.parquet           # optional (long build)
      disease_deg/mus_musculus.parquet

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


def build_disease_deg(
    organism: str,
    *,
    tissues: list[str] | None = None,
    max_workers: int = 8,
    cache_dir: str | Path | None = None,
    force: bool = False,
    progress: bool = True,
) -> pd.DataFrame:
    """Build disease-vs-normal DEG vectors for every (disease × tissue × cell type).

    For each tissue, enumerates its diseases (:func:`biodb.cellxgene.list_diseases`),
    the cell types present under each disease, and calls
    :func:`biodb.cellxgene.disease_vs_normal` for each. Results are cached to a
    per-(tissue, disease) parquet under ``cache_dir`` so the (long) build is
    **resumable** — an interrupted run continues where it stopped.

    Parameters
    ----------
    organism
        ``"Homo sapiens"`` or ``"Mus musculus"``.
    tissues
        Optional tissue subset (labels or ``UBERON:`` ids); defaults to all.
    max_workers
        Concurrent DE requests per (tissue, disease).
    cache_dir, force
        Per-(tissue, disease) parquet cache location / bypass.
    progress
        Show a tqdm bar over (tissue, disease) pairs.

    Returns
    -------
    pandas.DataFrame
        All DEG rows (columns per :data:`biodb.cellxgene.DE_COLUMNS`, minus the
        single-cell-type ``rank``; a global ``rank`` is not meaningful here).
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
    frames: list[pd.DataFrame] = []
    for tissue, disease_id, disease_label in iterator:
        cache_path = root / f"{organism_id}__{_slug(tissue)}__{disease_id}.parquet".replace(
            ":", "-"
        )
        if cache_path.exists() and not force:
            frames.append(pd.read_parquet(cache_path))
            continue
        tissue_id, _ = _resolve_tissue(tissue, organism_id)
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
        if not pair_df.empty:
            frames.append(pair_df)

    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=cellxgene.DE_COLUMNS)
    )


# ─── Dataset assembly + push ─────────────────────────────────────────────────


def build_dataset(
    out_dir: str | Path,
    *,
    organisms: list[str] | None = None,
    include_deg: bool = False,
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
        Also build + write the disease-DEG tables (the long step). Default
        False — markers only, per the "markers now, DEG follow-up" plan.
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

    markers = build_markers(cache_dir=cache_dir, force=force)
    markers["canonical"].to_parquet(out / "markers" / "canonical.parquet", index=False)
    written = {"canonical": len(markers["canonical"])}
    for species in organisms:
        if species in markers:
            path = out / "markers" / f"computational_{_slug(species)}.parquet"
            markers[species].to_parquet(path, index=False)
            written[species] = len(markers[species])

    deg_written: dict[str, int] = {}
    if include_deg:
        (out / "disease_deg").mkdir(parents=True, exist_ok=True)
        for species in organisms:
            deg = build_disease_deg(species, cache_dir=cache_dir, force=force)
            deg.to_parquet(out / "disease_deg" / f"{_slug(species)}.parquet", index=False)
            deg_written[species] = len(deg)

    (out / "README.md").write_text(
        _dataset_card(organisms, written, deg_written, include_deg), encoding="utf-8"
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
    if include_deg:
        for species in organisms:
            if species in deg_written:
                configs.append(
                    f"  - config_name: disease_deg_{_slug(species)}\n"
                    f"    data_files: disease_deg/{_slug(species)}.parquet"
                )
    counts = "\n".join(
        f"- **{k}**: {v:,} rows" for k, v in {**markers_written, **deg_written}.items()
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

## Contents

### Marker genes (`markers/`)
Per **species** computational markers + a shared **canonical** table, from the
CellGuide snapshot.

- `computational_<species>.parquet` — CZI's **Marker Score** (effect size) per
  gene × cell type × tissue, with `specificity`, `mean_expression`,
  `pct_expressing`, `rank`. One file per species (human, mouse).
- `canonical.parquet` — curated marker genes (literature / ASCT+B), cross-species
  (so `species` is null), with `publication` references where available.

### Disease DEG (`disease_deg/`){"" if include_deg else " — *not yet built (follow-up run)*"}
Disease-vs-normal differential expression per *(disease × tissue × cell type)*,
per species, from CELLxGENE's Differential Expression API: `effect_size`,
`log_fold_change`, `adjusted_p_value` (positive effect = up in disease).

## Row counts

{counts}

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
