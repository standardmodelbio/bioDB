"""Build + publish the curated-source bioDB HuggingFace datasets + collection.

Companion to :mod:`biodb.cellxgene_dataset` (which handles the CELLxGENE dataset).
This module packages the **curated** cell-type marker sources as HF datasets and
groups all of bioDB's cell-type datasets into one Hub **collection**:

* :func:`build_celltaxonomy_dataset` — Cell Taxonomy (CNCB-NGDC), **all species**.
* :func:`build_cellmarker_dataset` — CellMarker 2.0 (human + mouse).
* :func:`push_to_hub` — upload a built directory to the Hub.
* :func:`create_collection` — create/refresh a Hub collection and add datasets.

Each dataset is one ``markers/<species>.parquet`` per species (species encoded
in the file name, so no ``species``/``source`` columns) plus a README card.

Examples
--------
>>> from biodb import hf_datasets as hd
>>> hd.build_celltaxonomy_dataset("~/biodb_celltaxonomy")                # doctest: +SKIP
>>> hd.push_to_hub("~/biodb_celltaxonomy",                              # doctest: +SKIP
...                "standardmodelbio/biodb_celltaxonomy", private=True)
>>> hd.create_collection("bioDB", namespace="standardmodelbio",         # doctest: +SKIP
...     dataset_repo_ids=["standardmodelbio/biodb_cellxgene",
...                       "standardmodelbio/biodb_celltaxonomy",
...                       "standardmodelbio/biodb_cellmarker"])
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from biodb import cellmarker, celltaxonomy

logger = logging.getLogger(__name__)


# ─── shared writers ──────────────────────────────────────────────────────────


def _species_counts_md(df: pd.DataFrame) -> str:
    """Markdown table of per-species row / cell-type / CL / gene counts."""
    grouped = (
        df.groupby("species")
        .agg(
            rows=("gene_symbol", "size"),
            cell_types=("cell_type_name", "nunique"),
            cl_terms=("cell_ontology_id", "nunique"),
            genes=("gene_symbol", "nunique"),
        )
        .sort_values("rows", ascending=False)
    )
    lines = ["| species | marker rows | cell types | CL terms | genes |", "|---|--:|--:|--:|--:|"]
    for species, row in grouped.iterrows():
        lines.append(
            f"| {species} | {int(row['rows']):,} | {int(row['cell_types']):,} | "
            f"{int(row['cl_terms']):,} | {int(row['genes']):,} |"
        )
    return "\n".join(lines)


def _species_slug(species: str) -> str:
    """Filesystem-safe token for a species label (e.g. 'Macaca fascicularis')."""
    return re.sub(r"[^0-9a-z]+", "_", species.lower()).strip("_")


def _write_per_species(out_dir: str | Path, df: pd.DataFrame, *, readme_fn) -> Path:
    """Split ``df`` into one ``markers/<species>.parquet`` per species + README.

    ``species`` and ``source`` columns are dropped (species is in the file name;
    source is the dataset). ``readme_fn(species_files)`` renders the card given
    the ``{species: (slug, n_rows)}`` mapping.
    """
    out = Path(out_dir).expanduser()
    (out / "markers").mkdir(parents=True, exist_ok=True)
    species_files: dict[str, tuple[str, int]] = {}
    for species in sorted(df["species"].dropna().unique()):
        slug = _species_slug(species)
        sub = df[df["species"] == species].drop(columns=["species", "source"], errors="ignore")
        sub.to_parquet(out / "markers" / f"{slug}.parquet", index=False)
        species_files[species] = (slug, len(sub))
    (out / "README.md").write_text(readme_fn(species_files), encoding="utf-8")
    logger.info("Wrote %d per-species files to %s", len(species_files), out)
    return out


# ─── Cell Taxonomy dataset ───────────────────────────────────────────────────


def build_celltaxonomy_dataset(
    out_dir: str | Path,
    *,
    map_to_cl: bool = False,
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Build the ``biodb_celltaxonomy`` dataset (all species) at ``out_dir``.

    Parameters
    ----------
    out_dir
        Destination directory.
    map_to_cl
        Fill any blank CL ids via OLS (:func:`biodb.celltaxonomy.get_markers`).
    cache_dir, force
        Forwarded to :func:`biodb.celltaxonomy.get_markers`.

    Returns
    -------
    pathlib.Path
        ``out_dir``.
    """
    df = celltaxonomy.get_markers(map_to_cl=map_to_cl, cache_dir=cache_dir, force=force)
    return _write_per_species(out_dir, df, readme_fn=lambda sf: _celltaxonomy_card(df, sf))


def _species_configs(species_files: dict[str, tuple[str, int]]) -> str:
    """YAML `configs:` block, one per-species config -> markers/<slug>.parquet."""
    return "\n".join(
        f"  - config_name: {slug}\n    data_files: markers/{slug}.parquet"
        for _species, (slug, _n) in sorted(species_files.items())
    )


def _species_files_md(species_files: dict[str, tuple[str, int]]) -> str:
    lines = ["| species | file | rows |", "|---|---|--:|"]
    for species, (slug, n) in sorted(species_files.items(), key=lambda kv: -kv[1][1]):
        lines.append(f"| {species} | `markers/{slug}.parquet` | {n:,} |")
    return "\n".join(lines)


def _celltaxonomy_card(df: pd.DataFrame, species_files: dict[str, tuple[str, int]]) -> str:
    return f"""---
license: other
license_name: academic-use-only
license_link: https://ngdc.cncb.ac.cn/celltaxonomy/
language:
  - en
tags:
  - biology
  - single-cell
  - cell-ontology
  - marker-genes
  - cell-taxonomy
pretty_name: bioDB Cell Taxonomy cell-type markers (all species)
configs:
{_species_configs(species_files)}
---

# biodb_celltaxonomy

Curated cell-type **marker genes across all species**, from
[Cell Taxonomy](https://ngdc.cncb.ac.cn/celltaxonomy/) (CNCB-NGDC), packaged by
[`bioDB`](https://github.com/bschilder/bioDB) and keyed to the
**Cell Ontology (CL)**.

> **One file per species** — species is encoded in the file name
> (`markers/<species>.parquet`), so there is no `species` column (nor a constant
> `source` column). Load a species with its config name, e.g.
> `load_dataset("standardmodelbio/biodb_celltaxonomy", "homo_sapiens")`.

## What this is
Cell Taxonomy is a manually curated, cross-species catalog of cell types, each
mapped to a native CL term and annotated with literature-supported marker genes.
Each file is the normalized *(tissue, tissue_ontology_id, cell type, CL id, gene,
gene_id, score, rank)* table for one species — **{len(df):,} rows across
{df["species"].nunique()} species**.

## How it was made
`biodb.celltaxonomy.get_markers()` downloads the upstream
`Cell_Taxonomy_resource.txt`, then for each *(species, cell type, gene)*:
- takes the native `Specific_Cell_Ontology_ID` as `cell_ontology_id` and
  `Tissue_UberonOntology_ID` as `tissue_ontology_id`,
- sets `score` = **literature-support count** (distinct supporting PMIDs; falls
  back to record count) — Cell Taxonomy has no continuous per-gene score,
- sets `rank` = dense within-cell-type rank of `score` (1 = best-supported).

## How to use
```python
from datasets import load_dataset
mouse = load_dataset("standardmodelbio/biodb_celltaxonomy", "mus_musculus", split="train")
```
Or straight from `bioDB`: `from biodb import celltaxonomy; celltaxonomy.get_markers(species="Mus musculus")`.
To combine species/sources flexibly: `biodb.load_markers(source="celltaxonomy", species=None)`.

## Files / species coverage

{_species_files_md(species_files)}

## Provenance & license
Data from Cell Taxonomy (CNCB-NGDC), **free for academic use** (see the
[Cell Taxonomy site](https://ngdc.cncb.ac.cn/celltaxonomy/) for terms; not
formally CC-BY). Cite Jiang et al., *Nucleic Acids Research* 2023 (D853–D860).
Reproduce:
```python
from biodb import hf_datasets as hd
hd.build_celltaxonomy_dataset("biodb_celltaxonomy")
```
"""


# ─── CellMarker 2.0 dataset ──────────────────────────────────────────────────


def build_cellmarker_dataset(
    out_dir: str | Path,
    *,
    which: str = "all",
    map_to_cl: bool = False,
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Build the ``biodb_cellmarker`` dataset (human + mouse) at ``out_dir``.

    Parameters
    ----------
    out_dir
        Destination directory.
    which
        CellMarker bulk file to package (``"all"`` = human + mouse).
    map_to_cl
        Fill blank CL ids via OLS.
    cache_dir, force
        Forwarded to :func:`biodb.cellmarker.get_markers`.

    Returns
    -------
    pathlib.Path
        ``out_dir``.
    """
    df = cellmarker.get_markers(which=which, map_to_cl=map_to_cl, cache_dir=cache_dir, force=force)
    return _write_per_species(out_dir, df, readme_fn=lambda sf: _cellmarker_card(df, which, sf))


def _cellmarker_card(
    df: pd.DataFrame, which: str, species_files: dict[str, tuple[str, int]]
) -> str:
    return f"""---
license: other
license_name: academic-use-only
license_link: https://bio-bigdata.hrbmu.edu.cn/CellMarker/
language:
  - en
tags:
  - biology
  - single-cell
  - cell-ontology
  - marker-genes
  - cellmarker
pretty_name: bioDB CellMarker 2.0 cell-type markers (human + mouse)
configs:
{_species_configs(species_files)}
---

# biodb_cellmarker

Curated human & mouse cell-type **marker genes** from
[CellMarker 2.0](http://bio-bigdata.hrbmu.edu.cn/CellMarker/) (Harbin Medical
University), packaged by [`bioDB`](https://github.com/bschilder/bioDB) and keyed
to the **Cell Ontology (CL)**.

> **One file per species** — species is in the file name
> (`markers/<species>.parquet`), so there is no `species` (or constant `source`)
> column. Load with the config name, e.g.
> `load_dataset("standardmodelbio/biodb_cellmarker", "human")`.

## What this is
CellMarker 2.0 is a manually curated database of cell-type markers for human and
mouse (normal + cancer contexts), each row carrying a native `cellontology_id`.
Each file is the normalized *(tissue, tissue_ontology_id, cell type, CL id, gene,
gene_id, score, rank)* table for one species (from the `{which}` bulk file) —
**{len(df):,} rows**.

## How it was made
`biodb.cellmarker.get_markers()` downloads `Cell_marker_{which.capitalize()}.xlsx`,
then for each *(species, cell type, gene)*:
- normalizes the native `cellontology_id` (`CL_0000540` → `CL:0000540`) and
  `uberonongology_id` (`UBERON_0000955` → `UBERON:0000955`),
- sets `score` = **literature-support count** (distinct supporting PMIDs; falls
  back to record count) — CellMarker has no continuous per-gene score,
- sets `rank` = dense within-cell-type rank of `score`.

## How to use
```python
from datasets import load_dataset
human = load_dataset("standardmodelbio/biodb_cellmarker", "human", split="train")
```
Or from `bioDB`: `from biodb import cellmarker; cellmarker.query_markers("CL:0000084", which="human")`.
To combine species/sources flexibly: `biodb.load_markers(source="cellmarker")`.

## Files / species coverage

{_species_files_md(species_files)}

## Provenance & license
Data from CellMarker 2.0, free for academic use. Cite Hu et al., *Nucleic Acids
Research* 2023 (D870–D876). Reproduce:
```python
from biodb import hf_datasets as hd
hd.build_cellmarker_dataset("biodb_cellmarker")
```
"""


# ─── Publish + collection ────────────────────────────────────────────────────


def push_to_hub(
    local_dir: str | Path,
    repo_id: str,
    *,
    private: bool = True,
    token: str | None = None,
) -> str:
    """Upload a built dataset directory to the HuggingFace Hub.

    Parameters
    ----------
    local_dir
        Directory with ``markers.parquet`` + ``README.md``.
    repo_id
        e.g. ``"standardmodelbio/biodb_celltaxonomy"``.
    private, token
        Repo visibility / auth (falls back to cached login).

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


def create_collection(
    title: str,
    *,
    namespace: str,
    dataset_repo_ids: list[str],
    description: str | None = None,
    private: bool = True,
    token: str | None = None,
) -> str:
    """Create (or reuse) a Hub collection and add the given datasets to it.

    Parameters
    ----------
    title
        Collection title (e.g. ``"bioDB"``).
    namespace
        Owning user/org (e.g. ``"standardmodelbio"``).
    dataset_repo_ids
        Dataset repo ids to add as collection items.
    description, private, token
        Collection metadata / auth.

    Returns
    -------
    str
        The collection slug.
    """
    from huggingface_hub import add_collection_item
    from huggingface_hub import create_collection as _create_collection

    collection = _create_collection(
        title=title,
        namespace=namespace,
        description=description or "bioDB cell-type marker & DEG datasets.",
        private=private,
        exists_ok=True,
        token=token,
    )
    for repo_id in dataset_repo_ids:
        try:
            add_collection_item(
                collection.slug, item_id=repo_id, item_type="dataset", exists_ok=True, token=token
            )
        except Exception as exc:  # noqa: BLE001 - already-present items raise; keep going
            logger.warning("collection add %s: %s", repo_id, exc)
    return collection.slug


__all__ = [
    "build_celltaxonomy_dataset",
    "build_cellmarker_dataset",
    "push_to_hub",
    "create_collection",
]
