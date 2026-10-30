# Cell-type sources

bioDB provides curated Cell Taxonomy and CellMarker 2.0 marker tables,
CELLxGENE WMG/CellGuide markers, and CELLxGENE disease differential expression.
Source tables preserve CL cell-type IDs, tissue labels/UBERON IDs when
available, species, and gene identifiers.

## Choose a source

| Module | Targeted access | Bulk access / requirements |
|---|---|---|
| `celltaxonomy` | `query_markers` filters the normalized local table | `download`, `load_resource`, `get_markers`, `to_gmt`; core dependencies |
| `cellmarker` | `query_markers` filters the normalized local table | `download`, `load`, `get_markers`, `to_gmt`; `[celltype]` for XLSX |
| `cellxgene` (WMG) | `query_markers` for one cell type/tissue | `get_tissue_markers`, `get_all_markers`, `to_gmt`; core dependencies |
| `cellxgene` (CellGuide) | `cellguide_markers` for one cell type | `get_all_cellguide_markers`; core dependencies |
| `markers` | Filters over combined source tables | `load_markers`; `[celltype]` if CellMarker is selected |

Curated-source queries first acquire the bulk resource if needed. CELLxGENE
targeted queries use served APIs/snapshots, without downloading raw single-cell
matrices or requiring `cellxgene-census`.

## Curated markers

```python
from biodb import celltaxonomy, cellmarker

taxonomy = celltaxonomy.query_markers("CL:0000540", species="Homo sapiens")
cellmarker_neurons = cellmarker.query_markers(
    "CL:0000540", which="human", species="Homo sapiens"
)
```

Install CellMarker's Excel reader with `pip install "biodb[celltype]"`.
The downloader pins CellMarker 2.0's version-specific endpoint; the unversioned
site can serve another release and is not an automatic fallback.

Use `by="name"` for exact case-insensitive cell-type names. Native CL IDs
are normalized to CURIE form. `get_markers(map_to_cl=True)` optionally fills
missing CL IDs through OLS, adding network requests; the default is false.

The normalized schema is `species`, `tissue`, `tissue_ontology_id`,
`cell_type_name`, `cell_ontology_id`, `gene_symbol`, `gene_id`, `score`,
`rank`, `source`. Curated scores count distinct supporting PMIDs, falling
back to record counts, and rank is dense within cell type. These counts are
evidence support, not expression effect sizes.

## Tissue-specific WMG markers

```python
from biodb import cellxgene

cell_types = cellxgene.list_cell_types("spleen")
markers = cellxgene.query_markers("CL:0000236", tissue="spleen", n_top=10)
print(markers[["gene_symbol", "score", "rank"]])
```

WMG returns CZI's precomputed Marker Score for healthy-cell type specificity.
The tissue is required. A name may be used instead of a CL ID and is resolved
through OLS. `organism="Homo sapiens"` is the default; labels and NCBITaxon
IDs are accepted. `list_tissues` discovers available tissues.

`get_tissue_markers("spleen", n_top_per_type=25)` fetches all types in that
tissue and caches the table by organism, tissue, source snapshot, test, and
requested marker count. `get_all_markers(organism=..., tissues=...)` iterates
tissues and reuses those tables; omitting `tissues` selects the whole organism.

## CellGuide and aggregation

```python
from biodb import cellxgene

guide = cellxgene.cellguide_markers("CL:0000236", kind="both")
print(guide[["marker_type", "species", "tissue", "gene_symbol"]].head())
```

CellGuide returns all available tissues/species for a cell type, with
`kind="computational"`, `"canonical"`, or `"both"`. Computational rows have
`marker_score`, specificity, and source expression summaries. Canonical
curated reference markers are cross-species and carry null species/score.
Not every listed cell type has marker files; a successful missing file yields
an empty result. Corpus-wide acquisition is available via
`get_all_cellguide_markers`.

```python
from biodb.markers import load_markers

combined = load_markers(
    source=["celltaxonomy", "cellmarker"],
    species="Homo sapiens",
    cell_type="CL:0000540",
)
```

`load_markers` combines sources into a shared table with `source` and
`marker_type`. Each filter accepts a scalar, list, or `None` (all). Tissue
filters accept labels or UBERON IDs; cell-type filters accept names or CL IDs.
Filters apply after source tables are acquired, so a narrow filter can still
require a full source download. Selecting a species excludes cross-species
canonical CellGuide rows with null species.

The aggregator labels curated Cell Taxonomy/CellMarker rows as canonical and
renames CellGuide `marker_score` to `score`. Scores retain source-specific
meanings and should be interpreted within a source; concatenation does not
calibrate them across providers.

## Disease differential expression

```python
from biodb import cellxgene

diseases = cellxgene.list_diseases("lung")
cell_types = cellxgene.list_cell_types("lung")
if not diseases.empty and not cell_types.empty:
    deg = cellxgene.disease_vs_normal(
        cell_types.iloc[0]["cell_ontology_id"],
        tissue="lung",
        disease=diseases.iloc[0]["disease_ontology_term_id"],
        n_top=20,
    )
```

The discovery tables describe available tissue data; check that the selected
cell type is represented in both disease and normal groups for your contrast.
`disease_vs_normal` contrasts that cell type across disease and normal cells
using CZI's served differential-expression API. Positive `effect_size` is up
in disease. `log_fold_change` and `adjusted_p_value` are separate fields;
this is a different statistic from Marker Score. `n_top=None` retains the
returned genomewide vector. `include_descendants=True` pools available MONDO
descendant diseases into a disease-category contrast. The lower-level
`differential_expression` accepts explicit WMG filter dictionaries for both
groups and an overlap-handling option.

## Build local datasets or publish

```python
from biodb.cellxgene_dataset import build_dataset

# Corpus-wide marker acquisition; can take time on first use.
directory = build_dataset(
    "./biodb-cellxgene", organisms=["Homo sapiens"], include_deg=False
)
```

The builder writes computational marker Parquet files per species, a shared
`markers/canonical.parquet`, and a dataset card. Species/source/marker type
may be encoded in the file path rather than repeated as columns. Optional
`include_deg=True` and `include_deg_rollup=True` write leaf and pooled-disease
DEG shards respectively. These corpus operations can be large;
`build_disease_deg` concatenates a whole organism in memory, whereas the
dataset builder writes shards.

`biodb.hf_datasets.build_celltaxonomy_dataset` and
`build_cellmarker_dataset` package the curated sources as per-species marker
files and cards. Local building does not publish anything. The separate
`push_to_hub` functions and `hf_datasets.create_collection` require `[hf]`
(`huggingface_hub`) and credentials; dataset uploads default to private.

See the [API reference](api.rst) and
[cell-marker notebook](_generated/tutorials/15_cell_markers.ipynb).
