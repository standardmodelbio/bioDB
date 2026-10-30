# Quickstart

## Install

Install from the repository:

```bash
pip install "biodb @ git+https://github.com/standardmodelbio/bioDB.git"
```

Choose extras for the paths you use. Extras in this table are declared in
`pyproject.toml`; the core package includes pandas, Polars, PyArrow, requests,
and httpx.

| Path | Extra / additional dependency |
|---|---|
| OBO/OWL readers and graph expansion | `[ontology]` |
| UniProt XML parsing | `[protein]` |
| ClinVar VCF ingestion | `[clinvar]` |
| CellMarker XLSX and OmicsPred Excel | `[celltype]`, `[omicspred]` respectively |
| Indexed GWAS Catalog regions | `[gwas]` (`pysam`) |
| Pan-UKBB regions | `[panukbb]` (`pysam`) or the `tabix` executable |
| BigWig interval scores | `[bigwig]` |
| Gene ID conversion using g:Profiler's Python client | `[mapping]` |
| Hugging Face publication | `[hf]` |
| GENCODE GTF download | `pooch` (currently supplied by `[clinvar]`, or install directly) |
| Optional gget-backed helpers | `[gget]` |
| Plotting / token counting / docs build | `[viz]`, `[tokens]`, `[docs]` |

For example:

```bash
pip install "biodb[ontology,protein,celltype,gwas] @ git+https://github.com/standardmodelbio/bioDB.git"
```

## Run an offline transform

```python
import pandas as pd
from biodb.transform import create_gene_association_matrix

associations = pd.DataFrame({
    "sourceId": ["DIS:1", "DIS:1", "DIS:2"],
    "HGNC": ["BRCA1", "TP53", "EGFR"],
    "score": [0.9, 0.8, 0.7],
})
matrix, metadata = create_gene_association_matrix(associations, verbose=False)
print(matrix.shape)  # (2, 3)
```

## Query one Open Targets record

The GraphQL client needs only core dependencies:

```python
from biodb.opentargets_graphql import query_target

target = query_target("ENSG00000012048")
print(target)
```

For a bulk preview, explicitly limit the file count:

```python
from biodb.opentargets import DEFAULT_VERSION, scan_dataset

preview = scan_dataset(
    "target",
    version=DEFAULT_VERSION,
    limit_files=1,
    columns=["id", "approvedSymbol"],
).head(5).collect()
print(preview)
```

This acquires one whole Parquet shard locally. It is a preview of the source;
`.head(5)` limits rows after acquisition. Omit `limit_files` to acquire the
entire manifest membership. See [Open Targets](opentargets.md) for lazy remote
reads, raw nested fields, and lead/member credible-set semantics.

## Parse raw GWAS statistics locally

This example creates a tiny input and runs without network access:

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import polars as pl
from biodb import gwas_catalog

with TemporaryDirectory() as directory:
    path = Path(directory) / "example.tsv"
    path.write_text(
        "chromosome\tbase_pair_location\teffect_allele\tother_allele\tbeta\tp_value\n"
        "1\t100\tA\tG\t0.2\t0\n"
        "1\t200\tT\tC\tNA\t0.8\n"
    )
    batches = list(gwas_catalog.iter_sumstats(path))  # exhaust to validate EOF
    raw = pl.concat(batches)
    assert raw.height == 2
    assert raw["p"][0] == 0.0
```

The default is genomewide with no significance filter. Missing effects and
exact source tokens survive. [GWAS acquisition](gwas.md) covers real manifests,
region queries, provenance, durable receipts, and strict failure handling.

## Query one cell type

```python
from biodb import cellxgene

markers = cellxgene.query_markers("CL:0000236", tissue="spleen", n_top=10)
print(markers[["gene_symbol", "score", "rank"]])
```

This uses the served WMG API with core dependencies. The
[cell-type guide](cell-types.md) explains tissue-specific scores, CellGuide,
curated sources, and dataset builders.

## Resolve ontology terms

```python
from biodb.ols import find_term, get_descendants

term = find_term("breast carcinoma", ontology="efo")
if term is not None:
    print(term["obo_id"])
    descendants = get_descendants("efo", term["obo_id"])
```

See [OLS](ols.md) for ranked matching and bulk textual annotations, or
[GTR](gtr.md) for genetic test panels. The [source catalog](sources.md) maps
the remaining sources to their entry points and the
[tutorial index](_generated/tutorials/index.md) links runnable notebooks.
