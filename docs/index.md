# bioDB

`biodb` provides Python clients and bulk readers for biomedical knowledge
sources: gene–disease associations, GWAS statistics, ontologies, literature,
protein annotations, cell-type markers, and genomic tracks. Targeted queries
and cached bulk acquisition share a library; available modes depend on the
source.

Start with the [quickstart](quickstart.md), browse the [source catalog](sources.md),
or open the [tutorial notebooks](_generated/tutorials/index.md). The
[API reference](api.rst) documents every public source and helper module.

## Guides

| Task | Guide |
|---|---|
| Download Open Targets tables or inspect credible sets | [Open Targets](opentargets.md) |
| Acquire raw genomewide or regional GWAS statistics | [GWAS acquisition](gwas.md) |
| Attach publication, structure, constants, and gene annotations | [Raw annotations and enrichment](enrichment.md) |
| Query cell markers and disease differential expression | [Cell-type sources](cell-types.md) |
| Resolve ontology terms and preserve textual annotations | [OLS and SNOMED](ols.md) |
| Find genetic tests and build panel gene sets | [GTR](gtr.md) |

## A small offline example

```python
import pandas as pd
from biodb.transform import create_gene_association_matrix

associations = pd.DataFrame({
    "sourceId": ["trait:1", "trait:1", "trait:2"],
    "HGNC": ["BRCA1", "TP53", "EGFR"],
    "score": [0.9, 0.8, 0.7],
})
matrix, metadata = create_gene_association_matrix(associations, verbose=False)
assert matrix.shape == (2, 3)
```

Bulk examples in the guides can download large upstream files. File previews,
region selections, and significance filters are explicit options; they change
the coverage of the returned data. See each guide for cache and provenance
details.

```{toctree}
:maxdepth: 2
:hidden:

quickstart
sources
opentargets
gwas
enrichment
cell-types
ols
gtr
_generated/tutorials/index
api
changelog
```
