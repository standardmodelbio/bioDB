# biodb

`biodb` is a small standalone library of biomedical knowledge graph
helpers — Open Targets Platform, Monarch Initiative, OBO / OWL
ontologies, UniProt, Harmonizome, and ClinVar. Ported out of
`AoU.phenome` (plus helpers adapted from `VEP_protein`) so sibling
projects can depend on a narrow biomedical-KG library without pulling
the full All-of-Us pipeline.

## Why use it

- **`opentargets`** — pull and parse Open Targets Platform parquet/JSON
  dumps; render disease, drug, and pharmacogenomics summaries as
  markdown; build gene-association matrices.
- **`monarch`** — fetch Monarch Initiative TSVs (causal gene-to-disease
  associations + friends) into tidy DataFrames.
- **`ontology`** — N-hop keyword set expansion, Mondo / OWL loaders,
  hierarchical keyword set generation, attention analysis, gene-
  phenotype matrices, ontological similarity.
- **`clinvar`** — ClinVar VCF download / parse, CLNSIG long-tail to
  6-class simplification, BED + sites format converters.

## Quickstart

```python
from biodb.opentargets import list_datasets
from biodb.ontology import expand_keyword_sets_from_ontology

print(list_datasets())  # list available Open Targets parquet datasets

expanded = expand_keyword_sets_from_ontology(
    seed_keywords={"dementia": ["dementia"]},
    ontology_dict={"dementia": ["alzheimer's disease"]},
    n_hops=1,
)
```

See the [quickstart guide](quickstart.md) for usage patterns and the
[API reference](api.rst) for the complete surface.

```{toctree}
:maxdepth: 2
:hidden:

quickstart
api
changelog
```
