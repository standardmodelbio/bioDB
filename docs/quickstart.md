# Quickstart

## Install

```bash
pip install git+https://github.com/bschilder/bioDB
```

For the optional extras (ontology readers, plotting, tiktoken,
gget-based Open Targets queries):

```bash
pip install "biodb[ontology,viz,tokens,gget]"
```

## Open Targets

```python
from biodb.opentargets import list_datasets, get_dataset, df_to_markdown

# What parquet dumps are available?
print(list_datasets())

# Render a single target row as markdown (works with dict / Series rows).
row = {
    "approvedSymbol": "BRCA1",
    "approvedName": "BRCA1 DNA repair associated",
    "id": "ENSG00000012048",
    "biotype": "protein_coding",
}
print(df_to_markdown(row))
```

## Monarch Initiative

```python
from biodb.monarch import read_causal_gene_to_disease_association

df = read_causal_gene_to_disease_association()
print(df.head())
```

## Ontology expansion

```python
from biodb.ontology import expand_keyword_sets_from_ontology

ontology = {
    "dementia": ["alzheimer's disease", "vascular dementia"],
    "alzheimer's disease": ["early onset alzheimer's"],
}
expanded = expand_keyword_sets_from_ontology(
    seed_keywords={"dementia": ["dementia"]},
    ontology_dict=ontology,
    n_hops=2,
)
print(expanded["dementia"])
# ['dementia', "alzheimer's disease", 'vascular dementia',
#  "early onset alzheimer's"]
```

## Building a gene-association matrix

```python
import pandas as pd
from biodb.utils import create_gene_association_matrix

assoc = pd.DataFrame({
    "sourceId": ["DIS:1", "DIS:1", "DIS:2"],
    "HGNC":     ["BRCA1", "TP53",  "EGFR"],
    "score":    [0.9, 0.8, 0.7],
})
X, meta = create_gene_association_matrix(assoc, verbose=False)
print(X.shape, meta["metadata"]["sparsity"])
```
