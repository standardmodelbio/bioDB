# biodb

Phenotype-knowledge-graph helpers — Open Targets, Monarch Initiative, OBO
ontologies, and gene-weighting attention for clinical embeddings.

[![CI](https://github.com/bschilder/bioDB/actions/workflows/ci.yml/badge.svg)](https://github.com/bschilder/bioDB/actions/workflows/ci.yml)
[![Docs](https://github.com/bschilder/bioDB/actions/workflows/docs.yml/badge.svg)](https://bschilder.github.io/biodb/)
[![Docker](https://github.com/bschilder/bioDB/actions/workflows/docker.yml/badge.svg)](https://github.com/bschilder/bioDB/pkgs/container/biodb)
![coverage](docs/_static/coverage.svg)
![docker](docs/_static/docker.svg)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![License: PolyForm-NC 1.0.0](https://img.shields.io/badge/license-PolyForm--NC%201.0.0-lightgrey.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## What you get

* **`opentargets`** — download + parse Open Targets Platform
  parquet/JSON dumps (disease-to-gene, drug-to-gene, mouse-phenotype,
  expression, target-essentiality, pharmacogenomics, pathways),
  build gene-association matrices, and render disease / drug summary
  markdown.
* **`monarch`** — download + parse Monarch Initiative knowledge-graph
  TSVs (causal gene-to-disease associations and friends).
* **`ontology`** — expand keyword sets from OBO / OWL ontologies
  (Mondo by default), N-hop neighbour expansion, hierarchical keyword
  set generation, attention-weight analysis, gene-phenotype matrix
  construction, ontological similarity.
* **`gene_weighting`** — fast two-stage gene attention for clinical
  event embeddings, with lazy genomic sequence embedding cache and
  temporal / multi-condition weighting.

Each module was extracted from `AoU.phenome.{opentargets,monarch,
ontology,gene_weighting}` so sibling projects can depend on a narrow
phenotype-KG library without pulling the full All-of-Us pipeline.

## Install

Until released to PyPI, install from git:

```bash
pip install git+https://github.com/bschilder/bioDB
```

For the optional extras:

```bash
pip install "biodb[ontology,viz,tokens,gget]"
```

Local development:

```bash
git clone https://github.com/bschilder/bioDB
cd biodb
pip install -e ".[dev]"
```

## Quickstart

```python
import biodb

# Open Targets — list available parquet datasets
from biodb.opentargets import list_datasets
print(list_datasets())

# Monarch — read causal gene-to-disease associations
from biodb.monarch import read_causal_gene_to_disease_association
df = read_causal_gene_to_disease_association()

# Ontology — expand seeds into N-hop neighbourhoods
from biodb.ontology import expand_keyword_sets_from_ontology
ontology = {
    "dementia": ["alzheimer's disease", "vascular dementia"],
    "alzheimer's disease": ["early onset alzheimer's"],
}
seeds = {"dementia": ["dementia"]}
expanded = expand_keyword_sets_from_ontology(
    seed_keywords=seeds,
    ontology_dict=ontology,
    n_hops=2,
)

# Gene weighting — score gene relevance against event embeddings
from biodb.gene_weighting import GeneWeightingConfig, compute_gene_weights_fast
cfg = GeneWeightingConfig(top_k=50, temperature=0.1)
```

## License

PolyForm-Noncommercial-1.0.0 — see [LICENSE](LICENSE).
