# bioDB

Phenotype-knowledge-graph helpers — Open Targets, Monarch Initiative, and OBO ontologies.

[![CI](https://github.com/bschilder/bioDB/actions/workflows/ci.yml/badge.svg)](https://github.com/bschilder/bioDB/actions/workflows/ci.yml)
[![Docs](https://github.com/bschilder/bioDB/actions/workflows/docs.yml/badge.svg)](https://bschilder.github.io/biodb/)
[![Docker](https://github.com/bschilder/bioDB/actions/workflows/docker.yml/badge.svg)](https://github.com/bschilder/bioDB/pkgs/container/biodb)
![coverage](docs/_static/coverage.svg)
![docker](docs/_static/docker.svg)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![License: PolyForm-NC 1.0.0](https://img.shields.io/badge/license-PolyForm--NC%201.0.0-lightgrey.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## What this is

`bioDB` is a single library for accessing biomedical knowledge sources in **two complementary modes**:

|     | Targeted queries (API mode) | Bulk downloads (FTP mode) |
|---|---|---|
| **What** | Live lookups against the source's REST / GraphQL API — one gene, one disease, one ontology term at a time. | Whole-dataset Parquet / TSV / OBO downloads from each source's FTP server, cached locally and ready to load as DataFrames. |
| **For** | Interactive analysis, app backends, prototyping. | **Large-scale analyses and AI model training** — joining millions of associations, building gene-disease matrices, embedding gene/disease/drug knowledge into LMs, training models that need every variant or every pathway. |
| **Surface** | `query_*`, `fetch_*` functions (small payloads, fast, fresh). | `list_datasets()`, `get_dataset()`, parsers (large payloads, cached, reproducible by release). |

Each source module aims to provide **both** modes, so you can prototype against the API and then scale up to the bulk pull without rewriting your pipeline.

## Sources

| Source | Module | API queries | Bulk downloads |
|---|---|---|---|
| **[Open Targets Platform](https://platform-docs.opentargets.org/)** — gene-disease-drug associations, evidence, expression, essentiality, pharmacogenomics, pathways | [`biodb.opentargets`](src/biodb/opentargets.py) | ✅ GraphQL (`query_target`, `query_disease`, `query_drug`, `query_variant`, …) | ✅ FTP Parquet (`list_datasets`, `get_dataset`, `ensure_cached_shards`, versioned cache) |
| **[Monarch Initiative](https://monarchinitiative.org/)** — causal gene-to-disease, phenotype-to-disease | [`biodb.monarch`](src/biodb/monarch.py) | 🚧 [bioDB#TBD](https://github.com/bschilder/bioDB/issues) — Monarch BioLink API stub | ✅ TSV knowledge graphs (`get_gene_associations`, `read_causal_gene_to_disease_association`) |
| **OBO / OWL ontologies** ([Mondo](https://mondo.monarchinitiative.org/), HPO, EFO, …) | [`biodb.ontology`](src/biodb/ontology.py) | 🚧 [bioDB#TBD](https://github.com/bschilder/bioDB/issues) — per-term lookup helper | ✅ OBO / OWL download + parse, N-hop expansion, hierarchical keyword sets, attention-weight analysis, gene-phenotype matrix, ontological similarity |

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

### Targeted API queries

```python
from biodb.opentargets import query_target, query_disease

# One-gene-at-a-time GraphQL lookup — fast, fresh, no FTP needed.
brca1 = query_target("ENSG00000012048")
print(brca1["approvedSymbol"], brca1["biotype"])

# Same shape for disease + drug + variant.
mondo = query_disease("MONDO_0007254")
```

### Bulk downloads for large-scale analyses

```python
from biodb.opentargets import list_datasets, get_dataset

# Discover every Parquet dataset in the latest OT release.
datasets = list_datasets()  # {'target': '.../target', 'association_overall_direct': ..., ...}

# Pull a whole dataset for offline analysis / training input. Downloads
# every Parquet shard once, caches under ~/.cache/biodb/opentargets/<version>/<dataset>/,
# and concatenates into a pandas DataFrame on subsequent calls.
associations = get_dataset("association_overall_direct")
targets = get_dataset("target")
```

### Monarch + ontology

```python
from biodb.monarch import read_causal_gene_to_disease_association
from biodb.ontology import expand_keyword_sets_from_ontology

df = read_causal_gene_to_disease_association()

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
```

## Used by

- [`GeneDocs`](https://github.com/bschilder/GeneDocs) — biomedical gene knowledge base; uses `biodb.opentargets.get_dataset` to pull the OT release into a precomputed, queryable artifact + uses `biodb.opentargets.query_target` for live per-gene lookups.

## License

PolyForm-Noncommercial-1.0.0 — see [LICENSE](LICENSE).
