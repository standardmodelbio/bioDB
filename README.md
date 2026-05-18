# bioDB

Biomedical knowledge graph helpers — Open Targets, Monarch Initiative, OBO ontologies, UniProt, Harmonizome, and ClinVar.

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
| **[Monarch Initiative](https://monarchinitiative.org/)** — causal gene-to-disease, phenotype-to-disease, full KG | [`biodb.monarch`](src/biodb/monarch.py) | ✅ BioLink v3 REST (`query_entity`, `query_associations`, `query_gene_associations`) + Neo4j Cypher against the public KG (`query_cypher`, `query_neighbors`) | ✅ Per-association TSVs (`get_gene_associations`, `read_causal_gene_to_disease_association`) |
| **[Ontology Lookup Service (OLS)](https://www.ebi.ac.uk/ols4/)** — EBI-hosted gateway to ~280 OBO Foundry ontologies (Mondo, HPO, EFO, SO, GO, ChEBI, SNOMED, …) | [`biodb.ols`](src/biodb/ols.py), [`biodb.ontology`](src/biodb/ontology.py) | ✅ OLS4 REST — no local file needed (`ols.get_term`, `get_descendants`, `get_ancestors`, `search`) + generic owlready2 entity walk for callers with a local OWL file (`ontology.get_ontology`, …) | ✅ OBO / OWL download + parse, N-hop expansion, hierarchical keyword sets, attention-weight analysis, gene-phenotype matrix, ontological similarity |
| **[UniProt](https://www.uniprot.org/)** — protein sequences, features, cross-references | [`biodb.uniprot`](src/biodb/uniprot.py) | ✅ REST (`query_protein`, `get_sequences`, `get_features`, `get_dbxrefs`) | ✅ FTP FASTA (`download_swissprot_fasta`, `download_trembl_fasta`, `iter_fasta_records`, `count_swissprot_records`) |
| **[Harmonizome](https://maayanlab.cloud/Harmonizome/)** — ~114 curated gene-attribute datasets (CCLE, GTEx, ENCODE, HPA, KEGG, Reactome, …) | [`biodb.harmonizome`](src/biodb/harmonizome.py) | ✅ REST (`list_datasets`, `get_dataset_metadata`) | ✅ Bulk TSV/GMT (`download_datasets`, `get_gmt`, `load_gene_attribute_matrix`) |
| **[ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/)** — clinical significance + review status for human variants | [`biodb.clinvar`](src/biodb/clinvar.py) | ✅ NCBI E-utilities (`query_variant`, `query_gene`) | ✅ VCF download (`download_vcf`, `vcf_to_df`, `simplify_annotations`, `df_to_bed`, `df_to_sites`) |
| **[GWAS Atlas](https://atlas.ctglab.nl/)** — Watanabe et al. per-study gene-level MAGMA p-values across ~4k GWAS summary stats | [`biodb.gwas_atlas`](src/biodb/gwas_atlas.py) | ✅ Per-trait lookup via cached metadata (`query_trait`, `list_traits`) | ✅ Bulk download (`download_magma_p`, `load_magma_p`, `load_metadata`, `melt_magma_p`) |
| **[gProfiler](https://biit.cs.ut.ee/gprofiler/)** — University of Tartu gene-set library + functional enrichment | [`biodb.gprofiler`](src/biodb/gprofiler.py) | ✅ REST (`gost`) | ✅ Bulk GMT (`download_gmt`, `load_gmt`) |
| **[MSigDB](https://www.gsea-msigdb.org/gsea/msigdb/)** — Broad Institute Molecular Signatures DB (Hallmark + C1–C8) | [`biodb.msigdb`](src/biodb/msigdb.py) | ✅ Per-set JSON (`query_gene_set`, `query_genes`) | ✅ Bulk GMT (`download_gmt`, `load_gmt` for any collection/version) |
| **[PubMed](https://pubmed.ncbi.nlm.nih.gov/)** — NLM's 40 M+ biomedical citation database (titles, abstracts, authors, MeSH) | [`biodb.pubmed`](src/biodb/pubmed.py) | ✅ NCBI E-utilities (`search`, `query_pmid`, `query_summaries`, `query_abstract`) | ✅ Annual Baseline + Daily Update XML.gz (`list_baseline_files`, `download_baseline_file`, `parse_pubmed_xml`) |
| **[SNOMED CT](https://www.snomed.org/)** (OHDSI-flavoured) — clinical terminology: diagnoses, procedures, findings, body sites | [`biodb.snomed`](src/biodb/snomed.py) | ✅ Per-concept lookups via OLS4 (`query_concept`, `search_concepts`, `get_descendants`, `get_ancestors`, `get_children`, `get_parents`) | ✅ OHDSI CONCEPT.csv via GitHub release (`download_concept_csv`, `load_concept_csv`, `get_concept_csv_path`, `is_available`) |

### Cross-cutting helpers

* [`biodb.mapping`](src/biodb/mapping.py) — `map_gene_ids` — gProfiler-backed cross-namespace gene-ID conversion (Ensembl ↔ HGNC ↔ Entrez ↔ UniProt …)
* [`biodb.transform`](src/biodb/transform.py) — `create_gene_association_matrix` — pivot a long `(sourceId, targetId, score)` DataFrame into a sparse/dense (samples × genes) matrix with AnnData-shaped metadata
* [`biodb.utils`](src/biodb/utils.py) — `read_gmt` (GMT format reader used by MSigDB / Harmonizome / gProfiler), `filter_adaptive` (per-sample top-percentile keep), similarity helpers

## Install

Until released to PyPI, install from git:

```bash
pip install git+https://github.com/bschilder/bioDB
```

For the optional extras:

```bash
pip install "biodb[ontology,viz,tokens,gget,protein,clinvar]"
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

### UniProt protein lookups

```python
from biodb.uniprot import query_protein, get_features, get_dbxrefs

records = query_protein("P12345")           # list[Bio.SeqRecord]
features = get_features("P12345")           # DataFrame of protein features
xrefs = get_dbxrefs("P12345")               # DataFrame of cross-references
```

### Generic OWL ontology walks

```python
from biodb.ontology_owl import get_ontology, get_descendants, get_mrca, HPO_URL

hpo = get_ontology(HPO_URL)
kids = get_descendants("HP:0001250", ont=hpo, return_as="id|label")  # all descendants of "Seizure"
mrca = get_mrca(hpo, "HP:0001250", "HP:0002353")                      # most recent common ancestor
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

## Tutorials

Rendered Jupyter notebooks live under [`tutorials/`](tutorials/) — each
one runs offline against synthetic data and ships with executed
outputs so you can preview behavior without installing the package.
See [`tutorials/README.md`](tutorials/README.md) for the index.

## Used by

- [`GeneDocs`](https://github.com/bschilder/GeneDocs) — biomedical gene knowledge base; uses `biodb.opentargets.get_dataset` to pull the OT release into a precomputed, queryable artifact + uses `biodb.opentargets.query_target` for live per-gene lookups.
- [`seqlab`](https://github.com/standardmodelbio/seqlab) — variant analysis pipeline; uses `biodb.opentargets` for gene/variant queries and `biodb.clinvar` for clinical significance annotation.

## License

PolyForm-Noncommercial-1.0.0 — see [LICENSE](LICENSE).
