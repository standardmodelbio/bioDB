# bioDB tutorials

Rendered `.ipynb` notebooks (with executed outputs — open on GitHub /
nbviewer / Colab). Every per-source tutorial demonstrates **both** the
targeted API mode and the bulk-download mode.

To re-execute against the current `biodb` install:

```bash
pip install "biodb[clinvar,protein,ontology,mapping,test]" nbconvert ipykernel
jupyter nbconvert --to notebook --execute --inplace tutorials/*.ipynb
```

## Index

| # | Notebook | Source | API mode | Bulk mode |
|---|----------|--------|----------|-----------|
| 01 | [Utils quickstart](01_utils_quickstart.ipynb) | (cross-cutting) | — | — |
| 02 | [Open Targets](02_opentargets.ipynb) | `biodb.opentargets`, `biodb.opentargets_graphql` | GraphQL `query_target` / `query_disease` / `query_drug` | Parquet FTP — `list_datasets`, `get_dataset` |
| 03 | [Monarch](03_monarch.ipynb) | `biodb.monarch` | BioLink REST + Neo4j Cypher | Per-association TSV readers |
| 04 | [Ontology Lookup Service](04_ontology.ipynb) | `biodb.ols`, `biodb.ontology` | OLS4 REST + owlready2 entity walk | OBO/OWL download + parse, N-hop expansion |
| 05 | [UniProt](05_uniprot.ipynb) | `biodb.uniprot` | REST — `query_protein`, `get_features`, `get_dbxrefs` | FTP FASTA — `download_swissprot_fasta`, `iter_fasta_records` |
| 06 | [Harmonizome](06_harmonizome.ipynb) | `biodb.harmonizome` | REST — `list_datasets`, `get_dataset_metadata` | `download_datasets`, `get_gmt`, `load_gene_attribute_matrix` |
| 07 | [ClinVar](07_clinvar.ipynb) | `biodb.clinvar` | NCBI E-utils — `query_variant`, `query_gene` | VCF — `download_vcf`, `simplify_annotations` (`CLNSIG` long-tail buckets) |
| 08 | [GWAS Atlas](08_gwas_atlas.ipynb) | `biodb.gwas_atlas` | Cached per-trait `query_trait`, `list_traits` | MAGMA matrix — `download_magma_p`, `load_magma_p`, `melt_magma_p` |
| 09 | [gProfiler](09_gprofiler.ipynb) | `biodb.gprofiler` | REST enrichment — `gost` | Combined per-organism GMT — `download_gmt`, `load_gmt` |
| 10 | [MSigDB](10_msigdb.ipynb) | `biodb.msigdb` | Per-set JSON — `query_gene_set`, `query_genes` | Per-collection GMT — `download_gmt`, `load_gmt` |
| 11 | [PubMed](11_pubmed.ipynb) | `biodb.pubmed` | NCBI E-utils — `search`, `query_pmid`, `query_summaries`, `query_abstract` | Annual Baseline + Daily Update XML.gz — `list_baseline_files`, `download_baseline_file`, `parse_pubmed_xml` |
| 12 | [SNOMED CT](12_snomed.ipynb) | `biodb.snomed` | Per-concept via OLS4 — `query_concept`, `search_concepts`, `get_descendants`, `get_ancestors` | OHDSI CONCEPT.csv via GitHub release — `download_concept_csv`, `load_concept_csv` |

Notebooks 02–10 hit real upstream endpoints for the API-mode demos
(small payloads — fast) and execute bulk-mode calls whenever the
artifact is small enough to download in a tutorial (e.g. MSigDB
Hallmark ~50 KB). For multi-GB / multi-tens-of-GB artifacts (TrEMBL,
ClinVar VCF, full Open Targets bulk shards) the calls are shown in
markdown cells but not executed.

Notebook 01 stays offline against synthetic data — it covers the
cross-cutting helpers in `biodb.utils`, not any single source.
