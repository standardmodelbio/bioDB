# Source catalog

This catalog maps every public source and helper module to its supported
entry points. Targeted access may be a remote API or a filter over cached
bulk data; the table makes that distinction explicit. The
[API reference](api.rst) contains full signatures, return schemas, and module
documentation. The [tutorial index](_generated/tutorials/index.md) includes
offline examples and optional live cells.

## Open Targets

See the [Open Targets guide](opentargets.md) for full/preview acquisition and
correct lead/member PIP semantics.

| Module | Access and entry points |
|---|---|
| {mod}`biodb.opentargets` | Release discovery (`list_available_versions`, `list_datasets`), complete shard acquisition (`ensure_cached_shards`), raw eager tables (`get_dataset`), association/pathway/target helpers |
| {mod}`biodb.opentargets.readers` | Lazy local/remote `scan_dataset`; raw `get_diseases`, `get_traits`, `get_drugs` |
| {mod}`biodb.opentargets.variants` | Raw credible sets/annotations, matched-lead summaries, member tables, optional derived VEP aggregation |
| {mod}`biodb.opentargets.studies` | `get_study`, `attach_study_type` |
| {mod}`biodb.opentargets_graphql` | Targeted `query_target`, `query_disease`, `query_drug`, `query_variant`; symbol mapping and panel-score helpers |
| {mod}`biodb.opentargets.literature` | `fetch_literature_matches` from raw intermediate shards; exact pair/entity-type filtering |
| {mod}`biodb.opentargets.constants` | Source text/SHAs, parsed canonical maps (`fetch_constants`), complete OT SO labels |

## GWAS and molecular traits

See [raw GWAS acquisition](gwas.md) for unfiltered defaults, provenance,
indexed regions, and resumable ingestion.

| Module | Targeted / discovery access | Bulk / local access |
|---|---|---|
| {mod}`biodb.gwas_catalog` | `list_datasets` selects exact harmonised manifest files | `download_dataset`, `iter_sumstats`, `ingest_sumstats`, `load_ingest_result`; core, `[gwas]` for indexed ranges |
| {mod}`biodb.gwas_harvest` | `list_datasets` requires an actual source URL manifest | Same raw ingestion surface as Catalog; `local_root` for source-host archives |
| {mod}`biodb.panukbb` | `phenotype_manifest`, `select_traits`, `sumstats_region`, `sumstats_regions` | Regional dense statistics over tabix; GRCh37/hg19, `pysam` or `tabix` |
| {mod}`biodb.gwas_atlas` | `list_traits`, `query_trait` over atlas tables | Metadata and gene-level MAGMA matrix download/load/melt helpers |
| {mod}`biodb.aou_allxall` | `list_phenotypes`, `query_phenotype`, `get_gene_burden` via the public atlas API | `download_all_gene_burden`, `load_gene_burden`, `melt_gene_burden`, `iter_signature_variants`; public summary data |
| {mod}`biodb.omicspred` | Platforms/datasets, individual scores, search, performance/publication records | Excel metadata and scoring-file archives; `[omicspred]` for Excel |

These resources represent different units: variants, gene-level association,
gene burden, or molecular-trait scores. They do not share a universal effect
allele, reference build, or p-value transformation.

## Ontologies, diseases, genetic tests, and literature

See [OLS](ols.md), [GTR](gtr.md), and
[raw annotations](enrichment.md) for worked examples.

| Module | Targeted access | Bulk / local access |
|---|---|---|
| {mod}`biodb.ols` | Ranked `find_term` / `find_terms`, search, term/hierarchy lookups, CURIE/IRI helpers | `iter_terms`, version-aware `list_terms` cache |
| {mod}`biodb.ols_text` | `text_record` normalizes literal annotations from an OLS v2 entity | `download_text_cache` with resumable pages and ID/count audits |
| {mod}`biodb.snomed` | Concept/search/hierarchy wrappers over OLS | `bulk_concepts`, `bulk_text_concepts`; caller-supplied Athena `CONCEPT.csv`/ZIP parsers |
| {mod}`biodb.ontology` | Traversal, MRCA, labels/IDs, similarity and keyword-set helpers over local ontologies | OBO/OWL/Mondo loading; `[ontology]`, with `[viz]` for plotting paths |
| {mod}`biodb.ontology_owl` | Compatibility re-exports of generic OWL helpers | Deprecated import path; use `biodb.ontology` |
| {mod}`biodb.monarch` | `query_entity`, `query_associations`, `query_gene_associations`, `query_neighbors`, `query_cypher` | TSV `list_datasets` / `get_dataset`, causal gene–disease readers, association transforms |
| {mod}`biodb.orphanet` | `prevalence_map` over cached epidemiology | `download_prevalence`, `parse_prevalence`, `load_prevalence`; prevalence bands mapped explicitly |
| {mod}`biodb.gtr` | `search_tests`, `query_gene`, `query_condition`, `query_test`, `panel_text` | Daily TSV and full XML, gene-set aggregation/GMT |
| {mod}`biodb.pubmed` | E-utilities search, summaries, abstract records | Baseline/update XML downloads with MD5 verification and iterative XML parsing |
| {mod}`biodb.europepmc` | `fetch_metadata` for raw MED/core records keyed by PMID | Batched requests with per-PMID completion receipts |

OLS cache versions preserve already-acquired snapshots; they do not provide
an arbitrary historical release endpoint. `ols_text` preserves actual v2
literal labels/synonyms/definitions; logical axioms are not synthesized into
prose. SNOMED vocabulary licensing still applies to use and redistribution.

## Cell-type sources and dataset builders

See the [cell-type guide](cell-types.md) for schema, score meanings, source
filters, served DEG, and local dataset layout.

| Module | Access and entry points |
|---|---|
| {mod}`biodb.celltaxonomy` | Curated cross-species TSV; raw `load_resource`, normalized `get_markers`, cached-table `query_markers`, GMT export |
| {mod}`biodb.cellmarker` | Version-specific CellMarker 2.0 XLSX; raw `load`, normalized/query/GMT helpers; `[celltype]` |
| {mod}`biodb.cellxgene` | WMG tissue/type marker queries, CellGuide snapshots, served disease-vs-normal DEG, tissue/disease/type discovery |
| {mod}`biodb.markers` | `load_markers` aggregates selected sources/species/types/tissues and keeps provenance |
| {mod}`biodb.cellxgene_dataset` | Local `build_dataset` with markers and optional leaf/category DEG shards; separate `push_to_hub` |
| {mod}`biodb.hf_datasets` | Curated-source builders, separate `push_to_hub` and `create_collection`; `[hf]` for publication |

## Genes, proteins, gene sets, and interactions

See [raw annotations and enrichment](enrichment.md) for GENCODE, provider
structure JSON, and gene-set examples.

| Module | Targeted / derived access | Bulk / local access |
|---|---|---|
| {mod}`biodb.gencode` | `get_genes` (all biotypes), `fetch_mane_select` (tagged coding subset) | `download_gencode_gtf`; human release-pinned GTF, `pooch` |
| {mod}`biodb.uniprot` | `query_protein`, `get_sequences`, `get_features`, `get_dbxrefs` | Swiss-Prot/TrEMBL FASTA downloads and `iter_fasta_records`; `[protein]` |
| {mod}`biodb.protein_structure` | Raw AlphaFold/RCSB `fetch_provider`, `fetch_structure_for_accession` | Receipt-backed `build_structure_cache` export |
| {mod}`biodb.harmonizome` | Dataset discovery/metadata and compatibility entity clients | Downloaded GMT and gene–attribute matrices |
| {mod}`biodb.gprofiler` | Live `gost` functional enrichment | Annotation `download_gmt` / `load_gmt` |
| {mod}`biodb.msigdb` | `query_gene_set`, `query_genes` | Per-collection/version `download_gmt` / `load_gmt` |
| {mod}`biodb.string` | `physical_ppi_edges` keyed by gene symbol | Physical links and protein-info download/load helpers |
| {mod}`biodb.corum` | `complex_pairs` from curated complex subunits | `list_releases`, `download_complexes`, `load_complexes` |
| {mod}`biodb.coessentiality` | `coessential_pairs`, `pair_p_values` | Wainberg CRISPR-screen matrices and gene lists |
| {mod}`biodb.ensembl_compara` | `paralog_pairs`, `paralog_families` | Ensembl homology dump download/load helpers |

Interaction scores retain their source meanings: STRING physical evidence,
co-complex membership, and coessentiality are separate measurements. GMT
readers preserve set identity and membership; they do not run enrichment
unless an enrichment API is explicitly called.

## Variants, intervals, benchmarks, and tracks

| Module | Access and entry points | Coordinates / dependencies |
|---|---|---|
| {mod}`biodb.clinvar` | Targeted `query_variant` / `query_gene`; VCF download/parse, clinical significance simplification, BED/sites conversion | `[clinvar]` for VCF paths; inspect converter-specific schemas |
| {mod}`biodb.traitgym` | `load_variants`, `list_features`, `load_feature`, split download | Published benchmark at a pinned Hub revision; preserve row alignment with features |
| {mod}`biodb.constraint_tracks` | `track`, `open_track`, `score_intervals`, `score_positions`; GPN-Star download/load | hg38 BigWigs require `[bigwig]`; GPN-Star retains source `pos` plus derived `start` |
| {mod}`biodb.encode_re2g` | Published CRISPRi benchmark download/load and `crispri_pairs` | GRCh38, 0-based half-open tested enhancer intervals |
| {mod}`biodb.tads` | ENCODE discovery/download/load, `boundaries`, `same_domain`, `domain_metadata` | GRCh38 domain intervals; inspect biosample counts for pooled call sets |
| {mod}`biodb.rfam` | Release models/seeds/clans, `scan_fasta`, `load_hits`, `hit_structures`, `seed_structure` | Scanning needs Infernal executables (`cmpress`, `cmscan`); parsed hits are 0-based half-open |
| {mod}`biodb.intervals` | BED I/O, `overlap_join`, `nearest`, `merge` | Polars `chrom/start/end`, 0-based half-open; no liftover |

For example, interval overlap works entirely offline:

```python
import polars as pl
from biodb.intervals import overlap_join

left = pl.DataFrame({"chrom": ["chr1"], "start": [100], "end": [200]})
right = pl.DataFrame({"chrom": ["chr1"], "start": [150], "end": [250]})
overlaps = overlap_join(left, right)
assert overlaps.height == 1
```

Source-specific coordinate contracts take precedence over this interval helper
convention. GWAS Catalog/Harvest retain original positions, and Pan-UKBB is
GRCh37. Check build and coordinate origin before joining sources.

The [genomic-resources notebook](_generated/tutorials/16_genomic_resources.ipynb)
demonstrates local interval and benchmark workflows; the
[source-catalog notebook](_generated/tutorials/17_source_catalog.ipynb) covers
discovery and representative remaining source APIs.

## Shared helpers

| Module | Purpose |
|---|---|
| {mod}`biodb.mapping` | `map_gene_ids` via g:Profiler; `[mapping]` |
| {mod}`biodb.transform` | Public convenience import of `create_gene_association_matrix` |
| {mod}`biodb.utils` | Canonical GMT reader, matrix construction, adaptive filtering, similarities, random seed, optional token counting |

Modules beginning with `_` implement internal acquisition/cache/parsing
machinery. Use the documented source APIs above to obtain their supported
behavior and public result/error types.
