# bioDB

Biomedical and genomic data access, with versioned bulk acquisition, targeted queries, and reusable source readers.

[![CI](https://github.com/standardmodelbio/bioDB/actions/workflows/ci.yml/badge.svg)](https://github.com/standardmodelbio/bioDB/actions/workflows/ci.yml)
[![Docs](https://github.com/standardmodelbio/bioDB/actions/workflows/docs.yml/badge.svg)](https://standardmodel.bio/bioDB/)
[![Docker](https://github.com/standardmodelbio/bioDB/actions/workflows/docker.yml/badge.svg)](https://github.com/standardmodelbio/bioDB/pkgs/container/biodb)
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
| **GENCODE transcripts** — every annotated biotype, exon/CDS/UTR structure, native versions and RNA/protein sequences | [`biodb.gencode`](src/biodb/gencode.py) | `query_transcript` in a pinned cached release | `download_bundle`, `iter_transcripts`; disk-backed `SequenceStore` |
| **APPRIS / TRIFID** — protein-isoform features, principal status and scores | [`biodb.appris`](src/biodb/appris.py) | `query_transcript` in a pinned cached release | `get_dataset`, streaming score/feature readers |
| **MANE** — exact ENST–RefSeq transcript pairs and clinical reference flags | [`biodb.mane`](src/biodb/mane.py) | `query_transcript` in a pinned cached release | MANE 1.5 summary and RNA GenBank files |
| **RefSeq RNA** — curated transcript-variant descriptions with gene context separate | [`biodb.refseq`](src/biodb/refseq.py) | Versioned-accession `query_transcript` through EFetch | Pinned GRCh38 RNA GenBank dump and streaming parser |
| **GTEx transcript expression** — tissue TPM, detection and within-gene usage | [`biodb.gtex`](src/biodb/gtex.py) | Sequential V2 `query_expression` | Public file manifest, original matrices and two-pass disk-backed tissue summaries |
| **[Open Targets Platform](https://platform-docs.opentargets.org/)** — gene-disease-drug associations, evidence, expression, essentiality, pharmacogenomics, pathways, plus bulk variant fine-mapping (credible sets) and per-variant VEP scores | [`biodb.opentargets`](src/biodb/opentargets/) | ✅ GraphQL | ✅ FTP Parquet, versioned cache — incl. [`opentargets.variants.get_credible_set`](src/biodb/opentargets/variants.py), [`opentargets.variants.get_variant_effects`](src/biodb/opentargets/variants.py), [`opentargets.studies.get_study`](src/biodb/opentargets/studies.py) |
| **[Monarch Initiative](https://monarchinitiative.org/)** — causal gene-to-disease, phenotype-to-disease, full KG | [`biodb.monarch`](src/biodb/monarch.py) | ✅ BioLink v3 REST + Neo4j Cypher against the public KG | ✅ Per-association TSVs |
| **[Ontology Lookup Service (OLS)](https://www.ebi.ac.uk/ols4/)** — EBI-hosted gateway to ~280 OBO Foundry ontologies (Mondo, HPO, EFO, SO, GO, ChEBI, SNOMED, …) | [`biodb.ols`](src/biodb/ols.py), [`biodb.ontology`](src/biodb/ontology.py) | ✅ OLS4 REST — no local file needed, plus a generic owlready2 entity walk for callers with a local OWL file | ✅ OBO / OWL download + parse, N-hop expansion, hierarchical keyword sets, attention-weight analysis, gene-phenotype matrix, ontological similarity |
| **[UniProt](https://www.uniprot.org/)** — protein sequences, features, cross-references | [`biodb.uniprot`](src/biodb/uniprot.py) | ✅ REST | ✅ FTP FASTA |
| **[Harmonizome](https://maayanlab.cloud/Harmonizome/)** — ~114 curated gene-attribute datasets (CCLE, GTEx, ENCODE, HPA, KEGG, Reactome, …) | [`biodb.harmonizome`](src/biodb/harmonizome.py) | ✅ REST | ✅ Bulk TSV/GMT |
| **[ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/)** — clinical significance + review status for human variants | [`biodb.clinvar`](src/biodb/clinvar.py) | ✅ NCBI E-utilities | ✅ VCF download |
| **[GTR (Genetic Testing Registry)](https://www.ncbi.nlm.nih.gov/gtr/)** — NCBI catalog of ~64k genetic tests: per-test gene panels (Entrez IDs), conditions (MedGen CUIs), methodology, clinical validity/utility | [`biodb.gtr`](src/biodb/gtr.py) | ✅ NCBI E-utilities | ✅ FTP TSV + full XML, plus gene-set views and embeddable panel text |
| **GENCODE** — release-pinned human gene annotations, including all biotypes and MANE Select views | [`biodb.gencode`](src/biodb/gencode.py) | Cached gene and transcript views | GTF acquisition; zero-based half-open gene coordinates |
| **[GWAS Catalog](https://www.ebi.ac.uk/gwas/)** — exact harmonised source files, all files per study, raw statistics and provenance | [`biodb.gwas_catalog`](src/biodb/gwas_catalog.py) | Per-file streaming and explicit region queries | Complete manifest acquisition, resumable per-file ingestion; genome-wide/unfiltered by default |
| **[Europe PMC](https://europepmc.org/)** — original MED bibliographic records | [`biodb.europepmc`](src/biodb/europepmc.py) | PMID batches with completion receipts | Resumable acquisition of a caller-supplied PMID list |
| **[AlphaFold](https://alphafold.ebi.ac.uk/) / [RCSB](https://www.rcsb.org/)** — original protein structure metadata | [`biodb.protein_structure`](src/biodb/protein_structure.py) | Per-accession, per-provider responses | Resumable accession-list acquisition |
| **Harvest manifests** — caller-supplied lists of exact GWAS source files | [`biodb.gwas_harvest`](src/biodb/gwas_harvest.py) | Per-file streaming and local-source resolution | Resumable ingestion with source identity, scope and output receipts |
| **[Pan-UKBB](https://pan.ukbb.broadinstitute.org/)** — multi-ancestry summary statistics in GRCh37 coordinates | [`biodb.panukbb`](src/biodb/panukbb.py) | Phenotype selection and indexed region queries | Public manifest and remote tabix statistics slices; CLI or optional `pysam` backend |
| **[GWAS Atlas](https://atlas.ctglab.nl/)** — Watanabe et al. per-study gene-level MAGMA p-values across ~4k GWAS summary stats | [`biodb.gwas_atlas`](src/biodb/gwas_atlas.py) | ✅ Per-trait lookup via cached metadata | ✅ Bulk download |
| **[All of Us *All-by-All*](https://allbyall.researchallofus.org/)** — PheWAS atlas across ~3,600 phenotypes × ~414k WGS participants (lab / phecode / phecodeX / PFHH / drug / physical) with single-variant + gene-burden tests (pLoF / missenseLC / synonymous), by ancestry + meta | [`biodb.aou_allxall`](src/biodb/aou_allxall.py) | ✅ Public REST | ✅ Per-phenotype gene-burden tables + concurrent bulk pull |
| **[OmicsPred](https://www.omicspred.org/)** — Xu et al. *Nature 2023*; Bayesian-Ridge PRS atlas for ~17k+ molecular traits (Olink + SomaScan proteins, Metabolon + Nightingale metabolites, Illumina RNA-seq) trained on the INTERVAL cohort. Each score is already mapped to its cis Ensembl gene / UniProt protein. | [`biodb.omicspred`](src/biodb/omicspred.py) | ✅ Public REST | ✅ Box.com archives — PGS Catalog scoring-file format |
| **[Orphanet / Orphadata](https://www.orphadata.com/)** — rare-disease epidemiology: prevalence per ORPHAcode as a class band, by prevalence type, geography and validation status (CC-BY-4.0) | [`biodb.orphanet`](src/biodb/orphanet.py) | — | ✅ `en_product9_prev` XML streamed to a long table; `prevalence_map` gives one per-individual `K` per disease for liability-threshold models |
| **[ENCODE Hi-C](https://www.encodeproject.org/)** — topologically associating domains (Arrowhead contact domains, GRCh38) | [`biodb.tads`](src/biodb/tads.py) | 🚧 [bioDB#TBD] | ✅ Domain calls by accession, boundaries with flank, pair co-membership |
| **[TraitGym](https://huggingface.co/datasets/songlab/TraitGym)** — causal regulatory variant benchmarks (Mendelian OMIM and fine-mapped complex-trait positives, negatives matched on consequence / TSS distance / MAF) with precomputed predictor features (CADD, GPN-MSA, Enformer, Borzoi, Evo 2, …) | [`biodb.traitgym`](src/biodb/traitgym.py) | ✅ `list_features` | ✅ Split variant tables and row-aligned feature tables from the Hub at a pinned revision |
| **[CORUM](https://mips.helmholtz-muenchen.de/corum/)** — manually curated mammalian protein complexes (5.3: 7,812 human complexes) | [`biodb.corum`](src/biodb/corum.py) | ✅ `list_releases` | ✅ Complex table current or by archived version via the public API; `complex_pairs` gives every co-complex gene pair (CC BY-NC-SA, download only) |
| **[ENCODE-rE2G](https://github.com/EngreitzLab/ENCODE_Test_Dataset_Analysis)** — the CRISPRi enhancer–gene benchmark (Gschwind et al. 2023): 10,412 tested element–gene pairs in K562 with effect sizes, adjusted p-values and power, positives and tested negatives alike | [`biodb.encode_re2g`](src/biodb/encode_re2g.py) | — | ✅ Benchmark TSV pinned to a repository revision, as a labelled pair table with TSS distance |
| **[Rfam](https://rfam.org/)** — RNA families: covariance models, seed alignments with consensus structure, clan membership; plus Infernal `cmscan` annotation of any FASTA with each hit's structure projected onto its own bases | [`biodb.rfam`](src/biodb/rfam.py) | — | ✅ Release files from the EBI FTP at a pinned release; `scan_fasta` / `load_hits` / `hit_structures` (needs Infernal on PATH) |
| **[STRING](https://string-db.org/)** — protein–protein interaction network; the physical sub-network (direct-binding edges) with continuous combined-score weights | [`biodb.string`](src/biodb/string.py) | 🚧 [bioDB#TBD] | ✅ Bulk physical PPI edge list keyed by gene symbol, combined-score weighted |
| **[Co-essentiality network](https://mitra.stanford.edu/bassik/coessentiality/)** — Wainberg et al. 2021: DepMap CRISPR fitness profiles correlated by GLS across cell lines; the network as gene pairs at a BH FDR | [`biodb.coessentiality`](src/biodb/coessentiality.py) | ✅ Per-pair lookup (`pair_p_values`) | ✅ Whole matrices (2 × 2.5 GB, memory-mapped) |
| **[Ensembl Compara](https://www.ensembl.org/info/genome/compara/homology_method.html)** — gene-tree homologies; the human paralogue catalogue as gene pairs and connected families | [`biodb.ensembl_compara`](src/biodb/ensembl_compara.py) | 🚧 [bioDB#TBD] (REST `homology/id`) | ✅ Per-species homology TSV dump per release |
| **[UCSC conservation](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/) + [GPN-Star](https://huggingface.co/datasets/songlab/gpn-star-scores)** — per-base constraint: phyloP / phastCons bigWigs over the 241-, 100-, 447- and 470-way hg38 alignments, and GPN-Star calibrated entropy / LLR (Song lab) | [`biodb.constraint_tracks`](src/biodb/constraint_tracks.py) | ✅ Range reads over HTTPS (`open_track`, `score_intervals`, `score_positions`) | ✅ Whole-track download; GPN-Star per-chromosome Parquet at a pinned revision |
| **[gProfiler](https://biit.cs.ut.ee/gprofiler/)** — University of Tartu gene-set library + functional enrichment | [`biodb.gprofiler`](src/biodb/gprofiler.py) | ✅ REST | ✅ Bulk GMT |
| **[MSigDB](https://www.gsea-msigdb.org/gsea/msigdb/)** — Broad Institute Molecular Signatures DB (Hallmark + C1–C8) | [`biodb.msigdb`](src/biodb/msigdb.py) | ✅ Per-set JSON | ✅ Bulk GMT for any collection/version |
| **[PubMed](https://pubmed.ncbi.nlm.nih.gov/)** — NLM's 40 M+ biomedical citation database (titles, abstracts, authors, MeSH) | [`biodb.pubmed`](src/biodb/pubmed.py) | ✅ NCBI E-utilities | ✅ Annual Baseline + Daily Update XML.gz |
| **[SNOMED CT](https://www.snomed.org/)** (OHDSI-flavoured) — clinical terminology: diagnoses, procedures, findings, body sites | [`biodb.snomed`](src/biodb/snomed.py) | ✅ Per-concept lookups via OLS4 | ✅ OLS concept/text exports and local CONCEPT.csv / ZIP parsers; use licensed local vocabulary bundles where required. |
| **[Cell Taxonomy](https://ngdc.cncb.ac.cn/celltaxonomy/)** (CNCB-NGDC) — curated cross-species cell-type marker genes with **native Cell Ontology (CL) ids**; ~26k markers / ~3.1k cell types / 34 species | [`biodb.celltaxonomy`](src/biodb/celltaxonomy.py) | ✅ In-memory `query_markers` by CL id / name | ✅ Bulk TSV → normalized ranked gene lists per CL term (score = literature support), GMT export |
| **[CellMarker 2.0](https://bio-bigdata.hrbmu.edu.cn/CellMarker2.0/)** — manually curated human/mouse cell-type markers (~83k entries) carrying a native `cellontology_id` | [`biodb.cellmarker`](src/biodb/cellmarker.py) | ✅ In-memory `query_markers` by CL id / name | ✅ Bulk XLSX → normalized ranked gene lists per CL term (score = PMID support), GMT export |
| **[CZ CELLxGENE Discover](https://cellxgene.cziscience.com/)** — single-cell corpus; CZI's precomputed **Marker Score** per (CL cell type × tissue) + served **disease-vs-normal differential expression** — all via REST | [`biodb.cellxgene`](src/biodb/cellxgene.py) | ✅ `query_markers` (Marker Score) + `disease_vs_normal` (DE effect size / logFC) per CL term / tissue | ✅ `get_tissue_markers` / `get_all_markers` (whole corpus per species); GMT export |

### Cross-cutting helpers

* [`biodb.intervals`](src/biodb/intervals.py) — `read_bed` / `to_bed`, `overlap_join`, `nearest`, `merge` — 0-based half-open GRCh38 interval joins shared by the coordinate sources (no liftover by design)
* [`biodb.mapping`](src/biodb/mapping.py) — `map_gene_ids` — gProfiler-backed cross-namespace gene-ID conversion (Ensembl ↔ HGNC ↔ Entrez ↔ UniProt …)
* [`biodb.transform`](src/biodb/transform.py) — `create_gene_association_matrix` — pivot a long `(sourceId, targetId, score)` DataFrame into a sparse/dense (samples × genes) matrix with AnnData-shaped metadata
* [`biodb.utils`](src/biodb/utils.py) — `read_gmt` (GMT format reader used by MSigDB / Harmonizome / gProfiler), `filter_adaptive` (per-sample top-percentile keep), similarity helpers

* [`biodb.markers`](src/biodb/markers.py) — common ranked-marker views across cell sources
* [`biodb.cellxgene_dataset`](src/biodb/cellxgene_dataset.py) and [`biodb.hf_datasets`](src/biodb/hf_datasets.py) — local dataset builds and cards, with explicit optional Hub publication
* [`biodb.ols_text`](src/biodb/ols_text.py) — complete OLS text records and audited, resumable page caches
* [`biodb.opentargets.literature`](src/biodb/opentargets/literature.py) and [`biodb.opentargets.constants`](src/biodb/opentargets/constants.py) — raw literature matches and upstream label/source snapshots

## Install

Until released to PyPI, install from git:

```bash
pip install git+https://github.com/standardmodelbio/bioDB
```

For the optional extras:

```bash
pip install "biodb[ontology,viz,tokens,gget,protein,clinvar]"
```

Local development:

```bash
git clone https://github.com/standardmodelbio/bioDB
cd bioDB
pip install -e ".[dev]"
```

## Quickstart

### Targeted API queries

```python
from biodb.opentargets_graphql import query_target, query_disease

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

### Transcript / isoform inputs for IsoDocs

Install `biodb[transcripts]` for GenBank, streaming UniProt JSON and sequence
caches. `gencode.download_bundle("50")` fetches comprehensive CHR GTF,
transcript/protein FASTA and annotation metadata. `gencode.iter_transcripts`
streams one gene at a time and keeps all biotypes; the older
`fetch_mane_select` API is unchanged. Native versions and `_PAR_Y` suffixes
are preserved, and coordinates are 0-based half-open internally.

`uniprot.query_protein_json`, `download_human_entries` and
`iter_isoform_annotations` preserve isoform-qualified comments and mappings.
The bulk client checks `X-UniProt-Release` against the requested release;
it fails rather than silently mixing a newer live snapshot into an old build.
Source downloads retain original bytes plus `.source.json` manifests containing
the source, release, URL, assembly, acquisition time and SHA-256 checksum.

`appris.download_features` fetches all method GTFs (domains, residues,
conservation, structure, topology, signal sequences and peptide evidence).
`appris_data.annot.gtf.gz` is the input annotation, not those method results.
Recorded bounded source contracts are in `tests/fixtures/transcripts`, with
URLs, acquisition dates, release labels and excerpt hashes. Opt-in live checks
use `ISODOCS_NETWORK_TESTS=1 pytest -m network tests/test_transcript_contracts.py`.

For GTEx V11 summaries, provide the **complete GENCODE47** transcript→gene map
to `iter_expression_summaries`, not the target GENCODE50 subset. Usage is computed
per sample before tissue aggregation. The disk-backed denominator matrix is
intended for a compute worker; do not run the whole human matrix on a laptop.

Transcript summaries also retain TPM/usage quartiles, detection counts,
gene-detection support and the number of quantified siblings. RefSeq parsing
separates adjacent summary/variant/publication labels and exposes informative
RNA/regulatory features with native locations, fuzzy/compound boundaries,
evidence qualifiers and structured attributes. Propagated protein annotations
are not promoted to measured RNA mechanisms. Full original fields remain raw.

`biodb.transcript_annotations.iter_annotations(path, release=...)` accepts
additional provenance-bearing JSONL study annotations. Assertions declare their
biological kind, molecular scope, evidence type, source URL and optional
conditions/metrics/publications. It validates this interchange; the consumer
still must establish transcript/RNA/protein identity before attaching evidence.
Keep transcript and protein assertions in separate rows; protein assertions
require a peptide sequence for explicit correspondence. Normalizer capability
versions allow consumers to reject stale cached derivations after an upgrade.

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

### Cell-type marker genes (ranked, Cell Ontology-keyed)

```python
from biodb import celltaxonomy as ct

# Bulk: normalized (species, tissue, cell type, CL id, gene, score, rank).
markers = ct.get_markers(species="Homo sapiens", map_to_cl=True)

# Targeted: ranked markers for one cell type (CL:0000540 = neuron).
neuron = ct.query_markers("CL:0000540")
ct.to_gmt("celltaxonomy.gmt")               # one ranked gene set per CL term

# CellMarker 2.0 has the same API surface.
from biodb import cellmarker as cm
cm.query_markers("CL:0000540", which="human")

# CELLxGENE: CZI's precomputed Marker Score via the WMG REST API (no extra deps).
from biodb import cellxgene as cx
cx.query_markers("CL:0000236", tissue="spleen", n_top=25)   # B-cell markers
cx.get_all_markers(organism="Homo sapiens")                 # whole corpus, one species

# Disease-vs-normal differential per cell type (CZI's served DE, genome-wide).
cx.list_diseases("lung")                                    # what's available
cx.disease_vs_normal("CL:0000082", tissue="lung",
                     disease="interstitial lung disease", n_top=25)
```

## Complete and resumable raw acquisition

OpenTargets fine-mapped credible sets come from `biodb.opentargets`; original
GWAS summary statistics come from `biodb.gwas_catalog`. The latter does not
restrict acquisition to OpenTargets loci. Genome-wide/unfiltered is the default;
request `p_max=5e-8` for a significant subset, or explicit regional scope.

```python
from biodb import gwas_catalog
files = gwas_catalog.list_datasets(manifest="harmonised_list.txt")
# Rerun this call with the same output directory to resume validated progress.
result = gwas_catalog.ingest_sumstats(files, out_dir="gwas-run", p_max=5e-8)
```

Runs persist a frozen input/configuration manifest and validated per-file output
receipts, including successful zero-match files. Failed jobs never count as empty
successes. A failed strict run raises `IngestError` while retaining successful
work; `allow_partial=True` is required to return incomplete results. Consolidation
uses receipts rather than arbitrary Parquet files. Changed input identity, scope,
filters or parser schema requires a new run directory. Worker count may change.

HTTP downloads retain private `.part` files and resume with strong ETag/byte-range
validation where supported. Otherwise only the interrupted file restarts. Final
files are replaced atomically after validation. `force` bypasses finalized-cache
reuse while `resume` independently governs in-progress bytes.

OpenTargets caches now require authoritative shard manifests. A one-shard preview
cannot satisfy a later full request. Complete manifested caches work offline;
legacy cache directories need one remote listing before adoption. Missing/corrupt
shards are repaired; stale extra files are ignored. Default scans validate local
shards; `remote=True` explicitly projects remote Parquet columns without claiming
a complete raw local cache.

`get_credible_set` retains one lead row per credible set and matches its PIP by
variant ID. Missing/conflicting matches return null with `pipStatus`; list order
is never used as identity. `get_credible_set_members` exposes each member's own
statistics and 95/99-percent flags. Raw nested data remain available through
`scan_credible_sets`. `get_traits()`/`get_diseases()` and `get_drugs()` load the
complete disease and drug_molecule tables respectively.

Raw bibliography, OpenTargets literature and structure metadata are available
from `biodb.europepmc`, `biodb.opentargets.literature`, and
`biodb.protein_structure`. Corpus rendering, ranking, exports and worker deployment
remain in the consuming package. No bulk collection is launched by import.

## Tutorials

Executed Jupyter notebooks live under [`tutorials/`](tutorials/). They use small
offline fixtures with the real public readers; separate live recipes show how to
acquire production data. The published [documentation](https://standardmodel.bio/bioDB/)
renders the notebooks, and documentation CI re-executes them with network access
blocked. See [`tutorials/README.md`](tutorials/README.md) for the source-coverage index
and [the source guides](docs/sources.md) for API, optional-dependency and provenance details.

## Used by

- [`BioDocs`](https://github.com/standardmodelbio/BioDocs) — consumes bioDB raw sources to build gene, variant, trait and drug documents and indexes.
- [`hg-horizon`](https://github.com/standardmodelbio/hg-horizon) — consumes fine-mapping, GWAS statistics and annotations for scientific transforms and browser exports.
- [`seqlab`](https://github.com/standardmodelbio/seqlab) — variant analysis pipeline; uses `biodb.opentargets` for gene/variant queries and `biodb.clinvar` for clinical significance annotation.

## License

PolyForm-Noncommercial-1.0.0 — see [LICENSE](LICENSE).
