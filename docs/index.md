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
- **`gwas_atlas`** — Watanabe et al. GWAS Atlas per-study gene-level
  MAGMA p-values across ~4k summary stats; per-trait lookup +
  (gene × study) MAGMA-P matrix.
- **`aou_allxall`** — All of Us *All-by-All* PheWAS atlas client
  (~3,600 phenotypes × ~414k WGS participants). Public-API gene-burden
  ingest across the full variant grid (pLoF / missenseLC / synonymous,
  three MAF buckets, Burden / SKAT / SKAT-O), no Researcher Workbench
  enrollment required.
- **`omicspred`** — OmicsPred molecular-trait PRS atlas client. REST
  API + Box.com archive downloads for ~17k+ Bayesian-Ridge polygenic
  scores across proteins (Olink + SomaScan), metabolites (Metabolon +
  Nightingale), and RNA expression — each pre-mapped to its cis
  Ensembl gene / UniProt protein.
- **`ols`** — EBI OLS4 REST client for ~280 ontologies (Mondo, EFO,
  HPO, GO, SO, ChEBI, SNOMED CT, …). Ranked term search
  (`find_terms` / `find_term`), hierarchical traversal
  (`get_descendants` / `get_ancestors` / `get_children` / `get_parents`),
  versioned local cache for whole-ontology dumps (`list_terms`), and
  CURIE↔IRI conversion helpers. See [Searching ontologies with OLS](ols.md).
- **`gtr`** — NCBI Genetic Testing Registry client. Targeted E-utilities
  queries (`query_gene` / `query_condition` / `query_test`) plus bulk
  downloads (`download`, `load_test_condition_gene`, streaming
  `iter_full_records`). Materializes curated per-panel gene sets
  (`gene_sets` / `aggregate_gene_sets` / `to_gmt`) and embeddable panel
  text (`panel_text`) with a `support_count` importance prior. See
  [Querying genetic tests with GTR](gtr.md).
- **`traitgym`** — TraitGym causal-variant benchmarks with matched negatives and
  precomputed predictor features, at a pinned Hub revision.
- **`intervals`** — genomic-interval helpers: BED I/O, overlap joins, nearest
  neighbour, merge; 0-based half-open, GRCh38, no liftover.
- **`corum`** — CORUM curated protein complexes, current or archived release,
  and the co-complex gene-pair table they imply.
- **`encode_re2g`** — ENCODE-rE2G CRISPRi enhancer–gene benchmark (Gschwind
  et al. 2023): every tested element–gene pair with its effect size and
  label, as a polars table ready for interval joins.
- **`rfam`** — Rfam RNA families: models, seeds and clans from the EBI FTP,
  and Infernal-driven annotation of a FASTA with per-hit consensus structure.
- **`string`** — STRING protein–protein interaction client. Bulk physical
  sub-network (direct-binding) edges keyed by gene symbol with continuous
  combined-score weights (`download_physical_links`, `load_physical_links`,
  `physical_ppi_edges`).
- **`constraint_tracks`** — per-base constraint. UCSC phyloP / phastCons
  bigWigs for hg38 (241-, 100-, 447-, 470-way) and GPN-Star entropy bigWigs
  read in place over HTTPS (`open_track`, `score_intervals`,
  `score_positions`, `[bigwig]` extra), plus GPN-Star's canonical
  per-chromosome Parquet shards at a pinned revision (`load_gpn_star`).

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
ols
gtr
api
changelog
```
