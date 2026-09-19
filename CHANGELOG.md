# Changelog

All notable changes to `biodb` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

* `biodb.tads` — topologically associating domains from ENCODE Hi-C
  contact-domain calls on GRCh38. `list_domain_files` queries the portal
  (reporting every biosample a file's dataset covers, because ENCODE's
  biosample filter matches the dataset rather than the file and so returns
  pooled call sets for a single-cell-type query), `load_domains` reads one
  accession's bedpe as `chrom`/`start`/`end`, `boundaries(flank)` gives the
  domain edges widened and merged, and `same_domain` answers whether one
  domain contains both ends of a pair — the covariate a paired-position
  analysis has to condition on.

* `biodb.coessentiality` — Wainberg et al. 2021 co-essential network:
  `download_matrices` (genes, GLS p, GLS sign; memory-mapped), `load_genes`,
  `coessential_pairs` (positive pairs at a Benjamini–Hochberg FDR computed over
  every pair, or a raw p cutoff, optionally restricted to a gene list) and
  `pair_p_values` for explicit gene pairs.

* `biodb.ensembl_compara` — Ensembl Compara homology dumps (`download_homologies`,
  `load_homologies`) and the human paralogue catalogue derived from them:
  `paralog_pairs` (unordered, deduplicated, `within_species_paralog` +
  `other_paralog`, optional two-sided identity floor) and `paralog_families`
  (connected components of the paralogue graph, the exclusion set a
  coevolution or co-function benchmark needs).

* `biodb.constraint_tracks` — per-base constraint tracks. `TRACKS` pins the
  UCSC phyloP / phastCons bigWigs for hg38 (Zoonomia 241-way, 100-way,
  447-way whole and primates-only, 470-way) and GPN-Star's entropy bigWigs;
  `open_track` reads any of them in place over HTTPS (pyBigWig, `[bigwig]`
  extra) and `score_intervals` / `score_positions` pull only the bases
  asked for. `download_gpn_star` / `load_gpn_star` read GPN-Star's
  canonical per-chromosome Parquet shards (`entropy_calibrated`,
  `llr_calibrated`, `abs_llr_calibrated`) at a pinned dataset revision,
  keeping the published 1-based `pos` beside a derived 0-based `start`.

* `biodb.traitgym` — TraitGym causal-variant benchmarks: `download_split` /
  `load_variants` (chrom normalised to `chrN`), `list_features` and
  `load_feature` (row-aligned predictor features such as `GPN-MSA_absLLR`),
  read from the Hub at a pinned dataset revision over plain HTTPS.
* `biodb.intervals` — BED I/O and interval joins (`read_bed`, `to_bed`,
  `overlap_join`, `nearest`, `merge`) on 0-based half-open `chrom`/`start`/`end`
  polars frames; the cross-cutting helper the coordinate sources share.
* `biodb.corum` — CORUM protein complexes through the public FastAPI:
  `list_releases`, `download_complexes` / `load_complexes` (current or an
  archived version, human or complete), and `complex_pairs` (every
  unordered pair of subunit gene symbols per complex).
* `biodb.encode_re2g` — ENCODE-rE2G CRISPRi enhancer–gene benchmark:
  `download_crispri_benchmark` / `load_crispri_benchmark` (the published
  `EPCrisprBenchmark_ensemble_data_GRCh38.tsv`, pinned to a repository
  revision) and `crispri_pairs` (one labelled row per tested element–gene
  pair, 0-based half-open GRCh38, with TSS distance).
* `biodb.rfam` — Rfam release files (`download_models` / `download_seeds` /
  `download_clanin` at a pinned release), `press_models`, `scan_fasta`
  (Infernal `cmscan --cut_ga --rfam`), `load_hits` (0-based half-open hits
  from `--fmt 2` tblout) and `hit_structures` (each hit's WUSS consensus
  structure projected onto its own bases, local-end markers included), and
  `seed_structure`.
* `ols.find_terms` / `ols.find_term` — ranked term lookup wrapping the
  existing Solr-backed `search` with a deterministic exact-label /
  exact-synonym / prefix / regex re-ranker. Adds an explicit
  `match_quality` ordinal column (0..4) so callers can filter by tier
  (`df[df.match_quality >= 3]`) instead of depending on opaque Solr
  boost configs that drift between OLS releases. `find_term` is the
  singular best-match convenience for "I just need the ID" callers —
  returns `None` on no match rather than raising.
* `ols.ontology_id_from_curie` — centralised CURIE → OLS-slug mapping
  (`MONDO:0007254` → `"mondo"`, `EFO_0000311` → `"efo"`, with non-OBO
  aliases `SCTID`/`SNOMED` → `"snomed"` and `ORPHA`/`ORPHANET` →
  `"ordo"`). Lets downstream callers walk an ontology subtree via
  `get_descendants` without hard-coding the prefix → slug table.
* Dedicated [`docs/ols.md`](https://github.com/bschilder/bioDB/blob/main/docs/ols.md)
  user guide covering ranked search, hierarchy traversal, whole-ontology
  dumps, CURIE conversion, and a worked "expand a disease group"
  example. Includes an honest note on OLS's current lack of
  semantic / RAG search (Solr-only) and where to plug in an external
  embedding index if you need paraphrase-aware lookup.
* `aou_allxall` module — All of Us *All-by-All* PheWAS atlas client
  (~3,602 META phenotypes × ~414k WGS participants, CDR v8 / Feb 2025).
  Public-API REST wrapper for the unauthenticated browser backend at
  https://allbyall.researchallofus.org, captures the full gene-burden
  variant grid (`pLoF` / `missenseLC` / `pLoF;missenseLC` /
  `synonymous` × `0.01` / `0.001` / `0.0001` MAF × Burden / SKAT /
  SKAT-O). Concurrent bulk downloader, per-(phenotype, ancestry,
  max_maf) Parquet cache, consolidated bulk Parquet, and a
  `iter_signature_variants` helper that yields one melted long-frame
  per facet. 28 offline tests + 3 network-tagged live integration
  tests.
* `omicspred` module — [OmicsPred](https://www.omicspred.org/) client
  for the INTERVAL-trained Bayesian-Ridge molecular-trait PRS atlas
  (Xu et al. *Nature* 2023, PMID 36991119). Public REST API wrapper
  (`list_platforms`, `list_datasets`, `get_dataset`, `get_score`,
  `search_scores`, `get_performance`) + Box.com bulk-archive downloader
  for per-dataset metadata Excel and PGS Catalog-format scoring files
  (`download_metadata_excel`, `load_scores_metadata`,
  `load_performances_metadata`, `download_scoring_files`,
  `read_scoring_file`). Includes `melt_scores_to_gene_table` for
  reshaping the cis-gene-labeled scores into a long
  `(OPGS, gene, R²)` frame suitable for downstream gene-association
  matrix builders. 31 offline tests + 2 network-tagged live schema-drift
  tests. Optional `[omicspred]` extra: `pip install biodb[omicspred]`
  pulls in `openpyxl` for the Excel parser.
* `clinvar` module — ClinVar VCF download (via `pooch`), parsing
  (via `genoray`), CLNSIG long-tail → 6-class simplification, BED /
  sites format converters. Adapted from `bschilder/VEP_protein`.
* Optional `[clinvar]` extra: `pip install biodb[clinvar]` pulls in
  `genoray` + `pooch`.

### Fixed

* `filter_adaptive` lost the `source_id_col` on pandas 2.2+ where
  `groupby(..., group_keys=False).apply` stops auto-prepending the
  group key. Replaced with a sort + cumcount + rank-threshold filter
  that's version-agnostic.

### Removed

* `gene_weighting` module — moved to the `timeline_dataset` package
  (operates on patient event embeddings, not on the phenotype
  knowledge graph). Update imports: `from biodb.gene_weighting import …`
  → `from timeline_dataset.gene_weighting import …`.

## [0.1.0] — 2026-05-15

### Added

* Initial release, extracted from `AoU.phenome.{opentargets, monarch,
  ontology}` (~13,900 lines of vendored source).
* `opentargets` module — Open Targets Platform downloaders,
  disease/drug/PGx markdown summaries, gene-association matrix
  builders, pathway helpers.
* `monarch` module — Monarch Initiative association readers.
* `ontology` module — N-hop keyword-set expansion, Mondo / OWL
  loaders, hierarchical keyword set generation, attention-weight
  analysis, gene-phenotype matrix construction.
* `utils` module — `RANDOM_SEED`, `set_random_seed`, `count_tokens`,
  similarity helpers (`l2_normalize`, `cosine_similarity`,
  `euclidean_similarity`, `dot_product_similarity`),
  `create_gene_association_matrix`, `filter_adaptive`. All inlined so
  the package has no AoU dep at runtime.
* Sphinx docs scaffold, CPU + GPU Dockerfiles, multi-version CI matrix.
