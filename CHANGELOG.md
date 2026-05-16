# Changelog

All notable changes to `phenoref` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-05-15

### Added

* Initial release, extracted from `AoU.phenome.{opentargets, monarch,
  ontology, gene_weighting}` (~13,900 lines of vendored source).
* `opentargets` module — Open Targets Platform downloaders,
  disease/drug/PGx markdown summaries, gene-association matrix
  builders, pathway helpers.
* `monarch` module — Monarch Initiative association readers.
* `ontology` module — N-hop keyword-set expansion, Mondo / OWL
  loaders, hierarchical keyword set generation, attention-weight
  analysis, gene-phenotype matrix construction.
* `gene_weighting` module — fast two-stage gene attention,
  `GeneEmbeddingCache`, temporal / multi-condition weighting.
* `utils` module — `RANDOM_SEED`, `set_random_seed`, `count_tokens`,
  similarity helpers (`l2_normalize`, `cosine_similarity`,
  `euclidean_similarity`, `dot_product_similarity`),
  `create_gene_association_matrix`, `filter_adaptive`. All inlined so
  the package has no AoU dep at runtime.
* Sphinx docs scaffold, CPU + GPU Dockerfiles, multi-version CI matrix.
