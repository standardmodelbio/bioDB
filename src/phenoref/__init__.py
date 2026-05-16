"""phenoref: phenotype-knowledge-graph helpers, ported from ``AoU.phenome``.

Four modules carry the public surface:

* :mod:`phenoref.opentargets` -- Open Targets Platform downloaders and
  parsers (disease/drug/PGx/expression/essentiality/pathways), markdown
  summary generators, and gene-association matrix builders.
* :mod:`phenoref.monarch` -- Monarch Initiative association readers
  (causal gene-to-disease and friends).
* :mod:`phenoref.ontology` -- OBO / OWL / Mondo loaders, N-hop keyword
  set expansion, hierarchical keyword set generation, attention
  analysis, gene-phenotype matrix construction.
* :mod:`phenoref.gene_weighting` -- fast two-stage gene attention,
  :class:`GeneEmbeddingCache`, temporal / multi-condition weighting.

Shared utilities live in :mod:`phenoref.utils` (random seeding,
similarity helpers, token counting, ``create_gene_association_matrix``,
``filter_adaptive``).
"""

from phenoref import gene_weighting, monarch, ontology, opentargets, utils

# A small slice of high-frequency public symbols is re-exported at the
# top level for convenience. The full APIs live on the submodules.
from phenoref.gene_weighting import (
    GeneEmbeddingCache,
    GeneWeightingConfig,
    compute_gene_weights_fast,
    memory_efficient_gene_attention,
    multi_condition_gene_weighting,
    temporal_gene_weighting,
)
from phenoref.monarch import (
    get_gene_associations as monarch_get_gene_associations,
)
from phenoref.monarch import (
    read_causal_gene_to_disease_association,
)
from phenoref.opentargets import (
    df_to_markdown,
    diseases_to_markdown,
    drugs_to_markdown,
    get_dataset,
    get_gene_associations,
    get_pathways,
    get_targets,
    list_datasets,
    pharmacogenomics_to_markdown,
)
from phenoref.utils import (
    RANDOM_SEED,
    cosine_similarity,
    count_tokens,
    create_gene_association_matrix,
    dot_product_similarity,
    euclidean_similarity,
    filter_adaptive,
    l2_normalize,
    set_random_seed,
)

__version__ = "0.1.0"

__all__ = [
    "RANDOM_SEED",
    "GeneEmbeddingCache",
    "GeneWeightingConfig",
    "compute_gene_weights_fast",
    "cosine_similarity",
    "count_tokens",
    "create_gene_association_matrix",
    "df_to_markdown",
    "diseases_to_markdown",
    "dot_product_similarity",
    "drugs_to_markdown",
    "euclidean_similarity",
    "filter_adaptive",
    "gene_weighting",
    "get_dataset",
    "get_gene_associations",
    "get_pathways",
    "get_targets",
    "l2_normalize",
    "list_datasets",
    "memory_efficient_gene_attention",
    "monarch",
    "monarch_get_gene_associations",
    "multi_condition_gene_weighting",
    "ontology",
    "opentargets",
    "pharmacogenomics_to_markdown",
    "read_causal_gene_to_disease_association",
    "set_random_seed",
    "temporal_gene_weighting",
    "utils",
]
