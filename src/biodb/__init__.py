"""biodb: phenotype-knowledge-graph helpers, ported from ``AoU.phenome``.

Three modules carry the public surface:

* :mod:`biodb.opentargets` -- Open Targets Platform downloaders and
  parsers (disease/drug/PGx/expression/essentiality/pathways), markdown
  summary generators, and gene-association matrix builders.
* :mod:`biodb.monarch` -- Monarch Initiative association readers
  (causal gene-to-disease and friends).
* :mod:`biodb.ontology` -- OBO / OWL / Mondo loaders, N-hop keyword
  set expansion, hierarchical keyword set generation, attention
  analysis, gene-phenotype matrix construction.

Shared utilities live in :mod:`biodb.utils` (random seeding,
similarity helpers, token counting, ``create_gene_association_matrix``,
``filter_adaptive``).
"""

from biodb import monarch, ontology, opentargets, opentargets_graphql, utils

# A small slice of high-frequency public symbols is re-exported at the
# top level for convenience. The full APIs live on the submodules.
from biodb.monarch import (
    get_gene_associations as monarch_get_gene_associations,
)
from biodb.monarch import (
    read_causal_gene_to_disease_association,
)
from biodb.opentargets import (
    ensure_cached_shards,
    get_dataset,
    get_gene_associations,
    get_pathways,
    get_targets,
    list_available_versions,
    list_datasets,
)
from biodb.opentargets_graphql import (
    query_disease,
    query_drug,
    query_target,
    query_variant,
)
from biodb.utils import (
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
    "cosine_similarity",
    "count_tokens",
    "create_gene_association_matrix",
    "dot_product_similarity",
    "ensure_cached_shards",
    "euclidean_similarity",
    "filter_adaptive",
    "get_dataset",
    "get_gene_associations",
    "get_pathways",
    "get_targets",
    "l2_normalize",
    "list_available_versions",
    "list_datasets",
    "monarch",
    "monarch_get_gene_associations",
    "ontology",
    "opentargets",
    "opentargets_graphql",
    "query_disease",
    "query_drug",
    "query_target",
    "query_variant",
    "read_causal_gene_to_disease_association",
    "set_random_seed",
    "utils",
]
