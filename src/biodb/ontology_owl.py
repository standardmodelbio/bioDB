"""Back-compat shim — content merged into :mod:`biodb.ontology`.

The generic owlready2 helpers (``get_ontology``, ``get_descendants``,
``get_ancestors``, ``get_mrca``, ``get_id_map``, ``is_label_or_id``,
``map_terms``, …) and OBO URL constants now live inside
:mod:`biodb.ontology` alongside the MONDO-specific loaders. This module
is preserved purely so old call sites that wrote ``from biodb.ontology_owl
import X`` keep working — every public symbol is re-exported.

New code should prefer ``from biodb.ontology import X``.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "biodb.ontology_owl was merged into biodb.ontology; "
    "import directly from biodb.ontology instead.",
    DeprecationWarning,
    stacklevel=2,
)

from biodb.ontology import (  # noqa: F401, E402
    EFO_URL,
    GO_URL,
    HPO_URL,
    MONDO_URL,
    SEQUENCE_ONTOLOGY_URL,
    get_ancestors,
    get_descendants,
    get_id_map,
    get_ids,
    get_labels,
    get_mrca,
    get_mrca_counts,
    get_ontology,
    get_sequence_ontology,
    is_label_or_id,
    map_terms,
)

__all__ = [
    "EFO_URL",
    "GO_URL",
    "HPO_URL",
    "MONDO_URL",
    "SEQUENCE_ONTOLOGY_URL",
    "get_ancestors",
    "get_descendants",
    "get_id_map",
    "get_ids",
    "get_labels",
    "get_mrca",
    "get_mrca_counts",
    "get_ontology",
    "get_sequence_ontology",
    "is_label_or_id",
    "map_terms",
]
