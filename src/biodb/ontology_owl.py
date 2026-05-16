"""Generic owlready2-based ontology helpers.

Complements :func:`biodb.ontology.load_mondo_ontology` (which is
MONDO-specific) with **ontology-agnostic** primitives that work for any
OBO Foundry OWL file: Sequence Ontology (SO), HPO, EFO, GO, ChEBI, …

Ported and consolidated from
`VEP_protein <https://github.com/bschilder/VEP_protein>`_'s
``src/ontologies.py`` + ``src/owlready2.py`` with the following
hygiene fixes:

* Logging instead of bare ``print`` (so callers control verbosity).
* String defaults (``return_as="label"``, ``multi="first"``) instead of
  list-as-sentinel; the previous idiom required a ``one_only`` helper
  to pick the first element of a default list.
* MONDO-specific ``get_onto_mondo`` / ``get_onto_icdo`` helpers
  dropped — use :func:`get_ontology` with the URL, or the more
  featureful :func:`biodb.ontology.load_mondo_ontology` for MONDO.
* IRI-based ``get_ancestors`` variant dropped — keep one canonical
  label/id-based walker.

The owlready2 import is lazy so the rest of :mod:`biodb` works without
the ``[ontology]`` extra installed.
"""

from __future__ import annotations

import logging
from functools import partial
from itertools import chain
from typing import Any, Literal

logger = logging.getLogger(__name__)

# Common OBO Foundry ontology URLs — pass as ``url=`` to ``get_ontology``.
SEQUENCE_ONTOLOGY_URL = "http://purl.obolibrary.org/obo/so.owl"
HPO_URL = "http://purl.obolibrary.org/obo/hp.owl"
EFO_URL = "http://www.ebi.ac.uk/efo/efo.owl"
GO_URL = "http://purl.obolibrary.org/obo/go.owl"
MONDO_URL = "http://purl.obolibrary.org/obo/mondo.owl"

ReturnAs = Literal["entity", "label", "id", "id|label"]
Multi = Literal["join", "first", "all"]
KinType = Literal["descendants", "ancestors"]


def get_ontology(url: str = SEQUENCE_ONTOLOGY_URL, **kwargs: Any) -> Any:
    """Load an OWL ontology from ``url`` via :mod:`owlready2`.

    Parameters
    ----------
    url : str
        OWL or OWL/XML URL. Defaults to the Sequence Ontology.
    **kwargs
        Passed through to ``owlready2.get_ontology``.

    Returns
    -------
    owlready2.namespace.Ontology
        Loaded ontology object.
    """
    import owlready2

    return owlready2.get_ontology(url, **kwargs).load()


def get_sequence_ontology(**kwargs: Any) -> Any:
    """Load the OBO Foundry Sequence Ontology (``so.owl``)."""
    return get_ontology(SEQUENCE_ONTOLOGY_URL, **kwargs)


def _multi_handler(lst: list[Any], multi: Multi, sep: str = ".") -> Any:
    if multi == "join":
        return sep.join(lst)
    if multi == "first":
        return lst[0]
    if multi == "all":
        return lst
    raise ValueError(f"Invalid multi: {multi!r} (expected 'join', 'first', or 'all')")


def _return_as(x: Any, return_as: ReturnAs, multi: Multi = "first") -> Any:
    import owlready2

    if not isinstance(x, owlready2.entity.ThingClass):
        raise ValueError(f"Entity must be owlready2.entity.ThingClass, got {type(x)}")
    if return_as == "entity":
        return x
    if return_as == "label":
        return _multi_handler(x.label, multi)
    if return_as == "id":
        return _multi_handler(x.id, multi)
    if return_as == "id|label":
        return f"{_multi_handler(x.id, multi)}|{_multi_handler(x.label, multi)}"
    raise ValueError(
        f"Invalid return_as: {return_as!r} (expected 'entity', 'label', 'id', or 'id|label')"
    )


def get_labels(ont: Any, unnest: bool = True) -> list[Any]:
    """Return every class label in ``ont``.

    Parameters
    ----------
    ont : owlready2.namespace.Ontology
    unnest : bool, default True
        If True, flatten the list-of-lists from owlready2 (each class
        can carry multiple labels).
    """
    labels = [x.label for x in ont.classes()]
    if unnest:
        return list(chain.from_iterable(labels))
    return labels


def get_ids(ont: Any, unnest: bool = True) -> list[Any]:
    """Return every class id in ``ont`` (see :func:`get_labels`)."""
    ids = [x.id for x in ont.classes()]
    if unnest:
        return list(chain.from_iterable(ids))
    return ids


def get_id_map(ont: Any, first_only: bool = True) -> dict[str, Any]:
    """Map ``CURIE`` id → label for every class in ``ont``.

    Builds the mapping from ``entity.name`` (turning the underscore
    back into a colon) so e.g. ``MONDO_0007254 → "MONDO:0007254"``.
    """
    return {
        entity.name.replace("_", ":"): (entity.label.first() if first_only else list(entity.label))
        for entity in ont.classes()
    }


def is_label_or_id(label_or_id: str, ont: Any | None = None) -> str | None:
    """Return ``"id"``, ``"label"``, or ``None`` based on which class
    ``label_or_id`` matches in ``ont``.

    If ``ont`` is None, the Sequence Ontology is loaded as a fallback
    (matches the VEP_protein default).
    """
    if ont is None:
        ont = get_sequence_ontology()
    if label_or_id in get_ids(ont):
        return "id"
    if label_or_id in get_labels(ont):
        return "label"
    return None


def _get_kin(
    label_or_id: str | list[str],
    ont: Any,
    kin_type: KinType,
    include_self: bool = True,
    return_as: ReturnAs = "label",
    verbose: bool = False,
) -> Any:
    if isinstance(label_or_id, list) and len(label_or_id) == 1:
        label_or_id = label_or_id[0]
    if isinstance(label_or_id, list):
        return {
            term: _get_kin(
                term,
                ont=ont,
                kin_type=kin_type,
                include_self=include_self,
                return_as=return_as,
                verbose=verbose,
            )
            for term in label_or_id
        }
    if is_label_or_id(label_or_id, ont) == "label":
        entity = ont.search_one(label=label_or_id)
    else:
        entity = ont.search_one(id=label_or_id)
    if entity is None:
        if verbose:
            logger.info("No entity found for %r", label_or_id)
        return label_or_id if include_self else None
    kin = (
        entity.descendants(include_self=include_self)
        if kin_type == "descendants"
        else entity.ancestors(include_self=include_self)
    )
    if verbose:
        logger.info("Found %d %s of %r", len(kin), kin_type, label_or_id)
    return list(map(partial(_return_as, return_as=return_as), kin))


def get_descendants(
    label_or_id: str | list[str],
    ont: Any | None = None,
    include_self: bool = True,
    return_as: ReturnAs = "label",
    verbose: bool = False,
) -> Any:
    """All descendants (sub-classes) of ``label_or_id`` in ``ont``.

    Parameters
    ----------
    label_or_id : str or list[str]
        Class label or id. A list is mapped recursively into a dict.
    ont : owlready2.namespace.Ontology, optional
        Defaults to the Sequence Ontology.
    include_self : bool, default True
        Include the seed class itself.
    return_as : {"entity", "label", "id", "id|label"}
        How to render each returned class.
    """
    if ont is None:
        ont = get_sequence_ontology()
    return _get_kin(
        label_or_id,
        ont=ont,
        kin_type="descendants",
        include_self=include_self,
        return_as=return_as,
        verbose=verbose,
    )


def get_ancestors(
    label_or_id: str | list[str],
    ont: Any | None = None,
    include_self: bool = True,
    return_as: ReturnAs = "label",
    verbose: bool = False,
) -> Any:
    """All ancestors (super-classes) of ``label_or_id`` in ``ont``.

    Mirror of :func:`get_descendants`.
    """
    if ont is None:
        ont = get_sequence_ontology()
    return _get_kin(
        label_or_id,
        ont=ont,
        kin_type="ancestors",
        include_self=include_self,
        return_as=return_as,
        verbose=verbose,
    )


def map_terms(
    terms: str,
    ont: Any | None = None,
    return_as: ReturnAs = "label",
) -> Any:
    """Look up a single label-or-id and render it via :func:`_return_as`."""
    if ont is None:
        ont = get_sequence_ontology()
    if is_label_or_id(terms, ont) == "label":
        entity = ont.search_one(label=terms)
    else:
        entity = ont.search_one(id=terms)
    return _return_as(entity, return_as)


def get_mrca(ont: Any, id1: str, id2: str) -> Any | None:
    """Most-recent common ancestor of two classes (by id).

    Returns the owlready2 class with the **deepest** ancestor chain
    (i.e. furthest from the root), or ``None`` if either class is
    missing or there's no common ancestor.

    Parameters
    ----------
    ont : owlready2.namespace.Ontology
    id1, id2 : str
        Class ids (e.g. ``"MONDO:0007254"``). The colon is converted
        to an underscore so the owlready2 IRI lookup succeeds.
    """
    class1 = ont.search_one(iri=f"*{id1.replace(':', '_')}")
    class2 = ont.search_one(iri=f"*{id2.replace(':', '_')}")
    if not class1 or not class2:
        return None
    common = set(class1.ancestors()) & set(class2.ancestors())
    if not common:
        return None
    return max(common, key=lambda c: len(list(c.ancestors())))


def get_mrca_counts(
    ont: Any,
    ids: list[str],
    verbose: bool = True,
) -> list[tuple[str, int]]:
    """How often each ontology class appears as the MRCA of an id pair.

    Iterates over all unordered ``(id_i, id_j)`` pairs from ``ids``,
    computes :func:`get_mrca`, and tallies the resulting MRCA CURIEs.

    Parameters
    ----------
    ont : owlready2.namespace.Ontology
    ids : list[str]
        Class ids (e.g. ``["MONDO:0007254", "MONDO:0005148", …]``).
    verbose : bool, default True
        Show a tqdm progress bar.

    Returns
    -------
    list[tuple[str, int]]
        ``[(mrca_curie, count), …]`` sorted by descending count.
    """
    from tqdm import tqdm

    mrca_counts: dict[str, int] = {}
    for i in tqdm(
        range(len(ids)),
        desc="Computing MRCA counts",
        disable=not verbose,
        leave=False,
    ):
        for j in range(i + 1, len(ids)):
            mrca = get_mrca(ont, ids[i], ids[j])
            if mrca is None:
                continue
            mrca_id = mrca.iri.split("/")[-1].replace("_", ":")
            mrca_counts[mrca_id] = mrca_counts.get(mrca_id, 0) + 1
    return sorted(mrca_counts.items(), key=lambda x: x[1], reverse=True)
