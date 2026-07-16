"""Open Targets Platform access.

``biodb.opentargets`` is the bulk/FTP data-gathering surface. The historical
flat-module API (``get_dataset``, ``list_datasets``, ``ensure_cached_shards``,
``variants_for_target``, …) is re-exported here unchanged from the vendored
:mod:`biodb.opentargets._bulk`. Variant- and study-level readers live in the
:mod:`biodb.opentargets.variants` and :mod:`biodb.opentargets.studies`
submodules.
"""

from __future__ import annotations

from biodb.opentargets import _bulk

# Re-export every name from the vendored bulk module (public and private
# single-underscore alike, e.g. ``_resolve_gene_association_datasets``) so
# that both ``from biodb.opentargets import get_dataset`` and existing
# internal-name test access (``opentargets._resolve_gene_association_datasets``)
# keep working now that the flat module is a package. Dunder attributes
# (``__name__``, ``__file__``, ...) are skipped so the package keeps its own
# module identity. Done programmatically to keep ``_bulk.py`` a verbatim
# vendored copy (no ``__all__`` edit required).
_g = globals()
for _name in dir(_bulk):
    if _name.startswith("__") and _name.endswith("__"):
        continue
    _g[_name] = getattr(_bulk, _name)
del _g, _name

from biodb.opentargets import studies, variants  # noqa: E402,F401

__all__ = [  # noqa: F822 - names injected from _bulk above are intentional
    name for name in dir() if not name.startswith("_")
]
