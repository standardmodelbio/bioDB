"""Tabular → matrix transformations for downstream embedding work.

Currently exposes :func:`create_gene_association_matrix` — the
(samples × genes) sparse-or-dense matrix builder used by every
downstream pipeline (NMF, autoencoder, supervised retrieval, …).

The function lives in :mod:`biodb.utils` as a verbatim AoU port; this
module re-exports it so downstream callers can write the more obvious::

    from biodb.transform import create_gene_association_matrix

while ``from biodb.utils import create_gene_association_matrix``
continues to work for back-compat.
"""

from __future__ import annotations

from biodb.utils import create_gene_association_matrix

__all__ = ["create_gene_association_matrix"]
