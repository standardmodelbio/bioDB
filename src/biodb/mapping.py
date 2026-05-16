"""Gene-ID namespace mapping via gProfiler.

A thin wrapper around the gProfiler ``convert`` API that lets you map
between any pair of gene-id namespaces (Ensembl, HGNC, Entrez, UniProt,
RefSeq, …) for any supported organism.

Ported from ``AoU.utils.map_gene_ids`` with these hygiene fixes:

* Lazy ``gprofiler`` import (the dep is opt-in via the ``[mapping]`` extra
  so the rest of :mod:`biodb` works without it).
* Logging instead of bare ``print`` for verbose mode.
* Returns ``None`` when ``df`` is empty instead of round-tripping through
  gProfiler.

Examples
--------
>>> import pandas as pd
>>> from biodb.mapping import map_gene_ids
>>> df = pd.DataFrame({  # doctest: +SKIP
...     "targetId": ["ENSG00000157764", "ENSG00000141510", "ENSG00000139618"],
...     "score": [0.8, 0.6, 0.9],
... })
>>> map_gene_ids(df, target_id_col="targetId", target_namespace="HGNC")  # doctest: +SKIP
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def map_gene_ids(
    df: pd.DataFrame,
    target_id_col: str = "targetId",
    target_namespace: str = "HGNC",
    organism: str = "hsapiens",
    verbose: bool = True,
) -> pd.DataFrame:
    """Map gene identifiers in ``df[target_id_col]`` to ``target_namespace``.

    Parameters
    ----------
    df : pd.DataFrame
        Frame containing the gene identifiers to convert.
    target_id_col : str, default ``"targetId"``
        Column name holding the source gene IDs.
    target_namespace : str, default ``"HGNC"``
        Target gene-ID namespace. Common values:

        * ``"HGNC"`` — HUGO gene symbols (e.g. ``"BRCA1"``)
        * ``"ENSG"`` — Ensembl gene IDs
        * ``"ENTREZGENE_ACC"`` — Entrez gene IDs
        * ``"UNIPROTSWISSPROT"`` — UniProt accessions
        * ``"REFSEQ_MRNA"`` — RefSeq mRNA accessions

        See https://biit.cs.ut.ee/gprofiler/page/namespaces-list for the
        full list.
    organism : str, default ``"hsapiens"``
        gProfiler organism code (``"hsapiens"`` / ``"mmusculus"`` /
        ``"rnorvegicus"`` / …).
    verbose : bool, default True
        Log progress + mapping rate.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with an extra column named ``target_namespace``.
        Unmapped IDs fall back to the original ``target_id_col`` value so
        rows are never dropped.

    Raises
    ------
    ImportError
        If ``gprofiler-official`` is not installed. Install with
        ``pip install "biodb[mapping]"``.
    ValueError
        If ``target_id_col`` isn't in ``df``.
    """
    if target_id_col not in df.columns:
        raise ValueError(
            f"Column {target_id_col!r} not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    if df.empty:
        return df.copy()

    try:
        from gprofiler import GProfiler
    except ImportError as exc:
        raise ImportError(
            "gprofiler is required for map_gene_ids. Install with: pip install 'biodb[mapping]'"
        ) from exc

    if verbose:
        logger.info(
            "Mapping gene IDs from %r → %r (organism=%s); %d unique IDs",
            target_id_col,
            target_namespace,
            organism,
            df[target_id_col].nunique(),
        )

    unique_gene_ids = df[target_id_col].unique().tolist()
    gp = GProfiler(return_dataframe=True)
    gene_map = gp.convert(
        organism=organism,
        query=unique_gene_ids,
        target_namespace=target_namespace,
    )

    # gprofiler returns string ``"None"`` for unmapped — flip back to actual None.
    gene_map["converted"] = gene_map["converted"].replace("None", None)
    # Prefer the converted value; fall back to the incoming ID so we never lose rows.
    gene_map[target_namespace] = gene_map["converted"].fillna(gene_map["incoming"])

    mapping_dict: dict[Any, Any] = {}
    for incoming_id, group in gene_map.groupby("incoming"):
        converted_values = group[group["converted"].notna()][target_namespace].unique()
        mapping_dict[incoming_id] = (
            converted_values[0] if len(converted_values) > 0 else group["incoming"].iloc[0]
        )

    out = df.copy()
    out[target_namespace] = out[target_id_col].map(mapping_dict)
    # Fallback for IDs not in the gProfiler response at all.
    out[target_namespace] = out[target_namespace].fillna(out[target_id_col])

    if verbose:
        mapped = (out[target_namespace] != out[target_id_col]).sum()
        rate = mapped / len(out) * 100 if len(out) > 0 else 0
        logger.info(
            "  Mapped %d / %d rows (%.1f%%); %d unique %s values",
            mapped,
            len(out),
            rate,
            out[target_namespace].nunique(),
            target_namespace,
        )

    return out


__all__ = ["map_gene_ids"]
