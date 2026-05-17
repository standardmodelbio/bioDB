"""UniProt REST client — protein sequences, features, cross-references.

The **API mode** for UniProt; mirrors the dual-mode pattern in the
rest of :mod:`biodb` (one record at a time via REST, with future
bulk-FTP helpers planned).

Ported from `VEP_protein <https://github.com/bschilder/VEP_protein>`_'s
``src/unitprot.py`` with the following hygiene fixes:

* The VEP_protein version returned a Biopython ``SeqIO.parse`` iterator,
  which silently exhausts on first iteration — callers had to "reimport
  the records" to use them twice. Here we materialize to a list on
  every call so the return value is freely reusable.
* HTTP failures raise instead of printing and returning ``None``.
* Logging instead of bare ``print`` for verbose mode.
* Filename typo (``unitprot`` → ``uniprot``) fixed.

Endpoint: https://rest.uniprot.org/uniprotkb/

Examples
--------
>>> from biodb.uniprot import query_protein, get_features
>>> records = query_protein("P12345")  # doctest: +SKIP
>>> features = get_features("P12345")  # doctest: +SKIP
"""

from __future__ import annotations

import logging
from io import BytesIO
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

UNIPROT_REST_API = "https://rest.uniprot.org/uniprotkb"
"""UniProt REST root. Append ``/<accession>.<format>``."""

_DEFAULT_TIMEOUT_S: float = 30.0


def _fetch_records(
    uniprot_id: str,
    fmt: str = "xml",
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> list[Any]:
    """GET ``{UNIPROT_REST_API}/{uniprot_id}.{fmt}`` and parse to a list
    of Biopython records.

    Raises
    ------
    requests.HTTPError
        On non-2xx response.
    """
    from Bio import SeqIO

    url = f"{UNIPROT_REST_API}/{uniprot_id}.{fmt}"
    response = requests.get(url, timeout=timeout_s)
    response.raise_for_status()
    parser = "uniprot-xml" if fmt == "xml" else fmt
    # Biopython's uniprot-xml reader requires binary mode (it calls
    # ``defusedxml.ElementTree.iterparse`` which expects bytes). Use
    # ``response.content`` + ``BytesIO``, not ``response.text`` + ``StringIO``.
    return list(SeqIO.parse(BytesIO(response.content), parser))


def query_protein(
    uniprot_id: str,
    *,
    fmt: str = "xml",
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    verbose: bool = False,
) -> list[Any]:
    """Fetch one protein from UniProt and return parsed Biopython records.

    Parameters
    ----------
    uniprot_id : str
        UniProt accession (e.g. ``"P12345"``).
    fmt : str, default "xml"
        Response format. ``"xml"`` is parsed with the
        ``uniprot-xml`` Biopython parser.
    timeout_s : float, default 30.0
    verbose : bool, default False
        Log each record's id/name/description.

    Returns
    -------
    list[Bio.SeqRecord.SeqRecord]
        Materialized list (not an iterator), so it can be re-traversed.
    """
    records = _fetch_records(uniprot_id, fmt=fmt, timeout_s=timeout_s)
    if verbose:
        for record in records:
            logger.info("id=%s name=%s desc=%s", record.id, record.name, record.description)
    return records


def get_sequences(uniprot_id: str, **kwargs: Any) -> list[Any]:
    """Sequences for ``uniprot_id`` as a list of ``Bio.Seq`` objects.

    Pass extra arguments through to :func:`query_protein`.
    """
    return [record.seq for record in query_protein(uniprot_id, **kwargs)]


def get_features(uniprot_id: str, **kwargs: Any) -> pd.DataFrame:
    """Protein features as a DataFrame.

    Columns
    -------
    ``id``, ``type``, ``start``, ``end``, ``length``, plus any feature
    qualifier columns (e.g. ``description``, ``evidence``) unpacked from
    the Biopython record.

    ``length`` is computed as ``end - start``.
    """
    records = query_protein(uniprot_id, **kwargs)
    rows: list[dict[str, Any]] = []
    for record in records:
        for feature in record.features:
            row: dict[str, Any] = {
                "id": feature.id,
                "type": feature.type,
                "start": feature.location.start,
                "end": feature.location.end,
            }
            row.update(feature.qualifiers)
            rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["length"] = df["end"] - df["start"]
    return df


def get_dbxrefs(uniprot_id: str, **kwargs: Any) -> pd.DataFrame:
    """Database cross-references for ``uniprot_id`` as a DataFrame.

    Columns
    -------
    dbxref : str  -- original ``"db:id"`` string
    db : str
    id : str
    """
    records = query_protein(uniprot_id, **kwargs)
    xrefs: list[str] = []
    for record in records:
        xrefs.extend(record.dbxrefs)
    df = pd.DataFrame({"dbxref": xrefs})
    if not df.empty:
        df[["db", "id"]] = df["dbxref"].str.split(":", n=1, expand=True)
    return df
