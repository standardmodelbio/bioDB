"""SNOMED CT — per-concept lookups (via OLS4) + local CONCEPT.csv parser.

SNOMED CT is the canonical clinical-terminology vocabulary — diagnoses,
procedures, findings, body sites, etc. ``biodb.snomed`` covers both
modes, but the bulk path is deliberately **parser-only** because of
SNOMED CT's licensing posture:

* **API mode** — per-concept lookups via EBI's `OLS4
  <https://www.ebi.ac.uk/ols4/>`_ (~376 k SNOMED terms indexed).
  See :func:`query_concept`, :func:`search_concepts`,
  :func:`get_descendants`, :func:`get_ancestors`,
  :func:`get_children`, :func:`get_parents`. Each accepts an ``int``,
  a CURIE (``"SNOMED:38341003"``), or a full IRI. EBI handles the
  SNOMED CT license on their side — no token required.

* **Bulk mode (parser only)** — :func:`load_concept_csv` and
  :func:`load_concept_csv_from_zip` parse a CONCEPT.csv that the user
  obtained themselves from OHDSI Athena. **bioDB does not host or
  redistribute SNOMED CT bytes**, because SNOMED CT is a controlled
  vocabulary (free in Member countries via UMLS/IHTSDO license, paid
  Affiliate license elsewhere — neither permits onward redistribution
  to unlicensed parties).

How to get a CONCEPT.csv
------------------------
1. Register a free account at https://athena.ohdsi.org.
2. Open your profile and **accept the SNOMED CT license** (and any
   other vocabulary licenses you need — RxNorm, CPT, ICD, …).
3. From the Vocabularies page, select the vocabularies you want and
   click ``Download``. Athena builds a bundle asynchronously and
   emails you a download link.
4. Unzip the bundle locally, or pass the .zip directly to
   :func:`load_concept_csv_from_zip`.

Examples
--------
>>> from biodb import snomed
>>> snomed.query_concept(38341003)["label"]                        # doctest: +SKIP
'Hypertensive disorder'
>>> snomed.search_concepts("diabetes", rows=5)                     # doctest: +SKIP
>>> snomed.get_descendants(73211009)                               # doctest: +SKIP
>>> # Bulk: user obtained the Athena bundle manually
>>> df = snomed.load_concept_csv("~/Downloads/CONCEPT.csv")        # doctest: +SKIP
>>> df = snomed.load_concept_csv_from_zip(                         # doctest: +SKIP
...     "~/Downloads/vocabulary_download_v5_{uuid}_*.zip"
... )
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path("~/.cache/biodb/snomed").expanduser()
"""Optional cache root, only used by callers who want to stash a parsed
copy of CONCEPT.csv themselves. bioDB does not write to this directory
automatically — the parser returns the loaded DataFrame from whatever
path the caller passes."""

ATHENA_DOWNLOAD_PAGE = "https://athena.ohdsi.org/vocabulary/list"
"""Where users go to obtain a CONCEPT.csv (must accept SNOMED CT terms first)."""


# ─── Bulk parser — operates on user-supplied files ─────────────────────────


# OHDSI CDM's CONCEPT.csv schema:
#   https://ohdsi.github.io/CommonDataModel/cdm54.html#CONCEPT
_CONCEPT_DTYPES: dict[str, str] = {
    "concept_id": "Int64",
    "concept_name": "string",
    "domain_id": "string",
    "vocabulary_id": "string",
    "concept_class_id": "string",
    "standard_concept": "string",
    "concept_code": "string",
}
"""Sensible dtype overrides for CONCEPT.csv. The date columns are left
to pandas defaults so callers can choose whether to parse them."""


def load_concept_csv(
    path: str | Path,
    *,
    vocabulary_id: str | None = None,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Parse an OHDSI ``CONCEPT.csv`` into a DataFrame.

    Parameters
    ----------
    path
        Local path to ``CONCEPT.csv``. Obtain a copy from
        https://athena.ohdsi.org after accepting the relevant
        vocabulary licenses (SNOMED CT, RxNorm, CPT, …).
    vocabulary_id
        If provided, filter rows down to that vocabulary
        (e.g. ``"SNOMED"``, ``"RxNorm"``, ``"LOINC"``). The OHDSI
        bundle mixes many vocabularies into one file; this filter is
        the fast path when you only care about one.
    **read_csv_kwargs
        Forwarded to :func:`pandas.read_csv`. Sensible defaults for
        ``sep`` (tab) and ``dtype`` are applied first.
    """
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download CONCEPT.csv from "
            f"{ATHENA_DOWNLOAD_PAGE} (accepting the SNOMED CT license "
            f"and any other vocabulary licenses you need)."
        )
    defaults: dict = {"sep": "\t", "dtype": _CONCEPT_DTYPES}
    defaults.update(read_csv_kwargs)
    df = pd.read_csv(path, **defaults)
    if vocabulary_id is not None:
        df = df[df["vocabulary_id"] == vocabulary_id].reset_index(drop=True)
    return df


def load_concept_csv_from_zip(
    zip_path: str | Path,
    *,
    member: str = "CONCEPT.csv",
    vocabulary_id: str | None = None,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Parse ``CONCEPT.csv`` out of an Athena vocabulary bundle ``.zip``.

    Athena's web-UI downloads come as
    ``vocabulary_download_v5_{uuid}_{ts}.zip`` containing CONCEPT.csv,
    CONCEPT_RELATIONSHIP.csv, etc. as flat members.

    Parameters
    ----------
    zip_path
        Path to the downloaded zip.
    member
        Which member to extract; defaults to ``"CONCEPT.csv"``.
    vocabulary_id
        Same as :func:`load_concept_csv`.
    """
    zip_path = Path(zip_path).expanduser()
    if not zip_path.exists():
        raise FileNotFoundError(
            f"{zip_path} not found. Download a bundle from {ATHENA_DOWNLOAD_PAGE} first."
        )
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        target = next((m for m in members if Path(m).name == member), None)
        if target is None:
            raise KeyError(
                f"{member!r} not found in {zip_path}; available members: "
                f"{[Path(m).name for m in members][:8]}"
            )
        with zf.open(target) as handle:
            defaults: dict = {"sep": "\t", "dtype": _CONCEPT_DTYPES}
            defaults.update(read_csv_kwargs)
            df = pd.read_csv(handle, **defaults)
    if vocabulary_id is not None:
        df = df[df["vocabulary_id"] == vocabulary_id].reset_index(drop=True)
    return df


def get_snomed_data_dir() -> Path:
    """Return the SNOMED cache directory (creating it on first call).

    Provided for callers who want to stash their parsed Athena bundle
    somewhere stable. bioDB itself does not write here automatically.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


# ─── Per-concept lookups via OLS ───────────────────────────────────────────
# SNOMED CT is indexed on EBI's OLS4 (~376 k terms). For one-concept-at-a-time
# queries we go through OLS — EBI handles their own SNOMED CT licensing on
# the server side, so callers don't need a UMLS / IHTSDO Affiliate license
# just to look up a single concept's label.

OLS_ONTOLOGY_SLUG = "snomed"
"""The OLS slug for SNOMED CT."""


def _normalize_concept_id(concept_id: str | int) -> str:
    """Turn ``int``, bare digit string, ``SNOMED:xxx``, or full IRI → CURIE."""
    if isinstance(concept_id, int):
        return f"SNOMED:{concept_id}"
    text = str(concept_id).strip()
    if text.startswith(("http://", "https://")):
        return text
    if ":" in text:
        return text
    return f"SNOMED:{text}"


def query_concept(concept_id: str | int, *, timeout: int = 30) -> dict:
    """Look up one SNOMED concept via OLS4.

    Accepts a bare concept ID (``38341003``), a SNOMED CURIE
    (``"SNOMED:38341003"``), or the full SNOMED IRI
    (``"http://snomed.info/id/38341003"``).

    Returns the OLS term record — ``label``, ``description``, ``synonyms``,
    ``obo_id``, ``iri``, ``is_obsolete``, ``has_children``, ``is_root``.
    """
    from biodb import ols

    return ols.get_term(OLS_ONTOLOGY_SLUG, _normalize_concept_id(concept_id), timeout=timeout)


def get_descendants(
    concept_id: str | int,
    *,
    size: int = 500,
    timeout: int = 30,
) -> pd.DataFrame:
    """Return every transitive descendant of ``concept_id`` (via OLS)."""
    from biodb import ols

    return ols.get_descendants(
        OLS_ONTOLOGY_SLUG, _normalize_concept_id(concept_id), size=size, timeout=timeout
    )


def get_ancestors(
    concept_id: str | int,
    *,
    size: int = 500,
    timeout: int = 30,
) -> pd.DataFrame:
    """Return every transitive ancestor of ``concept_id`` (via OLS)."""
    from biodb import ols

    return ols.get_ancestors(
        OLS_ONTOLOGY_SLUG, _normalize_concept_id(concept_id), size=size, timeout=timeout
    )


def get_children(
    concept_id: str | int,
    *,
    size: int = 500,
    timeout: int = 30,
) -> pd.DataFrame:
    """Return the direct (one-hop) children of ``concept_id`` (via OLS)."""
    from biodb import ols

    return ols.get_children(
        OLS_ONTOLOGY_SLUG, _normalize_concept_id(concept_id), size=size, timeout=timeout
    )


def get_parents(
    concept_id: str | int,
    *,
    size: int = 500,
    timeout: int = 30,
) -> pd.DataFrame:
    """Return the direct (one-hop) parents of ``concept_id`` (via OLS)."""
    from biodb import ols

    return ols.get_parents(
        OLS_ONTOLOGY_SLUG, _normalize_concept_id(concept_id), size=size, timeout=timeout
    )


def search_concepts(
    query: str,
    *,
    rows: int = 20,
    exact: bool = False,
    timeout: int = 30,
) -> pd.DataFrame:
    """Full-text search SNOMED concepts via OLS (label / synonym / definition)."""
    from biodb import ols

    return ols.search(query, ontology=OLS_ONTOLOGY_SLUG, rows=rows, exact=exact, timeout=timeout)


__all__ = [
    "ATHENA_DOWNLOAD_PAGE",
    "CACHE_DIR",
    "OLS_ONTOLOGY_SLUG",
    "get_ancestors",
    "get_children",
    "get_descendants",
    "get_parents",
    "get_snomed_data_dir",
    "load_concept_csv",
    "load_concept_csv_from_zip",
    "query_concept",
    "search_concepts",
]
