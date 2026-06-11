# GTR Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add NCBI's Genetic Testing Registry (GTR) to bioDB as a first-party dual-mode source (`src/biodb/gtr.py`) — targeted E-utilities queries + bulk FTP downloads — that materializes curated per-panel gene sets (for GenForge) and embeddable per-panel free-text plus a source-grounded `support_count` prior (for HaploForge).

**Architecture:** Mirror `src/biodb/pubmed.py` (the closest existing template: E-utilities transport with polite rate-limiting + streaming `iterparse` XML + `_downloads.stream_to_file` bulk). Targeted mode normalizes esummary JSON into a `GTRTest` dataclass. Bulk mode has one `download()` entry point (light daily TSV always; 224 MB full XML on opt-in). Gene-set views and the embeddable `panel_text` hook are pure transforms over those. bioDB stops at producing clean gene lists + panel text — embedding/cosine-sim weighting happens downstream in HaploForge/biodocs.

**Tech Stack:** Python 3.10+, `requests` (core), stdlib `gzip` + `xml.etree.ElementTree.iterparse`, `pandas`, `biodb._downloads.stream_to_file`, `biodb.utils.read_gmt`. Tests use `responses` + pytest fixtures. No new optional-dependency extra.

---

## Confirmed upstream facts (verified live 2026-06-11)

These are hardcoded into the parsers — do not re-derive:

**E-utilities (`db=gtr`):**
- esearch: `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=gtr&term=...&retmode=json`
- esummary: `.../esummary.fcgi?db=gtr&id=<uid>&retmode=json`. Record at `result["<uid>"]`; `result["uids"]` lists uids.
- UID ≠ accession: accession is `GTR%09d` (uid `509983` → `GTR000509983`). esearch returns bare uids.
- Rate limit: 3 req/s un-keyed, 10 req/s with `api_key`.

**esummary JSON record keys (all lowercase):** `uid, id, source, accession, testname, testtype, conditionlist, conditionlist2, analytes, genelist, offerer, offererlocation, offererid, method, analyticalvalidity, targetpopulation, certifications, testtargetlist, conditioncount, genecount, country, testurl, testcode, cptcode, testpurpose, clinicalutility, clinicalvalidity, exons, ...`
- `conditionlist` items: `{"name","acronym","cui"}` (cui = MedGen/UMLS CUI, e.g. `"C0677776"`).
- `analytes` items: `{"analytetype":"Gene","name":"BRCA1","geneid":672,"location":"17q21.31"}` (geneid is an **int**).
- `method` items: `{"name","categoriesstring","categorylist":[{"name","code","methodlist":[...]}]}`.
- `clinicalvalidity`: object `{"description","pmid":[...],"url":[...],"citationtext":[...]}`.
- `clinicalutility`: **array** (often empty) of objects with the same shape.
- `testpurpose`: array of strings.

**Bulk light TSV** `https://ftp.ncbi.nlm.nih.gov/pub/GTR/data/test_condition_gene.txt` (daily, ~46 MB). Header (tab-sep, leading `#`):
```
#accession_version	test_type	object	GTR_identifier	MIM_number	object_name	gene_or_SNOMED_CT_ID	gene_symbol
```
- `object` column is `condition` OR `gene`.
- condition rows: `GTR_identifier`=MedGen CUI, `gene_or_SNOMED_CT_ID`=SNOMED CT id, `gene_symbol`=`N/A`.
- gene rows: `gene_or_SNOMED_CT_ID`=Entrez gene id, `gene_symbol`=symbol.
- Example rows:
```
GTR000004006.1	Clinical	condition	C0016667	300624	Fragile X syndrome	613003	N/A
GTR000004006.1	Clinical	gene	C1414649	309550	FMR1:fragile X messenger ribonucleoprotein 1	2332	FMR1
```

**Bulk full XML** `https://ftp.ncbi.nlm.nih.gov/pub/GTR/data/gtr_ftp.xml.gz` (~224 MB, weekly), validates against `GTRPublicData.xsd`. Root `<GTRPublicData Version="1.0">`. Structure:
```
GTRPublicData / GTRLabData / {GTRLab, GTRLabTest*, GTRLabResearchTest*}
```
- Per-clinical-test element to iterparse: **`GTRLabTest`** with attribs `GTRAccession`, `id`, `Version`. (Research tests are the sibling element `GTRLabResearchTest`.)
- `GTRLabTest/TestName` → test name.
- `GTRLabTest/Indications/TestType` → "Clinical"; `Indications/Purpose`, `Indications/TargetPop/Description`.
- Genes: `GTRLabTest//MeasureSet/Measure[@Type="Gene"]` → Entrez id at `Measure/XRef[@DB="Gene"]` **`ID` attribute**; symbol at `Measure/Symbol/ElementValue` (best-effort).
- Conditions: `GTRLabTest//TraitSet/Trait` → name at `Trait/Name/ElementValue`; CUI at `ClinVarSet/DescrSet` **`CUI` attribute** and/or `Trait/XRef[@DB="MedGen"]` `ID` attribute.
- Methodology: `GTRLabTest/Method/TopCategory[@Value]/Category[@Value][@code]/Methodology[@Value]`.
- Free-text (each is `Description` + optional `CitationText`/`PMID`/`URL`): `GTRLabTest/AnalyticalValidity`, `GTRLabTest/QualityControl/ClinicalValidity` (**nested under QualityControl**), `GTRLabTest/ClinicalUtility` (direct child, repeating), `GTRLabTest/TestStrategy`, `GTRLabTest/Indications/TargetPop`.

> **Schema note for the implementer:** the live XSD is at `https://ftp.ncbi.nlm.nih.gov/pub/GTR/documentation/GTRPublicData.xsd`. Task 8 step 1 re-confirms the `Symbol/ElementValue` and `Trait/Name/ElementValue` child element names against a real sample (the only best-effort items); everything attribute-based above is confirmed.

---

## File Structure

- **Create** `src/biodb/gtr.py` — the entire module (transport, `GTRTest`, targeted queries, bulk download, parsers, gene-set views, `panel_text`). One file, mirroring how every other source module is self-contained. ~600 lines projected — comparable to `pubmed.py`; if it materially exceeds `opentargets_graphql.py`-scale, that's expected for a dual-mode source and not a reason to split.
- **Create** `tests/test_gtr.py` — unit tests (offline, `responses`-mocked + fixtures).
- **Create** `tests/fixtures/gtr/` — `esummary.json`, `test_condition_gene.txt`, `gtr_sample.xml` test fixtures.
- **Create** `docs/gtr.md` — user guide mirroring `docs/ols.md`.
- **Modify** `src/biodb/__init__.py` — import `gtr`, re-export curated symbols, extend `__all__`.
- **Modify** `docs/api.rst` — add `gtr` to autosummary.
- **Modify** `docs/index.md` — add a `gtr` bullet + toctree entry.
- **Modify** `docs/quickstart.md` — add a short GTR snippet.
- **Modify** `README.md` — add a Sources-table row.

---

## Task 1: Module scaffold — docstring, constants, E-utilities transport

**Files:**
- Create: `src/biodb/gtr.py`
- Test: `tests/test_gtr.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_gtr.py`:

```python
"""Tests for :mod:`biodb.gtr`."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd
import pytest
import responses

from biodb import gtr

FIXTURES = Path(__file__).parent / "fixtures" / "gtr"


# ── module surface ──────────────────────────────────────────────────────────


def test_module_imports_offline() -> None:
    assert gtr.__name__ == "biodb.gtr"


def test_constants_present() -> None:
    assert gtr.NCBI_EUTILS_BASE_URL.endswith("/eutils")
    assert gtr.GTR_FTP_BASE_URL.endswith("/GTR/data")
    assert gtr.TEST_CONDITION_GENE_FILE == "test_condition_gene.txt"
    assert gtr.FULL_XML_FILE == "gtr_ftp.xml.gz"
    assert gtr.CACHE_DIR.exists()


def test_accession_from_uid() -> None:
    assert gtr.accession_from_uid("509983") == "GTR000509983"
    assert gtr.accession_from_uid(509983) == "GTR000509983"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gtr.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'biodb.gtr'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/biodb/gtr.py`:

```python
"""NCBI Genetic Testing Registry (GTR) client — E-utilities (API) + bulk FTP.

The `GTR <https://www.ncbi.nlm.nih.gov/gtr/>`_ catalogs genetic **tests**:
each record is *one test offered by one lab*. A record carries the lab, the
gene list / panel (each gene with an Entrez ID + cytoband), the conditions
(MedGen/UMLS CUIs + OMIM), a 3-level methodology taxonomy, a clinical-vs-
research flag, and clinical validity/utility free-text. ``bioDB`` exposes
both access modes:

* **API mode** — NCBI E-utilities (``esearch`` + ``esummary`` on ``db=gtr``).
  :func:`search_tests`, :func:`query_test`, :func:`query_gene`,
  :func:`query_condition`. Each call sleeps ~340 ms to respect the un-keyed
  3 req/sec cap; pass ``api_key`` to lift to 10 req/sec.

* **Bulk mode** — one :func:`download` entry point pulls the light daily
  ``test_condition_gene.txt`` TSV (test↔gene↔condition mapping) and, with
  ``full_xml=True``, also the 224 MB ``gtr_ftp.xml.gz`` (the only source
  carrying descriptions/methodology). :func:`load_test_condition_gene` and
  the streaming :func:`iter_full_records` read those files.

Gene-set views (:func:`gene_sets`, :func:`aggregate_gene_sets`,
:func:`to_gmt`) and the embeddable :func:`panel_text` hook are pure
transforms. bioDB produces clean gene lists + panel text; embedding /
cosine-similarity weighting is a downstream (HaploForge/biodocs) concern.

Cached files live at ``~/.cache/biodb/gtr/``.

Examples
--------
>>> from biodb import gtr
>>> accs = gtr.search_tests("BRCA1", field="SYMB", retmax=5)   # doctest: +SKIP
>>> test = gtr.query_test(accs[0])                             # doctest: +SKIP
>>> paths = gtr.download(full_xml=False)                       # doctest: +SKIP
>>> sets = gtr.aggregate_gene_sets(by="condition")             # doctest: +SKIP
>>> gtr.to_gmt("gtr_panels.gmt", by="test_name")               # doctest: +SKIP

References
----------
* GTR home: https://www.ncbi.nlm.nih.gov/gtr/
* E-utilities: https://www.ncbi.nlm.nih.gov/books/NBK25501/
* FTP data + docs: https://ftp.ncbi.nlm.nih.gov/pub/GTR/
"""

from __future__ import annotations

import gzip
import logging
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

import pandas as pd
import requests

from biodb._downloads import stream_to_file

logger = logging.getLogger(__name__)

NCBI_EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
"""NCBI E-utilities root. ``esearch`` / ``esummary`` live below."""

GTR_FTP_BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pub/GTR/data"
"""GTR bulk FTP data root (HTTPS-served)."""

GTR_DATA_SERVICE_URL = "https://www.ncbi.nlm.nih.gov/gtr/api/v1"
"""Undocumented-but-live richer JSON REST API (used only when rich=True)."""

TEST_CONDITION_GENE_FILE = "test_condition_gene.txt"
"""Light daily TSV: test ↔ gene(Entrez) ↔ condition(CUI/SNOMED) mapping."""

FULL_XML_FILE = "gtr_ftp.xml.gz"
"""Full weekly XML dump (~224 MB) — the only bulk source with descriptions."""

CACHE_DIR = Path("~/.cache/biodb/gtr").expanduser()
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_RATE_LIMIT_SLEEP_S = 0.34
"""Per-request sleep that keeps us under the un-keyed 3 req/sec cap."""

_USER_AGENT = "biodb/0.1 (+https://github.com/bschilder/bioDB)"


def accession_from_uid(uid: str | int) -> str:
    """Convert an Entrez GTR UID to its zero-padded ``GTR`` accession.

    GTR's Entrez UID is *not* the accession: uid ``509983`` is accession
    ``GTR000509983`` (the ``GTR%09d`` format). ``esearch`` returns bare
    uids, so callers that want accessions go through this helper.
    """
    return f"GTR{int(uid):09d}"


def _eutils_get(path: str, params: dict, *, timeout: int = 30) -> requests.Response:
    """GET an E-utilities endpoint with polite-rate-limit handling.

    Sleeps ~340 ms between calls and retries once on HTTP 429 — the same
    posture :mod:`biodb.pubmed` and :mod:`biodb.clinvar` use for E-utils.
    """
    url = f"{NCBI_EUTILS_BASE_URL}/{path.lstrip('/')}"
    response = requests.get(
        url, params=params, timeout=timeout, headers={"User-Agent": _USER_AGENT}
    )
    if response.status_code == 429:
        time.sleep(1.0)
        response = requests.get(
            url, params=params, timeout=timeout, headers={"User-Agent": _USER_AGENT}
        )
    response.raise_for_status()
    time.sleep(_RATE_LIMIT_SLEEP_S)
    return response


__all__ = [
    "CACHE_DIR",
    "FULL_XML_FILE",
    "GTR_DATA_SERVICE_URL",
    "GTR_FTP_BASE_URL",
    "NCBI_EUTILS_BASE_URL",
    "TEST_CONDITION_GENE_FILE",
    "accession_from_uid",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_gtr.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/biodb/gtr.py tests/test_gtr.py
git commit -m "feat(gtr): module scaffold + E-utilities transport + accession helper"
```

---

## Task 2: `GTRTest` record + esummary normalization

**Files:**
- Modify: `src/biodb/gtr.py`
- Create: `tests/fixtures/gtr/esummary.json`
- Test: `tests/test_gtr.py`

- [ ] **Step 1: Create the esummary fixture**

Create `tests/fixtures/gtr/esummary.json` (trimmed-but-real shape; the record is keyed by uid):

```json
{
  "header": {"type": "esummary", "version": "0.3"},
  "result": {
    "uids": ["509983"],
    "509983": {
      "uid": "509983",
      "id": 509983,
      "source": "GTR",
      "accession": "GTR000509983",
      "testname": "BRCA1 gene sequencing",
      "testtype": "Clinical",
      "conditionlist": [
        {"name": "Breast-ovarian cancer, familial 1", "acronym": "", "cui": "C0677776"},
        {"name": "Hereditary breast ovarian cancer syndrome", "acronym": "HBOC", "cui": "C0027672"}
      ],
      "analytes": [
        {"analytetype": "Gene", "name": "BRCA1", "geneid": 672, "location": "17q21.31"}
      ],
      "genelist": [],
      "offerer": "Example Genetics Lab",
      "offererid": 308659,
      "method": [
        {"name": "Molecular Genetics", "categoriesstring": "_C_",
         "categorylist": [
           {"name": "Sequence analysis of the entire coding region", "code": "C",
            "methodlist": ["Next-Generation (NGS)/Massively parallel sequencing (MPS)"]}
         ]}
      ],
      "analyticalvalidity": "Analytical sensitivity >99% for SNVs.",
      "targetpopulation": "Individuals with a personal or family history of breast cancer.",
      "testpurpose": ["Diagnosis", "Risk Assessment"],
      "clinicalvalidity": {
        "description": "Pathogenic BRCA1 variants confer high lifetime breast cancer risk.",
        "pmid": ["20301425"], "url": [], "citationtext": []
      },
      "clinicalutility": [],
      "testurl": "https://example.org/tests/brca1"
    }
  }
}
```

- [ ] **Step 2: Write the failing test**

Append to `tests/test_gtr.py`:

```python
def _esummary_record() -> dict:
    payload = json.loads((FIXTURES / "esummary.json").read_text())
    return payload["result"]["509983"]


def test_test_from_esummary_normalizes_core_fields() -> None:
    rec = gtr._test_from_esummary(_esummary_record())
    assert isinstance(rec, gtr.GTRTest)
    assert rec.accession == "GTR000509983"
    assert rec.uid == "509983"
    assert rec.name == "BRCA1 gene sequencing"
    assert rec.test_type == "Clinical"
    assert rec.lab == "Example Genetics Lab"
    assert rec.test_url == "https://example.org/tests/brca1"


def test_test_from_esummary_extracts_genes() -> None:
    rec = gtr._test_from_esummary(_esummary_record())
    assert rec.genes == [{"symbol": "BRCA1", "entrez": "672", "location": "17q21.31"}]


def test_test_from_esummary_extracts_conditions_and_methods() -> None:
    rec = gtr._test_from_esummary(_esummary_record())
    assert {"name": "Breast-ovarian cancer, familial 1", "cui": "C0677776"} in [
        {"name": c["name"], "cui": c["cui"]} for c in rec.conditions
    ]
    assert "Next-Generation (NGS)/Massively parallel sequencing (MPS)" in rec.methods
    assert rec.clinical_validity.startswith("Pathogenic BRCA1")
    assert rec.pmids == ["20301425"]
    assert rec.test_purpose == ["Diagnosis", "Risk Assessment"]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_gtr.py -k "esummary" -v`
Expected: FAIL with `AttributeError: module 'biodb.gtr' has no attribute 'GTRTest'`.

- [ ] **Step 4: Write minimal implementation**

In `src/biodb/gtr.py`, add the dataclass + normalizer after `_eutils_get` (and add their names to `__all__`):

```python
@dataclass
class GTRTest:
    """A normalized GTR test record (one test offered by one lab).

    Attributes
    ----------
    accession : str
        ``GTR`` accession (e.g. ``"GTR000509983"``).
    uid : str
        Entrez UID (numeric portion; see :func:`accession_from_uid`).
    name : str
        Test name.
    test_type : str
        ``"Clinical"`` or ``"Research"``.
    genes : list[dict]
        One ``{"symbol","entrez","location"}`` per gene analyte. ``entrez``
        is a string Entrez Gene ID.
    conditions : list[dict]
        One ``{"name","cui","omim"}`` per condition; ``cui`` is a MedGen/UMLS
        CUI, ``omim`` may be ``""``.
    methods : list[str]
        Flattened methodology leaf names (e.g. NGS/MPS).
    method_categories : list[str]
        Methodology category names (the middle taxonomy level).
    lab, lab_id : str
        Offering organization + its GTR org id.
    analytical_validity, clinical_validity, clinical_utility : str
        Free-text blocks (assembled descriptions).
    target_population : str
    test_purpose : list[str]
    pmids : list[str]
        PMIDs cited by the clinical-validity/utility blocks.
    test_url : str
    """

    accession: str
    uid: str = ""
    name: str = ""
    test_type: str = ""
    genes: list[dict] = field(default_factory=list)
    conditions: list[dict] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    method_categories: list[str] = field(default_factory=list)
    lab: str = ""
    lab_id: str = ""
    analytical_validity: str = ""
    clinical_validity: str = ""
    clinical_utility: str = ""
    target_population: str = ""
    test_purpose: list[str] = field(default_factory=list)
    pmids: list[str] = field(default_factory=list)
    test_url: str = ""


def _test_from_esummary(rec: dict) -> GTRTest:
    """Normalize one ``esummary`` (db=gtr, JSON) record into a :class:`GTRTest`."""
    genes = [
        {
            "symbol": a.get("name", ""),
            "entrez": str(a["geneid"]) if a.get("geneid") not in (None, "") else "",
            "location": a.get("location", ""),
        }
        for a in rec.get("analytes", []) or []
        if a.get("analytetype") == "Gene"
    ]
    conditions = [
        {"name": c.get("name", ""), "cui": c.get("cui", ""), "omim": ""}
        for c in rec.get("conditionlist", []) or []
    ]

    # Methodology: flatten method -> categorylist -> methodlist (leaf names)
    # and collect the category-level names separately.
    methods: list[str] = []
    method_categories: list[str] = []
    for m in rec.get("method", []) or []:
        for cat in m.get("categorylist", []) or []:
            if cat.get("name"):
                method_categories.append(cat["name"])
            methods.extend(cat.get("methodlist", []) or [])

    # clinicalvalidity is an object; clinicalutility is an array of objects.
    cv = rec.get("clinicalvalidity") or {}
    cv_desc = cv.get("description", "") if isinstance(cv, dict) else ""
    pmids = [str(p) for p in (cv.get("pmid", []) if isinstance(cv, dict) else [])]
    cu_items = rec.get("clinicalutility") or []
    cu_desc = "\n\n".join(
        u.get("description", "") for u in cu_items if isinstance(u, dict) and u.get("description")
    )
    for u in cu_items:
        if isinstance(u, dict):
            pmids.extend(str(p) for p in u.get("pmid", []) or [])

    return GTRTest(
        accession=rec.get("accession", ""),
        uid=str(rec.get("uid", "")),
        name=rec.get("testname", ""),
        test_type=rec.get("testtype", ""),
        genes=genes,
        conditions=conditions,
        methods=methods,
        method_categories=method_categories,
        lab=rec.get("offerer", ""),
        lab_id=str(rec.get("offererid", "")),
        analytical_validity=rec.get("analyticalvalidity", ""),
        clinical_validity=cv_desc,
        clinical_utility=cu_desc,
        target_population=rec.get("targetpopulation", ""),
        test_purpose=list(rec.get("testpurpose", []) or []),
        pmids=pmids,
        test_url=rec.get("testurl", ""),
    )
```

Add `"GTRTest"` to `__all__`.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_gtr.py -k "esummary" -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add src/biodb/gtr.py tests/test_gtr.py tests/fixtures/gtr/esummary.json
git commit -m "feat(gtr): GTRTest dataclass + esummary JSON normalizer"
```

---

## Task 3: `search_tests` (esearch)

**Files:**
- Modify: `src/biodb/gtr.py`
- Test: `tests/test_gtr.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gtr.py`:

```python
def test_search_tests_returns_accessions() -> None:
    url = f"{gtr.NCBI_EUTILS_BASE_URL}/esearch.fcgi"
    body = {"esearchresult": {"count": "2", "idlist": ["509983", "4006"],
                              "querytranslation": "BRCA1[SYMB]"}}
    with responses.RequestsMock() as rsps:
        rsps.add(responses.GET, url, json=body, status=200)
        accs = gtr.search_tests("BRCA1", field="SYMB", retmax=5)
    assert accs == ["GTR000509983", "GTR000004006"]


def test_search_tests_uids_mode() -> None:
    url = f"{gtr.NCBI_EUTILS_BASE_URL}/esearch.fcgi"
    body = {"esearchresult": {"count": "1", "idlist": ["509983"], "querytranslation": ""}}
    with responses.RequestsMock() as rsps:
        rsps.add(responses.GET, url, json=body, status=200)
        out = gtr.search_tests("BRCA1", as_accession=False)
    assert out == ["509983"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gtr.py -k "search_tests" -v`
Expected: FAIL with `AttributeError: ... has no attribute 'search_tests'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/biodb/gtr.py` (and `__all__`):

```python
def search_tests(
    term: str,
    *,
    field: str | None = None,
    retmax: int = 200,
    retstart: int = 0,
    as_accession: bool = True,
    api_key: str | None = None,
    timeout: int = 30,
) -> list[str]:
    """Run an ``esearch`` against ``db=gtr`` and return matching test ids.

    Parameters
    ----------
    term : str
        Query string. Free text works; scope to a GTR index with ``field``.
    field : str, optional
        GTR search index, appended as ``term[field]``. Common indices:
        ``"SYMB"`` (gene symbol), ``"GENEID"`` (Entrez id), ``"DISNAME"``
        (disease name), ``"DCUI"`` (condition CUI), ``"MTOD"`` (method).
    retmax, retstart : int
        Page size / offset.
    as_accession : bool, default True
        Return ``GTR`` accessions (via :func:`accession_from_uid`). Set
        False to get the raw Entrez uids that ``esummary`` expects.
    api_key : str, optional
        NCBI API key (lifts the rate limit to 10 req/sec).
    timeout : int

    Returns
    -------
    list[str]
        Accessions (default) or bare uids.
    """
    query = f"{term}[{field}]" if field else term
    params: dict = {
        "db": "gtr",
        "term": query,
        "retmax": retmax,
        "retstart": retstart,
        "retmode": "json",
    }
    if api_key:
        params["api_key"] = api_key
    payload = _eutils_get("esearch.fcgi", params, timeout=timeout).json()
    uids = payload.get("esearchresult", {}).get("idlist", [])
    return [accession_from_uid(u) for u in uids] if as_accession else list(uids)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_gtr.py -k "search_tests" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/biodb/gtr.py tests/test_gtr.py
git commit -m "feat(gtr): search_tests (esearch) with field scoping + accession mapping"
```

---

## Task 4: `query_test` / `_esummary_records` (esummary, single + batch)

**Files:**
- Modify: `src/biodb/gtr.py`
- Test: `tests/test_gtr.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gtr.py`:

```python
def test_query_test_by_accession() -> None:
    payload = json.loads((FIXTURES / "esummary.json").read_text())
    url = f"{gtr.NCBI_EUTILS_BASE_URL}/esummary.fcgi"
    with responses.RequestsMock() as rsps:
        rsps.add(responses.GET, url, json=payload, status=200)
        rec = gtr.query_test("GTR000509983")
    assert isinstance(rec, gtr.GTRTest)
    assert rec.accession == "GTR000509983"
    # the request must carry the bare uid, not the accession
    assert "id=509983" in rsps.calls[0].request.url


def test_query_test_accepts_bare_uid() -> None:
    payload = json.loads((FIXTURES / "esummary.json").read_text())
    url = f"{gtr.NCBI_EUTILS_BASE_URL}/esummary.fcgi"
    with responses.RequestsMock() as rsps:
        rsps.add(responses.GET, url, json=payload, status=200)
        rec = gtr.query_test(509983)
    assert rec.name == "BRCA1 gene sequencing"


def test_query_test_missing_raises() -> None:
    url = f"{gtr.NCBI_EUTILS_BASE_URL}/esummary.fcgi"
    with responses.RequestsMock() as rsps:
        rsps.add(responses.GET, url, json={"result": {"uids": []}}, status=200)
        with pytest.raises(KeyError):
            gtr.query_test("GTR000000001")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gtr.py -k "query_test" -v`
Expected: FAIL with `AttributeError: ... has no attribute 'query_test'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/biodb/gtr.py` (and `__all__`):

```python
def _uid_of(test_id: str | int) -> str:
    """Coerce a GTR accession or uid to the bare numeric uid esummary needs.

    ``"GTR000509983"`` and ``509983`` both yield ``"509983"``.
    """
    s = str(test_id).strip()
    if s.upper().startswith("GTR"):
        return str(int(s[3:]))
    return str(int(s))


def _esummary_records(
    test_ids: list[str | int],
    *,
    api_key: str | None = None,
    timeout: int = 30,
) -> list[GTRTest]:
    """Fetch + normalize ``esummary`` records for one or more GTR ids."""
    uids = [_uid_of(t) for t in test_ids]
    if not uids:
        return []
    params: dict = {"db": "gtr", "id": ",".join(uids), "retmode": "json"}
    if api_key:
        params["api_key"] = api_key
    payload = _eutils_get("esummary.fcgi", params, timeout=timeout).json()
    result = payload.get("result", {})
    return [_test_from_esummary(result[u]) for u in result.get("uids", []) if u in result]


def query_test(
    test_id: str | int,
    *,
    api_key: str | None = None,
    timeout: int = 30,
) -> GTRTest:
    """Fetch one GTR test record (by accession or uid) as a :class:`GTRTest`.

    Raises
    ------
    KeyError
        If GTR returns no record for ``test_id``.
    """
    records = _esummary_records([test_id], api_key=api_key, timeout=timeout)
    if not records:
        raise KeyError(f"GTR test {test_id!r} not found")
    return records[0]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_gtr.py -k "query_test" -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/biodb/gtr.py tests/test_gtr.py
git commit -m "feat(gtr): query_test + esummary batch normalizer (accession/uid coercion)"
```

---

## Task 5: `query_gene` / `query_condition` convenience wrappers

**Files:**
- Modify: `src/biodb/gtr.py`
- Test: `tests/test_gtr.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gtr.py`:

```python
def test_query_gene_searches_by_symbol_then_summarizes() -> None:
    payload = json.loads((FIXTURES / "esummary.json").read_text())
    esearch = f"{gtr.NCBI_EUTILS_BASE_URL}/esearch.fcgi"
    esummary = f"{gtr.NCBI_EUTILS_BASE_URL}/esummary.fcgi"
    with responses.RequestsMock() as rsps:
        rsps.add(responses.GET, esearch,
                 json={"esearchresult": {"count": "1", "idlist": ["509983"]}}, status=200)
        rsps.add(responses.GET, esummary, json=payload, status=200)
        recs = gtr.query_gene("BRCA1", retmax=5)
    assert len(recs) == 1 and recs[0].name == "BRCA1 gene sequencing"
    assert "BRCA1%5BSYMB%5D" in rsps.calls[0].request.url or "BRCA1[SYMB]" in rsps.calls[0].request.url


def test_query_gene_numeric_uses_geneid_field() -> None:
    payload = json.loads((FIXTURES / "esummary.json").read_text())
    esearch = f"{gtr.NCBI_EUTILS_BASE_URL}/esearch.fcgi"
    esummary = f"{gtr.NCBI_EUTILS_BASE_URL}/esummary.fcgi"
    with responses.RequestsMock() as rsps:
        rsps.add(responses.GET, esearch,
                 json={"esearchresult": {"count": "1", "idlist": ["509983"]}}, status=200)
        rsps.add(responses.GET, esummary, json=payload, status=200)
        gtr.query_gene(672)
    assert "GENEID" in rsps.calls[0].request.url


def test_query_condition_uses_cui_field_for_cui() -> None:
    payload = json.loads((FIXTURES / "esummary.json").read_text())
    esearch = f"{gtr.NCBI_EUTILS_BASE_URL}/esearch.fcgi"
    esummary = f"{gtr.NCBI_EUTILS_BASE_URL}/esummary.fcgi"
    with responses.RequestsMock() as rsps:
        rsps.add(responses.GET, esearch,
                 json={"esearchresult": {"count": "1", "idlist": ["509983"]}}, status=200)
        rsps.add(responses.GET, esummary, json=payload, status=200)
        gtr.query_condition("C0677776")
    assert "DCUI" in rsps.calls[0].request.url


def test_query_condition_uses_disname_for_text() -> None:
    payload = json.loads((FIXTURES / "esummary.json").read_text())
    esearch = f"{gtr.NCBI_EUTILS_BASE_URL}/esearch.fcgi"
    esummary = f"{gtr.NCBI_EUTILS_BASE_URL}/esummary.fcgi"
    with responses.RequestsMock() as rsps:
        rsps.add(responses.GET, esearch,
                 json={"esearchresult": {"count": "1", "idlist": ["509983"]}}, status=200)
        rsps.add(responses.GET, esummary, json=payload, status=200)
        gtr.query_condition("breast cancer")
    assert "DISNAME" in rsps.calls[0].request.url
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gtr.py -k "query_gene or query_condition" -v`
Expected: FAIL with `AttributeError: ... has no attribute 'query_gene'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/biodb/gtr.py` (and `__all__`). Note the regex pre-compiled at module scope:

```python
import re

_CUI_RE = re.compile(r"^C\d{7}$")
"""A UMLS/MedGen CUI looks like ``C`` + 7 digits (e.g. ``C0677776``)."""


def query_gene(
    gene: str | int,
    *,
    retmax: int = 200,
    api_key: str | None = None,
    timeout: int = 30,
) -> list[GTRTest]:
    """Find tests targeting ``gene`` (symbol or Entrez id) → list[GTRTest].

    A numeric ``gene`` searches the ``GENEID`` index; anything else searches
    the gene-symbol index ``SYMB``.
    """
    field = "GENEID" if str(gene).isdigit() else "SYMB"
    uids = search_tests(
        str(gene), field=field, retmax=retmax, as_accession=False,
        api_key=api_key, timeout=timeout,
    )
    return _esummary_records(uids, api_key=api_key, timeout=timeout)


def query_condition(
    condition: str,
    *,
    retmax: int = 200,
    api_key: str | None = None,
    timeout: int = 30,
) -> list[GTRTest]:
    """Find tests for ``condition`` → list[GTRTest].

    A CUI-shaped argument (``C`` + 7 digits) searches the ``DCUI`` index;
    free text searches the disease-name index ``DISNAME``.
    """
    field = "DCUI" if _CUI_RE.match(condition.strip()) else "DISNAME"
    uids = search_tests(
        condition, field=field, retmax=retmax, as_accession=False,
        api_key=api_key, timeout=timeout,
    )
    return _esummary_records(uids, api_key=api_key, timeout=timeout)
```

Move the `import re` to the top-of-file import block (alongside the others) rather than inline — placed inline here only to show where it's used.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_gtr.py -k "query_gene or query_condition" -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/biodb/gtr.py tests/test_gtr.py
git commit -m "feat(gtr): query_gene / query_condition convenience wrappers"
```

---

## Task 6: Bulk `download()` (light TSV always; full XML opt-in)

**Files:**
- Modify: `src/biodb/gtr.py`
- Test: `tests/test_gtr.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gtr.py`:

```python
def test_download_fetches_tsv_only_by_default(tmp_path) -> None:
    tsv_url = f"{gtr.GTR_FTP_BASE_URL}/{gtr.TEST_CONDITION_GENE_FILE}"
    with responses.RequestsMock() as rsps:
        rsps.add(responses.GET, tsv_url, body=b"#header\n", status=200)
        paths = gtr.download(cache_dir=tmp_path)
    assert paths["tsv"].exists()
    assert paths["xml"] is None


def test_download_full_xml_fetches_both(tmp_path) -> None:
    tsv_url = f"{gtr.GTR_FTP_BASE_URL}/{gtr.TEST_CONDITION_GENE_FILE}"
    xml_url = f"{gtr.GTR_FTP_BASE_URL}/{gtr.FULL_XML_FILE}"
    with responses.RequestsMock() as rsps:
        rsps.add(responses.GET, tsv_url, body=b"#header\n", status=200)
        rsps.add(responses.GET, xml_url, body=b"\x1f\x8bfake-gzip", status=200)
        paths = gtr.download(cache_dir=tmp_path, full_xml=True)
    assert paths["tsv"].exists() and paths["xml"].exists()


def test_download_uses_cache(tmp_path) -> None:
    (tmp_path / gtr.TEST_CONDITION_GENE_FILE).write_bytes(b"#cached\n")
    with responses.RequestsMock() as rsps:
        paths = gtr.download(cache_dir=tmp_path)
    assert len(rsps.calls) == 0
    assert paths["tsv"].read_bytes() == b"#cached\n"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gtr.py -k "download" -v`
Expected: FAIL with `AttributeError: ... has no attribute 'download'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/biodb/gtr.py` (and `__all__`):

```python
def download(
    cache_dir: str | Path | None = None,
    *,
    full_xml: bool = False,
    force: bool = False,
    progress: bool = True,
    timeout: int = 600,
) -> dict[str, Path | None]:
    """Download GTR bulk files into the local cache.

    Always fetches the light daily ``test_condition_gene.txt`` TSV
    (~46 MB; test↔gene↔condition mapping). With ``full_xml=True`` also
    fetches ``gtr_ftp.xml.gz`` (~224 MB) — the only bulk source carrying
    descriptions/methodology, needed by :func:`iter_full_records`.

    Parameters
    ----------
    cache_dir : str or Path, optional
        Cache root; defaults to :data:`CACHE_DIR`.
    full_xml : bool, default False
        Also download the 224 MB full XML dump.
    force : bool, default False
        Re-download even if a cached copy exists.
    progress : bool, default True
        Show a tqdm bar per file.
    timeout : int, default 600
        Per-request timeout — generous because the XML is large.

    Returns
    -------
    dict
        ``{"tsv": Path, "xml": Path | None}`` — ``"xml"`` is ``None`` unless
        ``full_xml=True``.
    """
    root = Path(cache_dir).expanduser() if cache_dir else CACHE_DIR
    root.mkdir(parents=True, exist_ok=True)

    tsv_path = root / TEST_CONDITION_GENE_FILE
    if force or not tsv_path.exists():
        stream_to_file(
            f"{GTR_FTP_BASE_URL}/{TEST_CONDITION_GENE_FILE}", tsv_path,
            headers={"User-Agent": _USER_AGENT}, timeout=timeout, progress=progress,
        )

    xml_path: Path | None = None
    if full_xml:
        xml_path = root / FULL_XML_FILE
        if force or not xml_path.exists():
            stream_to_file(
                f"{GTR_FTP_BASE_URL}/{FULL_XML_FILE}", xml_path,
                headers={"User-Agent": _USER_AGENT}, timeout=timeout, progress=progress,
            )

    return {"tsv": tsv_path, "xml": xml_path}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_gtr.py -k "download" -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/biodb/gtr.py tests/test_gtr.py
git commit -m "feat(gtr): single bulk download() with full_xml opt-in"
```

---

## Task 7: `load_test_condition_gene` (TSV parser)

**Files:**
- Modify: `src/biodb/gtr.py`
- Create: `tests/fixtures/gtr/test_condition_gene.txt`
- Test: `tests/test_gtr.py`

- [ ] **Step 1: Create the TSV fixture**

Create `tests/fixtures/gtr/test_condition_gene.txt` (tab-separated — ensure real tabs, not spaces):

```
#accession_version	test_type	object	GTR_identifier	MIM_number	object_name	gene_or_SNOMED_CT_ID	gene_symbol
GTR000004006.1	Clinical	condition	C0016667	300624	Fragile X syndrome	613003	N/A
GTR000004006.1	Clinical	gene	C1414649	309550	FMR1:fragile X messenger ribonucleoprotein 1	2332	FMR1
GTR000509983.1	Clinical	condition	C0677776	604370	Breast-ovarian cancer familial 1	718220	N/A
GTR000509983.1	Clinical	gene	C1414742	113705	BRCA1:BRCA1 DNA repair associated	672	BRCA1
```

- [ ] **Step 2: Write the failing test**

Append to `tests/test_gtr.py`:

```python
def test_load_test_condition_gene_splits_rows(tmp_path) -> None:
    src = (FIXTURES / "test_condition_gene.txt").read_bytes()
    (tmp_path / gtr.TEST_CONDITION_GENE_FILE).write_bytes(src)
    df = gtr.load_test_condition_gene(cache_dir=tmp_path)
    # Header '#' is stripped from the first column name.
    assert "accession_version" in df.columns
    assert set(df["object"].unique()) == {"condition", "gene"}
    gene_rows = df[df["object"] == "gene"]
    assert "672" in set(gene_rows["gene_or_SNOMED_CT_ID"])
    assert "BRCA1" in set(gene_rows["gene_symbol"])
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_gtr.py -k "load_test_condition_gene" -v`
Expected: FAIL with `AttributeError: ... has no attribute 'load_test_condition_gene'`.

- [ ] **Step 4: Write minimal implementation**

Add to `src/biodb/gtr.py` (and `__all__`):

```python
def load_test_condition_gene(
    cache_dir: str | Path | None = None,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """Load the light ``test_condition_gene.txt`` TSV into a DataFrame.

    Downloads via :func:`download` if not already cached. The upstream
    header's leading ``#`` is stripped from the first column name, so the
    columns are: ``accession_version, test_type, object, GTR_identifier,
    MIM_number, object_name, gene_or_SNOMED_CT_ID, gene_symbol``. Each row
    is either a ``condition`` row (``GTR_identifier`` = MedGen CUI) or a
    ``gene`` row (``gene_or_SNOMED_CT_ID`` = Entrez id, ``gene_symbol`` set).
    """
    root = Path(cache_dir).expanduser() if cache_dir else CACHE_DIR
    path = root / TEST_CONDITION_GENE_FILE
    if force or not path.exists():
        download(cache_dir=root, force=force, progress=False)
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    df = df.rename(columns={df.columns[0]: df.columns[0].lstrip("#")})
    return df
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_gtr.py -k "load_test_condition_gene" -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/biodb/gtr.py tests/test_gtr.py tests/fixtures/gtr/test_condition_gene.txt
git commit -m "feat(gtr): load_test_condition_gene TSV reader"
```

---

## Task 8: `iter_full_records` (streaming GTRPublicData XML parser)

**Files:**
- Modify: `src/biodb/gtr.py`
- Create: `tests/fixtures/gtr/gtr_sample.xml`
- Test: `tests/test_gtr.py`

- [ ] **Step 1: Confirm best-effort element names against the live XSD**

The attribute-based extractions below (`XRef/@ID`, `DescrSet/@CUI`, `*/@Value`) are confirmed. Two child-element names are best-effort and must be confirmed before writing the parser:

Run:
```bash
curl -s https://ftp.ncbi.nlm.nih.gov/pub/GTR/data/gtr_ftp.xml.gz \
  | gunzip 2>/dev/null | head -c 400000 \
  | grep -oE "<Symbol[^>]*>.*?</Symbol>|<Name[^>]*>.*?</Name>" | head
```
Confirm the gene symbol path under `Measure/Symbol` and the condition name path under `Trait/Name`. The data uses an `<ElementValue>` wrapper (e.g. `<Symbol Type="Preferred"><ElementValue Type="Preferred">BRCA1</ElementValue></Symbol>`). If the wrapper tag differs, adjust the two `findtext(...)` paths in Step 3 and the fixture in Step 2 to match — keep them identical.

- [ ] **Step 2: Create a small GTRPublicData XML fixture**

Create `tests/fixtures/gtr/gtr_sample.xml` (mirrors the confirmed real structure — two clinical tests):

```xml
<?xml version="1.0"?>
<GTRPublicData Version="1.0">
  <GTRLabData>
    <GTRLab>
      <Organization><Name>Example Genetics Lab</Name></Organization>
      <GeneTesting GeneID="672" test_count="1"><GeneSymbol>BRCA1</GeneSymbol></GeneTesting>
    </GTRLab>
    <GTRLabTest id="509983" GTRAccession="GTR000509983" Version="1">
      <TestName>BRCA1 gene sequencing</TestName>
      <Indications>
        <TestType>Clinical</TestType>
        <Purpose>Diagnosis</Purpose>
        <TargetPop><Description>Individuals with a family history of breast cancer.</Description></TargetPop>
      </Indications>
      <Method>
        <TopCategory Value="Molecular Genetics">
          <Category Value="Sequence analysis of the entire coding region" code="C">
            <Methodology Value="Next-Generation (NGS)/Massively parallel sequencing (MPS)"/>
          </Category>
        </TopCategory>
      </Method>
      <AnalyticalValidity><Description>Analytical sensitivity &gt;99% for SNVs.</Description></AnalyticalValidity>
      <QualityControl>
        <ClinicalValidity>
          <Description>Pathogenic BRCA1 variants confer high breast cancer risk.</Description>
          <PMID>20301425</PMID>
        </ClinicalValidity>
      </QualityControl>
      <ClinicalUtility>
        <Type>Diagnosis</Type>
        <Description>Confirms a hereditary breast cancer diagnosis.</Description>
      </ClinicalUtility>
      <ClinVarSet>
        <DescrSet Type="Preferred" CUI="C0677776"/>
        <ClinVarAssertion>
          <MeasureSet>
            <Measure Type="Gene">
              <Symbol Type="Preferred"><ElementValue Type="Preferred">BRCA1</ElementValue></Symbol>
              <XRef ID="672" DB="Gene"/>
            </Measure>
          </MeasureSet>
          <TraitSet>
            <Trait>
              <Name Type="Preferred"><ElementValue Type="Preferred">Breast-ovarian cancer, familial 1</ElementValue></Name>
              <XRef DB="MedGen" Type="CUI" ID="C0677776"/>
            </Trait>
          </TraitSet>
        </ClinVarAssertion>
      </ClinVarSet>
    </GTRLabTest>
    <GTRLabTest id="4006" GTRAccession="GTR000004006" Version="2">
      <TestName>FMR1 CGG repeat analysis</TestName>
      <Indications><TestType>Clinical</TestType></Indications>
      <ClinVarSet>
        <ClinVarAssertion>
          <MeasureSet>
            <Measure Type="Gene">
              <Symbol Type="Preferred"><ElementValue Type="Preferred">FMR1</ElementValue></Symbol>
              <XRef ID="2332" DB="Gene"/>
            </Measure>
          </MeasureSet>
          <TraitSet>
            <Trait>
              <Name Type="Preferred"><ElementValue Type="Preferred">Fragile X syndrome</ElementValue></Name>
              <XRef DB="MedGen" Type="CUI" ID="C0016667"/>
            </Trait>
          </TraitSet>
        </ClinVarAssertion>
      </ClinVarSet>
    </GTRLabTest>
  </GTRLabData>
</GTRPublicData>
```

- [ ] **Step 3: Write the failing test**

Append to `tests/test_gtr.py`:

```python
def test_iter_full_records_parses_plain_xml() -> None:
    recs = list(gtr.iter_full_records(FIXTURES / "gtr_sample.xml"))
    assert [r.accession for r in recs] == ["GTR000509983", "GTR000004006"]
    brca = recs[0]
    assert brca.name == "BRCA1 gene sequencing"
    assert brca.test_type == "Clinical"
    assert {"symbol": "BRCA1", "entrez": "672"} in [
        {"symbol": g["symbol"], "entrez": g["entrez"]} for g in brca.genes
    ]
    assert "C0677776" in [c["cui"] for c in brca.conditions]
    assert "Next-Generation (NGS)/Massively parallel sequencing (MPS)" in brca.methods
    assert brca.analytical_validity.startswith("Analytical sensitivity")
    assert brca.clinical_validity.startswith("Pathogenic BRCA1")
    assert "20301425" in brca.pmids


def test_iter_full_records_reads_gzip(tmp_path) -> None:
    raw = (FIXTURES / "gtr_sample.xml").read_bytes()
    gz = tmp_path / "gtr_ftp.xml.gz"
    with gzip.open(gz, "wb") as f:
        f.write(raw)
    recs = list(gtr.iter_full_records(gz))
    assert len(recs) == 2
```

- [ ] **Step 4: Run test to verify it fails**

Run: `pytest tests/test_gtr.py -k "iter_full_records" -v`
Expected: FAIL with `AttributeError: ... has no attribute 'iter_full_records'`.

- [ ] **Step 5: Write minimal implementation**

Add to `src/biodb/gtr.py` (and `__all__`). Note: `ElementTree` supports the `[@attr='v']` predicate but not `[@attr]` (presence-only) reliably across versions, so we filter attributes in Python where needed.

```python
def _first_desc(elem: ET.Element | None) -> str:
    """Return the ``Description`` text of a TextCitations-style block."""
    if elem is None:
        return ""
    return (elem.findtext("Description") or "").strip()


def _test_from_xml_element(test: ET.Element) -> GTRTest:
    """Normalize one ``<GTRLabTest>`` element into a :class:`GTRTest`.

    Genes/conditions are pulled from the nested ``ClinVarSet`` block (Entrez
    ids live in ``Measure/XRef[@DB='Gene']/@ID``; condition CUIs in
    ``Trait/XRef[@DB='MedGen']/@ID``). The unique value of the full XML over
    the light TSV is the free-text + methodology, so those are extracted in
    full; gene symbols are best-effort (the TSV is authoritative for symbols).
    """
    accession = test.get("GTRAccession", "")
    uid = test.get("id", "")
    name = (test.findtext("TestName") or "").strip()
    test_type = (test.findtext("Indications/TestType") or "Clinical").strip()

    genes: list[dict] = []
    for measure in test.findall(".//MeasureSet/Measure"):
        if measure.get("Type") != "Gene":
            continue
        entrez = ""
        for xref in measure.findall("XRef"):
            if xref.get("DB") == "Gene" and xref.get("ID"):
                entrez = xref.get("ID", "")
                break
        symbol = (measure.findtext("Symbol/ElementValue") or "").strip()
        genes.append({"symbol": symbol, "entrez": entrez, "location": ""})

    conditions: list[dict] = []
    seen_cui: set[str] = set()
    for trait in test.findall(".//TraitSet/Trait"):
        cui = ""
        for xref in trait.findall("XRef"):
            if xref.get("DB") == "MedGen" and xref.get("ID"):
                cui = xref.get("ID", "")
                break
        cname = (trait.findtext("Name/ElementValue") or "").strip()
        key = cui or cname
        if key and key not in seen_cui:
            seen_cui.add(key)
            conditions.append({"name": cname, "cui": cui, "omim": ""})

    methods: list[str] = []
    method_categories: list[str] = []
    for cat in test.findall("Method/TopCategory/Category"):
        if cat.get("Value"):
            method_categories.append(cat.get("Value", ""))
        for methodology in cat.findall("Methodology"):
            if methodology.get("Value"):
                methods.append(methodology.get("Value", ""))

    clinical_validity = _first_desc(test.find("QualityControl/ClinicalValidity"))
    clinical_utility = "\n\n".join(
        _first_desc(cu) for cu in test.findall("ClinicalUtility") if _first_desc(cu)
    )
    pmids: list[str] = []
    for block in (*test.findall("QualityControl/ClinicalValidity"), *test.findall("ClinicalUtility")):
        pmids.extend((p.text or "").strip() for p in block.findall("PMID") if (p.text or "").strip())

    return GTRTest(
        accession=accession,
        uid=uid,
        name=name,
        test_type=test_type,
        genes=genes,
        conditions=conditions,
        methods=methods,
        method_categories=method_categories,
        analytical_validity=_first_desc(test.find("AnalyticalValidity")),
        clinical_validity=clinical_validity,
        clinical_utility=clinical_utility,
        target_population=_first_desc(test.find("Indications/TargetPop")),
        test_purpose=[p.text.strip() for p in test.findall("Indications/Purpose") if p.text and p.text.strip()],
        pmids=pmids,
    )


def iter_full_records(
    path: str | Path | None = None,
    *,
    cache_dir: str | Path | None = None,
    force: bool = False,
    include_research: bool = True,
) -> Iterator[GTRTest]:
    """Stream :class:`GTRTest` records from the full ``gtr_ftp.xml.gz`` dump.

    Uses ``ElementTree.iterparse`` and clears each test element after read,
    so memory stays roughly flat over the full ~224 MB file. Downloads via
    :func:`download` (``full_xml=True``) if ``path`` is not given and no
    cached copy exists.

    Parameters
    ----------
    path : str or Path, optional
        An explicit ``.xml`` or ``.xml.gz`` file. If omitted, uses the
        cached ``gtr_ftp.xml.gz`` (downloading it if necessary).
    cache_dir : str or Path, optional
    force : bool, default False
    include_research : bool, default True
        Also yield ``<GTRLabResearchTest>`` elements (same parser); set
        False to restrict to clinical ``<GTRLabTest>`` records.

    Yields
    ------
    GTRTest
    """
    if path is None:
        root = Path(cache_dir).expanduser() if cache_dir else CACHE_DIR
        path = root / FULL_XML_FILE
        if force or not path.exists():
            download(cache_dir=root, full_xml=True, force=force, progress=False)
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    tags = {"GTRLabTest"}
    if include_research:
        tags.add("GTRLabResearchTest")

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        for _, elem in ET.iterparse(handle, events=("end",)):
            if elem.tag in tags:
                yield _test_from_xml_element(elem)
                elem.clear()
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_gtr.py -k "iter_full_records" -v`
Expected: PASS (2 tests).

- [ ] **Step 7: Commit**

```bash
git add src/biodb/gtr.py tests/test_gtr.py tests/fixtures/gtr/gtr_sample.xml
git commit -m "feat(gtr): streaming iter_full_records parser for gtr_ftp.xml.gz"
```

---

## Task 9: `panel_text` (embeddable free-text hook)

**Files:**
- Modify: `src/biodb/gtr.py`
- Test: `tests/test_gtr.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gtr.py`:

```python
def test_panel_text_assembles_fields() -> None:
    rec = gtr._test_from_esummary(_esummary_record())
    text = gtr.panel_text(rec)
    assert "BRCA1 gene sequencing" in text
    assert "Breast-ovarian cancer" in text
    assert "Pathogenic BRCA1" in text  # clinical validity
    assert "Next-Generation" in text   # methodology
    # default excludes nothing requested; lab name is not part of the blob
    assert "Example Genetics Lab" not in text


def test_panel_text_respects_include() -> None:
    rec = gtr._test_from_esummary(_esummary_record())
    text = gtr.panel_text(rec, include=("name",))
    assert text.strip() == "BRCA1 gene sequencing"


def test_panel_text_accepts_dict() -> None:
    rec = gtr._test_from_esummary(_esummary_record())
    text = gtr.panel_text(rec.__dict__, include=("name",))
    assert "BRCA1 gene sequencing" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gtr.py -k "panel_text" -v`
Expected: FAIL with `AttributeError: ... has no attribute 'panel_text'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/biodb/gtr.py` (and `__all__`):

```python
import dataclasses

_PANEL_TEXT_DEFAULT = (
    "name", "conditions", "clinical_validity", "clinical_utility",
    "analytical_validity", "target_population", "test_purpose", "methods",
)
"""Default field order for :func:`panel_text` — descriptive content only
(no lab/logistics), suited to embedding for HaploForge weighting."""


def panel_text(
    record: GTRTest | dict,
    include: tuple[str, ...] = _PANEL_TEXT_DEFAULT,
) -> str:
    """Assemble a clean free-text description of a panel/assay, ready to embed.

    bioDB stops here — embedding + cosine-similarity weighting is a
    downstream (HaploForge/biodocs) concern. The text concatenates the
    requested ``include`` fields (in order), flattening list fields
    (conditions → names; methods/test_purpose → joined) and dropping empties.

    Parameters
    ----------
    record : GTRTest or dict
        A record from :func:`query_test` / :func:`iter_full_records`, or its
        ``__dict__``.
    include : tuple[str, ...]
        Field names to include, in order. Defaults to descriptive fields
        only (see :data:`_PANEL_TEXT_DEFAULT`).

    Returns
    -------
    str
        Newline-joined text blob.
    """
    data = dataclasses.asdict(record) if isinstance(record, GTRTest) else dict(record)
    parts: list[str] = []
    for field_name in include:
        value = data.get(field_name)
        if not value:
            continue
        if field_name == "conditions":
            parts.append(", ".join(c.get("name", "") for c in value if c.get("name")))
        elif field_name in ("methods", "method_categories", "test_purpose"):
            parts.append(", ".join(str(v) for v in value if v))
        elif isinstance(value, str):
            parts.append(value)
    return "\n".join(p for p in parts if p.strip())
```

Move `import dataclasses` to the top import block.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_gtr.py -k "panel_text" -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/biodb/gtr.py tests/test_gtr.py
git commit -m "feat(gtr): panel_text embeddable free-text assembler"
```

---

## Task 10: `gene_sets` (raw per-test long frame)

**Files:**
- Modify: `src/biodb/gtr.py`
- Test: `tests/test_gtr.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gtr.py`:

```python
def test_gene_sets_builds_long_frame(tmp_path) -> None:
    (tmp_path / gtr.TEST_CONDITION_GENE_FILE).write_bytes(
        (FIXTURES / "test_condition_gene.txt").read_bytes()
    )
    df = gtr.gene_sets(cache_dir=tmp_path)
    assert set(df.columns) == {
        "panel_id", "panel_name", "condition_cui", "gene_symbol", "gene_entrez"
    }
    # FMR1 panel joins the gene row to the panel's condition CUI.
    fmr1 = df[df["gene_symbol"] == "FMR1"]
    assert "C0016667" in set(fmr1["condition_cui"])
    assert "2332" in set(fmr1["gene_entrez"])
    # Both panels present.
    assert set(df["panel_id"]) == {"GTR000004006.1", "GTR000509983.1"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gtr.py -k "gene_sets and not aggregate" -v`
Expected: FAIL with `AttributeError: ... has no attribute 'gene_sets'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/biodb/gtr.py` (and `__all__`):

```python
def gene_sets(
    cache_dir: str | Path | None = None,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """Build raw per-test gene sets from the light TSV.

    Returns a long DataFrame with one row per (test, gene), columns:
    ``panel_id`` (GTR accession.version), ``panel_name`` (test's condition
    name), ``condition_cui`` (the test's condition MedGen CUI),
    ``gene_symbol``, ``gene_entrez``. Highly redundant by design — the same
    panel recurs once per lab; see :func:`aggregate_gene_sets` to collapse.
    """
    df = load_test_condition_gene(cache_dir=cache_dir, force=force)
    genes = df[df["object"] == "gene"].copy()
    conds = df[df["object"] == "condition"].copy()
    # A panel's condition: take the first condition row per accession.
    cond_first = conds.groupby("accession_version", as_index=False).agg(
        condition_cui=("GTR_identifier", "first"),
        panel_name=("object_name", "first"),
    )
    out = genes.merge(cond_first, on="accession_version", how="left").fillna("")
    out = out.rename(
        columns={
            "accession_version": "panel_id",
            "gene_or_SNOMED_CT_ID": "gene_entrez",
        }
    )
    return out[
        ["panel_id", "panel_name", "condition_cui", "gene_symbol", "gene_entrez"]
    ].reset_index(drop=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_gtr.py -k "gene_sets and not aggregate" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/biodb/gtr.py tests/test_gtr.py
git commit -m "feat(gtr): gene_sets raw per-test long frame"
```

---

## Task 11: `aggregate_gene_sets` (dedup + support_count prior)

**Files:**
- Modify: `src/biodb/gtr.py`
- Test: `tests/test_gtr.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gtr.py`. The fixture has one panel per condition, so add a second FMR1 panel inline to exercise the support count:

```python
def _multi_panel_df() -> pd.DataFrame:
    # Two distinct tests for Fragile X, both including FMR1; one also AFF2.
    return pd.DataFrame(
        [
            {"panel_id": "GTR1", "panel_name": "Fragile X", "condition_cui": "C0016667",
             "gene_symbol": "FMR1", "gene_entrez": "2332"},
            {"panel_id": "GTR2", "panel_name": "Fragile X", "condition_cui": "C0016667",
             "gene_symbol": "FMR1", "gene_entrez": "2332"},
            {"panel_id": "GTR2", "panel_name": "Fragile X", "condition_cui": "C0016667",
             "gene_symbol": "AFF2", "gene_entrez": "2334"},
        ]
    )


def test_aggregate_gene_sets_by_condition_counts_support(monkeypatch) -> None:
    monkeypatch.setattr(gtr, "gene_sets", lambda **_: _multi_panel_df())
    agg = gtr.aggregate_gene_sets(by="condition")
    assert set(agg.columns) == {"set_id", "set_name", "gene_symbol", "gene_entrez", "support_count"}
    fmr1 = agg[(agg["set_id"] == "C0016667") & (agg["gene_symbol"] == "FMR1")]
    assert int(fmr1["support_count"].iloc[0]) == 2   # two distinct panels include FMR1
    aff2 = agg[(agg["set_id"] == "C0016667") & (agg["gene_symbol"] == "AFF2")]
    assert int(aff2["support_count"].iloc[0]) == 1


def test_aggregate_gene_sets_by_test_name(monkeypatch) -> None:
    monkeypatch.setattr(gtr, "gene_sets", lambda **_: _multi_panel_df())
    agg = gtr.aggregate_gene_sets(by="test_name")
    # grouped by panel_name "Fragile X"
    assert set(agg["set_id"]) == {"Fragile X"}
    assert int(agg[agg["gene_symbol"] == "FMR1"]["support_count"].iloc[0]) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gtr.py -k "aggregate_gene_sets" -v`
Expected: FAIL with `AttributeError: ... has no attribute 'aggregate_gene_sets'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/biodb/gtr.py` (and `__all__`):

```python
def aggregate_gene_sets(
    by: str = "test_name",
    *,
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Collapse the redundant per-test gene sets into deduplicated sets.

    Parameters
    ----------
    by : {"test_name", "condition"}, default "test_name"
        Grouping key. ``"condition"`` groups by the panel's MedGen CUI;
        ``"test_name"`` groups by the panel/condition display name.
    cache_dir, force
        Forwarded to :func:`gene_sets`.

    Returns
    -------
    pd.DataFrame
        Columns: ``set_id`` (the group key), ``set_name``, ``gene_symbol``,
        ``gene_entrez``, ``support_count``. ``support_count`` is the number
        of *distinct* tests/labs whose panel includes that gene — a free,
        source-grounded importance prior for downstream weighting.
    """
    if by not in ("test_name", "condition"):
        raise ValueError(f"by must be 'test_name' or 'condition', got {by!r}")
    raw = gene_sets(cache_dir=cache_dir, force=force)
    key = "condition_cui" if by == "condition" else "panel_name"

    # support_count = number of distinct panels (panel_id) in this group that
    # include this gene.
    grouped = (
        raw.groupby([key, "gene_entrez", "gene_symbol"], as_index=False)
        .agg(support_count=("panel_id", "nunique"))
    )
    name_map = (
        raw.groupby(key)["panel_name"].first()
        if by == "condition"
        else raw.groupby(key)[key].first()
    )
    grouped = grouped.rename(columns={key: "set_id"})
    grouped["set_name"] = grouped["set_id"].map(name_map).fillna(grouped["set_id"])
    return grouped[
        ["set_id", "set_name", "gene_symbol", "gene_entrez", "support_count"]
    ].reset_index(drop=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_gtr.py -k "aggregate_gene_sets" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/biodb/gtr.py tests/test_gtr.py
git commit -m "feat(gtr): aggregate_gene_sets with support_count importance prior"
```

---

## Task 12: `to_gmt` (GMT export)

**Files:**
- Modify: `src/biodb/gtr.py`
- Test: `tests/test_gtr.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gtr.py`:

```python
def test_to_gmt_raw_roundtrips(tmp_path, monkeypatch) -> None:
    from biodb.utils import read_gmt

    monkeypatch.setattr(gtr, "gene_sets", lambda **_: _multi_panel_df())
    out = tmp_path / "panels.gmt"
    gtr.to_gmt(out)  # by=None -> raw per-panel sets
    assert out.exists()
    parsed = read_gmt(out, return_format="dict")
    # GTR2 panel has FMR1 + AFF2
    gtr2_genes = {g for (sid, _desc), genes in parsed.items() if sid == "GTR2" for g in genes}
    assert gtr2_genes == {"FMR1", "AFF2"}


def test_to_gmt_aggregated(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(gtr, "gene_sets", lambda **_: _multi_panel_df())
    out = tmp_path / "agg.gmt"
    gtr.to_gmt(out, by="condition")
    lines = out.read_text().strip().splitlines()
    # one line per condition set
    assert any(line.startswith("C0016667\t") for line in lines)
    assert "FMR1" in lines[0] and "AFF2" in lines[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gtr.py -k "to_gmt" -v`
Expected: FAIL with `AttributeError: ... has no attribute 'to_gmt'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/biodb/gtr.py` (and `__all__`):

```python
def to_gmt(
    path: str | Path,
    *,
    by: str | None = None,
    gene_id: str = "symbol",
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Export GTR gene sets to a GMT file (``set\\tdescription\\tgene...``).

    Plugs straight into :func:`biodb.utils.read_gmt` / GenForge.

    Parameters
    ----------
    path : str or Path
        Output ``.gmt`` path.
    by : {None, "test_name", "condition"}
        ``None`` (default) exports raw per-test sets keyed by ``panel_id``;
        otherwise exports the deduplicated :func:`aggregate_gene_sets` output.
    gene_id : {"symbol", "entrez"}, default "symbol"
        Which gene identifier to emit.
    cache_dir, force
        Forwarded to the underlying builders.

    Returns
    -------
    pathlib.Path
        The written path.
    """
    col = "gene_symbol" if gene_id == "symbol" else "gene_entrez"
    if by is None:
        df = gene_sets(cache_dir=cache_dir, force=force)
        id_col, name_col = "panel_id", "panel_name"
    else:
        df = aggregate_gene_sets(by=by, cache_dir=cache_dir, force=force)
        id_col, name_col = "set_id", "set_name"

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for set_id, group in df.groupby(id_col, sort=False):
            genes = [g for g in dict.fromkeys(group[col]) if g and g != "N/A"]
            if not genes:
                continue
            description = str(group[name_col].iloc[0]) if name_col in group else ""
            handle.write("\t".join([str(set_id), description, *genes]) + "\n")
    return path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_gtr.py -k "to_gmt" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/biodb/gtr.py tests/test_gtr.py
git commit -m "feat(gtr): to_gmt export (raw or aggregated) for GenForge"
```

---

## Task 13: Top-level wiring (`__init__.py` + `api.rst`)

**Files:**
- Modify: `src/biodb/__init__.py`
- Modify: `docs/api.rst`
- Test: `tests/test_gtr.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gtr.py`:

```python
def test_top_level_reexports() -> None:
    import biodb

    assert biodb.gtr is gtr
    for name in ("gtr_search_tests", "gtr_query_test", "gtr_gene_sets", "gtr_to_gmt"):
        assert hasattr(biodb, name), name
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gtr.py -k "top_level" -v`
Expected: FAIL with `AttributeError: module 'biodb' has no attribute 'gtr'`.

- [ ] **Step 3: Write minimal implementation**

In `src/biodb/__init__.py`:

1. Add `gtr,` to the alphabetized submodule import block (after `gprofiler,` / before `gwas_atlas,`).
2. Add a re-export block (namespaced to avoid colliding with other sources' `query_*`/`gene_sets`):

```python
from biodb.gtr import (
    aggregate_gene_sets as gtr_aggregate_gene_sets,
)
from biodb.gtr import (
    gene_sets as gtr_gene_sets,
)
from biodb.gtr import (
    panel_text as gtr_panel_text,
)
from biodb.gtr import (
    query_test as gtr_query_test,
)
from biodb.gtr import (
    search_tests as gtr_search_tests,
)
from biodb.gtr import (
    to_gmt as gtr_to_gmt,
)
```

3. Add to `__all__` (alphabetized): `"gtr"`, `"gtr_aggregate_gene_sets"`, `"gtr_gene_sets"`, `"gtr_panel_text"`, `"gtr_query_test"`, `"gtr_search_tests"`, `"gtr_to_gmt"`.

In `docs/api.rst`, add `gtr` to the autosummary list (after `gprofiler` / before `msigdb` to match the README/source ordering — placement is cosmetic):

```rst
   gprofiler
   gtr
   msigdb
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_gtr.py -k "top_level" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/biodb/__init__.py docs/api.rst tests/test_gtr.py
git commit -m "feat(gtr): top-level re-exports + API-reference autosummary entry"
```

---

## Task 14: Documentation — `docs/gtr.md`, index, quickstart, README

**Files:**
- Create: `docs/gtr.md`
- Modify: `docs/index.md`
- Modify: `docs/quickstart.md`
- Modify: `README.md`

- [ ] **Step 1: Write `docs/gtr.md`**

Create `docs/gtr.md` (mirror `docs/ols.md`'s structure: intro, "which mode" table, worked examples for all four use shapes):

```markdown
# Querying genetic tests with GTR

The [`biodb.gtr`](api.rst) module is a client for NCBI's
[Genetic Testing Registry (GTR)](https://www.ncbi.nlm.nih.gov/gtr/) — a
catalog of ~64k genetic **tests**, where each record is *one test offered by
one lab*. A record carries the lab, the **gene list / panel** (each gene with
an Entrez ID + cytoband), the **conditions** (MedGen/UMLS CUIs + OMIM), a
3-level **methodology** taxonomy, a clinical-vs-research flag, and clinical
validity/utility free-text.

`biodb.gtr` exposes both bioDB modes:

| You want… | Use |
|---|---|
| Tests for one gene / condition / accession | `query_gene`, `query_condition`, `query_test` (E-utilities) |
| A raw test id search scoped to a GTR index | `search_tests(term, field=...)` |
| All test↔gene↔condition mappings (fast) | `download()` + `load_test_condition_gene()` (daily TSV) |
| The full records incl. descriptions/methodology | `download(full_xml=True)` + `iter_full_records()` (224 MB) |
| Curated gene sets for GenForge | `gene_sets`, `aggregate_gene_sets`, `to_gmt` |
| Embeddable panel text for HaploForge weighting | `panel_text` (+ the `support_count` prior) |

## Targeted lookups (API mode)

```python
from biodb import gtr

# Every test that targets BRCA1 (gene-symbol index).
tests = gtr.query_gene("BRCA1", retmax=50)
print(tests[0].name, tests[0].lab, [g["entrez"] for g in tests[0].genes])

# One record by accession (or bare uid).
brca = gtr.query_test("GTR000509983")
print(brca.clinical_validity, brca.methods)

# Tests for a condition by name or MedGen CUI.
hboc = gtr.query_condition("C0677776")     # CUI -> DCUI index
hboc = gtr.query_condition("breast cancer") # text -> DISNAME index
```

GTR rides NCBI E-utilities, so the 3 req/sec un-keyed cap applies; pass
`api_key="..."` to any query to lift it to 10 req/sec. The numeric Entrez UID
is **not** the accession — `accession_from_uid(509983) == "GTR000509983"`.

## Bulk gene sets for GenForge

```python
from biodb import gtr

# Raw per-test gene sets (one row per test×gene), highly redundant.
raw = gtr.gene_sets()

# Deduplicated sets with a source-grounded importance prior: support_count
# is the number of distinct labs/tests whose panel includes each gene.
panels = gtr.aggregate_gene_sets(by="condition")

# Export to GMT for GenForge (set -> gene vector signature).
gtr.to_gmt("gtr_condition_panels.gmt", by="condition")
```

## Panel text for HaploForge weighting

bioDB produces the clean inputs; the embedding + cosine-similarity weighting
lives downstream in HaploForge/biodocs.

```python
from biodb import gtr

brca = gtr.query_test("GTR000509983")
text = gtr.panel_text(brca)   # name + conditions + clinical text + methods
# -> feed `text` to your embedding backend; weight each gene in brca.genes
#    by cosine similarity to the panel embedding. The aggregate_gene_sets
#    support_count is a complementary, embedding-free prior.
```

For descriptions across *all* panels in one pass, stream the full dump:

```python
for rec in gtr.iter_full_records():          # downloads gtr_ftp.xml.gz once
    blob = gtr.panel_text(rec)
    ...                                        # embed / index downstream
```
```

- [ ] **Step 2: Wire `docs/gtr.md` into index + toctree**

In `docs/index.md`, add a bullet to the "Why use it" list (after the `ols` bullet):

```markdown
- **`gtr`** — NCBI Genetic Testing Registry client. Targeted E-utilities
  queries (`query_gene` / `query_condition` / `query_test`) plus bulk
  downloads (`download`, `load_test_condition_gene`, streaming
  `iter_full_records`). Materializes curated per-panel gene sets
  (`gene_sets` / `aggregate_gene_sets` / `to_gmt`) and embeddable panel
  text (`panel_text`) with a `support_count` importance prior. See
  [Querying genetic tests with GTR](gtr.md).
```

And add `gtr` to the hidden toctree (after `ols`):

```
quickstart
ols
gtr
api
changelog
```

- [ ] **Step 3: Add a quickstart snippet**

In `docs/quickstart.md`, append a short section:

```markdown
## Genetic tests + panels (GTR)

```python
from biodb import gtr

# Tests targeting a gene, and a panel's embeddable description.
tests = gtr.query_gene("BRCA1", retmax=10)
text = gtr.panel_text(tests[0])

# Curated gene sets -> GMT for GenForge.
gtr.to_gmt("gtr_panels.gmt", by="condition")
```

See [Querying genetic tests with GTR](gtr.md) for the full surface.
```

- [ ] **Step 4: Add the README Sources-table row**

In `README.md`, add a row to the Sources table (after the OmicsPred / gProfiler region — placement just needs to be inside the table):

```markdown
| **[GTR (Genetic Testing Registry)](https://www.ncbi.nlm.nih.gov/gtr/)** — NCBI catalog of ~64k genetic tests: per-test gene panels (Entrez IDs), conditions (MedGen CUIs), methodology, clinical validity/utility | [`biodb.gtr`](src/biodb/gtr.py) | ✅ NCBI E-utilities (`search_tests`, `query_test`, `query_gene`, `query_condition`) | ✅ FTP TSV + full XML (`download`, `load_test_condition_gene`, `iter_full_records`) + gene-set views (`gene_sets`, `aggregate_gene_sets`, `to_gmt`, `panel_text`) |
```

- [ ] **Step 5: Verify docs build cleanly**

Run:
```bash
sphinx-build -b html docs docs/_build/html 2>&1 | grep -iE "gtr|warning|error" | head -40
```
Expected: the GTR pages render; no new `WARNING`/`ERROR` lines referencing `gtr`. (Pre-existing unrelated warnings, if any, are acceptable — compare against a build of `main` if unsure.)

- [ ] **Step 6: Commit**

```bash
git add docs/gtr.md docs/index.md docs/quickstart.md README.md
git commit -m "docs(gtr): user guide, index/quickstart wiring, README Sources row"
```

---

## Task 15: Live-network smoke tests (marked `network`)

**Files:**
- Modify: `tests/test_gtr.py`

- [ ] **Step 1: Write the network-marked tests**

Append to `tests/test_gtr.py` (these hit real NCBI; skipped in CI via the `network` marker):

```python
@pytest.mark.network
def test_live_query_gene_brca1() -> None:
    tests = gtr.query_gene("BRCA1", retmax=3)
    assert tests
    assert all(t.accession.startswith("GTR") for t in tests)
    assert any("BRCA1" in (g["symbol"] for g in t.genes) for t in tests)


@pytest.mark.network
def test_live_search_then_query() -> None:
    accs = gtr.search_tests("BRCA1", field="SYMB", retmax=2)
    assert accs and accs[0].startswith("GTR")
    rec = gtr.query_test(accs[0])
    assert rec.name


@pytest.mark.network
@pytest.mark.slow
def test_live_tsv_download_and_gene_sets(tmp_path) -> None:
    df = gtr.load_test_condition_gene(cache_dir=tmp_path)
    assert "accession_version" in df.columns
    sets = gtr.gene_sets(cache_dir=tmp_path)
    assert {"panel_id", "gene_entrez"}.issubset(sets.columns)
    assert len(sets) > 1000
```

- [ ] **Step 2: Verify they're collected but skipped under the CI filter**

Run: `pytest tests/test_gtr.py -m "not slow and not network" -v`
Expected: the three `live_*` tests are deselected; all offline tests PASS.

Run (optional, real network): `pytest tests/test_gtr.py -m network -v`
Expected: PASS against live NCBI (may be slow; respects rate limits).

- [ ] **Step 3: Commit**

```bash
git add tests/test_gtr.py
git commit -m "test(gtr): live-network smoke tests (marked network/slow)"
```

---

## Task 16: Full verification sweep

**Files:** none (verification only)

- [ ] **Step 1: Lint**

Run: `ruff check src/biodb/gtr.py tests/test_gtr.py`
Expected: no errors. Fix any reported issues (the module is held to the full ruleset — it is NOT in `extend-exclude`). Note `gtr.py` is first-party, so do **not** add it to the ruff/coverage exclude lists.

- [ ] **Step 2: Format check**

Run: `ruff format --check src/biodb/gtr.py tests/test_gtr.py`
Expected: clean. If it reports a diff, run `ruff format src/biodb/gtr.py tests/test_gtr.py` and re-commit.

- [ ] **Step 3: Full offline suite + coverage**

Run: `pytest -m "not slow and not network" -q`
Expected: all tests pass, including the existing `tests/test_source_api_modes.py` (if it enumerates modules, confirm GTR doesn't break its expectations; add GTR to its coverage if that file maintains an explicit module list — check `tests/test_source_api_modes.py` first).

- [ ] **Step 4: Doctest sanity (import surface)**

Run: `python -c "import biodb; from biodb import gtr; print(gtr.__all__)"`
Expected: prints the `__all__` list without error.

- [ ] **Step 5: Final commit (if formatting/lint forced changes)**

```bash
git add -A
git commit -m "chore(gtr): lint + format pass" || echo "nothing to commit"
```

---

## Self-Review (completed during planning)

- **Spec coverage:** every spec section maps to a task — API mode (Tasks 3–5), bulk `download` (Task 6), TSV consumer (Task 7), rich XML consumer (Task 8), gene-set views (Tasks 10–12), `panel_text` hook (Task 9), `GTRTest` record (Task 2), testing (every task + 15), in-code docs (docstrings written inline in each task), website docs (Task 14), `__init__`/`api.rst` wiring (Task 13). Out-of-scope items (embedding, variant-level ClinVar, submission) are not implemented — correct.
- **Type consistency:** `GTRTest` fields are defined once (Task 2) and reused verbatim by both normalizers (Tasks 2, 8) and `panel_text` (Task 9). `gene_sets` emits `panel_id/panel_name/condition_cui/gene_symbol/gene_entrez`; `aggregate_gene_sets` consumes those exact names and emits `set_id/set_name/gene_symbol/gene_entrez/support_count`; `to_gmt` consumes both shapes by branching on `by`. Function names match the approved spec surface.
- **Placeholder scan:** no TBD/TODO; the one best-effort item (XML `Symbol/ElementValue` / `Trait/Name/ElementValue` child names) has an explicit confirmation step (Task 8 Step 1) with the exact command to verify and what to change — not a vague placeholder.
