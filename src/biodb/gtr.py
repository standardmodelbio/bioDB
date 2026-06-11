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
  ``full_xml=True``, also the ~224 MB ``gtr_ftp.xml.gz`` (the only source
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

import dataclasses
import gzip
import logging
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import requests

from biodb._downloads import stream_to_file

logger = logging.getLogger(__name__)

NCBI_EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
"""NCBI E-utilities root. ``esearch`` / ``esummary`` live below."""

GTR_FTP_BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pub/GTR/data"
"""GTR bulk FTP data root (HTTPS-served)."""

GTR_DATA_SERVICE_URL = "https://www.ncbi.nlm.nih.gov/gtr/api/v1"
"""Undocumented-but-live richer JSON REST API (reserved for future use)."""

TEST_CONDITION_GENE_FILE = "test_condition_gene.txt"
"""Light daily TSV: test ↔ gene(Entrez) ↔ condition(CUI/SNOMED) mapping."""

FULL_XML_FILE = "gtr_ftp.xml.gz"
"""Full weekly XML dump (~224 MB) — the only bulk source with descriptions."""

CACHE_DIR = Path("~/.cache/biodb/gtr").expanduser()
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_RATE_LIMIT_SLEEP_S = 0.34
"""Per-request sleep that keeps us under the un-keyed 3 req/sec cap."""

_USER_AGENT = "biodb/0.1 (+https://github.com/bschilder/bioDB)"

_CUI_RE = re.compile(r"^C\d{7}$")
"""A UMLS/MedGen CUI looks like ``C`` + 7 digits (e.g. ``C0677776``)."""


# ─── NCBI E-utilities transport ────────────────────────────────────────────


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


# ─── Normalized record ─────────────────────────────────────────────────────


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
        u.get("description", "")
        for u in cu_items
        if isinstance(u, dict) and u.get("description")
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


# ─── API / targeted mode ───────────────────────────────────────────────────


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
    return [
        _test_from_esummary(result[u]) for u in result.get("uids", []) if u in result
    ]


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
        str(gene),
        field=field,
        retmax=retmax,
        as_accession=False,
        api_key=api_key,
        timeout=timeout,
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
        condition,
        field=field,
        retmax=retmax,
        as_accession=False,
        api_key=api_key,
        timeout=timeout,
    )
    return _esummary_records(uids, api_key=api_key, timeout=timeout)


# ─── Bulk mode — download + readers ────────────────────────────────────────


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
        Also download the ~224 MB full XML dump.
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
            f"{GTR_FTP_BASE_URL}/{TEST_CONDITION_GENE_FILE}",
            tsv_path,
            headers={"User-Agent": _USER_AGENT},
            timeout=timeout,
            progress=progress,
        )

    xml_path: Path | None = None
    if full_xml:
        xml_path = root / FULL_XML_FILE
        if force or not xml_path.exists():
            stream_to_file(
                f"{GTR_FTP_BASE_URL}/{FULL_XML_FILE}",
                xml_path,
                headers={"User-Agent": _USER_AGENT},
                timeout=timeout,
                progress=progress,
            )

    return {"tsv": tsv_path, "xml": xml_path}


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


# ─── Bulk mode — streaming full-XML parser ─────────────────────────────────


def _preferred_text(parent: ET.Element, tag: str) -> str:
    """Text of ``parent``'s ``tag`` child, preferring ``Type="Preferred"``.

    GTR's ClinVar-style ``<Symbol>`` / ``<Name>`` elements repeat with a
    ``Type`` attribute (``Preferred`` + ``Alternate``s); we want the
    preferred display value, falling back to the first child if none is
    explicitly preferred.
    """
    children = parent.findall(tag)
    for child in children:
        if child.get("Type") == "Preferred" and (child.text or "").strip():
            return child.text.strip()
    for child in children:
        if (child.text or "").strip():
            return child.text.strip()
    return ""


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
    full; gene symbols come from the preferred ``<Symbol>`` child.
    """
    accession = test.get("GTRAccession", "")
    uid = test.get("id", "")
    name = (test.findtext("TestName") or "").strip()
    test_type = (test.findtext("Indications/TestType") or "Clinical").strip()

    genes: list[dict] = []
    seen_genes: set[str] = set()
    for measure in test.findall(".//MeasureSet/Measure"):
        if measure.get("Type") != "Gene":
            continue
        entrez = ""
        for xref in measure.findall("XRef"):
            if xref.get("DB") == "Gene" and xref.get("ID"):
                entrez = xref.get("ID", "")
                break
        symbol = _preferred_text(measure, "Symbol")
        key = entrez or symbol
        if key and key not in seen_genes:
            seen_genes.add(key)
            genes.append({"symbol": symbol, "entrez": entrez, "location": ""})

    conditions: list[dict] = []
    seen_cui: set[str] = set()
    for trait in test.findall(".//TraitSet/Trait"):
        cui = ""
        for xref in trait.findall("XRef"):
            if xref.get("DB") == "MedGen" and xref.get("ID"):
                cui = xref.get("ID", "")
                break
        cname = _preferred_text(trait, "Name")
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
    for block in (
        *test.findall("QualityControl/ClinicalValidity"),
        *test.findall("ClinicalUtility"),
    ):
        pmids.extend(
            (p.text or "").strip()
            for p in block.findall("PMID")
            if (p.text or "").strip()
        )

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
        test_purpose=[
            p.text.strip()
            for p in test.findall("Indications/Purpose")
            if p.text and p.text.strip()
        ],
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


# ─── Gene-set views + embeddable text ──────────────────────────────────────


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
    grouped = raw.groupby(
        [key, "gene_entrez", "gene_symbol"], as_index=False
    ).agg(support_count=("panel_id", "nunique"))
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


def to_gmt(
    path: str | Path,
    *,
    by: str | None = None,
    gene_id: str = "symbol",
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    r"""Export GTR gene sets to a GMT file (``set\tdescription\tgene...``).

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


_PANEL_TEXT_DEFAULT = (
    "name",
    "conditions",
    "clinical_validity",
    "clinical_utility",
    "analytical_validity",
    "target_population",
    "test_purpose",
    "methods",
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


__all__ = [
    "CACHE_DIR",
    "FULL_XML_FILE",
    "GTR_DATA_SERVICE_URL",
    "GTR_FTP_BASE_URL",
    "GTRTest",
    "NCBI_EUTILS_BASE_URL",
    "TEST_CONDITION_GENE_FILE",
    "accession_from_uid",
    "aggregate_gene_sets",
    "download",
    "gene_sets",
    "iter_full_records",
    "load_test_condition_gene",
    "panel_text",
    "query_condition",
    "query_gene",
    "query_test",
    "search_tests",
    "to_gmt",
]
