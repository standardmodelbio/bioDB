# bioDB GTR integration — design

**Date:** 2026-06-11
**Module:** `src/biodb/gtr.py`
**Status:** approved design, ready for implementation plan

## Goal

Add NCBI's **Genetic Testing Registry (GTR)** — <https://www.ncbi.nlm.nih.gov/gtr/> —
as a first-party bioDB source, following the library's dual-mode convention
(targeted API queries + bulk downloads). The primary downstream consumers are:

- **GenForge** — wants curated **gene sets** to feed in as gene-vector signatures.
- **HaploForge** — wants to subset genes to particular **assays/panels**, plus
  per-gene **importance-weighting priors**. bioDB supplies the raw materials
  (clean gene lists + an embeddable free-text description per panel); the actual
  embedding / cosine-similarity weighting happens downstream in HaploForge/biodocs,
  **not** in bioDB.

## What GTR is (verified against live endpoints, June 2026)

GTR catalogs **genetic tests**. Each record is *one test offered by one lab*
(~64k records, highly redundant — the same conceptual panel appears once per lab).
A record carries:

- **Identity:** GTR accession (`GTR%09d`, e.g. `GTR000509983`), test name, test
  type (`Clinical` | `Research`), test code, CPT code.
- **Genes / panel:** the `analytes` array — each gene as `{name, geneid (Entrez),
  location (cytoband)}`. A multi-gene panel just lists multiple Gene analytes;
  `genecount` gives the size. These are **curated, per-test gene lists**.
- **Conditions:** `conditionlist` — each `{name, cui (MedGen/UMLS), MIM}`.
- **Methodology:** a 3-level taxonomy — TopCategory (e.g. "Molecular Genetics")
  → Category (e.g. "Sequence analysis of the entire coding region",
  "Deletion/duplication analysis") → method (e.g. "NGS/MPS").
- **Clinical free-text:** `analyticalvalidity`, `clinicalvalidity`,
  `clinicalutility`, `targetpopulation`, `testpurpose`, each with description +
  PMIDs.
- **Lab + logistics:** offerer org/address, certifications (CLIA/CAP), URLs.

### Access paths

| Mode | Endpoint | Notes |
|---|---|---|
| **Targeted (API)** | E-utilities `db=gtr` — `esearch` + `esummary` (`retmode=json`) | 3 req/s (10 with API key). UID ≠ accession (`GTR%09d`). Rich indexed fields: `SYMB`, `GENEID`, `DISNAME`, `DCUI`, `MTOD`, … |
| **Targeted (richer, optional)** | `https://www.ncbi.nlm.nih.gov/gtr/api/v1/tests/{id}/` | Undocumented-but-live JSON REST; adds lab/ClinVar-participation fields. Throttle conservatively. |
| **Bulk (light)** | `https://ftp.ncbi.nlm.nih.gov/pub/GTR/data/test_condition_gene.txt` | ~46 MB TSV, **daily**. 8 columns: `accession_version, test_type, object (condition\|gene), GTR_identifier, MIM_number, object_name, gene_or_SNOMED_CT_ID, gene_symbol`. **No descriptions.** Effectively a ready-made test→gene(Entrez)→condition(CUI) table. |
| **Bulk (rich)** | `https://ftp.ncbi.nlm.nih.gov/pub/GTR/data/gtr_ftp.xml.gz` | ~214 MB gzip XML, **weekly**; validates against `GTRPublicData.xsd`. The only bulk source that carries descriptions/methodology. |

Cross-references: gene analytes → Entrez Gene IDs; conditions → MedGen/UMLS CUIs +
OMIM; `elink dbfrom=gtr` links to `gene`, `medgen`, `omim`. ClinVar is linked at
the lab/assertion level (flags, not an `elink` db).

## Design

### 1. Module shape & conventions

- First-party `src/biodb/gtr.py`, **not** vendored — held to the full ruff ruleset
  and full coverage report.
- Dual-mode, mirroring `gprofiler` / `omicspred` / `clinvar`.
- Cache root: `~/.cache/biodb/gtr/`.
- All bulk downloads funnel through `biodb._downloads.stream_to_file` (tqdm).
- **No new optional-dependency extra.** E-utilities is plain `requests` (already
  core, as in `clinvar`/`pubmed`); the 214 MB bulk file is handled by stdlib
  `gzip` + `xml.etree.ElementTree.iterparse` (streaming). Keeps the module as
  thin as `gprofiler`.

### 2. API / targeted mode

```python
search_tests(term, field=None, retmax=200, api_key=None, tool=None, email=None) -> list[str]
    # esearch db=gtr; returns GTR accessions. `field` maps to GTR indices
    # (SYMB, GENEID, DISNAME, DCUI, MTOD, ...).

query_test(test_id, rich=False, api_key=None, ...) -> GTRTest
    # esummary JSON for one record (rich=True layers /gtr/api/v1/tests/{id}/).

query_gene(symbol_or_entrez, retmax=200, ...) -> list[GTRTest]
    # convenience: search_tests(field=SYMB|GENEID) + batched esummary.

query_condition(name_or_cui, retmax=200, ...) -> list[GTRTest]
    # convenience: search_tests(field=DISNAME|DCUI) + batched esummary.
```

A normalized record (`GTRTest` dataclass or dict) holds:
`accession, name, alt_names, test_type, lab, genes [(symbol, entrez, location)],
conditions [(name, cui, omim)], methods (3-level), clinical_validity,
clinical_utility, analytical_validity, target_population, test_purpose, pmids,
test_url`.

An internal cooperative rate-limiter respects the 3/s (10/s with key) E-utilities
ceiling; `tool` + `email` params are forwarded per NCBI policy.

### 3. Bulk mode (one download entry point, layered consumers)

A single bulk-download function always fetches the light daily TSV and, with one
opt-in flag, *additionally* pulls the 214 MB full XML:

```python
download(cache_dir=None, full_xml=False, force=False, progress=True) -> dict[str, Path]
    # Always downloads the light daily test_condition_gene.txt TSV.
    # full_xml=True ALSO downloads gtr_ftp.xml.gz (214 MB).
    # Returns {"tsv": Path, "xml": Path | None} — "xml" is None unless requested.
    # Both files funnel through _downloads.stream_to_file (tqdm).
```

Consumers read whichever file they need:

```python
load_test_condition_gene(cache_dir=None, force=False) -> pd.DataFrame
    # Reads the TSV (downloading via download() if absent). 8 columns split into
    # a tidy frame: per-test (gene_entrez, gene_symbol, condition_cui) records.

iter_full_records(path=None, cache_dir=None, force=False) -> Iterator[GTRTest]
    # xml.etree.iterparse streaming generator over gtr_ftp.xml.gz (downloading
    # via download(full_xml=True) if absent); clears elements as it goes so
    # memory stays flat over the full 214 MB dump. Carries descriptions.
```

### 4. Gene-set views (the GenForge / HaploForge payload)

```python
gene_sets(cache_dir=None, force=False) -> pd.DataFrame
    # RAW per-test long frame: (panel_id, panel_name, condition_cui,
    # gene_symbol, gene_entrez). Built from the light TSV.

aggregate_gene_sets(by="test_name", cache_dir=None, force=False) -> pd.DataFrame
    # DEDUPLICATED gene sets grouped by `by` in {"test_name", "condition"}.
    # Adds `support_count` per gene = number of independent tests/labs whose
    # panel includes that gene -> a free, source-grounded importance prior.

to_gmt(path, by=None, cache_dir=None, force=False) -> Path
    # GMT export (set_name, description, gene...) so sets flow straight into
    # biodb.utils.read_gmt and GenForge as gene-vector signatures. by=None
    # exports raw per-test sets; by="test_name"/"condition" exports aggregated.
```

### 5. Embeddable-text hook (HaploForge — text only, no embedding)

```python
panel_text(record, include=("name","alt_names","conditions","clinical_validity",
                            "clinical_utility","target_population","test_purpose",
                            "methods")) -> str
    # Assemble a clean, deduplicated free-text blob describing the panel/assay,
    # ready to embed downstream. The rich GTRTest also carries this as a field.
```

bioDB explicitly stops here: producing clean gene lists + panel text. Embedding,
cosine-similarity, and per-gene cosine weights are HaploForge/biodocs concerns.

### 6. Testing & wiring

- Unit tests with `responses`-mocked E-utilities JSON (esearch + esummary), a
  tiny `test_condition_gene.txt` TSV fixture, and a small `gtr_ftp.xml` fixture
  exercising the streaming `iter_full_records` parser and `panel_text`.
- Any live-network call marked `@pytest.mark.network` (skipped in CI), matching
  the repo convention.
- Re-export a curated slice from `src/biodb/__init__.py`.

### 7. Documentation (thorough — both in-code and on the website)

Documentation is a first-class deliverable, not an afterthought. It spans two layers:

**In-code (docstrings).** Every public symbol — module, `GTRTest`, and all functions
in the public surface — gets a complete NumPy-style docstring (the convention
already used across `gprofiler`/`uniprot`/`gprofiler`), covering:

- A one-line summary plus an extended description explaining *what GTR concept*
  the function exposes (test record vs panel vs gene set) and *which access path*
  it uses (E-utilities vs FTP TSV vs full XML).
- Full `Parameters` / `Returns` / `Raises` sections with types.
- An `Examples` block with a runnable, `# doctest: +SKIP` snippet (skipped because
  it hits the network), mirroring the existing modules.
- Inline comments at every non-obvious decision: the `GTR%09d` UID↔accession gap,
  the 3/s vs 10/s rate-limit branch, why the XML parse streams + clears elements,
  and how `support_count` is derived. These are constraint/why comments, not
  narration.
- The module-level docstring opens with a `Sources` / access-paths overview and a
  short worked example for each mode (targeted, bulk-light, bulk-rich, gene-set).

**Website (Sphinx).** The docs site must document GTR as thoroughly as OLS:

- Add `docs/gtr.md` — a full user guide mirroring `docs/ols.md`: an intro to what
  GTR is, a table of the access paths, and worked end-to-end examples for each of
  the four use shapes (find tests for a gene; pull one rich record; bulk gene-set
  materialization for GenForge via `to_gmt`; panel-text extraction for HaploForge
  weighting via `panel_text` + the `support_count` prior). Include the dual-mode
  framing and rate-limit/API-key guidance.
- Wire `docs/gtr.md` into the Sphinx `toctree` (and any sources index/landing list
  that enumerates the other guides).
- Add an autodoc API-reference page (or section) for `biodb.gtr` so every public
  symbol's docstring renders on the site, matching how the other modules appear in
  the API reference.
- Add a row to the README **Sources** table (✅ both modes), in the same format as
  the existing rows, linking to `docs/gtr.md`.
- Verify the docs build cleanly: `sphinx-build -b html docs docs/_build/html`
  produces no new warnings for the GTR pages.

## Public surface summary

`search_tests`, `query_test`, `query_gene`, `query_condition`,
`download`, `load_test_condition_gene`, `iter_full_records`,
`gene_sets`, `aggregate_gene_sets`, `to_gmt`, `panel_text`,
plus the `GTRTest` record type and `CACHE_DIR` / URL constants.

## Out of scope

- Embedding / cosine-similarity weighting (HaploForge/biodocs).
- Variant-level ClinVar bridging beyond carrying the cross-reference IDs.
- The infectious-disease (`MICROBE_*`) test subset beyond what falls out of the
  generic parser.
- Writing/submission of GTR records (submission XSD path).
