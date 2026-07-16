# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`bioDB` is a Python library exposing biomedical knowledge sources (Open Targets, Monarch Initiative, OBO ontologies) under a **dual-mode API**:

- **API mode** — targeted `query_*` / `fetch_*` GraphQL or REST lookups, one record at a time.
- **FTP mode** — bulk `list_datasets()` / `get_dataset()` Parquet/TSV/OBO downloads, cached locally and concatenated into DataFrames.

When adding a new source or extending an existing one, **keep both modes symmetric**. The README's "Sources" table is the canonical map of what's implemented vs. stubbed (`🚧 [bioDB#TBD]`).

Project was renamed `phenoref → bioDB` (see commit `d2616cc`); the package name on disk is lowercase `biodb`.

## Commands

Install for development (CPU torch wheel is required up front to keep the resolver from pulling the ~2GB CUDA wheel — CI does this explicitly):

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -e ".[dev]"
```

Day-to-day:

```bash
pytest                                  # full suite (live-network tests are auto-skipped)
pytest tests/test_opentargets.py        # single file
pytest -k "test_get_dataset_signature"  # single test by name
pytest -m "not slow and not network"    # explicit marker filter (matches CI)
ruff check .                            # lint
ruff format --check .                   # formatting check (CI fails on diff)
ruff format .                           # apply formatting
```

Docs:

```bash
sphinx-build -b html docs docs/_build/html
```

Docker (matches the published `ghcr.io/bschilder/biodb` image):

```bash
docker build -f Dockerfile.cpu -t biodb:cpu .
docker build -f Dockerfile.gpu -t biodb:gpu .
```

## Architecture

Seven modules under `src/biodb/`, split between **vendored AoU ports** (heavy, frozen) and **first-party** thin wrappers:

| Module | Role | Notes |
|---|---|---|
| `opentargets/` | Open Targets **FTP/bulk** mode — now a **package**: `_bulk.py` is the original flat-module surface (`list_datasets`, `get_dataset`, `ensure_cached_shards`, parquet readers, gene-association matrix builders, pathway/expression/essentiality helpers); `variants.py` and `studies.py` are first-party submodules built on `_bulk.ensure_cached_shards`. | `_bulk.py` uses `gget` as a runtime backend for some paths (opt-in `[gget]` extra). 4.6k lines, **vendored**. `variants.py` — first-party: `get_credible_set` (variant coords + beta + p-value + pip + credibleSetSize, with optional `studyType` join/filter) and `get_variant_effects` (per-variant `vep_score` aggregated from `variantEffect[*].normalisedScore`). `studies.py` — first-party: `get_study` (studyId/studyType/traitFromSource/projectId/nSamples) and `attach_study_type` (left-joins `studyType` onto another frame). `opentargets/__init__.py` re-exports every `_bulk` name (public and private) programmatically so existing `from biodb.opentargets import get_dataset`-style imports are unaffected by the package split. |
| `opentargets_graphql.py` | Open Targets **API/targeted** mode — `query_target`, `query_disease`, `query_drug`, `query_variant`. | Independent `httpx`-based GraphQL client with exponential backoff. Intentionally separate from `opentargets.py` so the lightweight query path has no `gget` dep. First-party. Note: `httpx` is currently **not declared** in `pyproject.toml` — known bug, fix when next touching the file. |
| `monarch.py` | Monarch Initiative TSV association readers. | API mode is the `🚧` stub. **Vendored**. |
| `ontology.py` | OBO/OWL loaders, N-hop keyword-set expansion, hierarchical keyword sets, attention-weight analysis, gene-phenotype matrix, ontological similarity, **plus** the generic owlready2 primitives that work for any OBO Foundry OWL file (`get_ontology`, `get_descendants`, `get_ancestors`, `get_mrca`, …) — formerly in `ontology_owl.py`, merged here per user directive. | ~5.6k lines, **vendored** (AoU port + appended owl helpers). `matplotlib` / `datashader` / `owlready2` are lazy-imported only when needed. |
| `ontology_owl.py` | **Back-compat shim** — re-exports the generic owl helpers from `biodb.ontology` with a `DeprecationWarning`. | Preserved so old `from biodb.ontology_owl import X` keeps working. |
| `uniprot.py` | UniProt REST client — `query_protein`, `get_sequences`, `get_features`, `get_dbxrefs`. | First-party. Ported from `VEP_protein/src/unitprot.py` with the SeqIO-iterator footgun fixed (results are materialized lists, not exhaustible iterators). Requires the `[protein]` extra (Biopython). |
| `harmonizome.py` | Maayan-Lab Harmonizome client — `list_datasets`, `download_datasets`, `get_gmt`, `load_gene_attribute_matrix`, `get_dataset_metadata`, plus back-compat `Harmonizome` / `Entity` classes. | First-party. Ported from `AoU/phenome/harmonizome.py` with the **module-load-time HTTP call fixed** — `DOWNLOADS` / `DATASET_TO_PATH` now lazy-load via PEP 562 `__getattr__` so `import biodb.harmonizome` no longer requires network. Dead helpers using removed pandas/numpy APIs (`SparseDataFrame`, `np.object`) pruned; Python 2 compat shims dropped. |
| `gwas_atlas.py` | GWAS Atlas (Watanabe et al.) — `download_metadata`, `download_magma_p`, `load_metadata`, `load_magma_p`, `melt_magma_p`. | First-party. Wraps the per-study metadata TSV + (gene × study) MAGMA -log10 p-value matrix at `atlas.ctglab.nl/ukb2_sumstats/`. |
| `gprofiler.py` | g:Profiler — `download_gmt`, `load_gmt`, `gost`. | First-party. Bulk GMT + REST functional enrichment. Prefers `gprofiler-official` Python package; falls back to plain `requests.post`. |
| `msigdb.py` | Broad MSigDB — `download_gmt`, `load_gmt`. | First-party. Per-collection / per-version GMT downloader (Hallmark + C1–C8 + combined ``msigdb``). |
| `mapping.py` | Cross-namespace gene-ID mapping — `map_gene_ids` (gProfiler-backed). | First-party. Ported from `AoU.utils.map_gene_ids`. Requires `[mapping]` extra. |
| `transform.py` | Tabular → matrix transforms — `create_gene_association_matrix`. | First-party. Thin re-export from `biodb.utils` so callers can write the more obvious `from biodb.transform import …`. |
| `utils.py` | Shared helpers — `RANDOM_SEED=42`, `set_random_seed`, `count_tokens` (tiktoken), similarity (`cosine_similarity`, `l2_normalize`, …), `create_gene_association_matrix`, `filter_adaptive`, `read_gmt` (the canonical GMT-format reader reused by MSigDB / gProfiler / Harmonizome). | Verbatim ports from `AoU.utils` so `biodb` has no AoU runtime dep. First-party (despite being a port). |

`src/biodb/__init__.py` re-exports a curated slice of high-frequency symbols at the top level for convenience; the full APIs live on the submodules.

### Vendored AoU modules — important convention

`opentargets/_bulk.py`, `monarch.py`, and `ontology.py` are **verbatim ports from `AoU.phenome`** (~13,900 lines total). They are configured in `pyproject.toml` as out-of-bounds for both ruff and coverage:

```toml
[tool.ruff]
extend-exclude = ["src/biodb/opentargets/_bulk.py", "src/biodb/monarch.py", "src/biodb/ontology.py"]

[tool.coverage.run]
omit = ["src/biodb/opentargets/_bulk.py", "src/biodb/monarch.py", "src/biodb/ontology.py"]
```

The reason is stated inline: *"Treat as third-party — style fixes belong upstream in AoU, not here."* When editing these modules:

- Don't reflow / reformat for style — keep diffs minimal so re-syncing from AoU stays cheap.
- Tests (`tests/test_opentargets.py`, etc.) intentionally only cover the import surface and public-function signatures — live-network behaviour is not exercised in CI.
- `utils.py`, `opentargets_graphql.py`, `opentargets/variants.py`, `opentargets/studies.py`, `ontology_owl.py`, and `uniprot.py` are *not* vendored — they're held to the full ruff ruleset and the full coverage report. When adding **new** first-party functionality, prefer a sibling module to editing a vendored one.

### Caching

Each FTP-mode module writes to its own user-cache directory:

- `~/.cache/biodb/opentargets/<version>/<dataset>/*.parquet` (versioned, the modern layout)
- `~/.cache/opentargets/` (legacy single-flat-dir cache, kept readable for back-compat)
- `~/.cache/monarch/`

`opentargets.DEFAULT_VERSION = "25.12"` pins the default Open Targets release. **Bump only after testing against the new release** — schema shifts (especially in nested struct columns like `target_essentiality` / `expression`) routinely break the parsers.

### Test markers

`pytest.ini_options` defines two custom markers (and uses `--strict-markers`, so unknown markers fail):

- `slow` — tests > ~5s
- `network` — tests that hit a live remote API; **skipped in CI**

If you add a test that touches the network or takes more than a few seconds, mark it explicitly — otherwise `--strict-markers` won't flag it but CI will silently run a flaky test.

## CI

`.github/workflows/ci.yml` runs three jobs:

1. **Lint** — `ruff check` + `ruff format --check` (Python 3.11).
2. **Test** — matrix across Python 3.10 / 3.11 / 3.12, installs CPU-only torch first, runs `pytest --cov=biodb`.
3. **Coverage badge** — only on `push` to `main` from the 3.11 matrix cell, runs `coverage-badge -f -o docs/_static/coverage.svg` and commits the SVG back to the repo (using `github-actions[bot]`). The push step is path-ignored on the workflow trigger to avoid an infinite loop.

`docker.yml` builds + pushes to `ghcr.io/bschilder/biodb`. `docs.yml` builds the Sphinx site to GitHub Pages.

## License

PolyForm-Noncommercial-1.0.0. The `classifiers` field uses `"License :: Other/Proprietary License"` because PyPI's trove classifiers don't include PolyForm — don't "fix" this to a permissive classifier.
