# bioDB contributor guidance

## Purpose and boundaries

`bioDB` provides biomedical and genomic source acquisition, parsers, and reusable
queries. The import name is lowercase `biodb`; the repository is
`standardmodelbio/bioDB`. BioDocs and hg-horizon own document rendering, scientific
transforms, indexes, browser exports and worker deployment.

The README source table, `docs/sources.md`, API reference and tutorial index must
agree with the implementation. Describe each supported mode accurately: REST or
GraphQL, indexed region reads, whole-file acquisition, or local views. A preview
is not complete acquisition; a local lookup is not a remote API.

## Commands

```bash
pip install -e ".[test,docs]"
pytest tests/test_gwas_raw.py tests/test_opentargets_cache_manifest.py
pytest                                      # includes live integration checks
pytest -m "not network and not slow"         # filters explicitly marked tests
ruff check .
ruff format --check .
python scripts/validate_notebooks.py --execute
sphinx-build -W --keep-going -b html docs docs/_build/html
```

Some legacy live tests are not marked; a marker filter alone does not guarantee
zero network traffic. The notebook validator uses fresh kernels with outbound
networking blocked and never rewrites saved outputs.

Install optional extras for the features under test. Core CI installs CPU-only
torch before `.[test]`; acquisition and documentation imports must also work
without torch. `.[dev]` includes heavier plotting, ontology and backend dependencies.
Container builds use `Dockerfile.cpu` and `Dockerfile.gpu`; images live under
`ghcr.io/standardmodelbio/biodb`.

## Source layout

| Area | Modules |
|---|---|
| OpenTargets | `opentargets` bulk/readers, `variants`, `studies`, `literature`, `constants`; separate `opentargets_graphql` |
| GWAS and annotations | `gwas_catalog`, `gwas_harvest`, `gwas_atlas`, `panukbb`, `gencode`, `clinvar`, `traitgym` |
| Cell sources and builds | `celltaxonomy`, `cellmarker`, `cellxgene`, `markers`, `cellxgene_dataset`, `hf_datasets` |
| Ontologies and literature | `ols`, `ols_text`, `ontology`, deprecated `ontology_owl` shim, `monarch`, `pubmed`, `europepmc` |
| Genes, proteins and gene sets | `uniprot`, `protein_structure`, `harmonizome`, `gprofiler`, `msigdb`, `gtr`, `corum`, `string`, `coessentiality`, `ensembl_compara` |
| Other sources | `aou_allxall`, `omicspred`, `orphanet`, `tads`, `encode_re2g`, `rfam`, `constraint_tracks`, `snomed` |
| Shared transforms | `intervals`, `mapping`, `transform`, `utils` |

Private modules implement shared transport, receipts and parsing. Consumers use
public source APIs rather than duplicate cache layouts or infer completion from
file/study presence. Intended public functions belong in module `__all__` lists
so the API reference can discover them.

## Acquisition contracts

- OpenTargets defaults to release `25.12`. Pass `version` explicitly in examples;
  change the default only after checking release schema compatibility.
- Complete OT caches require authoritative membership and validated receipts.
  A limited preview cannot satisfy an unlimited request. Legacy caches require
  an authoritative listing; unlisted extras do not establish coverage.
- Fine-mapping comes from OpenTargets. Raw summary statistics come from GWAS
  Catalog or an explicit Harvest manifest. GWAS Atlas MAGMA gene statistics are
  a separate product.
- Raw GWAS acquisition defaults to genome-wide and unfiltered. Regions and
  significance thresholds are explicit. Preserve source precision, original text
  tokens, P=0, log-P, missing values and provenance. Keep derived fields separate.
- Run manifests freeze source membership and semantic configuration. Resume
  validates per-file receipts, including successful zero-match inputs. Failures
  raise by default; incomplete output requires explicit `allow_partial`.
- HTTP bytes resume only with safe source/range identity. Final files publish
  atomically after validation. Failed refreshes preserve prior valid finals.
- CellMarker uses the retained 2.0 endpoint and a versioned default cache.
  Explicit caller cache directories are used as supplied. Do not silently adopt
  an unversioned default cache or switch releases.

## Documentation and tests

For a new source/API, update the README source map, `docs/api.rst`, the appropriate
source guide, and tutorial coverage index. Add a tiny executable offline example
using actual public readers. Keep live acquisition recipes in labeled Markdown
cells. Tutorial execution must not collect a corpus, run a model, or publish data.

Notebooks opt into offline execution with `metadata.biodb.offline = true` and
ship current executed outputs. Validate schema, syntax, saved errors and offline
execution before committing. Docs CI renders copies under
`docs/_generated/tutorials/`, fails on Sphinx warnings, and publishes the site at
`https://standardmodel.bio/bioDB/` from main or a main manual dispatch.

Core CI runs live source checks and mocked regressions. Genuine connection,
gateway and interrupted-stream outages may skip through the narrow shared test
guard. Other HTTP errors, changed schemas, malformed payloads and assertion
failures must fail; do not introduce blanket service skips.

## Vendored code and licensing

`opentargets/_bulk.py`, `monarch.py`, `ontology.py`, and `clinvar.py` contain large
ported surfaces excluded from normal Ruff/coverage sweeps. Keep edits minimal;
do not reformat them wholesale. Prefer sibling modules for new functionality.
Docstring corrections for valid rendered documentation are appropriate; verify
that executable behavior is unchanged.

The license is PolyForm-Noncommercial-1.0.0. Preserve its metadata and links.
