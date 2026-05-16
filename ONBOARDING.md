# bioDB — handoff

[`bschilder/bioDB`](https://github.com/bschilder/bioDB) is a standalone Python library bundling phenotype-knowledge-graph helpers extracted from `AoU.phenome` (Open Targets / Monarch / ontology) plus a ClinVar helper adapted from `bschilder/VEP_protein`. Sibling projects (TimelineDataset, TimelineTransformer, autoencode) can depend on this narrow library without pulling the full All-of-Us pipeline.

**License:** PolyForm-Noncommercial-1.0.0 — not OSI-approved, no commercial use.

**Docs:** https://bschilder.github.io/bioDB/ (built by `.github/workflows/docs.yml`, served via GitHub Pages from the `gh-pages` artifact upload).

**Container:** `ghcr.io/bschilder/biodb` (built by `.github/workflows/docker.yml`).

## What's inside

```
src/biodb/
  __init__.py    Public re-exports + module docstring
  opentargets.py 7,116 LOC — vendored from AoU.phenome.opentargets
  ontology.py    5,290 LOC — vendored from AoU.phenome.ontology
  monarch.py       768 LOC — vendored from AoU.phenome.monarch
  clinvar.py       550 LOC — adapted from VEP_protein/src/clinvar.py
  utils.py         800 LOC — vendored from AoU.utils (the actively-tested surface)
tests/           pytest suite (60 utils tests, 9 clinvar smoke, 6 ontology, 6 opentargets, 5 monarch)
tutorials/       4 jupytext-percent .py + rendered .ipynb pairs (offline-runnable)
docs/            Sphinx site (myst-parser markdown + autosummary RST)
```

The 4 large modules (`opentargets`, `ontology`, `monarch`, `clinvar`) are **vendored ports** — treated as third-party. They're excluded from ruff lint (`pyproject.toml` `[tool.ruff].extend-exclude`) and coverage (`[tool.coverage.run].omit`) because the network-heavy, ontology-graph-walking, VCF-parsing branches inside them are unreachable without live remote APIs or heavy optional deps. The actively-tested surface is `utils.py` + `__init__.py`, currently at **94% coverage**.

If you ever want to bring a vendored module under coverage, the pattern is: write mock-API fixtures that exercise the parsing branches, then drop the file from the `omit =` list.

## Operating it

**Install (dev):**
```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/cpu torch    # brings in networkx
uv pip install -e ".[dev]"
```

Local CI also requires `pyarrow` only for one optional pandas-roundtrip clinvar test (skipped if absent).

**Run tests + coverage** (use `python -m pytest`, not bare `pytest`; the bare entrypoint hits a torch/numpy 2.x dual-load bug locally):
```bash
python -m pytest --cov=biodb -p no:cacheprovider
```

**Lint + format:**
```bash
ruff check . && ruff format --check .
```

**Build docs:**
```bash
sphinx-build -b html docs/ docs/_build/html
```

**Regenerate tutorials after editing the `.py` mirrors:**
```bash
jupytext --to ipynb tutorials/*.py --update
jupyter nbconvert --to notebook --execute --inplace tutorials/*.ipynb
```

## Quirks future-you should know

1. **`utils.filter_adaptive` was rewritten for pandas 2.2+** (commit `d4bf8d2`). The old `groupby(..., group_keys=False).apply(...)` lost the `source_id_col` from output columns because pandas 2.2 stopped auto-prepending the group key. The current impl uses sort + cumcount + rank-threshold filter and works across pandas versions. Don't revert to apply.

2. **`pytest` vs `python -m pytest` mismatch on macOS** — bare `pytest` triggers a `numpy._core._multiarray_umath: cannot load module more than once per process` error when run after the conftest imports torch. Reason unclear; always use `python -m pytest`. CI on Linux doesn't hit this.

3. **The `clinvar._CLNSIG_MAP` dict is verbatim from VEP_protein** — adding new long-tail entries upstream means re-syncing here. The `replace_strict(..., default=pl.lit("other"))` call means unknown strings won't raise, so out-of-sync maps degrade silently to `"other"`.

4. **`list_datasets()` returns a `dict`, not a list** — it maps dataset name → URL. Slice via `list(datasets)[:5]`, not `datasets[:5]`. Tutorial 04 demonstrates the right pattern.

5. **GHCR container was originally published as `phenoref`** — the package was renamed to `bioDB` (commit `d2616cc`), and the old `ghcr.io/bschilder/phenoref` package was deleted on 2026-05-16. The current image lives at `ghcr.io/bschilder/biodb`.

6. **`gene_weighting` is NOT here** — it was moved to `bschilder/TimelineDataset` in commit `3cf86e4` because it operates on patient event embeddings, not on the phenotype knowledge graph. `AoU.phenome.gene_weighting` is now a shim that re-exports from `timeline_dataset.gene_weighting`.

## CI matrix

`.github/workflows/ci.yml` runs three Python versions (3.10 / 3.11 / 3.12) plus a lint job on the same trigger. On push to `main`, the py3.11 test job also generates `docs/_static/coverage.svg` via `coverage-badge` and commits it back via `github-actions[bot]`. (Tip: PR runs skip the badge commit, so badge regenerates on merge.)

The Docker workflow builds CPU + GPU images and pushes to GHCR. The Docs workflow builds Sphinx + uploads via the GitHub Pages action.

## Related repos in this ecosystem

- **TimelineDataset** — patient-event timeline datasets + `gene_weighting` helper.
- **TimelineTransformer** — transformer on those timelines.
- **autoencode** — the autoencoder formerly at `AoU.phenome.autoencoder`.
- **AoU** — the original monorepo; `AoU.phenome.{opentargets,monarch,ontology,gene_weighting}` are now thin shims re-exporting from `biodb` / `timeline_dataset`.

## Open / deferred

- Three vendored modules (`opentargets`, `ontology`, `monarch`) have 0% measured coverage. Bringing any of them under coverage requires writing mock-API fixtures (Open Targets parquet shape, Monarch TSV layout, OBO file fragments).
- `clinvar.vcf_to_df` requires `genoray` + a live ClinVar VCF download — no end-to-end test for it yet (only the in-memory helpers are tested).
- Repo "About" panel `homepageUrl` still empty as of writing — could be set to `https://bschilder.github.io/bioDB/` via `gh repo edit bschilder/bioDB --homepage <url>`.
