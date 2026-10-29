# Open Targets `variants`/`studies` Readers — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add bulk, variant-level Open Targets readers to `biodb` (credible sets, study metadata, per-variant VEP scores), namespaced as `biodb.opentargets.variants.*` / `biodb.opentargets.studies.*`, by promoting the vendored `opentargets.py` module into a package without breaking any existing import.

**Architecture:** Move the frozen vendored module verbatim into `src/biodb/opentargets/_bulk.py` and add a package `__init__.py` that programmatically re-exports every public name from `_bulk` (so `from biodb.opentargets import get_dataset` keeps working). Two new **first-party** submodules, `variants.py` and `studies.py`, read the OT `credible_set`, `study`, and `variant` bulk Parquet via the existing `ensure_cached_shards()` + `polars.scan_parquet` path. This is the data-gathering prerequisite consumed by the separate `cordillera` project; `biodb` returns raw DataFrames and performs no scoring or cross-source joins.

**Tech Stack:** Python 3.10–3.12, polars (nested list-of-struct parsing), pandas, pyarrow, requests; pytest with `network`/`slow` markers; ruff + pyright; uv for env management.

## Global Constraints

- Python floor: **3.10** (matrix 3.10/3.11/3.12). Copied from CI.
- `src/biodb/opentargets/_bulk.py` is **vendored** — moved verbatim, NOT reformatted; stays excluded from ruff and coverage. Style fixes belong upstream in AoU.
- New first-party code (`__init__.py`, `variants.py`, `studies.py`) is held to the **full** ruff ruleset and coverage report.
- `--strict-markers` is on: any test hitting the network **must** be marked `@pytest.mark.network` (and `@pytest.mark.slow` if >~5s). CI runs `-m "not slow and not network"`.
- Live-network tests, when run, guard upstream flakiness with an `is_upstream_outage`-style skip (skip only on connection/timeout/502/503/504) — do NOT blanket-skip.
- OT release is pinned by `_bulk.DEFAULT_VERSION` (currently `"25.12"`); new readers default `version=DEFAULT_VERSION`.
- New readers return **polars** DataFrames by default (they parse nested columns).
- Do not add new hard dependencies; polars/pandas/pyarrow/requests already declared.

---

## File Structure

- `src/biodb/opentargets/__init__.py` — **new**, first-party. Re-exports `_bulk`'s public surface for back-compat; exposes `.variants` and `.studies` submodules.
- `src/biodb/opentargets/_bulk.py` — **moved** verbatim from `src/biodb/opentargets.py`. Vendored.
- `src/biodb/opentargets/studies.py` — **new**, first-party. `get_study()`, `attach_study_type()`.
- `src/biodb/opentargets/variants.py` — **new**, first-party. `get_credible_set()`, `get_variant_effects()`.
- `pyproject.toml` — **modify** ruff `extend-exclude` + coverage `omit` paths.
- `tests/test_opentargets_package.py` — **new**. Back-compat + structure.
- `tests/test_opentargets_studies.py` — **new**. Offline fixture tests.
- `tests/test_opentargets_variants.py` — **new**. Offline fixture tests.
- `CLAUDE.md`, `README.md` — **modify**. Reflect the package + new submodules.

## Setup (before Task 1)

```bash
cd ~/code/bioDB
git checkout main && git pull --ff-only 2>/dev/null || true
git checkout -b feat/opentargets-variants-readers
```

---

### Task 1: Promote `opentargets.py` to a package (verbatim move + back-compat)

**Files:**
- Move: `src/biodb/opentargets.py` → `src/biodb/opentargets/_bulk.py`
- Create: `src/biodb/opentargets/__init__.py`
- Modify: `pyproject.toml` (lines with `src/biodb/opentargets.py`, currently ~176 in `[tool.coverage.run] omit` and ~201 in `[tool.ruff] extend-exclude`)
- Test: `tests/test_opentargets_package.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `biodb.opentargets` as a package; `biodb.opentargets._bulk` module; every existing public name (`get_dataset`, `list_datasets`, `ensure_cached_shards`, `read_for_target`, `variants_for_target`, `get_targets`, `get_pathways`, `get_gene_associations`, `list_available_versions`, `DEFAULT_VERSION`, `DEFAULT_CACHE_DIR`, …) importable from `biodb.opentargets` exactly as before.

- [ ] **Step 1: Write the failing test**

Create `tests/test_opentargets_package.py`:

```python
"""Back-compat + package-structure tests for the opentargets package."""


def test_bulk_submodule_exists():
    # Fails before the move: there is no _bulk submodule yet.
    from biodb.opentargets import _bulk

    assert hasattr(_bulk, "get_dataset")


def test_public_names_still_importable_from_package():
    # The historical flat-module import surface must keep working.
    from biodb.opentargets import (  # noqa: F401
        DEFAULT_VERSION,
        ensure_cached_shards,
        get_dataset,
        get_gene_associations,
        get_pathways,
        get_targets,
        list_available_versions,
        list_datasets,
        read_for_target,
        variants_for_target,
    )

    assert isinstance(DEFAULT_VERSION, str)


def test_reexports_are_identical_objects():
    import biodb.opentargets as ot
    from biodb.opentargets import _bulk

    assert ot.get_dataset is _bulk.get_dataset
    assert ot.DEFAULT_VERSION == _bulk.DEFAULT_VERSION


def test_module_style_access_still_works():
    from biodb import opentargets

    assert callable(opentargets.get_dataset)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_opentargets_package.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'biodb.opentargets._bulk'`.

- [ ] **Step 3: Do the verbatim move**

```bash
cd ~/code/bioDB
mkdir -p src/biodb/opentargets
git mv src/biodb/opentargets.py src/biodb/opentargets/_bulk.py
```

- [ ] **Step 4: Create the package `__init__.py`**

Create `src/biodb/opentargets/__init__.py`:

```python
"""Open Targets Platform access.

``biodb.opentargets`` is the bulk/FTP data-gathering surface. The historical
flat-module API (``get_dataset``, ``list_datasets``, ``ensure_cached_shards``,
``variants_for_target``, …) is re-exported here unchanged from the vendored
:mod:`biodb.opentargets._bulk`. Variant- and study-level readers live in the
:mod:`biodb.opentargets.variants` and :mod:`biodb.opentargets.studies`
submodules.
"""

from __future__ import annotations

from biodb.opentargets import _bulk

# Re-export every public name from the vendored bulk module so that
# ``from biodb.opentargets import get_dataset`` (and friends) keep working
# after the flat module became a package. Done programmatically to keep
# ``_bulk.py`` a verbatim vendored copy (no ``__all__`` edit required).
_g = globals()
for _name in dir(_bulk):
    if not _name.startswith("_"):
        _g[_name] = getattr(_bulk, _name)
del _g, _name

from biodb.opentargets import studies, variants  # noqa: E402

__all__ = [  # noqa: F822 - names injected from _bulk above are intentional
    name for name in dir() if not name.startswith("_")
]
```

Note: `studies` and `variants` are imported at the bottom; they are created in
Tasks 2–3. Until then this import fails, so **temporarily** comment out the
`from biodb.opentargets import studies, variants` line and its use to let Task 1
pass in isolation, OR create empty stub modules now:

```bash
printf '"""OT study-level readers."""\n' > src/biodb/opentargets/studies.py
printf '"""OT variant-level readers."""\n' > src/biodb/opentargets/variants.py
```

Create the stubs (preferred — keeps the import line honest).

- [ ] **Step 5: Update `pyproject.toml` exclude/omit paths**

Change both occurrences of `src/biodb/opentargets.py` to `src/biodb/opentargets/_bulk.py`:

In `[tool.coverage.run]` `omit`:
```toml
    "src/biodb/opentargets/_bulk.py",
```
In `[tool.ruff]` `extend-exclude`:
```toml
    "src/biodb/opentargets/_bulk.py",
```

- [ ] **Step 6: Run tests + lint to verify pass**

Run: `pytest tests/test_opentargets_package.py tests/test_opentargets.py -v`
Expected: PASS (both the new back-compat file and the existing suite).

Run: `ruff check src/biodb/opentargets/__init__.py && ruff format --check src/biodb/opentargets/__init__.py`
Expected: clean (the `_bulk.py` move is excluded).

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor(opentargets): promote module to package with back-compat re-exports"
```

---

### Task 2: `opentargets.studies.get_study()` + `attach_study_type()`

**Files:**
- Modify: `src/biodb/opentargets/studies.py`
- Test: `tests/test_opentargets_studies.py`

**Interfaces:**
- Consumes: `biodb.opentargets._bulk.ensure_cached_shards`, `biodb.opentargets._bulk.DEFAULT_VERSION`.
- Produces:
  - `get_study(*, version: str = DEFAULT_VERSION, columns: list[str] | None = None, cache_dir=None, limit_files: int | None = None) -> pl.DataFrame` with at least columns `studyId, studyType, traitFromSource, projectId, nSamples`.
  - `attach_study_type(df: pl.DataFrame, study_df: pl.DataFrame, *, on: str = "studyId") -> pl.DataFrame` — left-joins `studyType` (and `traitFromSource`) onto `df`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_opentargets_studies.py`:

```python
"""Offline tests for biodb.opentargets.studies (synthetic parquet fixtures)."""

import polars as pl
import pytest

from biodb.opentargets import studies


@pytest.fixture
def study_fixture(tmp_path):
    """A tiny `study` parquet shard mimicking the OT schema."""
    df = pl.DataFrame(
        {
            "studyId": ["GCST001", "QTL_EQTL_1", "QTL_PQTL_1"],
            "studyType": ["gwas", "eqtl", "pqtl"],
            "traitFromSource": ["Height", "GENE1 expression", "PROT1 level"],
            "projectId": ["GCST", "GTEx", "UKB-PPP"],
            "nSamples": [500000, 800, 35000],
        }
    )
    path = tmp_path / "study-0.parquet"
    df.write_parquet(path)
    return path


def test_get_study_reads_shards(study_fixture, monkeypatch):
    monkeypatch.setattr(
        studies, "ensure_cached_shards", lambda *a, **k: [study_fixture]
    )
    out = studies.get_study()
    assert set(["studyId", "studyType", "traitFromSource"]).issubset(out.columns)
    assert out.height == 3
    assert out.filter(pl.col("studyType") == "eqtl").height == 1


def test_get_study_column_subset(study_fixture, monkeypatch):
    monkeypatch.setattr(
        studies, "ensure_cached_shards", lambda *a, **k: [study_fixture]
    )
    out = studies.get_study(columns=["studyId", "studyType"])
    assert out.columns == ["studyId", "studyType"]


def test_attach_study_type_left_join():
    cs = pl.DataFrame({"studyId": ["GCST001", "QTL_EQTL_1"], "beta": [0.1, 0.2]})
    st = pl.DataFrame(
        {
            "studyId": ["GCST001", "QTL_EQTL_1"],
            "studyType": ["gwas", "eqtl"],
            "traitFromSource": ["Height", "GENE1 expression"],
        }
    )
    out = studies.attach_study_type(cs, st)
    assert "studyType" in out.columns
    assert out.filter(pl.col("studyId") == "QTL_EQTL_1")["studyType"][0] == "eqtl"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_opentargets_studies.py -v`
Expected: FAIL — `AttributeError: module 'biodb.opentargets.studies' has no attribute 'get_study'`.

- [ ] **Step 3: Implement `studies.py`**

Replace the contents of `src/biodb/opentargets/studies.py`:

```python
"""Open Targets study-level bulk readers (data-gathering only)."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from biodb.opentargets._bulk import DEFAULT_VERSION, ensure_cached_shards

_DEFAULT_STUDY_COLUMNS = [
    "studyId",
    "studyType",
    "traitFromSource",
    "projectId",
    "nSamples",
]


def get_study(
    *,
    version: str = DEFAULT_VERSION,
    columns: list[str] | None = None,
    cache_dir: str | Path | None = None,
    limit_files: int | None = None,
) -> pl.DataFrame:
    """Read the OT ``study`` dataset as a flat polars DataFrame.

    Parameters
    ----------
    version : str
        OT release (defaults to the pinned :data:`DEFAULT_VERSION`).
    columns : list[str], optional
        Columns to select. Defaults to
        ``["studyId", "studyType", "traitFromSource", "projectId", "nSamples"]``.
    cache_dir, limit_files
        Forwarded to :func:`ensure_cached_shards`.

    Returns
    -------
    polars.DataFrame
    """
    cols = columns if columns is not None else _DEFAULT_STUDY_COLUMNS
    shards = ensure_cached_shards(
        "study", version=version, cache_dir=cache_dir, limit_files=limit_files
    )
    frames = [pl.scan_parquet(p).select(cols) for p in shards]
    return pl.concat(frames).collect()


def attach_study_type(
    df: pl.DataFrame,
    study_df: pl.DataFrame,
    *,
    on: str = "studyId",
) -> pl.DataFrame:
    """Left-join ``studyType`` and ``traitFromSource`` from ``study_df`` onto ``df``."""
    keep = [c for c in (on, "studyType", "traitFromSource") if c in study_df.columns]
    return df.join(study_df.select(keep), on=on, how="left")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_opentargets_studies.py -v`
Expected: PASS (3 tests).

Run: `ruff check src/biodb/opentargets/studies.py && ruff format --check src/biodb/opentargets/studies.py`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add src/biodb/opentargets/studies.py tests/test_opentargets_studies.py
git commit -m "feat(opentargets): add studies.get_study + attach_study_type readers"
```

---

### Task 3: `opentargets.variants.get_credible_set()`

**Files:**
- Modify: `src/biodb/opentargets/variants.py`
- Test: `tests/test_opentargets_variants.py`

**Interfaces:**
- Consumes: `ensure_cached_shards`, `DEFAULT_VERSION` from `_bulk`; `studies.get_study` + `studies.attach_study_type` from Task 2.
- Produces:
  - `get_credible_set(*, version=DEFAULT_VERSION, study_type: str | list[str] | None = None, cache_dir=None, limit_files=None) -> pl.DataFrame` with columns `variantId, chromosome, position, beta, pValueMantissa, pValueExponent, standardError, pip, credibleSetSize, finemappingMethod, studyLocusId, studyId, confidence, credibleSetlog10BF` (plus `studyType, traitFromSource` when `study_type` filtering is applied).
  - `pip` = `locus[0].posteriorProbability`; `credibleSetSize` = `locus.list.len()`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_opentargets_variants.py`:

```python
"""Offline tests for biodb.opentargets.variants (synthetic parquet fixtures)."""

import polars as pl
import pytest

from biodb.opentargets import variants


@pytest.fixture
def credible_set_fixture(tmp_path):
    """A tiny `credible_set` shard with the nested `locus` list-of-struct column."""
    df = pl.DataFrame(
        {
            "variantId": ["1_100_A_T", "2_200_C_G"],
            "chromosome": ["1", "2"],
            "position": [100, 200],
            "beta": [0.5, -0.3],
            "pValueMantissa": [2.0, 5.0],
            "pValueExponent": [-9, -12],
            "standardError": [0.1, 0.05],
            "finemappingMethod": ["SuSiE", "SuSiE"],
            "studyLocusId": ["sl1", "sl2"],
            "studyId": ["GCST001", "QTL_EQTL_1"],
            "confidence": ["high", "high"],
            "credibleSetlog10BF": [3.2, 4.1],
            "locus": [
                [
                    {"variantId": "1_100_A_T", "posteriorProbability": 0.8},
                    {"variantId": "1_101_C_G", "posteriorProbability": 0.2},
                ],
                [{"variantId": "2_200_C_G", "posteriorProbability": 0.95}],
            ],
        }
    )
    path = tmp_path / "credible_set-0.parquet"
    df.write_parquet(path)
    return path


@pytest.fixture
def study_fixture(tmp_path):
    df = pl.DataFrame(
        {
            "studyId": ["GCST001", "QTL_EQTL_1"],
            "studyType": ["gwas", "eqtl"],
            "traitFromSource": ["Height", "GENE1 expression"],
            "projectId": ["GCST", "GTEx"],
            "nSamples": [500000, 800],
        }
    )
    path = tmp_path / "study-0.parquet"
    df.write_parquet(path)
    return path


def test_get_credible_set_extracts_pip_and_size(credible_set_fixture, monkeypatch):
    monkeypatch.setattr(
        variants, "ensure_cached_shards", lambda *a, **k: [credible_set_fixture]
    )
    out = variants.get_credible_set()
    assert {"variantId", "chromosome", "position", "beta", "pip",
            "credibleSetSize"}.issubset(out.columns)
    row = out.filter(pl.col("variantId") == "1_100_A_T")
    assert abs(row["pip"][0] - 0.8) < 1e-6
    assert row["credibleSetSize"][0] == 2
    # locus struct column is not surfaced in the flat output
    assert "locus" not in out.columns


def test_get_credible_set_study_type_filter(
    credible_set_fixture, study_fixture, monkeypatch
):
    def fake_shards(dataset, **kwargs):
        return {"credible_set": [credible_set_fixture], "study": [study_fixture]}[
            dataset
        ]

    monkeypatch.setattr(variants, "ensure_cached_shards", fake_shards)
    # studies.get_study reads via its own ensure_cached_shards reference
    from biodb.opentargets import studies

    monkeypatch.setattr(studies, "ensure_cached_shards", lambda *a, **k: [study_fixture])

    out = variants.get_credible_set(study_type="gwas")
    assert out.height == 1
    assert out["studyId"][0] == "GCST001"
    assert out["studyType"][0] == "gwas"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_opentargets_variants.py -v`
Expected: FAIL — `AttributeError: module 'biodb.opentargets.variants' has no attribute 'get_credible_set'`.

- [ ] **Step 3: Implement `get_credible_set` in `variants.py`**

Replace the contents of `src/biodb/opentargets/variants.py`:

```python
"""Open Targets variant-level bulk readers (data-gathering only)."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from biodb.opentargets._bulk import DEFAULT_VERSION, ensure_cached_shards
from biodb.opentargets import studies

_CREDIBLE_SET_SCALAR_COLUMNS = [
    "variantId",
    "chromosome",
    "position",
    "beta",
    "pValueMantissa",
    "pValueExponent",
    "standardError",
    "finemappingMethod",
    "studyLocusId",
    "studyId",
    "confidence",
    "credibleSetlog10BF",
]


def get_credible_set(
    *,
    version: str = DEFAULT_VERSION,
    study_type: str | list[str] | None = None,
    cache_dir: str | Path | None = None,
    limit_files: int | None = None,
) -> pl.DataFrame:
    """Read the OT ``credible_set`` bulk Parquet into a flat polars DataFrame.

    The fine-mapping posterior (``pip``) is extracted from the first element of
    the nested ``locus`` list-of-struct column, and ``credibleSetSize`` from its
    length. ``studyType`` is not present in ``credible_set`` itself; pass
    ``study_type`` to join it from the ``study`` dataset and filter.

    Parameters
    ----------
    version : str
        OT release (defaults to the pinned :data:`DEFAULT_VERSION`).
    study_type : str | list[str], optional
        If given, join the ``study`` dataset on ``studyId`` and keep only rows
        whose ``studyType`` matches (e.g. ``"gwas"`` or ``["eqtl", "pqtl"]``).
    cache_dir, limit_files
        Forwarded to :func:`ensure_cached_shards`.

    Returns
    -------
    polars.DataFrame
    """
    shards = ensure_cached_shards(
        "credible_set", version=version, cache_dir=cache_dir, limit_files=limit_files
    )
    lazy = pl.concat([pl.scan_parquet(p) for p in shards])
    out = lazy.select(
        *_CREDIBLE_SET_SCALAR_COLUMNS,
        pl.col("locus")
        .list.first()
        .struct.field("posteriorProbability")
        .alias("pip"),
        pl.col("locus").list.len().alias("credibleSetSize"),
    ).collect()

    if study_type is not None:
        wanted = [study_type] if isinstance(study_type, str) else list(study_type)
        study_df = studies.get_study(
            version=version, cache_dir=cache_dir, limit_files=limit_files
        )
        out = studies.attach_study_type(out, study_df)
        out = out.filter(pl.col("studyType").is_in(wanted))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_opentargets_variants.py -v`
Expected: PASS (2 tests).

Run: `ruff check src/biodb/opentargets/variants.py && ruff format --check src/biodb/opentargets/variants.py`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add src/biodb/opentargets/variants.py tests/test_opentargets_variants.py
git commit -m "feat(opentargets): add variants.get_credible_set (pip + credibleSetSize + studyType join)"
```

---

### Task 4: `opentargets.variants.get_variant_effects(aggregate="max")`

**Files:**
- Modify: `src/biodb/opentargets/variants.py`
- Test: `tests/test_opentargets_variants.py` (append)

**Interfaces:**
- Consumes: `ensure_cached_shards`, `DEFAULT_VERSION` from `_bulk`.
- Produces: `get_variant_effects(*, version=DEFAULT_VERSION, aggregate: str = "max", cache_dir=None, limit_files=None) -> pl.DataFrame` with columns `variantId, vep_score`, where `vep_score` aggregates `variantEffect[*].normalisedScore` per variant. `aggregate` ∈ `{"max", "mean", "median"}`; unknown value raises `ValueError`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_opentargets_variants.py`:

```python
@pytest.fixture
def variant_fixture(tmp_path):
    """A tiny `variant` shard with the nested `variantEffect` list-of-struct column."""
    df = pl.DataFrame(
        {
            "variantId": ["1_100_A_T", "2_200_C_G"],
            "variantEffect": [
                [
                    {"method": "SIFT", "normalisedScore": 0.9},
                    {"method": "PolyPhen", "normalisedScore": 0.5},
                ],
                [{"method": "AlphaMissense", "normalisedScore": -0.2}],
            ],
        }
    )
    path = tmp_path / "variant-0.parquet"
    df.write_parquet(path)
    return path


def test_get_variant_effects_max(variant_fixture, monkeypatch):
    monkeypatch.setattr(
        variants, "ensure_cached_shards", lambda *a, **k: [variant_fixture]
    )
    out = variants.get_variant_effects(aggregate="max")
    assert out.columns == ["variantId", "vep_score"]
    row = out.filter(pl.col("variantId") == "1_100_A_T")
    assert abs(row["vep_score"][0] - 0.9) < 1e-6


def test_get_variant_effects_mean(variant_fixture, monkeypatch):
    monkeypatch.setattr(
        variants, "ensure_cached_shards", lambda *a, **k: [variant_fixture]
    )
    out = variants.get_variant_effects(aggregate="mean")
    row = out.filter(pl.col("variantId") == "1_100_A_T")
    assert abs(row["vep_score"][0] - 0.7) < 1e-6


def test_get_variant_effects_bad_aggregate(variant_fixture, monkeypatch):
    monkeypatch.setattr(
        variants, "ensure_cached_shards", lambda *a, **k: [variant_fixture]
    )
    with pytest.raises(ValueError, match="aggregate"):
        variants.get_variant_effects(aggregate="bogus")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_opentargets_variants.py -k variant_effects -v`
Expected: FAIL — `AttributeError: ... has no attribute 'get_variant_effects'`.

- [ ] **Step 3: Implement `get_variant_effects`**

Append to `src/biodb/opentargets/variants.py`:

```python
_VEP_AGGREGATES = {"max", "mean", "median"}


def get_variant_effects(
    *,
    version: str = DEFAULT_VERSION,
    aggregate: str = "max",
    cache_dir: str | Path | None = None,
    limit_files: int | None = None,
) -> pl.DataFrame:
    """Per-variant VEP-style deleteriousness score from the OT ``variant`` dataset.

    OT ships one ``normalisedScore`` (on a −1…+1 axis) per predictor method
    inside the ``variantEffect`` list-of-struct column, but no single scalar.
    This aggregates them per variant.

    Parameters
    ----------
    aggregate : {"max", "mean", "median"}
        How to combine predictor scores. ``"max"`` (default) takes the most
        deleterious predictor.

    Returns
    -------
    polars.DataFrame with columns ``variantId, vep_score``.
    """
    if aggregate not in _VEP_AGGREGATES:
        raise ValueError(
            f"aggregate must be one of {sorted(_VEP_AGGREGATES)}, got {aggregate!r}"
        )
    shards = ensure_cached_shards(
        "variant", version=version, cache_dir=cache_dir, limit_files=limit_files
    )
    lazy = pl.concat([pl.scan_parquet(p) for p in shards])
    score = (
        pl.col("variantEffect")
        .list.eval(pl.element().struct.field("normalisedScore"))
        .alias("_scores")
    )
    agg = {
        "max": pl.col("_scores").list.max(),
        "mean": pl.col("_scores").list.mean(),
        "median": pl.col("_scores").list.median(),
    }[aggregate]
    return (
        lazy.select("variantId", score)
        .select("variantId", agg.alias("vep_score"))
        .collect()
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_opentargets_variants.py -v`
Expected: PASS (all 5 tests in the file).

Run: `ruff check src/biodb/opentargets/variants.py && ruff format --check src/biodb/opentargets/variants.py`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add src/biodb/opentargets/variants.py tests/test_opentargets_variants.py
git commit -m "feat(opentargets): add variants.get_variant_effects (aggregated normalisedScore)"
```

---

### Task 5: Optional live smoke tests (network-marked) + full-suite green

**Files:**
- Modify: `tests/test_opentargets_variants.py`, `tests/test_opentargets_studies.py` (append network tests)

**Interfaces:**
- Consumes: the readers from Tasks 2–4.
- Produces: nothing new; guards real OT schema against drift when explicitly run.

- [ ] **Step 1: Write network-marked smoke tests**

Append to `tests/test_opentargets_variants.py`:

```python
@pytest.mark.network
@pytest.mark.slow
def test_get_credible_set_live_schema():
    """Live OT read (1 shard) — asserts the real schema still parses."""
    try:
        out = variants.get_credible_set(study_type="gwas", limit_files=1)
    except Exception as exc:  # noqa: BLE001
        import pytest as _pytest

        msg = str(exc).lower()
        if any(s in msg for s in ("timeout", "connection", "502", "503", "504")):
            _pytest.skip(f"upstream OT outage: {exc}")
        raise
    assert {"variantId", "pip", "beta", "studyType"}.issubset(out.columns)
    assert (out["studyType"] == "gwas").all()
```

- [ ] **Step 2: Verify it is collected but skipped in the CI filter**

Run: `pytest tests/test_opentargets_variants.py -m "not slow and not network" -v`
Expected: the live test is deselected; offline tests PASS.

- [ ] **Step 3: Run the full offline suite**

Run: `pytest -m "not slow and not network" -q`
Expected: PASS (whole repo, no regressions from the package move).

Run: `ruff check . && ruff format --check .`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add tests/test_opentargets_variants.py
git commit -m "test(opentargets): add network-marked live schema smoke test"
```

---

### Task 6: Docs — CLAUDE.md + README reflect the package + new readers

**Files:**
- Modify: `CLAUDE.md` (the module table row for `opentargets.py`; the "Vendored AoU modules" convention block referencing `src/biodb/opentargets.py`)
- Modify: `README.md` (Sources table / Open Targets entry)

**Interfaces:**
- Consumes: nothing.
- Produces: accurate docs.

- [ ] **Step 1: Update CLAUDE.md**

In the architecture table, change the `opentargets.py` row to describe the package: bulk mode split across `opentargets/_bulk.py` (vendored) + first-party `opentargets/variants.py` (`get_credible_set`, `get_variant_effects`) and `opentargets/studies.py` (`get_study`, `attach_study_type`).

In the "Vendored AoU modules" block, change the two `pyproject.toml` snippet paths and the prose from `src/biodb/opentargets.py` to `src/biodb/opentargets/_bulk.py`.

- [ ] **Step 2: Update README.md**

In the Sources table, note the new variant/study bulk readers under Open Targets (`opentargets.variants.get_credible_set`, `opentargets.variants.get_variant_effects`, `opentargets.studies.get_study`).

- [ ] **Step 3: Verify docs build / no broken references**

Run: `grep -rn "src/biodb/opentargets.py" . --include=*.md --include=*.toml`
Expected: no matches (all updated to `_bulk.py`).

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs(opentargets): document package restructure + variants/studies readers"
```

---

## Self-Review

**Spec coverage** (against `cordillera` spec §"Prerequisite: biodb PR"):
- Package restructure + back-compat re-exports → Task 1. ✅
- `get_credible_set` (variantId, chrom, pos, beta, pValues, pip, credibleSetSize, finemappingMethod, studyLocusId, studyId) → Task 3. ✅
- `get_study` + `attach_study_type` (studyType join) → Task 2. ✅
- `get_variant_effects(aggregate="max")` from `variantEffect[*].normalisedScore` → Task 4. ✅
- pyproject exclude/omit path update → Task 1 Step 5. ✅
- Back-compat import test → Task 1. ✅
- network+slow markers with outage guard → Task 5. ✅
- Docs → Task 6. ✅

**Placeholder scan:** No TBD/TODO; every code step shows complete code. ✅

**Type consistency:** `ensure_cached_shards`/`DEFAULT_VERSION` imported from `_bulk` in both new modules; `get_credible_set` calls `studies.get_study` + `studies.attach_study_type` with signatures defined in Task 2; `pip`/`credibleSetSize`/`vep_score` column names consistent across tasks and tests. ✅

**Note on Task 3 test monkeypatching:** `get_credible_set` and `studies.get_study` each reference `ensure_cached_shards` in their *own* module namespace, so the study-type-filter test patches both `variants.ensure_cached_shards` and `studies.ensure_cached_shards` — reflected in the test.
