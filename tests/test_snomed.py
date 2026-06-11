"""Tests for :mod:`biodb.snomed` — local CONCEPT.csv parser + OLS4-backed
per-concept lookups.

Two halves:

* **Bulk parser** — pure local I/O. Mocked-only, no network. Covers
  ``load_concept_csv`` + ``load_concept_csv_from_zip`` against synthetic
  fixtures matching the OHDSI CDM concept schema.
* **Per-concept lookups via OLS** — mixed mocked + live, same pattern as
  ``test_ols.py``. The CURIE normalisation is pure-function; the live
  tests round-trip a known stable SNOMED concept (38341003,
  "Hypertensive disorder").

There are deliberately no tests for a bulk **downloader**. ``biodb.snomed``
no longer ships one — SNOMED CT's licensing prohibits onward
redistribution from a public mirror. Users obtain CONCEPT.csv themselves
from https://athena.ohdsi.org after accepting the SNOMED CT license, and
the parser consumes whatever they bring.
"""

from __future__ import annotations

import gzip
import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest
import requests

from biodb import snomed

# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_module_surface_exposes_expected_names() -> None:
    for name in (
        "ATHENA_DOWNLOAD_PAGE",
        "CACHE_DIR",
        "OLS_ONTOLOGY_SLUG",
        "load_concept_csv",
        "load_concept_csv_from_zip",
        "get_snomed_data_dir",
        "query_concept",
        "search_concepts",
        "get_descendants",
        "get_ancestors",
        "get_children",
        "get_parents",
    ):
        assert hasattr(snomed, name), f"biodb.snomed.{name} should be exported"


def test_module_no_longer_exposes_bulk_downloader_constants() -> None:
    """Regression: the GitHub-release bulk path was removed in 2026-05-18
    for SNOMED CT licensing reasons. Catches the regression where someone
    re-adds it without thinking about distribution rights."""
    for name in (
        "GITHUB_REPO",
        "GITHUB_RELEASE_TAG",
        "GITHUB_ASSET_NAME",
        "SNOMED_RELEASE_URL",
        "download_concept_csv",
    ):
        assert not hasattr(snomed, name), (
            f"biodb.snomed.{name} was deliberately removed; do not re-add "
            f"without a licensing review (see module docstring)."
        )


def test_athena_download_page_points_at_athena() -> None:
    """If the docstring or tutorial breaks, the most important thing it
    must still convey is *where to get the data*. Pin the URL."""
    assert snomed.ATHENA_DOWNLOAD_PAGE.startswith("https://athena.ohdsi.org")


# ---------------------------------------------------------------------------
# Bulk parser — local CSV + zip
# ---------------------------------------------------------------------------

_CONCEPT_HEADER = (
    "concept_id\tconcept_name\tdomain_id\tvocabulary_id\tconcept_class_id\t"
    "standard_concept\tconcept_code\tvalid_start_date\tvalid_end_date\tinvalid_reason"
)

_CONCEPT_ROWS = (
    # Hypertensive disorder (SNOMED)
    "12345\tHypertensive disorder\tCondition\tSNOMED\tClinical Finding\t"
    "S\t38341003\t1970-01-01\t2099-12-31\t",
    # Type 2 diabetes mellitus (SNOMED)
    "67890\tType 2 diabetes mellitus\tCondition\tSNOMED\tClinical Finding\t"
    "S\t44054006\t1970-01-01\t2099-12-31\t",
    # Aspirin 81 mg (RxNorm) — different vocabulary, lets us test the filter
    "1112807\tAspirin 81 MG\tDrug\tRxNorm\tBranded Drug\tS\t315431\t1970-01-01\t2099-12-31\t",
)


def _concept_csv_bytes() -> bytes:
    return ("\n".join((_CONCEPT_HEADER, *_CONCEPT_ROWS)) + "\n").encode("utf-8")


def test_load_concept_csv_parses_documented_columns(tmp_path) -> None:
    csv_path = tmp_path / "CONCEPT.csv"
    csv_path.write_bytes(_concept_csv_bytes())

    df = snomed.load_concept_csv(csv_path)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 10)
    for col in (
        "concept_id",
        "concept_name",
        "domain_id",
        "vocabulary_id",
        "concept_class_id",
        "standard_concept",
        "concept_code",
    ):
        assert col in df.columns
    # concept_id should be Int64-typed thanks to the dtype overrides.
    assert df["concept_id"].dtype.kind in "iu"
    assert list(df["concept_id"]) == [12345, 67890, 1112807]


def test_load_concept_csv_vocabulary_filter_keeps_only_snomed(tmp_path) -> None:
    """The OHDSI bundle mixes SNOMED with RxNorm/LOINC/etc. The filter
    is the fast path for the common case 'I only want the SNOMED slice'."""
    csv_path = tmp_path / "CONCEPT.csv"
    csv_path.write_bytes(_concept_csv_bytes())

    df = snomed.load_concept_csv(csv_path, vocabulary_id="SNOMED")
    assert len(df) == 2
    assert set(df["vocabulary_id"]) == {"SNOMED"}
    assert "RxNorm" not in set(df["vocabulary_id"])


def test_load_concept_csv_raises_for_missing_file(tmp_path) -> None:
    """Missing-file error message must direct the user at Athena."""
    with pytest.raises(FileNotFoundError, match="athena.ohdsi.org"):
        snomed.load_concept_csv(tmp_path / "does_not_exist.csv")


def test_load_concept_csv_forwards_read_csv_kwargs(tmp_path) -> None:
    """``**read_csv_kwargs`` must reach pandas — e.g. nrows for a peek."""
    csv_path = tmp_path / "CONCEPT.csv"
    csv_path.write_bytes(_concept_csv_bytes())

    df = snomed.load_concept_csv(csv_path, nrows=1)
    assert len(df) == 1


def test_load_concept_csv_expands_user_tilde(tmp_path, monkeypatch) -> None:
    """``~/foo`` should be expanded to the user's home so callers can
    pass shell-style paths without thinking about it."""
    csv_path = tmp_path / "CONCEPT.csv"
    csv_path.write_bytes(_concept_csv_bytes())
    # Pretend tmp_path is $HOME and pass ``~/CONCEPT.csv``.
    monkeypatch.setenv("HOME", str(tmp_path))
    df = snomed.load_concept_csv("~/CONCEPT.csv")
    assert len(df) == 3


def test_load_concept_csv_from_zip_extracts_member(tmp_path) -> None:
    """Athena bundles arrive as a flat zip — verify we can read CONCEPT.csv
    out of it without staging to disk."""
    zip_path = tmp_path / "vocabulary_download_v5_abc_123.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("CONCEPT.csv", _concept_csv_bytes())
        zf.writestr("CONCEPT_RELATIONSHIP.csv", "concept_id_1\tconcept_id_2\n1\t2\n")

    df = snomed.load_concept_csv_from_zip(zip_path)
    assert len(df) == 3
    assert "concept_name" in df.columns


def test_load_concept_csv_from_zip_handles_nested_member(tmp_path) -> None:
    """Some Athena bundles wrap members under a single sub-directory.
    Match on basename so layout drift doesn't break us."""
    zip_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("vocab_2026_05/CONCEPT.csv", _concept_csv_bytes())

    df = snomed.load_concept_csv_from_zip(zip_path)
    assert len(df) == 3


def test_load_concept_csv_from_zip_filters_by_vocabulary(tmp_path) -> None:
    zip_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("CONCEPT.csv", _concept_csv_bytes())

    df = snomed.load_concept_csv_from_zip(zip_path, vocabulary_id="RxNorm")
    assert len(df) == 1
    assert list(df["vocabulary_id"]) == ["RxNorm"]


def test_load_concept_csv_from_zip_raises_on_missing_member(tmp_path) -> None:
    zip_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("README.txt", "no CONCEPT here")

    with pytest.raises(KeyError, match="CONCEPT.csv"):
        snomed.load_concept_csv_from_zip(zip_path)


def test_load_concept_csv_from_zip_raises_for_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="athena.ohdsi.org"):
        snomed.load_concept_csv_from_zip(tmp_path / "does_not_exist.zip")


def test_get_snomed_data_dir_is_a_path() -> None:
    """The cache helper is here for callers who want to stash their
    parsed bundle somewhere stable. bioDB doesn't auto-populate it."""
    d = snomed.get_snomed_data_dir()
    assert isinstance(d, Path)
    assert d.exists()


# ---------------------------------------------------------------------------
# Per-concept lookups via OLS — argument normalisation + live round-trip
# ---------------------------------------------------------------------------


def test_normalize_concept_id_accepts_int() -> None:
    assert snomed._normalize_concept_id(38341003) == "SNOMED:38341003"


def test_normalize_concept_id_accepts_bare_digit_string() -> None:
    assert snomed._normalize_concept_id("38341003") == "SNOMED:38341003"


def test_normalize_concept_id_passes_curie_through_unchanged() -> None:
    assert snomed._normalize_concept_id("SNOMED:38341003") == "SNOMED:38341003"


def test_normalize_concept_id_passes_iri_through_unchanged() -> None:
    iri = "http://snomed.info/id/38341003"
    assert snomed._normalize_concept_id(iri) == iri


def test_snomed_ols_slug_is_snomed() -> None:
    """If someone renames the OLS slug, all the per-concept helpers
    silently start hitting the wrong ontology — pin it."""
    assert snomed.OLS_ONTOLOGY_SLUG == "snomed"


def test_query_concept_hypertensive_disorder_round_trip() -> None:
    """Real OLS lookup: 38341003 → 'Hypertensive disorder'."""
    record = snomed.query_concept(38341003)
    assert record["obo_id"] == "SNOMED:38341003"
    assert record["label"] == "Hypertensive disorder"
    assert record["iri"] == "http://snomed.info/id/38341003"


def test_query_concept_accepts_curie_string() -> None:
    record = snomed.query_concept("SNOMED:73211009")  # diabetes mellitus
    assert record["obo_id"] == "SNOMED:73211009"
    assert "diabetes" in record["label"].lower()


def test_search_concepts_returns_dataframe_of_hits() -> None:
    hits = snomed.search_concepts("hypertension", rows=5)
    assert isinstance(hits, pd.DataFrame)
    assert len(hits) == 5
    for col in ("obo_id", "label", "iri"):
        assert col in hits.columns


def test_get_children_is_one_hop_subset_of_descendants() -> None:
    """Direct children should be a subset of all descendants."""
    # Live OLS call — tolerate a transient EBI connectivity blip (the CI
    # failure was a 30s ReadTimeout) rather than hard-failing. Scoped to
    # Timeout/ConnectionError ONLY: an HTTPError (4xx/5xx), parse error, or
    # the assertions below still fail loudly, so a real API change or
    # regression on our side is NOT masked.
    try:
        children = snomed.get_children(38341003, size=10)
        descendants = snomed.get_descendants(38341003, size=500)
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
        pytest.skip(f"OLS/EBI unreachable (transient connectivity): {exc}")
    assert set(children["obo_id"]) <= set(descendants["obo_id"])
    assert len(children) <= len(descendants)


# Silence unused-import warnings — gzip / io are kept around in case
# additional fixture helpers are added later.
_ = (gzip, io)
