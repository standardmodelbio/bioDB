"""Tests for :mod:`biodb.snomed` — SNOMED CT vocabulary downloader.

Mocked unit tests for the URL/path/decompression logic, plus a single
live integration test that HEADs the real bioDB release URL (no
download — the asset is 29 MB).
"""

from __future__ import annotations

import gzip

import pandas as pd
import pytest
import requests
import responses

from biodb import snomed

# ---------------------------------------------------------------------------
# Surface
# ---------------------------------------------------------------------------


def test_module_surface_exposes_expected_names() -> None:
    for name in (
        "GITHUB_REPO",
        "GITHUB_RELEASE_TAG",
        "GITHUB_ASSET_NAME",
        "SNOMED_RELEASE_URL",
        "CACHE_DIR",
        "download_concept_csv",
        "load_concept_csv",
        "get_concept_csv_path",
        "get_snomed_data_dir",
        "is_available",
    ):
        assert hasattr(snomed, name), f"biodb.snomed.{name} should be exported"


def test_default_url_points_at_biodb_release_not_synthlab() -> None:
    """The whole point of moving the asset: SNOMED_RELEASE_URL must
    target bioDB. Catches the regression where someone redoes the
    auth flow but forgets to flip the default URL back to bioDB."""
    assert snomed.GITHUB_REPO == "bschilder/bioDB"
    assert "bschilder/bioDB" in snomed.SNOMED_RELEASE_URL
    assert "synthlab" not in snomed.SNOMED_RELEASE_URL
    assert snomed.SNOMED_RELEASE_URL.endswith(snomed.GITHUB_ASSET_NAME)
    assert snomed.GITHUB_RELEASE_TAG in snomed.SNOMED_RELEASE_URL


# ---------------------------------------------------------------------------
# Mocked end-to-end: public download + decompression
# ---------------------------------------------------------------------------


def _make_gzipped_csv() -> bytes:
    """A minimal OHDSI CONCEPT.csv body, tab-separated."""
    csv = (
        "concept_id\tconcept_name\tdomain_id\tvocabulary_id\tconcept_class_id\t"
        "standard_concept\tconcept_code\tvalid_start_date\tvalid_end_date\tinvalid_reason\n"
        "12345\tHypertensive disorder\tCondition\tSNOMED\tClinical Finding\t"
        "S\t38341003\t1970-01-01\t2099-12-31\t\n"
        "67890\tType 2 diabetes mellitus\tCondition\tSNOMED\tClinical Finding\t"
        "S\t44054006\t1970-01-01\t2099-12-31\t\n"
    )
    return gzip.compress(csv.encode("utf-8"))


def test_download_concept_csv_round_trip(tmp_path, monkeypatch) -> None:
    """End-to-end: GET gzipped CSV → decompress → return path."""
    # Force the public-download path: disable gh CLI + clear env tokens.
    monkeypatch.setattr(snomed, "_gh_cli_available", lambda: False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)

    body = _make_gzipped_csv()
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(
            responses.GET,
            snomed.SNOMED_RELEASE_URL,
            body=body,
            status=200,
            headers={"content-length": str(len(body))},
        )
        path = snomed.download_concept_csv(output_dir=tmp_path, progress=False)

    assert path == tmp_path / "CONCEPT.csv"
    assert path.exists()
    # Decompressed CSV body should start with the header.
    content = path.read_text()
    assert content.startswith("concept_id\tconcept_name")
    assert "Hypertensive disorder" in content
    # The intermediate .gz file should be cleaned up after decompression.
    assert not (tmp_path / "CONCEPT.csv.gz").exists()


def test_download_concept_csv_uses_cache_when_present(tmp_path, monkeypatch) -> None:
    """A pre-existing CONCEPT.csv should short-circuit — no HTTP call."""
    cached = tmp_path / "CONCEPT.csv"
    cached.write_text("concept_id\tconcept_name\n1\tcached\n")
    monkeypatch.setattr(snomed, "_gh_cli_available", lambda: False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    with responses.RequestsMock() as mock_resp:
        # No mocks registered — any HTTP call would raise.
        path = snomed.download_concept_csv(output_dir=tmp_path)
        assert len(mock_resp.calls) == 0
    assert path == cached
    assert "cached" in cached.read_text()


def test_download_concept_csv_force_overrides_cache(tmp_path, monkeypatch) -> None:
    """``force=True`` should ignore the cached file and re-download."""
    cached = tmp_path / "CONCEPT.csv"
    cached.write_text("STALE CONTENTS\n")
    monkeypatch.setattr(snomed, "_gh_cli_available", lambda: False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)

    body = _make_gzipped_csv()
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(
            responses.GET,
            snomed.SNOMED_RELEASE_URL,
            body=body,
            status=200,
        )
        path = snomed.download_concept_csv(output_dir=tmp_path, force=True, progress=False)

    assert "Hypertensive" in path.read_text()
    assert "STALE" not in path.read_text()


def test_download_concept_csv_cleans_up_partial_on_error(tmp_path, monkeypatch) -> None:
    """If the public download succeeds but the gzip is corrupt, the
    decompression step should fail and clean up both files. No partial
    CONCEPT.csv should remain on disk to fool a future cache check."""
    monkeypatch.setattr(snomed, "_gh_cli_available", lambda: False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(
            responses.GET,
            snomed.SNOMED_RELEASE_URL,
            body=b"not actually gzipped",
            status=200,
        )
        with pytest.raises((OSError, EOFError, Exception)):  # gzip raises BadGzipFile
            snomed.download_concept_csv(output_dir=tmp_path, progress=False)
    # No leftover files.
    assert not (tmp_path / "CONCEPT.csv").exists()
    assert not (tmp_path / "CONCEPT.csv.gz").exists()


def test_download_concept_csv_raises_when_all_strategies_fail(tmp_path, monkeypatch) -> None:
    """No gh CLI, no token, public URL returns 404 → clean RuntimeError."""
    monkeypatch.setattr(snomed, "_gh_cli_available", lambda: False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(
            responses.GET,
            snomed.SNOMED_RELEASE_URL,
            json={"error": "not found"},
            status=404,
        )
        with pytest.raises(RuntimeError, match="Failed to download SNOMED"):
            snomed.download_concept_csv(output_dir=tmp_path, progress=False)


def test_load_concept_csv_returns_dataframe(tmp_path, monkeypatch) -> None:
    """``load_concept_csv`` should parse the fixture into the documented columns."""
    monkeypatch.setattr(snomed, "_gh_cli_available", lambda: False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setattr(snomed, "CACHE_DIR", tmp_path)

    body = _make_gzipped_csv()
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(
            responses.GET,
            snomed.SNOMED_RELEASE_URL,
            body=body,
            status=200,
        )
        df = snomed.load_concept_csv(progress=False)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 10)
    for col in (
        "concept_id",
        "concept_name",
        "domain_id",
        "vocabulary_id",
        "concept_class_id",
        "standard_concept",
        "concept_code",
        "valid_start_date",
        "valid_end_date",
        "invalid_reason",
    ):
        assert col in df.columns
    # concept_id should be integer-typed thanks to the dtype override.
    assert df["concept_id"].dtype.kind in "iu"
    assert list(df["concept_id"]) == [12345, 67890]


# ---------------------------------------------------------------------------
# Auth-helper unit tests (no real subprocess calls)
# ---------------------------------------------------------------------------


def test_get_github_token_prefers_GITHUB_TOKEN_over_GH_TOKEN(monkeypatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "primary")
    monkeypatch.setenv("GH_TOKEN", "secondary")
    assert snomed._get_github_token() == "primary"


def test_get_github_token_falls_back_to_GH_TOKEN(monkeypatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("GH_TOKEN", "secondary")
    assert snomed._get_github_token() == "secondary"


def test_get_github_token_returns_none_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    assert snomed._get_github_token() is None


def test_is_available_reflects_cache_state(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(snomed, "CACHE_DIR", tmp_path)
    assert snomed.is_available() is False
    (tmp_path / "CONCEPT.csv").write_text("seeded\n")
    assert snomed.is_available() is True


# ---------------------------------------------------------------------------
# Live integration — HEAD the real release asset, don't download
# ---------------------------------------------------------------------------


def test_release_asset_url_is_alive() -> None:
    """HEAD the actual bioDB release URL — verify the asset is reachable
    without downloading the 29 MB body."""
    response = requests.head(snomed.SNOMED_RELEASE_URL, timeout=15, allow_redirects=True)
    assert response.status_code == 200
    if "content-length" in response.headers:
        size = int(response.headers["content-length"])
        # Asset is ~29 MB; anything < 1 MB is an error page.
        assert size > 1_000_000, f"CONCEPT.csv.gz reports size {size} bytes — suspicious"
