"""Tests for :mod:`biodb.string`.

Network access is mocked with :mod:`responses`; the :mod:`pytest.mark.network`
smoke test at the bottom hits the real STRING download endpoint on demand.
"""

from __future__ import annotations

import gzip
import io
from pathlib import Path

import pandas as pd
import pytest
import responses

from biodb import string as string_mod

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _info_text() -> str:
    return (
        "#string_protein_id\tpreferred_name\tprotein_size\tannotation\n"
        "9606.ENSP00000000001\tBRCA1\t1863\tBreast cancer type 1 susceptibility protein\n"
        "9606.ENSP00000000002\tTP53\t393\tCellular tumor antigen p53\n"
        "9606.ENSP00000000003\tMDM2\t491\tE3 ubiquitin-protein ligase Mdm2\n"
    )


def _links_text() -> str:
    # Symmetric: each undirected edge appears twice. Also include a self-edge
    # (filtered out) and an unmapped protein (also filtered).
    return (
        "protein1 protein2 combined_score\n"
        "9606.ENSP00000000001 9606.ENSP00000000002 900\n"
        "9606.ENSP00000000002 9606.ENSP00000000001 900\n"
        "9606.ENSP00000000002 9606.ENSP00000000003 850\n"
        "9606.ENSP00000000003 9606.ENSP00000000002 850\n"
        "9606.ENSP00000000001 9606.ENSP00000000001 999\n"  # self-edge -> filtered
        "9606.ENSP00000000001 9606.ENSP_UNMAPPED 200\n"  # unmapped -> filtered
    )


def _gzip_bytes(text: str) -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(text.encode("utf-8"))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_module_imports_offline() -> None:
    assert string_mod.__name__ == "biodb.string"


def test_constants_present() -> None:
    assert string_mod.STRING_DOWNLOAD_BASE.startswith("https://")
    assert string_mod.DEFAULT_VERSION == "12.0"
    assert string_mod.DEFAULT_ORGANISM == "9606"
    assert string_mod.CACHE_DIR.exists()


def test_public_api_signatures_stable() -> None:
    for name in (
        "download_physical_links",
        "download_protein_info",
        "load_protein_info",
        "load_physical_links",
        "physical_ppi_edges",
    ):
        assert hasattr(string_mod, name)


def test_url_builders_use_version_and_organism() -> None:
    p_url = string_mod._physical_links_url("12.0", "9606")
    i_url = string_mod._protein_info_url("12.0", "9606")
    assert "12.0" in p_url and "9606" in p_url
    assert "12.0" in i_url and "9606" in i_url
    assert p_url.endswith(".txt.gz")
    assert i_url.endswith(".txt.gz")


# ---------------------------------------------------------------------------
# download / load — happy path + cache
# ---------------------------------------------------------------------------


@responses.activate
def test_download_physical_links_writes_gz(tmp_path: Path) -> None:
    url = string_mod._physical_links_url("12.0", "9606")
    body = _gzip_bytes(_links_text())
    responses.add(responses.GET, url, body=body, status=200)
    out = string_mod.download_physical_links(cache_dir=tmp_path, progress=False)
    assert out.exists()
    assert out.suffix == ".gz"
    assert out.read_bytes() == body


@responses.activate
def test_download_physical_links_skips_when_cached(tmp_path: Path) -> None:
    url = string_mod._physical_links_url("12.0", "9606")
    body = _gzip_bytes(_links_text())
    # First call → hits the mocked HTTP.
    responses.add(responses.GET, url, body=body, status=200)
    string_mod.download_physical_links(cache_dir=tmp_path, progress=False)
    n_calls_before = len(responses.calls)
    # Second call without force= → should NOT issue another HTTP request.
    string_mod.download_physical_links(cache_dir=tmp_path, progress=False)
    assert len(responses.calls) == n_calls_before


@responses.activate
def test_load_protein_info_returns_dataframe(tmp_path: Path) -> None:
    url = string_mod._protein_info_url("12.0", "9606")
    responses.add(responses.GET, url, body=_gzip_bytes(_info_text()), status=200)
    df = string_mod.load_protein_info(cache_dir=tmp_path)
    assert isinstance(df, pd.DataFrame)
    assert "preferred_name" in df.columns
    assert len(df) == 3


@responses.activate
def test_load_physical_links_applies_min_score_filter(tmp_path: Path) -> None:
    url = string_mod._physical_links_url("12.0", "9606")
    responses.add(responses.GET, url, body=_gzip_bytes(_links_text()), status=200)
    df = string_mod.load_physical_links(cache_dir=tmp_path, min_combined_score=900)
    # 3 rows pass the score gate: two symmetric BRCA1-TP53 rows (score 900)
    # plus the self-edge (score 999). ``load_physical_links`` is the raw
    # loader; self-edge filtering only happens in ``physical_ppi_edges``.
    assert len(df) == 3
    assert (df["combined_score"] >= 900).all()


# ---------------------------------------------------------------------------
# physical_ppi_edges — joining + deduplication
# ---------------------------------------------------------------------------


@responses.activate
def test_physical_ppi_edges_normalizes_and_dedups(tmp_path: Path) -> None:
    responses.add(
        responses.GET,
        string_mod._protein_info_url("12.0", "9606"),
        body=_gzip_bytes(_info_text()),
        status=200,
    )
    responses.add(
        responses.GET,
        string_mod._physical_links_url("12.0", "9606"),
        body=_gzip_bytes(_links_text()),
        status=200,
    )
    edges = string_mod.physical_ppi_edges(cache_dir=tmp_path)

    # Schema
    assert set(edges.columns) == {"gene_a", "gene_b", "combined_score", "score"}

    # Self-edge + unmapped row filtered out → only the two true undirected
    # pairs (BRCA1-TP53 and MDM2-TP53) remain, deduplicated.
    assert len(edges) == 2

    # gene_a <= gene_b lex on every row
    assert (edges["gene_a"] <= edges["gene_b"]).all()

    # Symbols upper-cased
    assert edges["gene_a"].str.isupper().all()
    assert edges["gene_b"].str.isupper().all()

    # Continuous score is the raw / 1000
    assert (edges["score"] - edges["combined_score"] / 1000.0).abs().max() < 1e-9

    # Sorted by combined_score descending
    assert edges["combined_score"].is_monotonic_decreasing


@responses.activate
def test_physical_ppi_edges_respects_min_score_filter(tmp_path: Path) -> None:
    responses.add(
        responses.GET,
        string_mod._protein_info_url("12.0", "9606"),
        body=_gzip_bytes(_info_text()),
        status=200,
    )
    responses.add(
        responses.GET,
        string_mod._physical_links_url("12.0", "9606"),
        body=_gzip_bytes(_links_text()),
        status=200,
    )
    # min=900 keeps only BRCA1-TP53 (score 900); MDM2-TP53 at 850 drops out.
    edges = string_mod.physical_ppi_edges(cache_dir=tmp_path, min_combined_score=900)
    assert len(edges) == 1
    assert set(edges.iloc[0][["gene_a", "gene_b"]].tolist()) == {"BRCA1", "TP53"}


# ---------------------------------------------------------------------------
# Live smoke (opt-in) — hits the real STRING endpoint
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_live_physical_ppi_edges_smoke(tmp_path: Path) -> None:
    edges = string_mod.physical_ppi_edges(cache_dir=tmp_path, min_combined_score=700)
    assert len(edges) > 10_000  # human physical PPI at score >= 700
    assert (edges["combined_score"] >= 700).all()
    assert (edges["score"] >= 0.7).all()
