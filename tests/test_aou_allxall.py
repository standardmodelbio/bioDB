"""Tests for :mod:`biodb.aou_allxall`.

The All-by-All API is a public unauthenticated REST endpoint. Offline
tests use :mod:`responses` to mock the HTTP layer. A handful of
``@pytest.mark.network`` tests at the bottom probe the live server to
catch upstream schema drift.
"""

from __future__ import annotations

import json

import pandas as pd
import polars as pl
import pytest
import responses

from biodb import aou_allxall

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FAKE_ANALYSES = [
    {
        "analysis_id": "100001",
        "ancestry_group": "META",
        "category": "lab_measurement",
        "description": "Total cholesterol",
        "description_more": "Total cholesterol (mg/dL)",
        "keep_pheno_burden": True,
        "keep_pheno_skat": True,
        "keep_pheno_skato": True,
        "lambda_gc_acaf": 1.01,
        "lambda_gc_exome": 1.00,
        "lambda_gc_gene_burden_001": 1.02,
        "n_cases": 12345,
        "n_controls": 200000,
        "pheno_sex": "both",
        "trait_type": "continuous",
    },
    {
        "analysis_id": "200002",
        "ancestry_group": "META",
        "category": "pfhh_survey",
        "description": "Family history of diabetes",
        "description_more": "Family history of diabetes (PFHH survey)",
        "keep_pheno_burden": True,
        "keep_pheno_skat": True,
        "keep_pheno_skato": True,
        "lambda_gc_acaf": 1.00,
        "lambda_gc_exome": 1.00,
        "lambda_gc_gene_burden_001": 1.00,
        "n_cases": 50000,
        "n_controls": 150000,
        "pheno_sex": "both",
        "trait_type": "binary",
    },
]

_FAKE_GENE_BURDEN = [
    {
        "gene_id": "ENSG00000000001",
        "gene_symbol": "GENEA",
        "annotation": "pLoF",
        "max_maf": 0.001,
        "analysis_id": "100001",
        "ancestry_group": "meta",
        "pvalue": 1e-9,
        "neg_log10_p": 9.0,
        "pvalue_burden": 1e-10,
        "neg_log10_p_burden": 10.0,
        "pvalue_skat": 1e-8,
        "neg_log10_p_skat": 8.0,
        "beta_burden": 2.3,
        "mac": 500,
        "contig": "chr1",
        "gene_start_position": 1000000,
    },
    {
        "gene_id": "ENSG00000000002",
        "gene_symbol": "GENEB",
        "annotation": "missenseLC",
        "max_maf": 0.001,
        "analysis_id": "100001",
        "ancestry_group": "meta",
        "pvalue": 1e-5,
        "neg_log10_p": 5.0,
        "pvalue_burden": 1e-6,
        "neg_log10_p_burden": 6.0,
        "pvalue_skat": 1e-4,
        "neg_log10_p_skat": 4.0,
        "beta_burden": -1.2,
        "mac": 200,
        "contig": "chr2",
        "gene_start_position": 2000000,
    },
    {
        "gene_id": "ENSG00000000003",
        "gene_symbol": "GENEC",
        "annotation": "pLoF",
        "max_maf": 0.001,
        "analysis_id": "100001",
        "ancestry_group": "meta",
        "pvalue": 1e-3,
        "neg_log10_p": 3.0,
        "pvalue_burden": 1e-3,
        "neg_log10_p_burden": 3.0,
        "pvalue_skat": 1e-2,
        "neg_log10_p_skat": 2.0,
        "beta_burden": 0.5,
        "mac": 100,
        "contig": "chr3",
        "gene_start_position": 3000000,
    },
]


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Redirect ``CACHE_DIR`` to a per-test tmpdir so cache hits don't leak across tests."""
    monkeypatch.setattr(aou_allxall, "CACHE_DIR", tmp_path)
    # Reset the in-process analyses cache too.
    monkeypatch.setattr(aou_allxall, "_ANALYSES_CACHE", {})
    return tmp_path


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_module_imports_offline() -> None:
    assert aou_allxall.__name__ == "biodb.aou_allxall"


def test_constants_present() -> None:
    assert aou_allxall.BASE_URL == "https://allbyall.researchallofus.org"
    assert aou_allxall.API_URL == "https://allbyall.researchallofus.org/api"
    assert aou_allxall.DEFAULT_VERSION == "v1"
    assert "meta" in aou_allxall.ANCESTRY_CODES
    assert "pLoF" in aou_allxall.BURDEN_SETS
    assert "missenseLC" in aou_allxall.BURDEN_SETS
    assert "synonymous" in aou_allxall.BURDEN_SETS
    assert aou_allxall.MAF_THRESHOLDS == (0.01, 0.001, 0.0001)
    assert aou_allxall.BURDEN_TESTS == ("burden", "skat", "skato")


def test_public_api_signatures_stable() -> None:
    for name in (
        "get_config",
        "list_categories",
        "list_analyses",
        "list_assets",
        "get_assets_summary",
        "get_gene_burden",
        "download_all_gene_burden",
        "load_gene_burden",
        "melt_gene_burden",
        "iter_signature_variants",
        "query_phenotype",
        "list_phenotypes",
    ):
        assert hasattr(aou_allxall, name), f"missing public symbol {name}"


# ---------------------------------------------------------------------------
# _request_json — retry/backoff behavior
# ---------------------------------------------------------------------------


def test_request_json_returns_payload() -> None:
    with responses.RequestsMock() as mock:
        mock.add(
            responses.GET,
            f"{aou_allxall.API_URL}/some/path",
            json={"hello": "world"},
            status=200,
        )
        payload = aou_allxall._request_json("/some/path")
        assert payload == {"hello": "world"}


def test_request_json_retries_on_5xx_then_succeeds() -> None:
    with responses.RequestsMock() as mock:
        mock.add(responses.GET, f"{aou_allxall.API_URL}/p", status=503)
        mock.add(responses.GET, f"{aou_allxall.API_URL}/p", json=[1, 2], status=200)
        result = aou_allxall._request_json("/p", max_retries=3)
        assert result == [1, 2]
        assert len(mock.calls) == 2


def test_request_json_gives_up_after_max_retries() -> None:
    with responses.RequestsMock() as mock:
        for _ in range(5):
            mock.add(responses.GET, f"{aou_allxall.API_URL}/fail", status=502)
        with pytest.raises(RuntimeError, match="Failed to GET"):
            aou_allxall._request_json("/fail", max_retries=5)


# ---------------------------------------------------------------------------
# list_analyses / list_categories / get_config — cache layer
# ---------------------------------------------------------------------------


def test_list_analyses_caches_to_parquet(isolated_cache) -> None:
    with responses.RequestsMock() as mock:
        mock.add(
            responses.GET,
            f"{aou_allxall.API_URL}/analyses",
            json=_FAKE_ANALYSES,
            status=200,
            match=[responses.matchers.query_param_matcher({"ancestry_group": "meta"})],
        )
        df = aou_allxall.list_analyses(ancestry="meta")
    assert len(df) == 2
    assert (isolated_cache / "analyses_meta.parquet").exists()


def test_list_analyses_reads_from_cache_on_second_call(isolated_cache) -> None:
    """A second call with cache present must not hit the network."""
    with responses.RequestsMock() as mock:
        mock.add(
            responses.GET,
            f"{aou_allxall.API_URL}/analyses",
            json=_FAKE_ANALYSES,
            status=200,
        )
        aou_allxall.list_analyses(ancestry="meta")
        assert len(mock.calls) == 1

    # Second call → no .add(); any HTTP attempt would explode.
    with responses.RequestsMock() as mock:
        df = aou_allxall.list_analyses(ancestry="meta")
        assert len(df) == 2
        assert len(mock.calls) == 0


def test_list_analyses_force_refetches(isolated_cache) -> None:
    cache_file = isolated_cache / "analyses_meta.parquet"
    cache_file.write_bytes(b"")  # stale cache
    # The empty file would explode parquet read; force=True must skip the read.
    with responses.RequestsMock() as mock:
        mock.add(
            responses.GET,
            f"{aou_allxall.API_URL}/analyses",
            json=_FAKE_ANALYSES,
            status=200,
        )
        df = aou_allxall.list_analyses(ancestry="meta", force=True)
        assert len(df) == 2


def test_list_analyses_rejects_unknown_ancestry() -> None:
    with pytest.raises(ValueError, match="ancestry"):
        aou_allxall.list_analyses(ancestry="klingon")


def test_list_categories_caches_as_parquet(isolated_cache) -> None:
    with responses.RequestsMock() as mock:
        mock.add(
            responses.GET,
            f"{aou_allxall.API_URL}/categories",
            json=[{"category": "lab", "color": "#abc", "analyses": ["1", "2"]}],
            status=200,
        )
        df = aou_allxall.list_categories()
    assert len(df) == 1
    assert (isolated_cache / "categories.parquet").exists()


def test_get_config_caches_as_json(isolated_cache) -> None:
    payload = {
        "ancestry_codes": list(aou_allxall.ANCESTRY_CODES),
        "burden_sets": list(aou_allxall.BURDEN_SETS),
    }
    with responses.RequestsMock() as mock:
        mock.add(responses.GET, f"{aou_allxall.API_URL}/config", json=payload, status=200)
        cfg = aou_allxall.get_config()
    assert cfg == payload
    assert (isolated_cache / "config.json").exists()
    assert json.loads((isolated_cache / "config.json").read_text()) == payload


# ---------------------------------------------------------------------------
# get_gene_burden
# ---------------------------------------------------------------------------


def test_get_gene_burden_caches_per_phenotype(isolated_cache) -> None:
    with responses.RequestsMock() as mock:
        mock.add(
            responses.GET,
            f"{aou_allxall.API_URL}/phenotype/100001/genes",
            json=_FAKE_GENE_BURDEN,
            status=200,
            match=[responses.matchers.query_param_matcher({"max_maf": "0.001"})],
        )
        df = aou_allxall.get_gene_burden("100001")
    assert len(df) == 3
    # Filename embeds the MAF so per-MAF pulls don't collide.
    assert (isolated_cache / "gene_burden" / "100001_maf0.001.parquet").exists()


def test_get_gene_burden_handles_empty_response(isolated_cache) -> None:
    """Some (phenotype, ancestry) pairs return an empty list — must produce an empty frame."""
    with responses.RequestsMock() as mock:
        mock.add(
            responses.GET,
            f"{aou_allxall.API_URL}/phenotype/999999/genes",
            json=[],
            status=200,
        )
        df = aou_allxall.get_gene_burden("999999")
    assert len(df) == 0
    assert isinstance(df, pd.DataFrame)


def test_get_gene_burden_with_ancestry_and_maf_suffix(isolated_cache) -> None:
    """Passing ``ancestry`` and ``max_maf`` should embed both in the cache filename."""
    with responses.RequestsMock() as mock:
        mock.add(
            responses.GET,
            f"{aou_allxall.API_URL}/phenotype/100001/genes",
            json=_FAKE_GENE_BURDEN,
            status=200,
            match=[
                responses.matchers.query_param_matcher({"ancestry_group": "eur", "max_maf": "0.01"})
            ],
        )
        aou_allxall.get_gene_burden("100001", ancestry="eur", max_maf=0.01)
    assert (isolated_cache / "gene_burden" / "100001_eur_maf0.01.parquet").exists()


def test_get_gene_burden_rejects_unknown_maf() -> None:
    with pytest.raises(ValueError, match="max_maf"):
        aou_allxall.get_gene_burden("100001", max_maf=0.5)


# ---------------------------------------------------------------------------
# melt_gene_burden — pure pandas
# ---------------------------------------------------------------------------


def test_melt_gene_burden_default_burden_test_no_filter() -> None:
    """Default test='burden', no burden_set/max_maf filter → all rows kept."""
    df = pd.DataFrame(_FAKE_GENE_BURDEN)
    long = aou_allxall.melt_gene_burden(df)
    assert len(long) == 3
    assert set(long["test"]) == {"burden"}
    # GENEA neg_log10_p_burden=10, beta=+2.3 → signed score = +10
    geneA = long[long["targetId"] == "ENSG00000000001"].iloc[0]
    assert geneA["score"] == pytest.approx(10.0)
    # GENEB neg_log10_p_burden=6, beta=-1.2 → signed score = -6
    geneB = long[long["targetId"] == "ENSG00000000002"].iloc[0]
    assert geneB["score"] == pytest.approx(-6.0)


def test_melt_gene_burden_filters_to_burden_set() -> None:
    df = pd.DataFrame(_FAKE_GENE_BURDEN)
    long = aou_allxall.melt_gene_burden(df, burden_set="pLoF")
    assert len(long) == 2
    assert set(long["annotation"]) == {"pLoF"}


def test_melt_gene_burden_skat_uses_skat_column() -> None:
    """test='skat' must pull scores from neg_log10_p_skat, not neg_log10_p_burden."""
    df = pd.DataFrame(_FAKE_GENE_BURDEN)
    long = aou_allxall.melt_gene_burden(df, test="skat", burden_set="pLoF")
    # GENEA neg_log10_p_skat=8, beta=+2.3 → signed = +8 (vs. +10 for burden).
    geneA = long[long["targetId"] == "ENSG00000000001"].iloc[0]
    assert geneA["score"] == pytest.approx(8.0)


def test_melt_gene_burden_skato_uses_combined_column() -> None:
    """test='skato' must pull scores from neg_log10_p (the bare combined column)."""
    df = pd.DataFrame(_FAKE_GENE_BURDEN)
    long = aou_allxall.melt_gene_burden(df, test="skato", burden_set="pLoF")
    geneA = long[long["targetId"] == "ENSG00000000001"].iloc[0]
    assert geneA["score"] == pytest.approx(9.0)


def test_melt_gene_burden_unsigned_when_beta_signed_false() -> None:
    df = pd.DataFrame(_FAKE_GENE_BURDEN)
    long = aou_allxall.melt_gene_burden(df, beta_signed=False)
    assert (long["score"] >= 0).all()


def test_melt_gene_burden_rejects_unknown_test() -> None:
    df = pd.DataFrame(_FAKE_GENE_BURDEN)
    with pytest.raises(ValueError, match="test"):
        aou_allxall.melt_gene_burden(df, test="not_a_test")


def test_melt_gene_burden_accepts_polars_input() -> None:
    pdf = pl.DataFrame(_FAKE_GENE_BURDEN)
    long = aou_allxall.melt_gene_burden(pdf, burden_set="pLoF")
    assert isinstance(long, pd.DataFrame)
    assert len(long) == 2


def test_iter_signature_variants_enumerates_full_grid() -> None:
    """The full grid is len(tests) × len(burden_sets) × len(max_mafs) facets."""
    df = pd.DataFrame(_FAKE_GENE_BURDEN)
    variants = list(aou_allxall.iter_signature_variants(df))
    expected = (
        len(aou_allxall.BURDEN_TESTS)
        * len(aou_allxall.BURDEN_SETS)
        * len(aou_allxall.MAF_THRESHOLDS)
    )
    assert len(variants) == expected
    # Each facet is a (dict, DataFrame) pair.
    facet, long = variants[0]
    assert {"test", "burden_set", "max_maf"} == set(facet)
    assert isinstance(long, pd.DataFrame)


def test_iter_signature_variants_respects_custom_grid() -> None:
    df = pd.DataFrame(_FAKE_GENE_BURDEN)
    variants = list(
        aou_allxall.iter_signature_variants(
            df,
            tests=("burden",),
            burden_sets=("pLoF",),
            max_mafs=(0.001,),
        )
    )
    assert len(variants) == 1
    facet, long = variants[0]
    assert facet == {"test": "burden", "burden_set": "pLoF", "max_maf": 0.001}
    # Two pLoF rows in the fake data.
    assert len(long) == 2


# ---------------------------------------------------------------------------
# query_phenotype
# ---------------------------------------------------------------------------


def test_query_phenotype_by_id(isolated_cache) -> None:
    with responses.RequestsMock() as mock:
        mock.add(
            responses.GET,
            f"{aou_allxall.API_URL}/analyses",
            json=_FAKE_ANALYSES,
            status=200,
        )
        hits = aou_allxall.query_phenotype(100001)
    assert len(hits) == 1
    assert hits.iloc[0]["description"] == "Total cholesterol"


def test_query_phenotype_substring_search(isolated_cache) -> None:
    with responses.RequestsMock() as mock:
        mock.add(
            responses.GET,
            f"{aou_allxall.API_URL}/analyses",
            json=_FAKE_ANALYSES,
            status=200,
        )
        hits = aou_allxall.query_phenotype("diabetes")
    assert len(hits) == 1
    assert hits.iloc[0]["analysis_id"] == "200002"


def test_query_phenotype_unknown_column_raises(isolated_cache) -> None:
    with responses.RequestsMock() as mock:
        mock.add(
            responses.GET,
            f"{aou_allxall.API_URL}/analyses",
            json=_FAKE_ANALYSES,
            status=200,
        )
        with pytest.raises(KeyError):
            aou_allxall.query_phenotype("anything", column="does_not_exist")


# ---------------------------------------------------------------------------
# Live integration tests — gated behind --run-network in CI
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_live_get_config(isolated_cache) -> None:
    cfg = aou_allxall.get_config(force=True)
    assert "burden_sets" in cfg
    assert set(cfg["burden_sets"]) >= {"pLoF", "missenseLC"}
    assert set(cfg["ancestry_codes"]) == set(aou_allxall.ANCESTRY_CODES)


@pytest.mark.network
def test_live_list_analyses_returns_expected_scale(isolated_cache) -> None:
    df = aou_allxall.list_analyses(ancestry="meta", force=True)
    assert 3000 < len(df) < 5000, (
        f"Got {len(df)} meta analyses — All-by-All normally ~3,600. "
        "Upstream schema may have changed."
    )
    assert "analysis_id" in df.columns


@pytest.mark.network
@pytest.mark.slow
def test_live_get_gene_burden_for_first_analysis(isolated_cache) -> None:
    analyses = aou_allxall.list_analyses(ancestry="meta", force=True)
    aid = analyses.iloc[0]["analysis_id"]
    df = aou_allxall.get_gene_burden(aid, force=True)
    assert len(df) > 0
    for col in ("gene_id", "gene_symbol", "annotation", "neg_log10_p_burden", "beta_burden"):
        assert col in df.columns, f"missing column {col}"
