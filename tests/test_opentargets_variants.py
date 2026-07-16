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
    monkeypatch.setattr(variants, "ensure_cached_shards", lambda *a, **k: [credible_set_fixture])
    out = variants.get_credible_set()
    assert {"variantId", "chromosome", "position", "beta", "pip", "credibleSetSize"}.issubset(
        out.columns
    )
    row = out.filter(pl.col("variantId") == "1_100_A_T")
    assert abs(row["pip"][0] - 0.8) < 1e-6
    assert row["credibleSetSize"][0] == 2
    # locus struct column is not surfaced in the flat output
    assert "locus" not in out.columns


def test_get_credible_set_study_type_filter(credible_set_fixture, study_fixture, monkeypatch):
    def fake_shards(dataset, **kwargs):
        return {"credible_set": [credible_set_fixture], "study": [study_fixture]}[dataset]

    monkeypatch.setattr(variants, "ensure_cached_shards", fake_shards)
    # studies.get_study reads via its own ensure_cached_shards reference
    from biodb.opentargets import studies

    monkeypatch.setattr(studies, "ensure_cached_shards", lambda *a, **k: [study_fixture])

    out = variants.get_credible_set(study_type="gwas")
    assert out.height == 1
    assert out["studyId"][0] == "GCST001"
    assert out["studyType"][0] == "gwas"
