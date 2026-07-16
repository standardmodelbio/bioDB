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
    monkeypatch.setattr(studies, "ensure_cached_shards", lambda *a, **k: [study_fixture])
    out = studies.get_study()
    assert set(["studyId", "studyType", "traitFromSource"]).issubset(out.columns)
    assert out.height == 3
    assert out.filter(pl.col("studyType") == "eqtl").height == 1


def test_get_study_column_subset(study_fixture, monkeypatch):
    monkeypatch.setattr(studies, "ensure_cached_shards", lambda *a, **k: [study_fixture])
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
