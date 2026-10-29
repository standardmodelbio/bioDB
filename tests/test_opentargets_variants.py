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
    monkeypatch.setattr(variants, "ensure_cached_shards", lambda *a, **k: [variant_fixture])
    out = variants.get_variant_effects(aggregate="max")
    assert out.columns == ["variantId", "vep_score"]
    row = out.filter(pl.col("variantId") == "1_100_A_T")
    assert abs(row["vep_score"][0] - 0.9) < 1e-6


def test_get_variant_effects_mean(variant_fixture, monkeypatch):
    monkeypatch.setattr(variants, "ensure_cached_shards", lambda *a, **k: [variant_fixture])
    out = variants.get_variant_effects(aggregate="mean")
    row = out.filter(pl.col("variantId") == "1_100_A_T")
    assert abs(row["vep_score"][0] - 0.7) < 1e-6


def test_get_variant_effects_bad_aggregate(variant_fixture, monkeypatch):
    monkeypatch.setattr(variants, "ensure_cached_shards", lambda *a, **k: [variant_fixture])
    with pytest.raises(ValueError, match="aggregate"):
        variants.get_variant_effects(aggregate="bogus")


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
    assert out.height > 0  # guard: an empty frame makes the .all() below vacuous
    assert {"variantId", "pip", "beta", "studyType"}.issubset(out.columns)
    assert (out["studyType"] == "gwas").all()


def test_summary_matches_lead_id_preserves_raw_columns_and_frame_type():
    raw = pl.DataFrame(
        {
            "variantId": ["lead", "absent", "conflict", "null", "empty"],
            "zScore": [4.2] * 5,
            "studyType": ["gwas"] * 5,
            "locus": [
                [
                    {"variantId": "other", "posteriorProbability": 0.9},
                    {"variantId": "lead", "posteriorProbability": 0.1},
                ],
                [{"variantId": "other", "posteriorProbability": 0.9}],
                [
                    {"variantId": "conflict", "posteriorProbability": 0.2},
                    {"variantId": "conflict", "posteriorProbability": 0.3},
                ],
                [{"variantId": "null", "posteriorProbability": None}],
                [],
            ],
        }
    )
    out = variants.summarize_credible_sets(raw)
    assert out["pip"].to_list() == [0.1, None, None, None, None]
    assert out["pipStatus"].to_list() == [
        "matched",
        "lead_not_found",
        "conflicting_lead",
        "missing_probability",
        "missing_locus",
    ]
    assert out["credibleSetSize"].to_list() == [2, 1, 2, 1, 0]
    assert out.select(raw.columns).equals(raw)
    lazy = variants.summarize_credible_sets(raw.lazy())
    assert isinstance(lazy, pl.LazyFrame)
    assert lazy.collect().equals(out)


def test_lead_reader_keeps_existing_study_type_and_scalar_zscore(tmp_path, monkeypatch):
    path = tmp_path / "credible.parquet"
    pl.DataFrame(
        {
            "variantId": ["lead"],
            "studyId": ["s1"],
            "studyType": ["gwas"],
            "zScore": [6.2],
            "locus": [
                [
                    {"variantId": "other", "posteriorProbability": 0.7},
                    {"variantId": "lead", "posteriorProbability": 0.3},
                ]
            ],
        }
    ).write_parquet(path)
    monkeypatch.setattr(variants, "ensure_cached_shards", lambda *a, **k: [path])
    monkeypatch.setattr(
        variants.studies, "get_study", lambda **kwargs: pytest.fail("unnecessary study join")
    )
    out = variants.get_credible_set(study_type="gwas")
    assert out["pip"].to_list() == [0.3]
    assert out["studyType"].to_list() == ["gwas"]
    assert out["zScore"].to_list() == [6.2]


def test_member_reader_uses_member_statistics_and_retains_flags(tmp_path, monkeypatch):
    path = tmp_path / "credible.parquet"
    pl.DataFrame(
        {
            "variantId": ["1_10_A_G", "empty"],
            "studyId": ["s1", "s2"],
            "beta": [9.0, 9.0],
            "zScore": [99.0, 99.0],
            "locus": [
                [
                    {
                        "variantId": "1_10_A_G",
                        "posteriorProbability": 0.8,
                        "beta": 0.2,
                        "standardError": 0.1,
                        "is95CredibleSet": True,
                        "is99CredibleSet": True,
                    },
                    {
                        "variantId": "1_11_A_T",
                        "posteriorProbability": 0.2,
                        "beta": None,
                        "standardError": None,
                        "is95CredibleSet": False,
                        "is99CredibleSet": True,
                    },
                ],
                [],
            ],
        }
    ).write_parquet(path)
    monkeypatch.setattr(variants, "ensure_cached_shards", lambda *a, **k: [path])
    out = variants.get_credible_set_members()
    assert out["variantId"].to_list() == ["1_10_A_G", "1_11_A_T"]
    assert out["pip"].to_list() == [0.8, 0.2]
    assert out["beta"].to_list() == [0.2, None]
    assert "zScore" not in out.columns  # lead zScore cannot become a member effect
    assert out["leadZScore"].to_list() == [99.0, 99.0]
    assert out["is95CredibleSet"].to_list() == [True, False]
    assert out["is99CredibleSet"].to_list() == [True, True]
    assert variants.get_credible_set_variants().equals(out)


def test_raw_and_annotation_scans_project_without_collecting(tmp_path, monkeypatch):
    path = tmp_path / "variant.parquet"
    raw = pl.DataFrame(
        {"variantId": ["1_1_A_G", "1_2_C_T"], "nested": [[{"score": 0.2}], [{"score": 0.7}]]}
    )
    raw.write_parquet(path)
    monkeypatch.setattr(variants, "ensure_cached_shards", lambda *a, **k: [path])
    assert variants.scan_credible_sets().collect().equals(raw)
    lazy = variants.scan_variant_annotations(columns=["variantId"])
    assert isinstance(lazy, pl.LazyFrame)
    assert "PROJECT 1/2 COLUMNS" in lazy.explain()
    assert lazy.filter(pl.col("variantId") == "1_2_C_T").collect()["variantId"].to_list() == [
        "1_2_C_T"
    ]
    assert variants.get_variant_annotations().equals(raw)
