"""Tests for :mod:`biodb.traitgym` — downloads mocked with :mod:`responses`."""

from __future__ import annotations

import io
from pathlib import Path

import polars as pl
import pytest
import responses

from biodb import traitgym


def _parquet_bytes(frame: pl.DataFrame) -> bytes:
    buf = io.BytesIO()
    frame.write_parquet(buf)
    return buf.getvalue()


def _variants() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "chrom": ["1", "19", "X"],
            "pos": [1425822, 5000, 100],
            "ref": ["C", "G", "A"],
            "alt": ["G", "A", "T"],
            "OMIM": [None, "123", None],
            "consequence": ["PLS", "dELS", "PLS"],
            "label": [False, True, False],
            "tss_dist": [48, 2000, 10],
            "match_group": ["PLS_4", "dELS_1", "PLS_4"],
        }
    )


def test_module_imports_offline() -> None:
    assert traitgym.__name__ == "biodb.traitgym"
    assert "mendelian_traits_matched_9" in traitgym.SPLITS
    with pytest.raises(ValueError, match="unknown split"):
        traitgym.download_split("nope")


@responses.activate
def test_load_variants_normalises_chrom_and_caches(tmp_path: Path) -> None:
    url = traitgym._resolve_url("mendelian_traits_matched_9/test.parquet", "abc")
    responses.add(responses.GET, url, body=_parquet_bytes(_variants()), status=200)
    frame = traitgym.load_variants(
        "mendelian_traits_matched_9", revision="abc", cache_dir=tmp_path, progress=False
    )
    assert frame["chrom"].to_list() == ["chr1", "chr19", "chrX"]
    assert frame["label"].to_list() == [False, True, False]
    assert (tmp_path / "abc" / "mendelian_traits_matched_9" / "test.parquet").exists()
    again = traitgym.load_variants(
        "mendelian_traits_matched_9", revision="abc", cache_dir=tmp_path, progress=False
    )
    assert again.equals(frame) and len(responses.calls) == 1


@responses.activate
def test_load_feature_is_row_aligned(tmp_path: Path) -> None:
    url = traitgym._resolve_url("mendelian_traits_matched_9/features/GPN-MSA_absLLR.parquet", "abc")
    feature = pl.DataFrame({"score": [0.1, 2.5, 0.3]})
    responses.add(responses.GET, url, body=_parquet_bytes(feature), status=200)
    got = traitgym.load_feature(
        "mendelian_traits_matched_9",
        "GPN-MSA_absLLR",
        revision="abc",
        cache_dir=tmp_path,
        progress=False,
    )
    assert got["score"].to_list() == [0.1, 2.5, 0.3]


@pytest.mark.network
def test_live_mendelian_split_has_the_published_shape(tmp_path: Path) -> None:
    from tests.conftest import is_upstream_outage

    try:
        frame = traitgym.load_variants(
            "mendelian_traits_matched_9", cache_dir=tmp_path, progress=False
        )
        features = traitgym.list_features("mendelian_traits_matched_9")
    except Exception as exc:  # pragma: no cover
        if is_upstream_outage(exc):
            pytest.skip(f"upstream outage: {exc}")
        raise
    assert frame.height == 3380 and int(frame["label"].sum()) == 338
    assert "GPN-MSA_absLLR" in features
