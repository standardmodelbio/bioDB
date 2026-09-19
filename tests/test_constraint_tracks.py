"""Tests for :mod:`biodb.constraint_tracks` — a tiny bigWig written on the
fly for the track readers, :mod:`responses` for the GPN-Star shard."""

from __future__ import annotations

import io
from pathlib import Path

import polars as pl
import pytest
import responses

from biodb import constraint_tracks as ct

pyBigWig = pytest.importorskip("pyBigWig")


@pytest.fixture
def small_bigwig(tmp_path: Path) -> Path:
    """chrT is 100 bases; bases 10..19 score 1.0, 20..29 score 3.0, the rest unscored."""
    path = tmp_path / "small.bw"
    bw = pyBigWig.open(str(path), "w")
    bw.addHeader([("chrT", 100)])
    bw.addEntries(["chrT", "chrT"], [10, 20], ends=[20, 30], values=[1.0, 3.0])
    bw.close()
    return path


def test_registry_names_every_track_with_a_url_and_terms() -> None:
    for name, item in ct.TRACKS.items():
        assert item.name == name
        assert item.url.startswith("https://")
        assert item.assembly == "hg38"
        assert item.license
    assert ct.track("phylop241").url.endswith("cactus241way.phyloP.bw")
    assert ct.track("gpn_star_m447_entropy").url.startswith(
        "https://huggingface.co/datasets/songlab/gpn-star-scores/resolve/"
    )
    with pytest.raises(ValueError, match="unknown track"):
        ct.track("nope")


def test_score_intervals_summarises_half_open_intervals(small_bigwig: Path) -> None:
    bw = ct.open_track(small_bigwig)
    frame = pl.DataFrame(
        {
            "chrom": ["chrT", "chrT", "chrT", "chrX"],
            "start": [10, 15, 40, 0],
            "end": [20, 25, 50, 10],
        }
    )
    out = ct.score_intervals(bw, frame, stat="mean")
    assert out["mean"].to_list()[:2] == [1.0, 2.0]
    assert out["mean"][2] is None  # unscored bases
    assert out["mean"][3] is None  # contig the track does not carry
    peak = ct.score_intervals(bw, frame, stat="max", column="peak")
    assert peak["peak"].to_list()[:2] == [1.0, 3.0]
    with pytest.raises(ValueError, match="stat must be"):
        ct.score_intervals(bw, frame, stat="median")


def test_score_positions_reads_runs_and_leaves_gaps_null(small_bigwig: Path) -> None:
    bw = ct.open_track(small_bigwig)
    frame = pl.DataFrame(
        {
            "chrom": ["chrT", "chrT", "chrT", "chrT", "chrX"],
            "start": [25, 12, 50, 19, 3],
        }
    )
    out = ct.score_positions(bw, frame)
    assert out["score"].to_list() == [3.0, 1.0, None, 1.0, None]


def _parquet_bytes(frame: pl.DataFrame) -> bytes:
    buf = io.BytesIO()
    frame.write_parquet(buf)
    return buf.getvalue()


@responses.activate
def test_load_gpn_star_filters_positions_and_derives_zero_based_start(tmp_path: Path) -> None:
    shard = pl.DataFrame(
        {
            "chrom": ["19", "19", "19"],
            "pos": [100, 101, 102],
            "ref": ["A", "C", "G"],
            "entropy_calibrated": pl.Series([0.5, 1.0, 1.2], dtype=pl.Float32),
        }
    )
    url = ct._gpn_star_url("data/gpn-star-hg38-m447-200m/entropy/entropy_chr19.parquet", "abc")
    responses.add(responses.GET, url, body=_parquet_bytes(shard), status=200)
    frame = ct.load_gpn_star(
        "19", positions=[100, 102], revision="abc", cache_dir=tmp_path, progress=False
    )
    assert frame["chrom"].to_list() == ["chr19", "chr19"]
    assert frame["pos"].to_list() == [100, 102]
    assert frame["start"].to_list() == [99, 101]
    assert frame["entropy_calibrated"].to_list() == pytest.approx([0.5, 1.2])
    again = ct.load_gpn_star("chr19", revision="abc", cache_dir=tmp_path, progress=False)
    assert again.height == 3 and len(responses.calls) == 1
    with pytest.raises(ValueError, match="unknown model"):
        ct.download_gpn_star("chr19", model="x")
    with pytest.raises(ValueError, match="kind must be"):
        ct.download_gpn_star("chr19", kind="x")


@pytest.mark.network
def test_remote_phylop_reads_a_chr19_window() -> None:
    from tests.conftest import is_upstream_outage

    try:
        bw = ct.open_track("phylop241")
        frame = pl.DataFrame({"chrom": ["chr19"], "start": [1_000_000], "end": [1_000_100]})
        out = ct.score_intervals(bw, frame)
    except Exception as exc:  # noqa: BLE001
        if is_upstream_outage(exc):
            pytest.skip(f"upstream outage: {exc}")
        raise
    assert out["mean"][0] is not None
