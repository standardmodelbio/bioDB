"""Tests for :mod:`biodb.intervals` — pure functions, no network."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from biodb import intervals


def _left() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1", "chr2"],
            "start": [100, 400, 800, 100],
            "end": [200, 500, 900, 200],
            "name": ["a", "b", "c", "d"],
        }
    )


def _right() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1", "chr3"],
            "start": [150, 200, 450, 100],
            "end": [180, 300, 460, 200],
            "gene": ["G1", "G2", "G3", "G4"],
        }
    )


def test_read_and_write_bed_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "x.bed.gz"
    intervals.to_bed(_left(), path, extra_columns=("name",))
    back = intervals.read_bed(path, extra_columns=("name",))
    assert back.equals(_left())
    assert back.schema["start"] == pl.Int64
    (tmp_path / "bad.bed").write_text("chr1\t5\n")
    with pytest.raises(ValueError, match="three columns"):
        intervals.read_bed(tmp_path / "bad.bed")


def test_overlap_join_is_half_open_and_per_chromosome() -> None:
    joined = intervals.overlap_join(_left(), _right())
    # a [100,200) overlaps G1 [150,180) but not G2 [200,300) (abutting is not overlap);
    # b [400,500) overlaps G3; c and d overlap nothing; chr3's G4 never matches.
    assert joined.select("name", "gene_right").rows() == [("a", "G1"), ("b", "G3")]
    empty = intervals.overlap_join(_left().filter(pl.col("chrom") == "chr2"), _right())
    assert empty.height == 0 and "gene_right" in empty.columns


def test_nearest_reports_distance_and_keeps_unmatched() -> None:
    near = intervals.nearest(_left(), _right(), max_distance=150)
    by = {row["name"]: row for row in near.iter_rows(named=True)}
    assert by["a"]["gene_right"] == "G1" and by["a"]["distance"] == 0
    assert by["b"]["gene_right"] == "G3" and by["b"]["distance"] == 0
    # c [800,900): nearest is G3 ending at 460, gap 340 > 150 -> null.
    assert by["c"]["gene_right"] is None and by["c"]["distance"] is None
    assert by["d"]["gene_right"] is None  # chr2 has no right rows
    near2 = intervals.nearest(_left(), _right(), max_distance=400)
    by2 = {row["name"]: row for row in near2.iter_rows(named=True)}
    assert by2["c"]["gene_right"] == "G3" and by2["c"]["distance"] == 340


def test_merge_collapses_overlapping_and_abutting() -> None:
    frame = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1", "chr1", "chr2"],
            "start": [10, 15, 30, 40, 5],
            "end": [20, 25, 40, 50, 6],
        }
    )
    merged = intervals.merge(frame)
    assert merged.rows() == [("chr1", 10, 25), ("chr1", 30, 50), ("chr2", 5, 6)]
