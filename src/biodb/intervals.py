"""Genomic intervals — BED I/O and overlap joins shared by coordinate sources.

Several sources here are tables of genomic intervals in GRCh38 (ENCODE-rE2G
elements and TSSs, Rfam hits, ClinVar sites, GENCODE genes, TAD calls), and
joining them is the same operation every time. This module holds it once.

Conventions, stated because the off-by-one between them is the whole
difficulty: intervals are **0-based, half-open** (BED), as
``chrom``/``start``/``end`` columns of a polars DataFrame; two intervals
overlap when ``a.start < b.end and b.start < a.end``; a point at base ``p``
is the interval ``[p, p + 1)``. No liftover is provided — every source in
this package is already on GRCh38, and a build mismatch is a failure to
refuse at the source, not to paper over here.

* :func:`read_bed` / :func:`to_bed` — BED3+ files (optionally gzipped),
  extra columns by position.
* :func:`overlap_join` — every pair of overlapping rows across two tables,
  per chromosome, by binary search on a sorted right table.
* :func:`nearest` — for each left row, the nearest right row within a
  maximum distance (0 when overlapping), ties to the left-most.
* :func:`merge` — collapse overlapping (and abutting) intervals per chromosome.

Examples
--------
>>> import polars as pl
>>> from biodb.intervals import overlap_join
>>> genes = pl.DataFrame({"chrom": ["chr1"], "start": [100], "end": [500], "gene": ["A"]})
>>> sites = pl.DataFrame({"chrom": ["chr1", "chr1"], "start": [120, 900], "end": [121, 901]})
>>> overlap_join(sites, genes).select("start", "gene")
shape: (1, 2)
┌───────┬──────┐
│ start ┆ gene │
│ ---   ┆ ---  │
│ i64   ┆ str  │
╞═══════╪══════╡
│ 120   ┆ A    │
└───────┴──────┘
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import IO

import numpy as np
import polars as pl

_REQUIRED = ("chrom", "start", "end")


def _check(frame: pl.DataFrame, name: str) -> None:
    missing = [c for c in _REQUIRED if c not in frame.columns]
    if missing:
        raise ValueError(f"{name} lacks interval columns {missing}")


def _open(path: str | Path, mode: str) -> IO[str]:
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8")  # type: ignore[return-value]
    return path.open(mode, encoding="utf-8")


def read_bed(path: str | Path, *, extra_columns: tuple[str, ...] = ()) -> pl.DataFrame:
    """Read a BED3+ file (``.gz`` allowed) into ``chrom``, ``start``, ``end`` [+ extras].

    ``extra_columns`` names the columns after the third, in order; further
    columns are dropped. Comment, ``track`` and ``browser`` lines are skipped.
    """
    rows: list[list[str]] = []
    width = 3 + len(extra_columns)
    with _open(path, "r") as handle:
        for line in handle:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                raise ValueError(f"{path}: a BED row needs at least three columns: {line!r}")
            fields = (fields + [""] * width)[:width]
            rows.append(fields)
    names = list(_REQUIRED) + list(extra_columns)
    frame = pl.DataFrame(
        {name: [row[i] for row in rows] for i, name in enumerate(names)},
        schema={name: pl.Utf8 for name in names},
    )
    frame = frame.with_columns(pl.col("start").cast(pl.Int64), pl.col("end").cast(pl.Int64))
    for name in extra_columns:
        # Best-effort typing of extras: integers, then floats, else strings.
        for dtype in (pl.Int64, pl.Float64):
            try:
                frame = frame.with_columns(pl.col(name).cast(dtype, strict=True))
                break
            except pl.exceptions.InvalidOperationError:
                continue
    return frame


def to_bed(frame: pl.DataFrame, path: str | Path, *, extra_columns: tuple[str, ...] = ()) -> Path:
    """Write ``chrom``/``start``/``end`` [+ ``extra_columns``] as BED (``.gz`` allowed)."""
    _check(frame, "frame")
    path = Path(path)
    columns = list(_REQUIRED) + list(extra_columns)
    with _open(path, "w") as handle:
        for row in frame.select(columns).iter_rows():
            handle.write("\t".join("" if v is None else str(v) for v in row) + "\n")
    return path


def overlap_join(
    left: pl.DataFrame, right: pl.DataFrame, *, suffix: str = "_right"
) -> pl.DataFrame:
    """Every (left row, right row) pair whose intervals overlap on one chromosome.

    Left columns keep their names; right columns other than ``chrom`` get
    ``suffix``. Rows with no partner are absent (an inner join). Each side is
    processed per chromosome with the right side sorted by start, so the cost
    is ``O((n + m) log m)`` plus the output size.
    """
    _check(left, "left")
    _check(right, "right")
    left_index: list[int] = []
    right_index: list[int] = []
    left_pos = np.arange(left.height)
    right_pos = np.arange(right.height)
    left_chrom = left["chrom"].to_numpy()
    right_chrom = right["chrom"].to_numpy()
    l_start = left["start"].to_numpy()
    l_end = left["end"].to_numpy()
    r_start = right["start"].to_numpy()
    r_end = right["end"].to_numpy()
    for chrom in np.unique(left_chrom):
        li = left_pos[left_chrom == chrom]
        ri = right_pos[right_chrom == chrom]
        if li.size == 0 or ri.size == 0:
            continue
        order = ri[np.argsort(r_start[ri], kind="stable")]
        starts = r_start[order]
        # Running maximum of ends lets a bracket on starts bound the scan.
        ends_max = np.maximum.accumulate(r_end[order])
        for i in li.tolist():
            hi = int(np.searchsorted(starts, l_end[i], side="left"))
            lo = int(np.searchsorted(ends_max, l_start[i], side="right"))
            for j in order[lo:hi].tolist():
                if r_start[j] < l_end[i] and l_start[i] < r_end[j]:
                    left_index.append(i)
                    right_index.append(j)
    l_rows = left[left_index] if left_index else left.clear()
    r_rows = right[right_index] if right_index else right.clear()
    r_rows = r_rows.drop("chrom").rename(
        {c: f"{c}{suffix}" for c in r_rows.columns if c != "chrom"}
    )
    return pl.concat([l_rows, r_rows], how="horizontal")


def nearest(
    left: pl.DataFrame, right: pl.DataFrame, *, max_distance: int, suffix: str = "_right"
) -> pl.DataFrame:
    """For each left row, the nearest right row within ``max_distance`` bases.

    ``distance`` is 0 for overlap, else the gap between the closer edges.
    Left rows with nothing within reach are kept with nulls on the right.
    Ties go to the right row with the smaller start.
    """
    _check(left, "left")
    _check(right, "right")
    best_index = np.full(left.height, -1, dtype=np.int64)
    best_distance = np.full(left.height, -1, dtype=np.int64)
    left_chrom = left["chrom"].to_numpy()
    right_chrom = right["chrom"].to_numpy()
    l_start = left["start"].to_numpy()
    l_end = left["end"].to_numpy()
    r_start = right["start"].to_numpy()
    r_end = right["end"].to_numpy()
    right_pos = np.arange(right.height)
    for chrom in np.unique(left_chrom):
        li = np.nonzero(left_chrom == chrom)[0]
        ri = right_pos[right_chrom == chrom]
        if ri.size == 0:
            continue
        order = ri[np.argsort(r_start[ri], kind="stable")]
        starts = r_start[order]
        ends = r_end[order]
        for i in li.tolist():
            gap_after = np.maximum(starts - l_end[i], 0)
            gap_before = np.maximum(l_start[i] - ends, 0)
            distance = np.maximum(gap_after, gap_before)
            k = int(np.argmin(distance))
            if distance[k] <= max_distance:
                best_index[i] = order[k]
                best_distance[i] = distance[k]
    found = best_index >= 0
    r_rows = right[best_index[found].tolist()] if found.any() else right.clear()
    r_rows = r_rows.drop("chrom").rename(
        {c: f"{c}{suffix}" for c in r_rows.columns if c != "chrom"}
    )
    filler = pl.DataFrame(
        {c: [None] * int((~found).sum()) for c in r_rows.columns}, schema=r_rows.schema
    )
    right_part = pl.concat([r_rows, filler]) if (~found).any() else r_rows
    order = np.concatenate([np.nonzero(found)[0], np.nonzero(~found)[0]])
    out = pl.concat([left[order.tolist()], right_part], how="horizontal")
    out = out.with_columns(
        pl.Series(
            "distance", np.concatenate([best_distance[found], np.full(int((~found).sum()), -1)])
        )
    ).with_columns(
        pl.when(pl.col("distance") < 0).then(None).otherwise(pl.col("distance")).alias("distance")
    )
    return out.sort(["chrom", "start"])


def merge(frame: pl.DataFrame) -> pl.DataFrame:
    """Collapse overlapping and abutting intervals per chromosome; extras dropped."""
    _check(frame, "frame")
    out_chrom: list[str] = []
    out_start: list[int] = []
    out_end: list[int] = []
    for chrom, group in frame.sort(["chrom", "start"]).group_by("chrom", maintain_order=True):
        current_start: int | None = None
        current_end = 0
        for start, end in group.select("start", "end").iter_rows():
            if current_start is None:
                current_start, current_end = start, end
            elif start <= current_end:
                current_end = max(current_end, end)
            else:
                out_chrom.append(str(chrom[0]))
                out_start.append(current_start)
                out_end.append(current_end)
                current_start, current_end = start, end
        if current_start is not None:
            out_chrom.append(str(chrom[0]))
            out_start.append(current_start)
            out_end.append(current_end)
    return pl.DataFrame(
        {"chrom": out_chrom, "start": out_start, "end": out_end},
        schema={"chrom": pl.Utf8, "start": pl.Int64, "end": pl.Int64},
    )
