"""Tests for :mod:`biodb.encode_re2g` — network mocked with :mod:`responses`."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
import responses

from biodb import encode_re2g

HEADER = (
    "dataset\tchrom\tchromStart\tchromEnd\tname\tEffectSize\tchrTSS\tstartTSS\tendTSS\t"
    "measuredGeneSymbol\tSignificant\tpValueAdjusted\tPowerAtEffectSize25\tValidConnection\t"
    "CellType\tReference\tRegulated\tPowerAtEffectSize10\tPowerAtEffectSize15\t"
    "PowerAtEffectSize20\tPowerAtEffectSize50\tpair_uid\tmerged_uid\tmerged_start\tmerged_end"
)


def _row(
    chrom: str,
    start: int,
    end: int,
    gene: str,
    tss: int | str,
    regulated: str,
    cell: str = "K562",
    valid: str = "TRUE",
) -> str:
    return (
        f"FlowFISH_K562\t{chrom}\t{start}\t{end}\t{gene}|{chrom}:{start}-{end}:*\t-0.3\t{chrom}\t{tss}\t"
        f"{tss if tss == 'NA' else int(tss) + 1}\t{gene}\tTRUE\t0.01\t0.8\t{valid}\t{cell}\tUlirsch2016\t"
        f"{regulated}\tNA\tNA\tNA\tNA\tFlowFISH_K562|{gene}|{chrom}:{start}-{end}:*\t1\t{start}\t{end}"
    )


def _text() -> str:
    rows = [
        HEADER,
        _row("chr19", 1000, 1500, "GENEA", 5000, "TRUE"),
        _row("chr19", 2000, 2500, "GENEB", 9000, "FALSE"),
        _row("chr1", 100, 600, "GENEC", "NA", "FALSE"),
        _row("chr19", 3000, 3500, "GENED", 4000, "TRUE", cell="HCT116"),
        _row("chr19", 4000, 4500, "GENEE", 8000, "FALSE", valid="FALSE"),
    ]
    return "\n".join(rows) + "\n"


def test_module_imports_offline() -> None:
    assert encode_re2g.__name__ == "biodb.encode_re2g"
    assert encode_re2g.DEFAULT_REVISION


@responses.activate
def test_download_and_load_are_cached_and_typed(tmp_path: Path) -> None:
    url = encode_re2g._benchmark_url("abc123")
    responses.add(responses.GET, url, body=_text(), status=200)
    first = encode_re2g.download_crispri_benchmark(
        revision="abc123", cache_dir=tmp_path, progress=False
    )
    assert first.exists() and first.parent == tmp_path
    second = encode_re2g.download_crispri_benchmark(
        revision="abc123", cache_dir=tmp_path, progress=False
    )
    assert second == first and len(responses.calls) == 1  # cached

    frame = encode_re2g.load_crispri_benchmark(
        revision="abc123", cache_dir=tmp_path, progress=False
    )
    assert frame.height == 5
    assert frame.schema["chromStart"] == pl.Int64 and frame.schema["startTSS"] == pl.Int64
    assert frame.schema["Regulated"] == pl.Boolean and frame.schema["ValidConnection"] == pl.Boolean
    assert frame["startTSS"].null_count() == 1  # the NA row


@responses.activate
def test_crispri_pairs_labels_filters_and_distances(tmp_path: Path) -> None:
    responses.add(responses.GET, encode_re2g._benchmark_url("abc123"), body=_text(), status=200)
    pairs = encode_re2g.crispri_pairs(revision="abc123", cache_dir=tmp_path, progress=False)
    # K562 and valid only: drops the HCT116 row and the ValidConnection=FALSE row.
    assert pairs.height == 3
    assert pairs.columns[:7] == ["chrom", "start", "end", "gene", "tss_chrom", "tss", "label"]
    assert pairs["label"].to_list() == [
        0,
        1,
        0,
    ]  # sorted by chrom, start: chr1, chr19 1000, chr19 2000
    assert pairs.filter(pl.col("gene") == "GENEA")["distance"].item() == 5000 - 1250
    assert pairs.filter(pl.col("gene") == "GENEC")["distance"].null_count() == 1
    everything = encode_re2g.crispri_pairs(
        cell_type=None, valid_only=False, revision="abc123", cache_dir=tmp_path, progress=False
    )
    assert everything.height == 5


@pytest.mark.network
def test_live_benchmark_has_the_published_shape(tmp_path: Path) -> None:
    from tests.conftest import is_upstream_outage

    try:
        pairs = encode_re2g.crispri_pairs(cache_dir=tmp_path, progress=False)
    except Exception as exc:  # pragma: no cover - outage guard
        if is_upstream_outage(exc):
            pytest.skip(f"upstream outage: {exc}")
        raise
    assert pairs.height > 9000
    assert pairs["label"].sum() > 400
    assert pairs.filter(pl.col("chrom") == "chr19").height > 2000
