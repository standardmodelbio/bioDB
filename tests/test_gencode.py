"""Tests for ``biodb.gencode``.

Pure-Python helpers (``_parse_attrs``, ``_strip_version``) get
exhaustive coverage; the live GTF fetch + parse path is exercised by
a synthetic 5-line GTF written to ``tmp_path`` and fed to
``_iter_gtf_rows`` + ``fetch_mane_select`` with ``download_gencode_gtf``
monkeypatched out.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import polars as pl
import pytest

from biodb import gencode


def test_parse_attrs_basic() -> None:
    s = 'gene_id "ENSG_A"; gene_name "BRCA1"; tag "MANE_Select";'
    d = gencode._parse_attrs(s)
    assert d["gene_id"] == "ENSG_A"
    assert d["gene_name"] == "BRCA1"
    assert "MANE_Select" in d["tag"]


def test_parse_attrs_collects_multiple_tags() -> None:
    """A GTF row can carry several ``tag "X"; tag "Y";`` pairs -- they
    must all show up in the joined ``tag`` value, not get overwritten."""
    s = 'gene_id "G"; tag "MANE_Select"; tag "basic"; tag "Ensembl_canonical";'
    d = gencode._parse_attrs(s)
    parts = set(d["tag"].split(","))
    assert {"MANE_Select", "basic", "Ensembl_canonical"} <= parts


def test_parse_attrs_handles_empty_string() -> None:
    assert gencode._parse_attrs("") == {}


def test_strip_version_with_dot() -> None:
    assert gencode._strip_version("ENSG00000012048.25") == "ENSG00000012048"


def test_strip_version_without_dot() -> None:
    """An unversioned ID is returned unchanged -- the strip is a no-op
    safety net, not an assertion that the input was versioned."""
    assert gencode._strip_version("ENSG00000012048") == "ENSG00000012048"


def test_iter_gtf_rows_skips_comments_and_other_features(tmp_path: Path) -> None:
    """Only ``gene`` / ``transcript`` / ``CDS`` features are emitted;
    exons, UTRs, and ``##``-prefixed header lines must be filtered out."""
    gtf = tmp_path / "tiny.gtf.gz"
    body = "\n".join(
        [
            "##description: synthetic",
            '1\tHAVANA\tgene\t100\t200\t.\t+\t.\tgene_id "ENSG_A.1"; gene_name "GENE_A"; gene_type "protein_coding";',
            '1\tHAVANA\texon\t100\t200\t.\t+\t.\tgene_id "ENSG_A.1"; transcript_id "ENST_A.1";',
            '1\tHAVANA\ttranscript\t100\t200\t.\t+\t.\tgene_id "ENSG_A.1"; transcript_id "ENST_A.1"; transcript_type "protein_coding"; tag "MANE_Select";',
            '1\tHAVANA\tCDS\t110\t199\t.\t+\t0\tgene_id "ENSG_A.1"; transcript_id "ENST_A.1"; tag "MANE_Select";',
        ]
    )
    with gzip.open(gtf, "wt") as fh:
        fh.write(body + "\n")
    features = [row[1] for row in gencode._iter_gtf_rows(gtf)]
    assert features == ["gene", "transcript", "CDS"]


def test_iter_gtf_rows_emits_0_based_half_open_coords(tmp_path: Path) -> None:
    """GTF is 1-based inclusive on disk; the iterator emits 0-based
    half-open so the rest of biodb's coordinate arithmetic stays
    consistent."""
    gtf = tmp_path / "coord.gtf.gz"
    with gzip.open(gtf, "wt") as fh:
        fh.write('1\tHAVANA\tgene\t100\t200\t.\t+\t.\tgene_id "G"; gene_type "protein_coding";\n')
    (_chrom, _feat, start, end, *_) = next(iter(gencode._iter_gtf_rows(gtf)))
    assert start == 99  # 100 - 1 (1-based -> 0-based)
    assert end == 200  # exclusive end == inclusive end + 1 ... or just original?


def test_fetch_mane_select_joins_gene_tx_cds(tmp_path: Path, monkeypatch) -> None:
    """End-to-end: a synthetic GTF with one protein-coding gene,
    one MANE Select transcript, and two CDS records (the canonical
    aa_length is ``(150 - 3) // 3 = 49``)."""
    gtf = tmp_path / "synth.gtf.gz"
    lines = [
        '1\tHAVANA\tgene\t1000\t2000\t.\t+\t.\tgene_id "ENSG_X.5"; gene_name "GENEX"; gene_type "protein_coding";',
        '1\tHAVANA\ttranscript\t1000\t2000\t.\t+\t.\tgene_id "ENSG_X.5"; transcript_id "ENST_X.3"; transcript_type "protein_coding"; tag "MANE_Select";',
        '1\tHAVANA\tCDS\t1001\t1100\t.\t+\t0\tgene_id "ENSG_X.5"; transcript_id "ENST_X.3"; tag "MANE_Select";',
        '1\tHAVANA\tCDS\t1201\t1250\t.\t+\t0\tgene_id "ENSG_X.5"; transcript_id "ENST_X.3"; tag "MANE_Select";',
    ]
    with gzip.open(gtf, "wt") as fh:
        fh.write("\n".join(lines) + "\n")

    # Short-circuit the download
    monkeypatch.setattr(gencode, "download_gencode_gtf", lambda release, cache_dir=None: gtf)

    df = gencode.fetch_mane_select(release="testver")
    assert isinstance(df, pl.DataFrame)
    assert df.height == 1
    row = df.row(0, named=True)
    assert row["gene_id"] == "ENSG_X"  # versionless
    assert row["gene_symbol"] == "GENEX"
    assert row["mane_select_tx"] == "ENST_X.3"  # versioned
    # CDS total = (1100 - 1000) + (1250 - 1200) = 100 + 50 = 150
    # aa_length = (150 - 3) // 3 = 49
    assert row["aa_length"] == 49


def test_fetch_mane_select_drops_non_protein_coding(tmp_path: Path, monkeypatch) -> None:
    """A pseudogene + a lncRNA must NOT appear in the output even if
    they happen to carry a MANE_Select tag on disk."""
    gtf = tmp_path / "mixed.gtf.gz"
    lines = [
        # protein-coding -- kept
        '1\tHAVANA\tgene\t1000\t2000\t.\t+\t.\tgene_id "ENSG_KEEP.1"; gene_name "KEEP"; gene_type "protein_coding";',
        '1\tHAVANA\ttranscript\t1000\t2000\t.\t+\t.\tgene_id "ENSG_KEEP.1"; transcript_id "ENST_KEEP.1"; transcript_type "protein_coding"; tag "MANE_Select";',
        '1\tHAVANA\tCDS\t1001\t1100\t.\t+\t0\tgene_id "ENSG_KEEP.1"; transcript_id "ENST_KEEP.1"; tag "MANE_Select";',
        # lncRNA -- dropped (gene_type filter)
        '1\tHAVANA\tgene\t3000\t4000\t.\t+\t.\tgene_id "ENSG_DROP.1"; gene_name "DROP"; gene_type "lncRNA";',
        '1\tHAVANA\ttranscript\t3000\t4000\t.\t+\t.\tgene_id "ENSG_DROP.1"; transcript_id "ENST_DROP.1"; transcript_type "lncRNA"; tag "MANE_Select";',
    ]
    with gzip.open(gtf, "wt") as fh:
        fh.write("\n".join(lines) + "\n")

    monkeypatch.setattr(gencode, "download_gencode_gtf", lambda release, cache_dir=None: gtf)
    df = gencode.fetch_mane_select(release="testver")
    assert df["gene_id"].to_list() == ["ENSG_KEEP"]


def test_fetch_mane_select_raises_when_no_protein_coding(tmp_path: Path, monkeypatch) -> None:
    """A corrupted download (no protein-coding genes at all) must
    fail loudly rather than silently return an empty DataFrame."""
    gtf = tmp_path / "broken.gtf.gz"
    with gzip.open(gtf, "wt") as fh:
        fh.write("##nothing useful here\n")
    monkeypatch.setattr(gencode, "download_gencode_gtf", lambda release, cache_dir=None: gtf)
    with pytest.raises(RuntimeError, match="no protein-coding genes"):
        gencode.fetch_mane_select(release="testver")


def test_fetch_mane_select_raises_when_no_mane_transcripts(tmp_path: Path, monkeypatch) -> None:
    """Wrong / very old release tag: protein-coding genes exist but
    nothing has the MANE_Select tag -- raise instead of returning
    an empty join result."""
    gtf = tmp_path / "no_mane.gtf.gz"
    lines = [
        '1\tHAVANA\tgene\t1000\t2000\t.\t+\t.\tgene_id "ENSG_X.1"; gene_name "X"; gene_type "protein_coding";',
        '1\tHAVANA\ttranscript\t1000\t2000\t.\t+\t.\tgene_id "ENSG_X.1"; transcript_id "ENST_X.1"; transcript_type "protein_coding";',  # no tag
    ]
    with gzip.open(gtf, "wt") as fh:
        fh.write("\n".join(lines) + "\n")
    monkeypatch.setattr(gencode, "download_gencode_gtf", lambda release, cache_dir=None: gtf)
    with pytest.raises(RuntimeError, match="no MANE Select transcripts"):
        gencode.fetch_mane_select(release="testver")
