"""Tests for :mod:`biodb.rfam` — downloads mocked, parsers on real Infernal output."""

from __future__ import annotations

import gzip
import io
from pathlib import Path

import polars as pl
import pytest
import responses

from biodb import rfam

FIXTURES = Path(__file__).parent / "fixtures" / "rfam"


def _gz(text: str) -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(text.encode("utf-8"))
    return buf.getvalue()


def test_module_imports_offline() -> None:
    assert rfam.__name__ == "biodb.rfam"
    assert rfam.DEFAULT_RELEASE == "CURRENT"


@responses.activate
def test_downloads_are_cached_per_release(tmp_path: Path) -> None:
    responses.add(responses.GET, rfam._url("15.1", "Rfam.clanin"), body="CL00001\tRF00005\n")
    first = rfam.download_clanin(release="15.1", cache_dir=tmp_path, progress=False)
    second = rfam.download_clanin(release="15.1", cache_dir=tmp_path, progress=False)
    assert first == second == tmp_path / "15.1" / "Rfam.clanin"
    assert len(responses.calls) == 1


def test_load_hits_parses_fmt2_tblout_with_half_open_coordinates() -> None:
    hits = rfam.load_hits(FIXTURES / "slice.tblout")
    assert hits.height == 3
    assert hits.columns[:7] == ["rfam_acc", "rfam_id", "clan", "seq_name", "start", "end", "strand"]
    u6 = hits.filter(pl.col("rfam_id") == "U6").row(0, named=True)
    # Infernal printed 53685..53791 (1-based inclusive) on +: half-open 53684..53791.
    assert (u6["start"], u6["end"], u6["strand"]) == (53684, 53791, "+")
    assert u6["clan"] == "CL00009" and u6["rfam_acc"] == "RF00026"
    assert u6["score"] == pytest.approx(107.6)
    assert not u6["truncated"]
    assert hits["overlap"].to_list() == ["*", "*", "*"]


def test_hit_structures_project_the_consensus_onto_every_hit_base() -> None:
    structures = rfam.hit_structures(FIXTURES / "slice.cmscan.txt")
    assert structures.height == 3
    span = (structures["seq_to"] - structures["seq_from"]).abs() + 1
    assert structures["structure"].str.len_chars().to_list() == span.to_list()
    mir = structures.filter(pl.col("rfam_id") == "mir-512").row(0, named=True)
    assert mir["strand"] == "+"
    assert mir["structure"].count("<") == mir["structure"].count(">") > 10  # a hairpin
    u6 = structures.filter(pl.col("rfam_id") == "U6").row(0, named=True)
    assert u6["seq_from"] == 53685 and u6["seq_to"] == 53791


def test_project_handles_deletions_insertions_and_local_ends() -> None:
    # model:  A C - G   target: A - u G   (target deletes C, inserts u)
    assert rfam._project("<<.>", "AC-G", "A-uG") == "<.>"
    # A local-end marker occupies the same columns on every line and stands
    # for N unaligned target bases (the model's own count may differ).
    assert rfam._project("<*[ 2]*>", "A*[20]*U", "A*[ 2]*U") == "<..>"


def test_seed_structure_reads_one_family_from_the_seed_file(tmp_path: Path) -> None:
    seeds = tmp_path / "CURRENT"
    seeds.mkdir()
    text = (
        "# STOCKHOLM 1.0\n#=GF AC   RF00001\nseqA ACGU\n#=GC SS_cons ((..\n//\n"
        "# STOCKHOLM 1.0\n#=GF AC   RF00002\nseqB ACG\nseqC ACG\n#=GC SS_cons (.)\n//\n"
    )
    (seeds / "Rfam.seed.gz").write_bytes(_gz(text))
    sequences, structure = rfam.seed_structure("RF00002", cache_dir=tmp_path, progress=False)
    assert sequences == {"seqB": "ACG", "seqC": "ACG"} and structure == "(.)"
    with pytest.raises(KeyError):
        rfam.seed_structure("RF99999", cache_dir=tmp_path, progress=False)


def test_scan_fasta_needs_infernal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(rfam.shutil, "which", lambda name: None)
    fasta = tmp_path / "x.fa"
    fasta.write_text(">x\nACGT\n")
    with pytest.raises(RuntimeError, match="cmscan"):
        rfam.scan_fasta(
            fasta, models=tmp_path / "Rfam.cm", clanin=tmp_path / "c", genome_size_mb=1.0
        )
