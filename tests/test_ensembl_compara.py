"""Tests for :mod:`biodb.ensembl_compara` — the dump mocked with :mod:`responses`."""

from __future__ import annotations

import gzip
from pathlib import Path

import polars as pl
import pytest
import responses

from biodb import ensembl_compara as ec

_HEADER = (
    "gene_stable_id\tprotein_stable_id\tspecies\tidentity\thomology_type\t"
    "homology_gene_stable_id\thomology_protein_stable_id\thomology_species\t"
    "homology_identity\tdn\tds\tgoc_score\twga_coverage\tis_high_confidence\thomology_id\n"
)


def _row(
    a: str, b: str, kind: str, ia: float, ib: float, hid: int, species_b: str = "homo_sapiens"
) -> str:
    hc = "NULL" if species_b == "homo_sapiens" else "1"
    return (
        f"{a}\tP{a}\thomo_sapiens\t{ia}\t{kind}\t{b}\tP{b}\t{species_b}\t{ib}\tNULL\tNULL\t"
        f"NULL\tNULL\t{hc}\t{hid}\n"
    )


def _dump() -> bytes:
    rows = [
        _row("ENSG3", "ENSG1", "within_species_paralog", 80.0, 78.0, 1),
        _row("ENSG1", "ENSG3", "within_species_paralog", 78.0, 80.0, 2),  # the mirror row
        _row("ENSG1", "ENSG2", "other_paralog", 30.0, 31.0, 3),
        _row("ENSG5", "ENSG4", "within_species_paralog", 90.0, 90.0, 4),
        _row("ENSG1", "ENSG6", "gene_split", 99.0, 99.0, 5),
        _row("ENSG1", "ENSMUSG1", "ortholog_one2one", 85.0, 85.0, 6, species_b="mus_musculus"),
    ]
    return gzip.compress((_HEADER + "".join(rows)).encode())


def test_module_imports_offline() -> None:
    assert ec.PARALOG_TYPES == ("within_species_paralog", "other_paralog")
    with pytest.raises(ValueError, match="protein"):
        ec.download_homologies(kind="dna")


@responses.activate
def test_load_homologies_reads_nulls_and_confidence(tmp_path: Path) -> None:
    responses.add(responses.GET, ec._homologies_url("116", "homo_sapiens", "protein"), body=_dump())
    frame = ec.load_homologies(cache_dir=tmp_path, progress=False)
    assert frame.height == 6
    assert frame["goc_score"].is_null().all()
    assert frame.filter(pl.col("homology_type") == "ortholog_one2one")[
        "is_high_confidence"
    ].to_list() == [True]
    assert frame.filter(pl.col("homology_type") == "gene_split")[
        "is_high_confidence"
    ].to_list() == [None]
    again = ec.load_homologies(cache_dir=tmp_path, progress=False, homology_types=("gene_split",))
    assert again.height == 1 and len(responses.calls) == 1


@responses.activate
def test_paralog_pairs_are_unordered_deduplicated_and_within_species(tmp_path: Path) -> None:
    responses.add(responses.GET, ec._homologies_url("116", "homo_sapiens", "protein"), body=_dump())
    pairs = ec.paralog_pairs(cache_dir=tmp_path, progress=False)
    assert pairs.select("gene_a", "gene_b").rows() == [
        ("ENSG1", "ENSG3"),
        ("ENSG1", "ENSG2"),
        ("ENSG4", "ENSG5"),
    ]
    row = pairs.filter(pl.col("gene_a") == "ENSG1", pl.col("gene_b") == "ENSG3").row(0, named=True)
    assert (row["identity_a"], row["identity_b"]) == (
        78.0,
        80.0,
    )  # identity seen from ENSG1, then ENSG3
    strict = ec.paralog_pairs(cache_dir=tmp_path, progress=False, min_identity=50.0)
    assert strict.select("gene_a", "gene_b").rows() == [("ENSG1", "ENSG3"), ("ENSG4", "ENSG5")]


def test_paralog_families_are_connected_components() -> None:
    pairs = pl.DataFrame(
        {
            "gene_a": ["ENSG1", "ENSG1", "ENSG4", "ENSG7"],
            "gene_b": ["ENSG3", "ENSG2", "ENSG5", "ENSG8"],
        }
    )
    families = ec.paralog_families(pairs)
    by_gene = dict(zip(families["gene_stable_id"], families["family"], strict=True))
    assert by_gene["ENSG1"] == by_gene["ENSG2"] == by_gene["ENSG3"] == 0
    assert by_gene["ENSG4"] == by_gene["ENSG5"] == 1
    assert by_gene["ENSG7"] == by_gene["ENSG8"] == 2
    sizes = dict(zip(families["gene_stable_id"], families["family_size"], strict=True))
    assert sizes["ENSG1"] == 3 and sizes["ENSG7"] == 2
