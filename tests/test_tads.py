"""Tests for :mod:`biodb.tads`."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import polars as pl
import pytest
import responses

from biodb import tads

SEARCH_URL = f"{tads.ENCODE_BASE}/search/"


def _graph(*items: dict) -> dict:
    return {"@graph": list(items), "total": len(items)}


def _file(accession: str, *biosamples: str) -> dict:
    """A search hit. The live portal gives `biosample_ontology` as a list."""

    return {
        "accession": accession,
        "biosample_ontology": [{"term_name": name} for name in biosamples],
        "file_format": "bedpe",
        "href": f"/files/{accession}/@@download/{accession}.bedpe.gz",
        "dataset": "/experiments/ENCSR000AAA/",
        "status": "released",
    }


def _bedpe(rows: list[tuple[str, int, int, str, int, int]]) -> bytes:
    header = "#chr1\tx1\tx2\tchr2\ty1\ty2\n"
    body = "".join("\t".join(str(v) for v in row) + "\n" for row in rows)
    return gzip.compress((header + body).encode("utf-8"))


def test_module_imports_offline() -> None:
    assert tads.DEFAULT_ASSEMBLY == "GRCh38"
    assert "contact domains" in tads.DOMAIN_OUTPUT_TYPES


@responses.activate
def test_list_domain_files_filters_by_biosample_and_sorts() -> None:
    responses.add(
        responses.GET,
        SEARCH_URL,
        json=_graph(_file("ENCFF222BBB", "K562"), _file("ENCFF111AAA", "K562")),
        status=200,
    )
    frame = tads.list_domain_files(biosample="K562")
    assert frame["accession"].to_list() == ["ENCFF111AAA", "ENCFF222BBB"]
    assert frame["biosamples"].to_list() == [["K562"], ["K562"]]
    assert frame["biosample_count"].to_list() == [1, 1]
    query = responses.calls[0].request.url
    assert "biosample_ontology.term_name=K562" in query
    assert "output_type=contact+domains" in query or "output_type=contact%20domains" in query


@responses.activate
def test_list_domain_files_returns_an_empty_typed_frame_when_the_portal_404s() -> None:
    """ENCODE answers a search with no hits with 404; that is empty, not an error."""

    responses.add(responses.GET, SEARCH_URL, status=404)
    frame = tads.list_domain_files(biosample="no-such-cell")
    assert frame.height == 0
    assert frame.schema["accession"] == pl.Utf8


def test_list_domain_files_refuses_an_unknown_output_type() -> None:
    with pytest.raises(ValueError, match="output_type"):
        tads.list_domain_files(output_type="loops")


@responses.activate
def test_load_domains_spans_the_arrowhead_corners_and_drops_interchromosomal(
    tmp_path: Path,
) -> None:
    """A domain runs from the first corner's start to the second corner's end."""

    accession = "ENCFF111AAA"
    responses.add(
        responses.GET,
        f"{tads.ENCODE_BASE}/files/{accession}/@@download/{accession}.bedpe.gz",
        body=_bedpe(
            [
                ("chr1", 1000, 1500, "chr1", 4000, 4500),
                ("2", 100, 200, "2", 800, 900),  # bare contig name is normalised
                ("chr3", 10, 20, "chr9", 30, 40),  # not a domain
            ]
        ),
        status=200,
    )
    frame = tads.load_domains(accession, cache_dir=tmp_path, progress=False)
    assert frame.height == 2, "the interchromosomal row is not a domain"
    first = frame.filter(pl.col("chrom") == "chr1").row(0, named=True)
    assert (first["start"], first["end"]) == (1000, 4500)
    assert (first["corner_start"], first["corner_end"]) == (1000, 4500)
    assert frame["chrom"].to_list() == ["chr1", "chr2"]
    # Cached: a second read does not fetch again.
    tads.load_domains(accession, cache_dir=tmp_path, progress=False)
    assert len(responses.calls) == 1


@responses.activate
def test_load_domains_refuses_a_file_with_no_domains(tmp_path: Path) -> None:
    accession = "ENCFF333CCC"
    responses.add(
        responses.GET,
        f"{tads.ENCODE_BASE}/files/{accession}/@@download/{accession}.bedpe.gz",
        body=_bedpe([("chr1", 10, 20, "chr7", 30, 40)]),
        status=200,
    )
    with pytest.raises(ValueError, match="no intra-chromosomal domains"):
        tads.load_domains(accession, cache_dir=tmp_path, progress=False)


def test_download_domains_refuses_a_non_accession(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ENCFF"):
        tads.download_domains("hg38.TADs.zip", cache_dir=tmp_path)


def _domains() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [100, 400, 100],
            "end": [300, 900, 900],  # the third nests the first two
            "corner_start": [100, 400, 100],
            "corner_end": [300, 900, 900],
            "accession": ["x", "x", "x"],
        }
    )


def test_boundaries_widens_each_edge_and_merges_what_overlaps() -> None:
    edges = tads.boundaries(_domains(), flank=50)
    spans = list(zip(edges["start"].to_list(), edges["end"].to_list(), strict=True))
    # Edges at 100 (twice), 300, 400, 900 (twice): 100±50 merges with itself,
    # 300±50 and 400±50 abut and merge into one wall, 900±50 merges with itself.
    assert spans == [(50, 151), (250, 451), (850, 951)]
    assert edges["chrom"].unique().to_list() == ["chr1"]


def test_boundaries_with_no_flank_are_single_bases() -> None:
    edges = tads.boundaries(_domains(), flank=0)
    assert all(end - start == 1 for start, end in zip(edges["start"], edges["end"], strict=True))


def test_boundaries_refuse_a_negative_flank() -> None:
    with pytest.raises(ValueError, match="flank"):
        tads.boundaries(_domains(), flank=-1)


def test_same_domain_counts_nesting_and_answers_false_off_the_calls() -> None:
    """A pair in no common domain is False with a count of zero, never null."""

    pairs = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1", "chr2"],
            "left": [150, 150, 500, 10],
            "right": [250, 500, 800, 20],
        }
    )
    scored = tads.same_domain(pairs, _domains())
    # 150-250 sits in the small domain and the nesting one; 150-500 crosses the
    # 300/400 wall but both ends are inside the nesting domain; 500-800 is in
    # two; chr2 has no calls at all.
    assert scored["shared_domains"].to_list() == [2, 1, 2, 0]
    assert scored["same_domain"].to_list() == [True, True, True, False]
    assert scored["same_domain"].null_count() == 0


def test_same_domain_is_orientation_independent() -> None:
    forward = pl.DataFrame({"chrom": ["chr1"], "left": [150], "right": [250]})
    reverse = pl.DataFrame({"chrom": ["chr1"], "left": [250], "right": [150]})
    assert (
        tads.same_domain(forward, _domains())["shared_domains"].to_list()
        == tads.same_domain(reverse, _domains())["shared_domains"].to_list()
    )


def test_same_domain_refuses_missing_columns() -> None:
    with pytest.raises(ValueError, match="left"):
        tads.same_domain(pl.DataFrame({"chrom": ["chr1"], "right": [1]}), _domains())


@responses.activate
def test_domain_metadata_reads_the_portal_record() -> None:
    accession = "ENCFF111AAA"
    responses.add(
        responses.GET,
        f"{tads.ENCODE_BASE}/files/{accession}/",
        json={
            "accession": accession,
            "assembly": "GRCh38",
            "output_type": "contact domains",
            "biosample_ontology": [{"term_name": "K562"}],
            "dataset": "/experiments/ENCSR000AAA/",
            "md5sum": "0" * 32,
            "date_created": "2023-01-01T00:00:00.000000+00:00",
        },
        status=200,
    )
    record = tads.domain_metadata(accession)
    assert record["biosamples"] == ["K562"] and record["assembly"] == "GRCh38"
    assert record["md5sum"] == "0" * 32


@responses.activate
def test_a_pooled_call_set_is_reported_with_every_biosample_it_covers() -> None:
    """The portal's filter matches the dataset, so one cell type can return a pool.

    ENCFF549OBE really is returned by a K562 query while covering HepG2, PC-3
    and a dozen others; a caller who took the first row as "K562 domains"
    would be using the wrong cell type. The list makes that visible, and
    `biosample_count` is what to filter on.
    """

    responses.add(
        responses.GET,
        SEARCH_URL,
        json=_graph(
            _file("ENCFF111AAA", "K562"),
            _file("ENCFF549OBE", "HepG2", "PC-3", "K562"),
        ),
        status=200,
    )
    frame = tads.list_domain_files(biosample="K562")
    assert frame["biosample_count"].to_list() == [1, 3]
    specific = frame.filter(pl.col("biosample_count") == 1)
    assert specific["accession"].to_list() == ["ENCFF111AAA"]
    assert frame.filter(pl.col("accession") == "ENCFF549OBE")["biosamples"][0].to_list() == [
        "HepG2",
        "K562",
        "PC-3",
    ]


def test_biosamples_accepts_the_singular_dict_shape_too() -> None:
    assert tads._biosamples({"biosample_ontology": {"term_name": "K562"}}) == ["K562"]
    assert tads._biosamples({"biosample_ontology": None}) == []
    assert tads._biosamples({}) == []


@pytest.mark.network
def test_live_encode_publishes_k562_domain_calls() -> None:
    """The portal still serves GRCh38 contact domains for the K562 benchmark cell."""

    frame = tads.list_domain_files(biosample="K562", limit=25)
    assert frame.height > 0
    assert set(frame["file_format"].unique().to_list()) <= {"bedpe"}
    assert all(a.startswith("ENCFF") for a in frame["accession"].to_list())
    # Every hit must name the biosample asked for somewhere, pooled or not.
    assert all("K562" in names for names in frame["biosamples"].to_list())
    record = tads.domain_metadata(frame["accession"][0])
    assert record["assembly"] == "GRCh38"
    assert json.dumps(record)  # the record is JSON-serialisable for provenance
