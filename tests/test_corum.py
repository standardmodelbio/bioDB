"""Tests for :mod:`biodb.corum` — network mocked with :mod:`responses`."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
import responses

from biodb import corum

HEADER = (
    "complex_id\tcomplex_name\tsynonyms\torganism\tcell_line\tpmid\tcomment_complex\t"
    "comment_members\tcomment_disease\tcomment_drug\tcomment_drug_formal\tsubunits_uniprot_id\t"
    "subunits_gene_name\tsubunits_gene_name_synonyms\tsubunits_protein_name\t"
    "subunits_proteinname_synonyms\tsubunits_organism\tsubunits_drugs\tsubunits_mims\t"
    "subunits_stoechiometrie\tpurification_methods\tfunctions_evi\tfunctions_pmid\t"
    "functions_go_id\tfunctions_go_name\tfcgs_id\tfcgs_name\tfcgs_category_name"
)


def _row(cid: int, name: str, organism: str, genes: str) -> str:
    cells = [str(cid), name, "", organism, "", "1", "", "", "", "", "", "P1;P2", genes] + [""] * 15
    return "\t".join(cells)


def _text() -> str:
    return (
        "\n".join(
            [
                HEADER,
                _row(1, "BCL6-HDAC4 complex", "Human", "BCL6;HDAC4"),
                _row(2, "Trio", "Human", "a;B;c"),
                _row(3, "Mouse thing", "Mouse", "X;Y"),
                _row(4, "Solo", "Human", "ONLY"),
                _row(5, "Repeat", "Human", "BCL6;HDAC4;HDAC4"),
            ]
        )
        + "\n"
    )


def test_module_imports_offline() -> None:
    assert corum.__name__ == "biodb.corum"
    assert corum.DEFAULT_VERSION == "current" and corum.DEFAULT_FILE == "human"


def test_urls_distinguish_current_and_archived() -> None:
    assert "download_current_file?file_id=human&file_format=txt" in corum._download_url(
        "human", "current"
    )
    assert (
        "download_archived_file?file_id=complete&file_format=txt&version=5.0"
        in corum._download_url("complete", "5.0")
    )
    with pytest.raises(ValueError, match="file_id"):
        corum.download_complexes(file_id="mouse")


@responses.activate
def test_load_and_pairs(tmp_path: Path) -> None:
    responses.add(responses.GET, corum._download_url("human", "5.3"), body=_text(), status=200)
    frame = corum.load_complexes(version="5.3", cache_dir=tmp_path, progress=False)
    assert frame.height == 5 and frame.schema["complex_id"] == pl.Int64
    assert (tmp_path / "5.3" / "corum_humanComplexes.txt").exists()

    pairs = corum.complex_pairs(version="5.3", cache_dir=tmp_path, progress=False)
    assert pairs.columns == ["gene_a", "gene_b", "complex_id", "complex_name", "subunits"]
    # Human only; the solo complex is skipped; the duplicated subunit collapses.
    assert pairs.filter(pl.col("complex_id") == 3).height == 0
    assert pairs.filter(pl.col("complex_id") == 4).height == 0
    trio = pairs.filter(pl.col("complex_id") == 2)
    assert trio.height == 3 and set(trio["gene_a"]) <= {"A", "B", "C"}
    assert bool((pairs["gene_a"] < pairs["gene_b"]).all())
    repeat = pairs.filter(pl.col("complex_id") == 5)
    assert repeat.height == 1 and repeat["subunits"].item() == 2
    assert (
        pairs.unique(["gene_a", "gene_b"]).height == 4
    )  # BCL6-HDAC4 once across complexes 1 and 5
    assert len(responses.calls) == 1


@pytest.mark.network
def test_live_release_and_human_pairs(tmp_path: Path) -> None:
    from tests.conftest import is_upstream_outage

    try:
        releases = corum.list_releases()
        pairs = corum.complex_pairs(cache_dir=tmp_path, progress=False)
    except Exception as exc:  # pragma: no cover
        if is_upstream_outage(exc):
            pytest.skip(f"upstream outage: {exc}")
        raise
    assert releases.filter(pl.col("is_current")).height == 1
    assert pairs.unique(["gene_a", "gene_b"]).height > 50_000
