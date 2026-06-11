"""Tests for :mod:`biodb.gtr`."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd
import pytest
import responses

from biodb import gtr

FIXTURES = Path(__file__).parent / "fixtures" / "gtr"


# ── module surface ──────────────────────────────────────────────────────────


def test_module_imports_offline() -> None:
    assert gtr.__name__ == "biodb.gtr"


def test_constants_present() -> None:
    assert gtr.NCBI_EUTILS_BASE_URL.endswith("/eutils")
    assert gtr.GTR_FTP_BASE_URL.endswith("/GTR/data")
    assert gtr.TEST_CONDITION_GENE_FILE == "test_condition_gene.txt"
    assert gtr.FULL_XML_FILE == "gtr_ftp.xml.gz"
    assert gtr.CACHE_DIR.exists()


def test_accession_from_uid() -> None:
    assert gtr.accession_from_uid("509983") == "GTR000509983"
    assert gtr.accession_from_uid(509983) == "GTR000509983"


# ── esummary normalization ───────────────────────────────────────────────────


def _esummary_record() -> dict:
    payload = json.loads((FIXTURES / "esummary.json").read_text())
    return payload["result"]["509983"]


def test_test_from_esummary_normalizes_core_fields() -> None:
    rec = gtr._test_from_esummary(_esummary_record())
    assert isinstance(rec, gtr.GTRTest)
    assert rec.accession == "GTR000509983"
    assert rec.uid == "509983"
    assert rec.name == "BRCA1 gene sequencing"
    assert rec.test_type == "Clinical"
    assert rec.lab == "Example Genetics Lab"
    assert rec.test_url == "https://example.org/tests/brca1"


def test_test_from_esummary_extracts_genes() -> None:
    rec = gtr._test_from_esummary(_esummary_record())
    assert rec.genes == [{"symbol": "BRCA1", "entrez": "672", "location": "17q21.31"}]


def test_test_from_esummary_extracts_conditions_and_methods() -> None:
    rec = gtr._test_from_esummary(_esummary_record())
    assert {"name": "Breast-ovarian cancer, familial 1", "cui": "C0677776"} in [
        {"name": c["name"], "cui": c["cui"]} for c in rec.conditions
    ]
    assert "Next-Generation (NGS)/Massively parallel sequencing (MPS)" in rec.methods
    assert rec.clinical_validity.startswith("Pathogenic BRCA1")
    assert rec.pmids == ["20301425"]
    assert rec.test_purpose == ["Diagnosis", "Risk Assessment"]


# ── search_tests ─────────────────────────────────────────────────────────────


def test_search_tests_returns_accessions() -> None:
    url = f"{gtr.NCBI_EUTILS_BASE_URL}/esearch.fcgi"
    body = {
        "esearchresult": {
            "count": "2",
            "idlist": ["509983", "4006"],
            "querytranslation": "BRCA1[SYMB]",
        }
    }
    with responses.RequestsMock() as rsps:
        rsps.add(responses.GET, url, json=body, status=200)
        accs = gtr.search_tests("BRCA1", field="SYMB", retmax=5)
    assert accs == ["GTR000509983", "GTR000004006"]


def test_search_tests_uids_mode() -> None:
    url = f"{gtr.NCBI_EUTILS_BASE_URL}/esearch.fcgi"
    body = {"esearchresult": {"count": "1", "idlist": ["509983"], "querytranslation": ""}}
    with responses.RequestsMock() as rsps:
        rsps.add(responses.GET, url, json=body, status=200)
        out = gtr.search_tests("BRCA1", as_accession=False)
    assert out == ["509983"]


# ── query_test ───────────────────────────────────────────────────────────────


def test_query_test_by_accession() -> None:
    payload = json.loads((FIXTURES / "esummary.json").read_text())
    url = f"{gtr.NCBI_EUTILS_BASE_URL}/esummary.fcgi"
    with responses.RequestsMock() as rsps:
        rsps.add(responses.GET, url, json=payload, status=200)
        rec = gtr.query_test("GTR000509983")
        assert "id=509983" in rsps.calls[0].request.url
    assert isinstance(rec, gtr.GTRTest)
    assert rec.accession == "GTR000509983"


def test_query_test_accepts_bare_uid() -> None:
    payload = json.loads((FIXTURES / "esummary.json").read_text())
    url = f"{gtr.NCBI_EUTILS_BASE_URL}/esummary.fcgi"
    with responses.RequestsMock() as rsps:
        rsps.add(responses.GET, url, json=payload, status=200)
        rec = gtr.query_test(509983)
    assert rec.name == "BRCA1 gene sequencing"


def test_query_test_missing_raises() -> None:
    url = f"{gtr.NCBI_EUTILS_BASE_URL}/esummary.fcgi"
    with responses.RequestsMock() as rsps:
        rsps.add(responses.GET, url, json={"result": {"uids": []}}, status=200)
        with pytest.raises(KeyError):
            gtr.query_test("GTR000000001")


# ── query_gene / query_condition ─────────────────────────────────────────────


def _mock_search_then_summary(rsps: responses.RequestsMock) -> None:
    payload = json.loads((FIXTURES / "esummary.json").read_text())
    rsps.add(
        responses.GET,
        f"{gtr.NCBI_EUTILS_BASE_URL}/esearch.fcgi",
        json={"esearchresult": {"count": "1", "idlist": ["509983"]}},
        status=200,
    )
    rsps.add(
        responses.GET,
        f"{gtr.NCBI_EUTILS_BASE_URL}/esummary.fcgi",
        json=payload,
        status=200,
    )


def test_query_gene_searches_by_symbol_then_summarizes() -> None:
    with responses.RequestsMock() as rsps:
        _mock_search_then_summary(rsps)
        recs = gtr.query_gene("BRCA1", retmax=5)
        assert "SYMB" in rsps.calls[0].request.url
    assert len(recs) == 1 and recs[0].name == "BRCA1 gene sequencing"


def test_query_gene_numeric_uses_geneid_field() -> None:
    with responses.RequestsMock() as rsps:
        _mock_search_then_summary(rsps)
        gtr.query_gene(672)
        assert "GENEID" in rsps.calls[0].request.url


def test_query_condition_uses_cui_field_for_cui() -> None:
    with responses.RequestsMock() as rsps:
        _mock_search_then_summary(rsps)
        gtr.query_condition("C0677776")
        assert "DCUI" in rsps.calls[0].request.url


def test_query_condition_uses_disname_for_text() -> None:
    with responses.RequestsMock() as rsps:
        _mock_search_then_summary(rsps)
        gtr.query_condition("breast cancer")
        assert "DISNAME" in rsps.calls[0].request.url


# ── download ─────────────────────────────────────────────────────────────────


def test_download_fetches_tsv_only_by_default(tmp_path) -> None:
    tsv_url = f"{gtr.GTR_FTP_BASE_URL}/{gtr.TEST_CONDITION_GENE_FILE}"
    with responses.RequestsMock() as rsps:
        rsps.add(responses.GET, tsv_url, body=b"#header\n", status=200)
        paths = gtr.download(cache_dir=tmp_path)
    assert paths["tsv"].exists()
    assert paths["xml"] is None


def test_download_full_xml_fetches_both(tmp_path) -> None:
    tsv_url = f"{gtr.GTR_FTP_BASE_URL}/{gtr.TEST_CONDITION_GENE_FILE}"
    xml_url = f"{gtr.GTR_FTP_BASE_URL}/{gtr.FULL_XML_FILE}"
    with responses.RequestsMock() as rsps:
        rsps.add(responses.GET, tsv_url, body=b"#header\n", status=200)
        rsps.add(responses.GET, xml_url, body=b"\x1f\x8bfake-gzip", status=200)
        paths = gtr.download(cache_dir=tmp_path, full_xml=True)
    assert paths["tsv"].exists() and paths["xml"].exists()


def test_download_uses_cache(tmp_path) -> None:
    (tmp_path / gtr.TEST_CONDITION_GENE_FILE).write_bytes(b"#cached\n")
    with responses.RequestsMock() as rsps:
        paths = gtr.download(cache_dir=tmp_path)
    assert len(rsps.calls) == 0
    assert paths["tsv"].read_bytes() == b"#cached\n"


# ── load_test_condition_gene ─────────────────────────────────────────────────


def test_load_test_condition_gene_splits_rows(tmp_path) -> None:
    src = (FIXTURES / "test_condition_gene.txt").read_bytes()
    (tmp_path / gtr.TEST_CONDITION_GENE_FILE).write_bytes(src)
    df = gtr.load_test_condition_gene(cache_dir=tmp_path)
    assert "accession_version" in df.columns
    assert set(df["object"].unique()) == {"condition", "gene"}
    gene_rows = df[df["object"] == "gene"]
    assert "672" in set(gene_rows["gene_or_SNOMED_CT_ID"])
    assert "BRCA1" in set(gene_rows["gene_symbol"])


# ── iter_full_records ────────────────────────────────────────────────────────


def test_iter_full_records_parses_plain_xml() -> None:
    recs = list(gtr.iter_full_records(FIXTURES / "gtr_sample.xml"))
    assert [r.accession for r in recs] == ["GTR000509983", "GTR000004006"]
    brca = recs[0]
    assert brca.name == "BRCA1 gene sequencing"
    assert brca.test_type == "Clinical"
    assert {"symbol": "BRCA1", "entrez": "672"} in [
        {"symbol": g["symbol"], "entrez": g["entrez"]} for g in brca.genes
    ]
    assert "C0677776" in [c["cui"] for c in brca.conditions]
    assert "Next-Generation (NGS)/Massively parallel sequencing (MPS)" in brca.methods
    assert brca.analytical_validity.startswith("Analytical sensitivity")
    assert brca.clinical_validity.startswith("Pathogenic BRCA1")
    assert "20301425" in brca.pmids


def test_iter_full_records_reads_gzip(tmp_path) -> None:
    raw = (FIXTURES / "gtr_sample.xml").read_bytes()
    gz = tmp_path / "gtr_ftp.xml.gz"
    with gzip.open(gz, "wb") as f:
        f.write(raw)
    recs = list(gtr.iter_full_records(gz))
    assert len(recs) == 2


# ── panel_text ───────────────────────────────────────────────────────────────


def test_panel_text_assembles_fields() -> None:
    rec = gtr._test_from_esummary(_esummary_record())
    text = gtr.panel_text(rec)
    assert "BRCA1 gene sequencing" in text
    assert "Breast-ovarian cancer" in text
    assert "Pathogenic BRCA1" in text
    assert "Next-Generation" in text
    assert "Example Genetics Lab" not in text


def test_panel_text_respects_include() -> None:
    rec = gtr._test_from_esummary(_esummary_record())
    text = gtr.panel_text(rec, include=("name",))
    assert text.strip() == "BRCA1 gene sequencing"


def test_panel_text_accepts_dict() -> None:
    rec = gtr._test_from_esummary(_esummary_record())
    text = gtr.panel_text(rec.__dict__, include=("name",))
    assert "BRCA1 gene sequencing" in text


# ── gene_sets ────────────────────────────────────────────────────────────────


def test_gene_sets_builds_long_frame(tmp_path) -> None:
    (tmp_path / gtr.TEST_CONDITION_GENE_FILE).write_bytes(
        (FIXTURES / "test_condition_gene.txt").read_bytes()
    )
    df = gtr.gene_sets(cache_dir=tmp_path)
    assert set(df.columns) == {
        "panel_id",
        "panel_name",
        "condition_cui",
        "gene_symbol",
        "gene_entrez",
    }
    fmr1 = df[df["gene_symbol"] == "FMR1"]
    assert "C0016667" in set(fmr1["condition_cui"])
    assert "2332" in set(fmr1["gene_entrez"])
    assert set(df["panel_id"]) == {"GTR000004006.1", "GTR000509983.1"}


# ── aggregate_gene_sets ──────────────────────────────────────────────────────


def _multi_panel_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "panel_id": "GTR1",
                "panel_name": "Fragile X",
                "condition_cui": "C0016667",
                "gene_symbol": "FMR1",
                "gene_entrez": "2332",
            },
            {
                "panel_id": "GTR2",
                "panel_name": "Fragile X",
                "condition_cui": "C0016667",
                "gene_symbol": "FMR1",
                "gene_entrez": "2332",
            },
            {
                "panel_id": "GTR2",
                "panel_name": "Fragile X",
                "condition_cui": "C0016667",
                "gene_symbol": "AFF2",
                "gene_entrez": "2334",
            },
        ]
    )


def test_aggregate_gene_sets_by_condition_counts_support(monkeypatch) -> None:
    monkeypatch.setattr(gtr, "gene_sets", lambda **_: _multi_panel_df())
    agg = gtr.aggregate_gene_sets(by="condition")
    assert set(agg.columns) == {
        "set_id",
        "set_name",
        "gene_symbol",
        "gene_entrez",
        "support_count",
    }
    fmr1 = agg[(agg["set_id"] == "C0016667") & (agg["gene_symbol"] == "FMR1")]
    assert int(fmr1["support_count"].iloc[0]) == 2
    aff2 = agg[(agg["set_id"] == "C0016667") & (agg["gene_symbol"] == "AFF2")]
    assert int(aff2["support_count"].iloc[0]) == 1


def test_aggregate_gene_sets_by_test_name(monkeypatch) -> None:
    monkeypatch.setattr(gtr, "gene_sets", lambda **_: _multi_panel_df())
    agg = gtr.aggregate_gene_sets(by="test_name")
    assert set(agg["set_id"]) == {"Fragile X"}
    assert int(agg[agg["gene_symbol"] == "FMR1"]["support_count"].iloc[0]) == 2


def test_aggregate_gene_sets_rejects_bad_by() -> None:
    with pytest.raises(ValueError):
        gtr.aggregate_gene_sets(by="nonsense")


# ── to_gmt ───────────────────────────────────────────────────────────────────


def test_to_gmt_raw_roundtrips(tmp_path, monkeypatch) -> None:
    from biodb.utils import read_gmt

    monkeypatch.setattr(gtr, "gene_sets", lambda **_: _multi_panel_df())
    out = tmp_path / "panels.gmt"
    gtr.to_gmt(out)
    assert out.exists()
    parsed = read_gmt(out, return_format="dict")
    gtr2_genes = {g for (sid, _desc), genes in parsed.items() if sid == "GTR2" for g in genes}
    assert gtr2_genes == {"FMR1", "AFF2"}


def test_to_gmt_aggregated(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(gtr, "gene_sets", lambda **_: _multi_panel_df())
    out = tmp_path / "agg.gmt"
    gtr.to_gmt(out, by="condition")
    lines = out.read_text().strip().splitlines()
    assert any(line.startswith("C0016667\t") for line in lines)
    assert "FMR1" in lines[0] and "AFF2" in lines[0]


# ── top-level re-exports ─────────────────────────────────────────────────────


def test_top_level_reexports() -> None:
    import biodb

    assert biodb.gtr is gtr
    for name in ("gtr_search_tests", "gtr_query_test", "gtr_gene_sets", "gtr_to_gmt"):
        assert hasattr(biodb, name), name


# ── live-network smoke tests (skipped in CI) ─────────────────────────────────


@pytest.mark.network
def test_live_query_gene_brca1() -> None:
    tests = gtr.query_gene("BRCA1", retmax=3)
    assert tests
    assert all(t.accession.startswith("GTR") for t in tests)
    assert any("BRCA1" in {g["symbol"] for g in t.genes} for t in tests)


@pytest.mark.network
def test_live_search_then_query() -> None:
    accs = gtr.search_tests("BRCA1", field="SYMB", retmax=2)
    assert accs and accs[0].startswith("GTR")
    rec = gtr.query_test(accs[0])
    assert rec.name


@pytest.mark.network
@pytest.mark.slow
def test_live_tsv_download_and_gene_sets(tmp_path) -> None:
    df = gtr.load_test_condition_gene(cache_dir=tmp_path)
    assert "accession_version" in df.columns
    sets = gtr.gene_sets(cache_dir=tmp_path)
    assert {"panel_id", "gene_entrez"}.issubset(sets.columns)
    assert len(sets) > 1000
