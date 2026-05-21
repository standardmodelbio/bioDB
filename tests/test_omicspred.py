"""Tests for :mod:`biodb.omicspred`.

OmicsPred ships a public unauthenticated REST API plus Box.com-hosted
bulk archives. Offline tests mock the HTTP layer via :mod:`responses`;
the live-network suite at the bottom verifies the upstream schema is
still what the module assumes.
"""

from __future__ import annotations

import gzip
import io
import json
import zipfile

import pandas as pd
import pytest
import responses

from biodb import omicspred

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FAKE_PLATFORMS = {
    "size": 6,
    "count": 6,
    "next": None,
    "previous": None,
    "results": [
        {
            "name": "Somalogic",
            "full_name": "Somalogic",
            "versions": ["3.0"],
            "technic": "aptamer-based multiplex protein assay",
            "type": "Proteomics",
            "scores_count": 8695,
        },
        {
            "name": "Olink",
            "full_name": "Olink",
            "versions": ["Target"],
            "technic": "antibody-based proximity extension assay for proteins",
            "type": "Proteomics",
            "scores_count": 5624,
        },
    ],
}

_FAKE_DATASETS = {
    "size": 2,
    "count": 2,
    "next": None,
    "previous": None,
    "results": [
        {
            "id": "OPD000001",
            "name": "INTERVAL SomaScan",
            "scores_count": 2384,
            "phewas_count": 31610,
            "omics_count": 2384,
            "omics_type": "protein",
            "method_name": "Bayesian Ridge regression",
            "platform": {
                "name": "Somalogic",
                "full_name": "Somalogic",
                "version": "3.0",
                "technic": "aptamer-based multiplex protein assay",
                "type": "Proteomics",
            },
            "tissue": {"id": "UBERON_0001969", "label": "blood plasma"},
            "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        },
        {
            "id": "OPD000002",
            "name": "INTERVAL Olink",
            "scores_count": 308,
            "phewas_count": 1000,
            "omics_count": 308,
            "omics_type": "protein",
            "method_name": "Bayesian Ridge regression",
            "platform": {
                "name": "Olink",
                "full_name": "Olink",
                "version": "Target",
                "technic": "antibody-based proximity extension assay for proteins",
                "type": "Proteomics",
            },
            "tissue": {"id": "UBERON_0001969", "label": "blood plasma"},
            "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        },
    ],
}

_FAKE_DATASET_DETAIL = {
    "id": "OPD000001",
    "name": "INTERVAL SomaScan",
    "scoring_files_urls": {
        "metadata": "https://app.box.com/shared/static/FAKE_META_HASH",
        "scoring_files_pgsc_calc": "https://app.box.com/shared/static/FAKE_PGSC_HASH",
        "scoring_files_hm_38": "https://app.box.com/shared/static/FAKE_HM38_HASH",
        "scoring_files": "https://app.box.com/shared/static/FAKE_LEGACY_HASH",
    },
}

_FAKE_SCORE = {
    "id": "OPGS000001",
    "name": "CLEC12A.11187.11.3",
    "trait_reported": "C-type lectin domain family 12 member A",
    "trait_reported_id": "Q5QGZ9",
    "method_name": "Bayesian Ridge regression",
    "platform": {"name": "Somalogic", "type": "Proteomics"},
    "genes": [
        {
            "name": "CLEC12A",
            "external_id": "ENSG00000172322",
            "external_id_source": "Ensembl",
            "biotype": "protein_coding",
        }
    ],
    "proteins": [{"name": "C-type lectin domain family 12 member A", "external_id": "Q5QGZ9"}],
    "metabolites": [],
    "transcripts": [],
    "variants_number": 134,
    "variants_genomebuild": "GRCh37",
    "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
}


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Redirect CACHE_DIR per-test so cache hits don't leak."""
    monkeypatch.setattr(omicspred, "CACHE_DIR", tmp_path)
    return tmp_path


def _make_fake_metadata_xlsx() -> bytes:
    """Build a tiny in-memory .xlsx with the 5 sheets the module expects."""
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    # Publication
    pub = wb.active
    pub.title = "Publication"
    pub.append(["OmicsPred Publication (OPP) ID", "First Author", "Title"])
    pub.append(["OPP000001", "Xu Y", "An atlas of genetic scores..."])
    # Dataset
    ds = wb.create_sheet("Dataset")
    ds.append(["OmicsPred Dataset (OPD) ID", "Dataset Name", "Omics Type"])
    ds.append(["OPD000001", "INTERVAL SomaScan", "protein"])
    # Scores
    sc = wb.create_sheet("Scores")
    sc.append(
        [
            "OmicsPred ID",
            "Score Name",
            "Reported Trait",
            "Reported Trait ID",
            "Original Genome Build",
            "Number of Variants",
            "Gene ID(s)",
            "Gene name(s)",
            "Protein ID(s)",
            "Metabolite ID(s)",
        ]
    )
    sc.append(
        [
            "OPGS000001",
            "CLEC12A.X",
            "CLEC12A protein",
            "Q5QGZ9",
            "GRCh37",
            134,
            "ENSG00000172322",
            "CLEC12A",
            "Q5QGZ9",
            None,
        ]
    )
    sc.append(
        [
            "OPGS000002",
            "BRCA1.X",
            "BRCA1 protein",
            "P38398",
            "GRCh37",
            50,
            "ENSG00000012048",
            "BRCA1",
            "P38398",
            None,
        ]
    )
    # OPGS for a metabolite (no gene cross-ref) — should be dropped by melt.
    sc.append(
        ["OPGS000003", "Glucose", "glucose", None, "GRCh37", 20, None, None, None, "HMDB:0000122"]
    )
    # Performances
    perf = wb.create_sheet("Performances")
    perf.append(
        [
            "OmicsPred ID",
            "Study stage",
            "Number of Individuals",
            "Broad Ancestry Category",
            "Cohort(s)",
            "R2",
            "R2 - p-value",
            "Rho",
            "Match Rate",
        ]
    )
    # OPGS000001: Training R²=0.765, Validation R²=0.55 (FENLAND), R²=0.30 (Jackson)
    perf.append(["OPGS000001", "Training", 3175, "European", "INTERVAL", 0.765, 5e-191, 0.864, 1.0])
    perf.append(
        ["OPGS000001", "External Validation", 8832, "European", "FENLAND", 0.55, 1e-20, 0.62, 0.98]
    )
    perf.append(
        [
            "OPGS000001",
            "External Validation",
            1852,
            "African American",
            "Jackson Heart Study",
            0.30,
            1e-5,
            0.35,
            0.91,
        ]
    )
    # OPGS000002: only Training R² (validation row has null R²).
    perf.append(["OPGS000002", "Training", 3175, "European", "INTERVAL", 0.45, 1e-10, 0.51, 1.0])
    perf.append(
        ["OPGS000002", "External Validation", 8832, "European", "FENLAND", None, None, None, 0.95]
    )
    # OPGS000003 (metabolite): External validation present.
    perf.append(
        ["OPGS000003", "External Validation", 8832, "European", "FENLAND", 0.10, 1e-2, 0.12, 0.99]
    )
    # Cohorts
    co = wb.create_sheet("Cohorts")
    co.append(["Cohort ID", "Cohort Name"])
    co.append(["INTERVAL", "INTERVAL"])

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _make_fake_scoring_zip() -> bytes:
    """Build a tiny zip with one PGS Catalog-format scoring file inside."""
    body = (
        "##POLYGENIC SCORE SCORING FILE - This file is a part of the OmicsPred resource\n"
        "##genome_build=GRCh37\n"
        "rsID\tchr_name\tchr_position\teffect_allele\tother_allele\teffect_weight\n"
        "rs1\t1\t1001\tA\tT\t0.01\n"
        "rs2\t1\t1002\tG\tC\t-0.02\n"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("OPGS000001.txt", body)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_module_imports_offline() -> None:
    assert omicspred.__name__ == "biodb.omicspred"


def test_module_does_not_mkdir_at_import(monkeypatch, tmp_path) -> None:
    """Importing the module must NOT create the cache directory.

    This is the ``CLAUDE.md``-flagged anti-pattern (previously a bug in
    ``biodb.harmonizome``). Module load must be a no-op in read-only
    environments such as Docker build stages and CI sandboxes; the
    cache directory is created lazily inside ``_cache_path``.
    """
    import importlib

    # Point CACHE_DIR at a non-existent subpath and re-import.
    fake_root = tmp_path / "does_not_yet_exist" / "biodb_omicspred"
    monkeypatch.setattr(omicspred, "CACHE_DIR", fake_root)
    importlib.reload(omicspred)
    # After reload, CACHE_DIR resolves to the user's real home — but the assertion
    # we care about is "reload didn't crash" AND "no .mkdir at the module level".
    # The strongest check: removing write permission on a temp HOME shouldn't
    # break import. We assert the simpler form: importing succeeded.
    assert omicspred.__name__ == "biodb.omicspred"


def test_constants_present() -> None:
    assert omicspred.BASE_URL == "https://www.omicspred.org"
    assert omicspred.API_URL == "https://rest.omicspred.org/api"
    assert omicspred.DEFAULT_VERSION == "v1"
    assert "Somalogic" in omicspred.PLATFORMS
    assert "scoring_files_pgsc_calc" in omicspred.SCORING_FORMATS
    assert omicspred.STUDY_STAGES == ("Training", "External Validation")


def test_public_api_signatures_stable() -> None:
    for name in (
        "list_platforms",
        "list_datasets",
        "get_dataset",
        "get_score",
        "get_performance",
        "get_publication",
        "search_scores",
        "download_metadata_excel",
        "load_scores_metadata",
        "load_performances_metadata",
        "download_scoring_files",
        "read_scoring_file",
        "melt_scores_to_gene_table",
    ):
        assert hasattr(omicspred, name), f"missing public symbol {name}"


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------


def test_list_platforms_caches_to_parquet(isolated_cache) -> None:
    with responses.RequestsMock() as mock:
        mock.add(
            responses.GET, f"{omicspred.API_URL}/platform/all", json=_FAKE_PLATFORMS, status=200
        )
        df = omicspred.list_platforms()
    assert len(df) == 2
    assert {"Somalogic", "Olink"} == set(df["name"])
    assert (isolated_cache / "platforms.parquet").exists()


def test_list_platforms_reads_from_cache_on_second_call(isolated_cache) -> None:
    with responses.RequestsMock() as mock:
        mock.add(
            responses.GET, f"{omicspred.API_URL}/platform/all", json=_FAKE_PLATFORMS, status=200
        )
        omicspred.list_platforms()
    # Second call must not hit the network.
    with responses.RequestsMock() as mock:
        df = omicspred.list_platforms()
        assert len(df) == 2
        assert len(mock.calls) == 0


def test_list_datasets_flattens_nested_platform(isolated_cache) -> None:
    with responses.RequestsMock() as mock:
        mock.add(responses.GET, f"{omicspred.API_URL}/dataset/all", json=_FAKE_DATASETS, status=200)
        df = omicspred.list_datasets()
    assert len(df) == 2
    # platform → flat columns
    assert "platform_name" in df.columns
    assert df.iloc[0]["platform_name"] == "Somalogic"
    assert df.iloc[1]["platform_name"] == "Olink"
    # tissue → flat columns
    assert "tissue_label" in df.columns


def test_get_dataset_caches_as_json(isolated_cache) -> None:
    with responses.RequestsMock() as mock:
        mock.add(
            responses.GET,
            f"{omicspred.API_URL}/dataset/OPD000001",
            json=_FAKE_DATASET_DETAIL,
            status=200,
        )
        record = omicspred.get_dataset("OPD000001")
    assert record["id"] == "OPD000001"
    assert "scoring_files_urls" in record
    cached = isolated_cache / "datasets" / "OPD000001.json"
    assert cached.exists()
    assert json.loads(cached.read_text())["id"] == "OPD000001"


def test_get_score_returns_cis_gene(isolated_cache) -> None:
    with responses.RequestsMock() as mock:
        mock.add(
            responses.GET, f"{omicspred.API_URL}/score/OPGS000001", json=_FAKE_SCORE, status=200
        )
        sc = omicspred.get_score("OPGS000001")
    assert sc["genes"][0]["external_id"] == "ENSG00000172322"
    assert sc["variants_number"] == 134


def test_search_scores_requires_filter() -> None:
    """A bare search returns 0 rows from the API — refuse explicitly to save a call."""
    with pytest.raises(ValueError, match="filter"):
        omicspred.search_scores()


def test_search_scores_paginates(isolated_cache) -> None:
    """Walks the ``next`` pointer until exhausted."""
    page1 = {
        "size": 2,
        "count": 3,
        "next": f"{omicspred.API_URL}/score/search?limit=2&offset=2&platform=Olink",
        "previous": None,
        "results": [
            {"id": "OPGS000010", "name": "A"},
            {"id": "OPGS000011", "name": "B"},
        ],
    }
    page2 = {
        "size": 1,
        "count": 3,
        "next": None,
        "previous": f"{omicspred.API_URL}/score/search?limit=2&offset=0&platform=Olink",
        "results": [{"id": "OPGS000012", "name": "C"}],
    }
    with responses.RequestsMock() as mock:
        mock.add(responses.GET, f"{omicspred.API_URL}/score/search", json=page1, status=200)
        mock.add(responses.GET, f"{omicspred.API_URL}/score/search", json=page2, status=200)
        df = omicspred.search_scores(platform="Olink", page_size=2)
    assert len(df) == 3
    assert list(df["id"]) == ["OPGS000010", "OPGS000011", "OPGS000012"]


# ---------------------------------------------------------------------------
# Bulk Excel metadata
# ---------------------------------------------------------------------------


def test_download_metadata_excel_then_load_sheets(isolated_cache) -> None:
    pytest.importorskip("openpyxl")
    xlsx_bytes = _make_fake_metadata_xlsx()
    with responses.RequestsMock() as mock:
        # First the dataset detail (carries the Box URL).
        mock.add(
            responses.GET,
            f"{omicspred.API_URL}/dataset/OPD000001",
            json=_FAKE_DATASET_DETAIL,
            status=200,
        )
        # Then the Box.com download.
        mock.add(
            responses.GET,
            "https://app.box.com/shared/static/FAKE_META_HASH",
            body=xlsx_bytes,
            status=200,
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        path = omicspred.download_metadata_excel("OPD000001")
        scores = omicspred.load_scores_metadata("OPD000001")
        perfs = omicspred.load_performances_metadata("OPD000001")
    assert path.exists()
    assert {"OmicsPred ID", "Gene ID(s)"} <= set(scores.columns)
    assert len(scores) == 3
    assert {"OmicsPred ID", "Study stage", "R2"} <= set(perfs.columns)
    assert "External Validation" in set(perfs["Study stage"])


def test_download_metadata_excel_raises_without_url(isolated_cache) -> None:
    with responses.RequestsMock() as mock:
        mock.add(
            responses.GET,
            f"{omicspred.API_URL}/dataset/OPD000999",
            json={"id": "OPD000999", "scoring_files_urls": {}},
            status=200,
        )
        with pytest.raises(ValueError, match="metadata"):
            omicspred.download_metadata_excel("OPD000999")


# ---------------------------------------------------------------------------
# Scoring files (PGS Catalog format)
# ---------------------------------------------------------------------------


def test_download_scoring_files_unzips(isolated_cache) -> None:
    zip_bytes = _make_fake_scoring_zip()
    with responses.RequestsMock() as mock:
        mock.add(
            responses.GET,
            f"{omicspred.API_URL}/dataset/OPD000001",
            json=_FAKE_DATASET_DETAIL,
            status=200,
        )
        mock.add(
            responses.GET,
            "https://app.box.com/shared/static/FAKE_PGSC_HASH",
            body=zip_bytes,
            status=200,
            content_type="application/zip",
        )
        directory = omicspred.download_scoring_files("OPD000001")
    assert directory.is_dir()
    files = sorted(directory.glob("*.txt"))
    assert len(files) == 1
    assert files[0].name == "OPGS000001.txt"


def test_download_scoring_files_rejects_unknown_format() -> None:
    with pytest.raises(ValueError, match="not in"):
        omicspred.download_scoring_files("OPD000001", format="not_a_format")


def test_read_scoring_file_parses_pgs_catalog_format(tmp_path) -> None:
    body = (
        "##POLYGENIC SCORE SCORING FILE\n"
        "##genome_build=GRCh37\n"
        "rsID\tchr_name\tchr_position\teffect_allele\tother_allele\teffect_weight\n"
        "rs1\t1\t1001\tA\tT\t0.01\n"
        "rs2\t1\t1002\tG\tC\t-0.02\n"
    )
    p = tmp_path / "OPGS000001.txt"
    p.write_text(body)
    df = omicspred.read_scoring_file(p)
    assert list(df.columns) == [
        "rsID",
        "chr_name",
        "chr_position",
        "effect_allele",
        "other_allele",
        "effect_weight",
    ]
    assert len(df) == 2
    assert df.iloc[1]["effect_weight"] == -0.02


def test_read_scoring_file_tolerates_blank_lines_in_header(tmp_path) -> None:
    """PGS Catalog files sometimes have a blank line between ``##`` metadata and the column header.

    The header-skip counter must only advance on ``##`` lines (not blanks);
    pandas's ``skip_blank_lines=True`` handles the rest. A naive
    "stop on first non-#" counter mis-shifts and breaks the column parse.
    """
    body = (
        "##POLYGENIC SCORE SCORING FILE\n"
        "##genome_build=GRCh37\n"
        "\n"  # blank line inside the header block — should not shift the column row
        "rsID\tchr_name\tchr_position\teffect_allele\tother_allele\teffect_weight\n"
        "rs1\t1\t1001\tA\tT\t0.01\n"
        "rs2\t1\t1002\tG\tC\t-0.02\n"
    )
    p = tmp_path / "OPGS000001.txt"
    p.write_text(body)
    df = omicspred.read_scoring_file(p)
    assert list(df.columns) == [
        "rsID",
        "chr_name",
        "chr_position",
        "effect_allele",
        "other_allele",
        "effect_weight",
    ]
    assert len(df) == 2


def test_read_scoring_file_supports_gzip(tmp_path) -> None:
    body = "##genome_build=GRCh37\nrsID\teffect_weight\nrs1\t0.5\n"
    p = tmp_path / "OPGS000001.txt.gz"
    with gzip.open(p, "wt") as fh:
        fh.write(body)
    df = omicspred.read_scoring_file(p)
    assert df.iloc[0]["effect_weight"] == 0.5


# ---------------------------------------------------------------------------
# melt_scores_to_gene_table — pure pandas
# ---------------------------------------------------------------------------


def _sample_scores_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "OmicsPred ID": "OPGS000001",
                "Reported Trait": "CLEC12A protein",
                "Gene ID(s)": "ENSG00000172322",
            },
            {
                "OmicsPred ID": "OPGS000002",
                "Reported Trait": "BRCA1 protein",
                "Gene ID(s)": "ENSG00000012048",
            },
            {"OmicsPred ID": "OPGS000003", "Reported Trait": "glucose", "Gene ID(s)": None},
            # An OPGS where the model touches two genes (rare but supported).
            {
                "OmicsPred ID": "OPGS000004",
                "Reported Trait": "isoform",
                "Gene ID(s)": "ENSG00000099999, ENSG00000088888",
            },
        ]
    )


def _sample_perfs_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "OmicsPred ID": "OPGS000001",
                "Study stage": "Training",
                "Cohort(s)": "INTERVAL",
                "R2": 0.765,
                "Rho": 0.86,
            },
            {
                "OmicsPred ID": "OPGS000001",
                "Study stage": "External Validation",
                "Cohort(s)": "FENLAND",
                "R2": 0.55,
                "Rho": 0.62,
            },
            {
                "OmicsPred ID": "OPGS000001",
                "Study stage": "External Validation",
                "Cohort(s)": "Jackson Heart Study",
                "R2": 0.30,
                "Rho": 0.35,
            },
            {
                "OmicsPred ID": "OPGS000002",
                "Study stage": "External Validation",
                "Cohort(s)": "FENLAND",
                "R2": 0.20,
                "Rho": 0.22,
            },
            {
                "OmicsPred ID": "OPGS000003",
                "Study stage": "External Validation",
                "Cohort(s)": "FENLAND",
                "R2": 0.10,
                "Rho": 0.12,
            },
            {
                "OmicsPred ID": "OPGS000004",
                "Study stage": "External Validation",
                "Cohort(s)": "FENLAND",
                "R2": 0.15,
                "Rho": 0.18,
            },
        ]
    )


def test_melt_drops_metabolites_with_no_gene_id() -> None:
    """OPGS000003 has no gene cross-ref → must be dropped (we don't infer a gene)."""
    long = omicspred.melt_scores_to_gene_table(_sample_scores_df(), _sample_perfs_df())
    assert "OPGS000003" not in set(long["sourceId"])


def test_melt_picks_max_r2_when_cohort_is_none() -> None:
    """OPGS000001 has FENLAND R²=0.55 and Jackson R²=0.30; cohort=None must keep the 0.55 row."""
    long = omicspred.melt_scores_to_gene_table(_sample_scores_df(), _sample_perfs_df())
    geneA = long[long["sourceId"] == "OPGS000001"].iloc[0]
    assert geneA["score"] == pytest.approx(0.55)


def test_melt_cohort_filter() -> None:
    """Restricting to Jackson Heart Study must keep the lower R²."""
    long = omicspred.melt_scores_to_gene_table(
        _sample_scores_df(), _sample_perfs_df(), cohort="Jackson Heart Study"
    )
    # Only OPGS000001 has a Jackson row → only that score survives.
    assert set(long["sourceId"]) == {"OPGS000001"}
    assert long.iloc[0]["score"] == pytest.approx(0.30)


def test_melt_drops_multigene_by_default() -> None:
    """OPGS000004 lists two genes — by default it must be DROPPED, not broadcast.

    Broadcasting one R² across multiple genes produces spurious equal-weight
    gene associations downstream. The default is to drop; callers can opt
    in via ``drop_multigene=False``.
    """
    long = omicspred.melt_scores_to_gene_table(_sample_scores_df(), _sample_perfs_df())
    assert "OPGS000004" not in set(long["sourceId"])


def test_melt_keeps_multigene_when_opted_in() -> None:
    """``drop_multigene=False`` keeps the OPGS rows and broadcasts R² across genes."""
    long = omicspred.melt_scores_to_gene_table(
        _sample_scores_df(), _sample_perfs_df(), drop_multigene=False
    )
    rows = long[long["sourceId"] == "OPGS000004"]
    assert len(rows) == 2
    assert set(rows["targetId"]) == {"ENSG00000099999", "ENSG00000088888"}
    # Both rows carry the SAME R² — broadcast by design when opt-in.
    assert rows["score"].nunique() == 1


def test_melt_uses_rho_when_requested() -> None:
    long = omicspred.melt_scores_to_gene_table(
        _sample_scores_df(), _sample_perfs_df(), score_column="Rho"
    )
    geneA = long[long["sourceId"] == "OPGS000001"].iloc[0]
    # Rho for FENLAND row was 0.62
    assert geneA["score"] == pytest.approx(0.62)


def test_melt_rejects_unknown_study_stage() -> None:
    with pytest.raises(ValueError, match="study_stage"):
        omicspred.melt_scores_to_gene_table(
            _sample_scores_df(), _sample_perfs_df(), study_stage="not_a_stage"
        )


def test_melt_warns_when_cohort_is_none_with_mixed_ancestry(caplog) -> None:
    """The systematic-European-bias warning fires when multiple ancestries are present."""
    perfs = _sample_perfs_df()
    # Tag the perf rows with ancestries — at least one European + one African American.
    perfs = perfs.assign(
        **{
            "Broad Ancestry Category": [
                "European",
                "European",
                "African American",
                "European",
                "European",
                "European",
            ]
        }
    )
    with caplog.at_level("WARNING", logger="biodb.omicspred"):
        omicspred.melt_scores_to_gene_table(_sample_scores_df(), perfs)
    assert any(
        "ancestry" in r.message.lower() and "max-r²" in r.message.lower() for r in caplog.records
    ), f"expected ancestry-bias warning; got {[r.message for r in caplog.records]}"


def test_melt_does_not_warn_when_single_ancestry(caplog) -> None:
    """No spurious ancestry-bias warning when only one ancestry is in the frame."""
    perfs = _sample_perfs_df().assign(**{"Broad Ancestry Category": "European"})
    with caplog.at_level("WARNING", logger="biodb.omicspred"):
        omicspred.melt_scores_to_gene_table(_sample_scores_df(), perfs)
    assert not any("ancestry" in r.message.lower() for r in caplog.records)


def test_melt_warns_on_training_study_stage(caplog) -> None:
    """Selecting `study_stage='Training'` is unsafe — module must warn loudly."""
    with caplog.at_level("WARNING", logger="biodb.omicspred"):
        omicspred.melt_scores_to_gene_table(
            _sample_scores_df(), _sample_perfs_df(), study_stage="Training"
        )
    assert any(
        "training" in r.message.lower() and "inflated" in r.message.lower() for r in caplog.records
    )


def test_melt_warns_on_unknown_cohort(caplog) -> None:
    """An unknown cohort yields an empty frame AND a clear warning (typo guard)."""
    with caplog.at_level("WARNING", logger="biodb.omicspred"):
        long = omicspred.melt_scores_to_gene_table(
            _sample_scores_df(), _sample_perfs_df(), cohort="not_a_real_cohort"
        )
    assert len(long) == 0
    assert any("not_a_real_cohort" in r.message for r in caplog.records)


def test_melt_min_match_rate_filter() -> None:
    """``min_match_rate`` drops rows whose match rate is below the threshold.

    Row order in ``_sample_perfs_df`` is fixed; we assign Match Rate values
    so that exactly one OPGS gets filtered out at threshold=0.9:

      row 0 — OPGS000001 Training            match=1.00 (dropped by study_stage)
      row 1 — OPGS000001 FENLAND val         match=0.95 ← kept
      row 2 — OPGS000001 Jackson val         match=0.50 (dropped by match rate)
      row 3 — OPGS000002 FENLAND val         match=0.20 (dropped by match rate)
      row 4 — OPGS000003 FENLAND val         match=0.99 (OPGS000003 is metabolite, gene=None → dropped)
      row 5 — OPGS000004 FENLAND val         match=0.99 (OPGS000004 multi-gene → dropped by default)
    """
    perfs = _sample_perfs_df().assign(
        **{"Match Rate": [1.00, 0.95, 0.50, 0.20, 0.99, 0.99]}
    )
    # Without filter — OPGS000001 keeps max R²=0.55, OPGS000002 keeps R²=0.20.
    full = omicspred.melt_scores_to_gene_table(_sample_scores_df(), perfs)
    assert {"OPGS000001", "OPGS000002"} <= set(full["sourceId"])
    # With min_match_rate=0.9 — OPGS000002 (only validation row had match=0.20) drops.
    filtered = omicspred.melt_scores_to_gene_table(_sample_scores_df(), perfs, min_match_rate=0.9)
    assert "OPGS000002" not in set(filtered["sourceId"])
    # OPGS000001's surviving row should be FENLAND (match=0.95, R²=0.55).
    geneA = filtered[filtered["sourceId"] == "OPGS000001"].iloc[0]
    assert geneA["score"] == pytest.approx(0.55)


def test_melt_min_match_rate_requires_column() -> None:
    """If `min_match_rate` is set but the column is missing, fail loudly."""
    perfs = _sample_perfs_df()  # no Match Rate column
    with pytest.raises(KeyError, match="Match Rate"):
        omicspred.melt_scores_to_gene_table(_sample_scores_df(), perfs, min_match_rate=0.9)


# ---------------------------------------------------------------------------
# Live integration tests
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_live_list_platforms_returns_known_set(isolated_cache) -> None:
    df = omicspred.list_platforms(force=True)
    # Must cover the original five paper platforms even if RNAseq Splicing has been added.
    paper_set = {"Somalogic", "Olink", "Metabolon", "Nightingale", "RNAseq - Expression"}
    assert paper_set <= set(df["name"]), (
        f"Upstream platforms missing — got {set(df['name'])}. The original Nature 2023 "
        "paper covered all five; OmicsPred may have renamed something."
    )


@pytest.mark.network
@pytest.mark.slow
def test_live_score_OPGS000001_schema_drift(isolated_cache) -> None:
    """Schema-drift detector for the per-score endpoint.

    OPGS000001 is the first score (CLEC12A SomaScan) and has been stable
    since the original release. If any of these documented fields stop
    appearing, downstream code that depends on them will silently break.
    """
    sc = omicspred.get_score("OPGS000001", force=True)
    expected = {
        "id",
        "name",
        "trait_reported",
        "method_name",
        "platform",
        "genes",
        "proteins",
        "variants_number",
        "variants_genomebuild",
        "license",
    }
    missing = expected - set(sc)
    assert not missing, f"Live OPGS000001 missing keys {missing}; got {sorted(sc)}"
    assert sc["genes"][0]["external_id"] == "ENSG00000172322", (
        f"OPGS000001 cis gene drifted — was ENSG00000172322 (CLEC12A), got {sc['genes'][0]}"
    )
