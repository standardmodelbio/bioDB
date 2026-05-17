"""Tests for :mod:`biodb.harmonizome`.

Network access is mocked with :mod:`responses` for ``requests``-based code
and with ``unittest.mock`` for ``urllib.request.urlopen``-based code so CI
never touches Maayan-Lab. The handful of ``@pytest.mark.network`` smoke
tests at the bottom hit the real API on demand.
"""

from __future__ import annotations

import gzip
import inspect
import io
import json
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError

import pandas as pd
import pytest
import responses

from biodb import harmonizome

# ---------------------------------------------------------------------------
# Fixtures: a small but realistic ``script_config`` payload + GMT body
# ---------------------------------------------------------------------------


def _fake_script_config() -> dict:
    return {
        "downloads": [
            "gene_set_library_crisp.gmt.gz",
            "gene_attribute_matrix.txt.gz",
            "attribute_list.txt.gz",
        ],
        "datasets": {
            "Achilles Cell Line Gene Essentiality Profiles": "achilles",
            "GTEx Tissue Gene Expression Profiles": "gtex_tissue",
            "HuRI Human Reference Interactome": "huri",
        },
    }


def _fake_dataset_metadata(name: str = "GTEx Tissue Gene Expression Profiles") -> dict:
    return {
        "description": "RNA-seq expression in tissues",
        "measurement": "TPM",
        "association": "tissue",
        "category": "transcriptomics",
        "resource": {"name": "GTEx Consortium"},
        "citations": ["doi:1", "doi:2"],
        "lastUpdated": "2024-01-01",
        "stats": {"genes": 30000, "tissues": 53},
    }


def _gmt_body() -> str:
    return (
        "BRAIN\thttp://example/brain\tBRCA1\tTP53\tAPOE\nLIVER\thttp://example/liver\tALB\tCYP2A6\n"
    )


@pytest.fixture(autouse=True)
def _clear_config_cache():
    """``_load_config`` is ``@cache``'d. Reset between tests so each test
    can install its own fake payload via ``mock.patch``."""
    harmonizome._load_config.cache_clear()
    yield
    harmonizome._load_config.cache_clear()


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_module_imports_offline() -> None:
    assert harmonizome.__name__ == "biodb.harmonizome"
    assert harmonizome.VERSION == "1.0"


def test_constants_present() -> None:
    assert harmonizome.API_URL.endswith("/Harmonizome/api")
    assert harmonizome.DOWNLOAD_URL.endswith("/data")
    assert harmonizome.CACHE_DIR.exists()


def test_public_api_signatures_stable() -> None:
    expected = {
        "list_datasets",
        "download_datasets",
        "get_gmt",
        "load_gene_attribute_matrix",
        "get_dataset_metadata",
        "json_from_url",
        "Entity",
        "Harmonizome",
    }
    missing = [name for name in expected if not hasattr(harmonizome, name)]
    assert not missing


def test_get_gmt_signature_uses_string_defaults() -> None:
    sig = inspect.signature(harmonizome.get_gmt)
    assert sig.parameters["file_types"].default == "gene_set*.gmt.gz"
    assert sig.parameters["force"].default == 0
    assert sig.parameters["verbose"].default == 1


def test_entity_constants() -> None:
    assert harmonizome.Entity.DATASET == "dataset"
    assert harmonizome.Entity.GENE == "gene"
    assert harmonizome.Entity.GENE_SET == "gene_set"
    assert harmonizome.Entity.ATTRIBUTE == "attribute"


# ---------------------------------------------------------------------------
# json_from_url + _load_config + module __getattr__
# ---------------------------------------------------------------------------


def test_json_from_url_decodes_response() -> None:
    payload = {"hello": "world"}
    fake_resp = mock.MagicMock()
    fake_resp.read.return_value = json.dumps(payload).encode("utf-8")
    fake_resp.__enter__ = mock.MagicMock(return_value=fake_resp)
    fake_resp.__exit__ = mock.MagicMock(return_value=False)

    with mock.patch.object(harmonizome, "urlopen", return_value=fake_resp):
        out = harmonizome.json_from_url("https://example/x")
    assert out == payload


def test_load_config_caches_result() -> None:
    with mock.patch.object(
        harmonizome, "json_from_url", return_value=_fake_script_config()
    ) as mocked:
        cfg1 = harmonizome._load_config()
        cfg2 = harmonizome._load_config()
    assert cfg1 == _fake_script_config()
    assert cfg2 is cfg1  # cached
    assert mocked.call_count == 1  # only the first call hit the network


def test_module_getattr_downloads_and_datasets() -> None:
    with mock.patch.object(harmonizome, "json_from_url", return_value=_fake_script_config()):
        downloads = harmonizome.DOWNLOADS
        datasets = harmonizome.DATASET_TO_PATH
    assert isinstance(downloads, list)
    assert "gene_set_library_crisp.gmt.gz" in downloads
    assert isinstance(datasets, dict)
    assert "GTEx Tissue Gene Expression Profiles" in datasets


def test_module_getattr_rejects_unknown_name() -> None:
    with pytest.raises(AttributeError):
        harmonizome.NOT_A_REAL_THING  # noqa: B018


# ---------------------------------------------------------------------------
# _download_file + _download_and_decompress_file
# ---------------------------------------------------------------------------


def test_download_file_writes_streamed_chunks(tmp_path) -> None:
    fake_response = mock.MagicMock()
    fake_response.iter_content.return_value = [b"chunk1", b"chunk2", b""]
    out = tmp_path / "subdir" / "file.bin"
    harmonizome._download_file(fake_response, out)
    assert out.read_bytes() == b"chunk1chunk2"


def test_download_and_decompress_file_writes_decompressed(tmp_path) -> None:
    body = b"hello harmonizome"
    compressed = gzip.compress(body)

    class _RawStream:
        def __init__(self) -> None:
            self._buf = io.BytesIO(compressed)

        def read(self, n: int) -> bytes:
            return self._buf.read(n)

    fake_response = mock.MagicMock()
    fake_response.raw = _RawStream()
    target = tmp_path / "out.txt.gz"
    harmonizome._download_and_decompress_file(fake_response, target)
    decompressed = target.with_suffix("")
    assert decompressed.read_bytes() == body


# ---------------------------------------------------------------------------
# download_datasets — caching, decompress, request errors
# ---------------------------------------------------------------------------


def test_download_datasets_uses_cache_when_file_exists(tmp_path) -> None:
    cached_dir = tmp_path / "achilles"
    cached_dir.mkdir(parents=True)
    cached_file = cached_dir / "gene_set_library_crisp.gmt.gz"
    cached_file.write_bytes(b"cached")
    out = harmonizome.download_datasets(
        selected_datasets=[("achilles", "achilles")],
        selected_downloads=["gene_set_library_crisp.gmt.gz"],
        cache_dir=tmp_path,
    )
    assert out["gene_set_library_crisp.gmt.gz"] == str(cached_file.resolve())


def test_download_datasets_downloads_when_missing(tmp_path) -> None:
    url = f"{harmonizome.DOWNLOAD_URL}/achilles/gene_set_library_crisp.gmt.gz"
    with responses.RequestsMock() as mocks:
        mocks.add(responses.GET, url, body=b"raw-bytes", status=200, stream=True)
        out = harmonizome.download_datasets(
            selected_datasets=[("achilles", "achilles")],
            selected_downloads=["gene_set_library_crisp.gmt.gz"],
            cache_dir=tmp_path,
        )
    assert "gene_set_library_crisp.gmt.gz" in out
    assert Path(out["gene_set_library_crisp.gmt.gz"]).read_bytes() == b"raw-bytes"


def test_download_datasets_skips_404s(tmp_path) -> None:
    """A 404 (dataset doesn't have this file) silently drops the entry."""
    url = f"{harmonizome.DOWNLOAD_URL}/achilles/missing.gmt.gz"
    with responses.RequestsMock() as mocks:
        mocks.add(responses.GET, url, status=404)
        out = harmonizome.download_datasets(
            selected_datasets=[("achilles", "achilles")],
            selected_downloads=["missing.gmt.gz"],
            cache_dir=tmp_path,
        )
    assert out == {}


def test_download_datasets_skips_network_errors(tmp_path) -> None:
    """``requests.RequestException`` is swallowed and the file omitted."""
    url = f"{harmonizome.DOWNLOAD_URL}/achilles/file.gmt.gz"
    with responses.RequestsMock(assert_all_requests_are_fired=False) as mocks:
        from requests.exceptions import ConnectionError as ReqConnError

        mocks.add(responses.GET, url, body=ReqConnError("offline"))
        out = harmonizome.download_datasets(
            selected_datasets=[("achilles", "achilles")],
            selected_downloads=["file.gmt.gz"],
            cache_dir=tmp_path,
        )
    assert out == {}


def test_download_datasets_decompress_inplace(tmp_path) -> None:
    body = b"decompressed body"
    compressed = gzip.compress(body)
    url = f"{harmonizome.DOWNLOAD_URL}/achilles/gene_set_library_crisp.txt.gz"
    with responses.RequestsMock() as mocks:
        mocks.add(responses.GET, url, body=compressed, status=200, stream=True)
        out = harmonizome.download_datasets(
            selected_datasets=[("achilles", "achilles")],
            selected_downloads=["gene_set_library_crisp.txt.gz"],
            decompress=True,
            cache_dir=tmp_path,
            verbose=True,
        )
    decompressed_path = tmp_path / "achilles" / "gene_set_library_crisp.txt"
    assert decompressed_path.exists()
    assert decompressed_path.read_bytes() == body
    assert Path(out["gene_set_library_crisp.txt.gz"]) == decompressed_path


# ---------------------------------------------------------------------------
# list_datasets — pagination
# ---------------------------------------------------------------------------


def test_list_datasets_paginates_until_no_next() -> None:
    page1 = {
        "entities": [{"name": "DS_A", "href": "/api/1.0/dataset/DS_A"}],
        "next": "/api/1.0/dataset?cursor=1",
    }
    page2 = {
        "entities": [{"name": "DS_B", "href": "/api/1.0/dataset/DS_B"}],
        "next": None,
    }
    responses_seq = iter([page1, page2])
    with mock.patch.object(
        harmonizome, "json_from_url", side_effect=lambda url: next(responses_seq)
    ):
        df = harmonizome.list_datasets(as_df=True)
    assert isinstance(df, pd.DataFrame)
    assert list(df["name"]) == ["DS_A", "DS_B"]


def test_list_datasets_as_list() -> None:
    payload = {
        "entities": [{"name": "DS_A", "href": "/x"}, {"name": "DS_B", "href": "/y"}],
        "next": None,
    }
    with mock.patch.object(harmonizome, "json_from_url", return_value=payload):
        out = harmonizome.list_datasets(as_df=False)
    assert isinstance(out, list)
    assert {d["name"] for d in out} == {"DS_A", "DS_B"}


def test_list_datasets_handles_http_error() -> None:
    """A 5xx mid-pagination breaks the loop instead of crashing."""

    def boom(url):
        raise HTTPError(url, 503, "boom", {}, None)  # type: ignore[arg-type]

    with mock.patch.object(harmonizome, "json_from_url", side_effect=boom):
        df = harmonizome.list_datasets()
    assert df.empty


def test_list_datasets_next_with_http_prefix() -> None:
    """``next: "https://...://"`` (absolute URL) is followed verbatim."""
    page1 = {
        "entities": [{"name": "A", "href": "/x"}],
        "next": "https://maayanlab.cloud/Harmonizome/api/1.0/dataset?cursor=1",
    }
    page2 = {"entities": [{"name": "B", "href": "/y"}], "next": None}
    iter_pages = iter([page1, page2])
    with mock.patch.object(harmonizome, "json_from_url", side_effect=lambda url: next(iter_pages)):
        df = harmonizome.list_datasets()
    assert list(df["name"]) == ["A", "B"]


def test_list_datasets_next_bare_dataset_string() -> None:
    """``next: "dataset?cursor=N"`` (bare) is treated as ``/api/<VERSION>/<bare>``."""
    page1 = {"entities": [{"name": "A", "href": "/x"}], "next": "dataset?cursor=1"}
    page2 = {"entities": [{"name": "B", "href": "/y"}], "next": None}
    iter_pages = iter([page1, page2])
    urls_seen: list[str] = []

    def stub(url):
        urls_seen.append(url)
        return next(iter_pages)

    with mock.patch.object(harmonizome, "json_from_url", side_effect=stub):
        df = harmonizome.list_datasets()
    assert list(df["name"]) == ["A", "B"]
    assert urls_seen[1].endswith("/dataset?cursor=1")


# ---------------------------------------------------------------------------
# get_dataset_metadata — happy + degraded shapes
# ---------------------------------------------------------------------------


def test_get_dataset_metadata_flattens_nested_fields() -> None:
    with mock.patch.object(harmonizome, "json_from_url", return_value=_fake_dataset_metadata()):
        out = harmonizome.get_dataset_metadata("GTEx Tissue Gene Expression Profiles")
    # All values are strings (parquet-safe).
    assert all(isinstance(v, str) for v in out.values())
    assert "GTEx" in out["resource"]
    assert "doi:1" in out["citations"]
    assert "genes: 30000" in out["stats"]
    assert out["last_updated"] == "2024-01-01"


def test_get_dataset_metadata_returns_empty_on_error() -> None:
    def boom(url):
        raise HTTPError(url, 404, "missing", {}, None)  # type: ignore[arg-type]

    with mock.patch.object(harmonizome, "json_from_url", side_effect=boom):
        out = harmonizome.get_dataset_metadata("BOGUS")
    assert out == harmonizome._EMPTY_METADATA


def test_get_dataset_metadata_handles_string_citations() -> None:
    """Some datasets return ``citations`` as a single string, not a list."""
    payload = {**_fake_dataset_metadata(), "citations": "single citation"}
    with mock.patch.object(harmonizome, "json_from_url", return_value=payload):
        out = harmonizome.get_dataset_metadata("any")
    assert out["citations"] == "single citation"


def test_get_dataset_metadata_handles_string_resource() -> None:
    payload = {**_fake_dataset_metadata(), "resource": "GTEx Consortium"}
    with mock.patch.object(harmonizome, "json_from_url", return_value=payload):
        out = harmonizome.get_dataset_metadata("any")
    assert out["resource"] == "GTEx Consortium"


# ---------------------------------------------------------------------------
# _decompress_inplace + _read_one_gmt
# ---------------------------------------------------------------------------


def test_decompress_inplace_skips_non_gz(tmp_path) -> None:
    plain = tmp_path / "x.txt"
    plain.write_text("not gzip")
    assert harmonizome._decompress_inplace(plain) == plain


def test_decompress_inplace_creates_decompressed(tmp_path) -> None:
    body = b"decompressed"
    gz = tmp_path / "x.gmt.gz"
    gz.write_bytes(gzip.compress(body))
    out = harmonizome._decompress_inplace(gz)
    assert out == gz.with_suffix("")
    assert out.read_bytes() == body
    # Idempotent: a second call returns the same path without re-extracting.
    out2 = harmonizome._decompress_inplace(gz)
    assert out2 == out


def test_read_one_gmt_returns_tagged_dataframe(tmp_path) -> None:
    gmt = tmp_path / "tiny.gmt"
    gmt.write_text(_gmt_body())
    df = harmonizome._read_one_gmt("DS", "/href", str(gmt))
    assert df is not None
    assert set(df["dataset"]) == {"DS"}
    assert set(df["dataset_href"]) == {"/href"}
    assert "BRCA1" in set(df["gene"])


def test_read_one_gmt_returns_none_on_read_error(tmp_path) -> None:
    df = harmonizome._read_one_gmt("DS", "/href", str(tmp_path / "doesnotexist.gmt"))
    assert df is None


# ---------------------------------------------------------------------------
# get_gmt — orchestration cases (cache hit, none, custom datasets)
# ---------------------------------------------------------------------------


def test_get_gmt_returns_cached_parquet_when_available(tmp_path) -> None:
    cached_df = pd.DataFrame({"dataset": ["A"], "id": ["g"], "label": ["l"], "gene": ["X"]})
    merged = tmp_path / "gmt_merged.parquet"
    cached_df.to_parquet(merged, index=False)
    out = harmonizome.get_gmt(cache_dir=tmp_path, verbose=0)
    pd.testing.assert_frame_equal(out.reset_index(drop=True), cached_df)


def test_get_gmt_returns_empty_when_no_gmt_downloads(tmp_path) -> None:
    """When the config has no GMT-shaped downloadables, return empty."""
    with mock.patch.object(
        harmonizome,
        "_load_config",
        return_value={"downloads": ["only_a_text.txt"], "datasets": {}},
    ):
        out = harmonizome.get_gmt(
            datasets=["whatever"], cache_dir=tmp_path, save_path=False, verbose=0
        )
    assert out.empty


def test_get_gmt_returns_empty_when_dataset_unknown(tmp_path) -> None:
    """Requested datasets that aren't in DATASET_TO_PATH are dropped."""
    cfg = _fake_script_config()
    with (
        mock.patch.object(harmonizome, "_load_config", return_value=cfg),
        mock.patch.object(
            harmonizome, "list_datasets", return_value=pd.DataFrame(columns=["name", "href"])
        ),
    ):
        out = harmonizome.get_gmt(
            datasets=["NEVER_HEARD_OF_IT"],
            cache_dir=tmp_path,
            save_path=False,
            verbose=0,
        )
    assert out.empty


def test_get_gmt_uses_cached_gmt_files(tmp_path) -> None:
    """When cache hits for every dataset, no downloads happen and the
    merged DataFrame is constructed from disk."""
    cfg = _fake_script_config()
    dataset = "GTEx Tissue Gene Expression Profiles"

    # Seed the cache with one cached GMT for the dataset.
    ds_dir = tmp_path / dataset
    ds_dir.mkdir(parents=True)
    gmt_file = ds_dir / "gene_set_library_crisp.gmt"
    gmt_file.write_text(_gmt_body())
    # also need the .gz to satisfy the loop's filename probe
    gz_file = ds_dir / "gene_set_library_crisp.gmt.gz"
    gz_file.write_bytes(gzip.compress(_gmt_body().encode()))

    metadata_payload = _fake_dataset_metadata()
    with (
        mock.patch.object(harmonizome, "_load_config", return_value=cfg),
        mock.patch.object(
            harmonizome,
            "list_datasets",
            return_value=pd.DataFrame({"name": [dataset], "href": ["/x"]}),
        ),
        mock.patch.object(harmonizome, "get_dataset_metadata", return_value=metadata_payload),
    ):
        out = harmonizome.get_gmt(
            datasets=[dataset], cache_dir=tmp_path, save_path=False, verbose=0
        )
    assert not out.empty
    assert set(out["dataset"]) == {dataset}
    assert "description" in out.columns


def test_get_gmt_save_path_false_skips_caching(tmp_path) -> None:
    """``save_path=False`` means no parquet is written, even if a result is produced."""
    cfg = _fake_script_config()
    dataset = "GTEx Tissue Gene Expression Profiles"
    ds_dir = tmp_path / dataset
    ds_dir.mkdir(parents=True)
    (ds_dir / "gene_set_library_crisp.gmt.gz").write_bytes(gzip.compress(_gmt_body().encode()))
    (ds_dir / "gene_set_library_crisp.gmt").write_text(_gmt_body())

    with (
        mock.patch.object(harmonizome, "_load_config", return_value=cfg),
        mock.patch.object(
            harmonizome,
            "list_datasets",
            return_value=pd.DataFrame({"name": [dataset], "href": ["/x"]}),
        ),
        mock.patch.object(
            harmonizome, "get_dataset_metadata", return_value=_fake_dataset_metadata()
        ),
    ):
        harmonizome.get_gmt(datasets=[dataset], cache_dir=tmp_path, save_path=False, verbose=0)
    # The default parquet path should NOT exist.
    assert not (tmp_path / "gmt_merged.parquet").exists()


def test_get_gmt_limit_truncates_datasets(tmp_path) -> None:
    """``limit=N`` caps the number of datasets processed."""
    cfg = _fake_script_config()
    datasets = list(cfg["datasets"].keys())
    # Seed cache for all three.
    for ds in datasets:
        ds_dir = tmp_path / ds
        ds_dir.mkdir(parents=True)
        (ds_dir / "gene_set_library_crisp.gmt.gz").write_bytes(gzip.compress(_gmt_body().encode()))
        (ds_dir / "gene_set_library_crisp.gmt").write_text(_gmt_body())

    with (
        mock.patch.object(harmonizome, "_load_config", return_value=cfg),
        mock.patch.object(
            harmonizome,
            "list_datasets",
            return_value=pd.DataFrame(
                {"name": datasets, "href": [f"/{i}" for i in range(len(datasets))]}
            ),
        ),
        mock.patch.object(
            harmonizome, "get_dataset_metadata", return_value=_fake_dataset_metadata()
        ),
    ):
        out = harmonizome.get_gmt(
            datasets=datasets, cache_dir=tmp_path, save_path=False, limit=1, verbose=0
        )
    assert out["dataset"].nunique() == 1


# ---------------------------------------------------------------------------
# load_gene_attribute_matrix
# ---------------------------------------------------------------------------


def _gene_attr_matrix_text() -> str:
    """Tiny but realistic gene_attribute_matrix.txt body."""
    return (
        "# Description: example matrix\n"
        "GeneSym\tNA\tGeneID\tBRAIN\tLIVER\tKIDNEY\n"
        "na\tna\tna\tBR\tLV\tKD\n"
        "BRCA1\tna\t672\t0.1\t0.4\t0.2\n"
        "TP53\tna\t7157\t0.3\tna\t0.5\n"
    )


def test_load_gene_attribute_matrix_parses_data(tmp_path) -> None:
    dataset = "demo"
    ds_dir = tmp_path / dataset
    ds_dir.mkdir(parents=True)
    f = ds_dir / "gene_attribute_matrix.txt"
    f.write_text(_gene_attr_matrix_text())
    # The loader looks for the ``.gz`` first; provide a sibling so the
    # fall-through to ``.txt`` triggers.
    df, tissue_meta, col_meta = harmonizome.load_gene_attribute_matrix(
        dataset_name=dataset, cache_dir=tmp_path, include_col_metadata=True
    )
    assert {"GeneSym", "GeneID", "BRAIN", "LIVER", "KIDNEY"}.issubset(df.columns)
    assert len(df) == 2
    assert set(df["GeneSym"]) == {"BRCA1", "TP53"}


def test_load_gene_attribute_matrix_raises_for_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        harmonizome.load_gene_attribute_matrix(dataset_name="nope", cache_dir=tmp_path)


def test_load_gene_attribute_matrix_raises_for_unexpected_header(tmp_path) -> None:
    ds_dir = tmp_path / "demo"
    ds_dir.mkdir(parents=True)
    f = ds_dir / "gene_attribute_matrix.txt"
    f.write_text("WRONG_HEADER\tNA\tx\nrow\tna\t1\n")
    with pytest.raises(ValueError, match="GeneSym"):
        harmonizome.load_gene_attribute_matrix(dataset_name="demo", cache_dir=tmp_path)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_looks_like_gene_set_name() -> None:
    assert harmonizome._looks_like_gene_set_name("*Marker")
    assert harmonizome._looks_like_gene_set_name("__special")
    assert harmonizome._looks_like_gene_set_name("001N03")
    assert not harmonizome._looks_like_gene_set_name("BRCA1")
    assert not harmonizome._looks_like_gene_set_name("")
    assert not harmonizome._looks_like_gene_set_name(None)  # type: ignore[arg-type]


def test_reverse_excel_date_conversion() -> None:
    assert harmonizome._reverse_excel_date("1-MAR") == "MARCH1"
    assert harmonizome._reverse_excel_date("2-SEP") == "SEPT2"
    assert harmonizome._reverse_excel_date("10-DEC") == "DEC10"
    assert harmonizome._reverse_excel_date("BRCA1") == "BRCA1"
    assert harmonizome._reverse_excel_date("not-a-date") == "not-a-date"


def test_matches_file_type_variants() -> None:
    assert harmonizome._matches_file_type("gene_set_library_crisp.gmt", "gene_set*.gmt")
    assert harmonizome._matches_file_type("gene_set_library_crisp.gmt.gz", "gene_set*.gmt")
    assert not harmonizome._matches_file_type("attribute_set_library.gmt", "gene_set*.gmt")
    assert harmonizome._matches_file_type("any.gmt", None)
    assert harmonizome._matches_file_type("any.gmt.gz", None)
    assert not harmonizome._matches_file_type("any.txt", None)
    assert harmonizome._matches_file_type(
        "gene_set_library_up_crisp.gmt", ["gene_set_library_up_crisp.gmt"]
    )
    with pytest.raises(TypeError):
        harmonizome._matches_file_type("x.gmt", 123)  # type: ignore[arg-type]


def test_safe_float() -> None:
    assert harmonizome._safe_float("1.5") == 1.5
    assert pd.isna(harmonizome._safe_float("not-a-number"))


def test_detect_attribute_start_falls_back_when_no_obvious_attribute() -> None:
    """When the heuristic doesn't recognise an attribute column, return 3."""
    header = ["GeneSym", "NA", "GeneID"]
    assert harmonizome._detect_attribute_start(header) == 3


def test_detect_attribute_start_finds_first_real_attribute() -> None:
    header = ["GeneSym", "NA", "GeneID", "BRAIN", "LIVER", "KIDNEY"]
    assert harmonizome._detect_attribute_start(header) == 3


def test_build_column_metadata_empty_when_no_rows() -> None:
    assert harmonizome._build_column_metadata([], ["a", "b"]) is None


def test_build_column_metadata_assembles_frame() -> None:
    rows = [
        ["#", "#", "CellLine", "K562", "HEK293"],
        ["#", "#", "Tissue", "blood", "kidney"],
    ]
    cols = ["K562", "HEK293"]
    out = harmonizome._build_column_metadata(rows, cols)
    assert out is not None
    assert "CellLine" in out.columns
    assert "Tissue" in out.columns
    assert list(out.index) == cols
    assert out.loc["K562", "Tissue"] == "blood"


# ---------------------------------------------------------------------------
# Harmonizome legacy class
# ---------------------------------------------------------------------------


def test_legacy_harmonizome_get_by_name() -> None:
    """``Harmonizome.get(entity, name=...)`` URL-encodes the name."""
    captured: list[str] = []

    def stub(url):
        captured.append(url)
        return {"ok": True}

    with mock.patch.object(harmonizome, "json_from_url", side_effect=stub):
        harmonizome.Harmonizome.get("dataset", name="GTEx Tissue Gene Expression Profiles")
    assert any("GTEx" in url for url in captured)
    assert any("+" in url or "%20" in url for url in captured)


def test_legacy_harmonizome_get_with_cursor() -> None:
    with mock.patch.object(harmonizome, "json_from_url", return_value={"ok": True}) as mocked:
        harmonizome.Harmonizome.get("dataset", start_at=42)
    assert "cursor=42" in mocked.call_args.args[0]


def test_legacy_harmonizome_get_default_url() -> None:
    with mock.patch.object(harmonizome, "json_from_url", return_value={"ok": True}) as mocked:
        harmonizome.Harmonizome.get("dataset")
    assert mocked.call_args.args[0].endswith("/dataset")


def test_legacy_harmonizome_next_parses_response() -> None:
    response = {"next": "/api/1.0/dataset?cursor=100"}
    with mock.patch.object(harmonizome.Harmonizome, "get") as mocked_get:
        harmonizome.Harmonizome.next(response)
    mocked_get.assert_called_once_with(entity="dataset", start_at=100)


def test_legacy_harmonizome_download_refuses_no_dataset_list() -> None:
    with pytest.raises(ValueError, match="explicit list of datasets"):
        list(harmonizome.Harmonizome.download(datasets=None))


def test_legacy_harmonizome_download_rejects_unknown_dataset() -> None:
    cfg = _fake_script_config()
    with (
        mock.patch.object(harmonizome, "_load_config", return_value=cfg),
        pytest.raises(AttributeError, match="not a valid dataset name"),
    ):
        list(harmonizome.Harmonizome.download(datasets=["TOTALLY_FAKE"]))


def test_legacy_harmonizome_download_writes_and_yields(tmp_path, monkeypatch) -> None:
    """Happy path through ``Harmonizome.download`` — urlopen → write → yield."""
    cfg = _fake_script_config()
    body = b"some bytes"
    compressed = gzip.compress(body)
    fake_response = mock.MagicMock()
    fake_response.read.return_value = compressed
    monkeypatch.chdir(tmp_path)

    with (
        mock.patch.object(harmonizome, "_load_config", return_value=cfg),
        mock.patch.object(harmonizome, "urlopen", return_value=fake_response),
    ):
        filenames = list(
            harmonizome.Harmonizome.download(
                datasets=["GTEx Tissue Gene Expression Profiles"],
                what=["gene_set_library_crisp.gmt.gz"],
            )
        )

    assert len(filenames) == 1
    out = Path(filenames[0])
    assert out.read_bytes() == body
    assert "GTEx" in str(out.parent)


def test_legacy_harmonizome_download_http_error_with_explicit_what(monkeypatch) -> None:
    """When ``what`` is explicitly passed, an HTTPError surfaces as RuntimeError."""
    cfg = _fake_script_config()

    def boom(url):
        raise HTTPError(url, 500, "boom", {}, None)  # type: ignore[arg-type]

    with (
        mock.patch.object(harmonizome, "_load_config", return_value=cfg),
        mock.patch.object(harmonizome, "urlopen", side_effect=boom),
        pytest.raises(RuntimeError, match="Error downloading"),
    ):
        list(
            harmonizome.Harmonizome.download(
                datasets=["GTEx Tissue Gene Expression Profiles"],
                what=["gene_set_library_crisp.gmt.gz"],
            )
        )


def test_legacy_harmonizome_download_skips_http_errors_in_default_what(monkeypatch) -> None:
    """Without an explicit ``what``, HTTP errors are quietly skipped."""
    cfg = _fake_script_config()

    def boom(url):
        raise HTTPError(url, 404, "missing", {}, None)  # type: ignore[arg-type]

    with (
        mock.patch.object(harmonizome, "_load_config", return_value=cfg),
        mock.patch.object(harmonizome, "urlopen", side_effect=boom),
    ):
        filenames = list(
            harmonizome.Harmonizome.download(
                datasets=["GTEx Tissue Gene Expression Profiles"],
            )
        )
    assert filenames == []


# ---------------------------------------------------------------------------
# get_gmt — force=2 (redownload) + parquet error path
# ---------------------------------------------------------------------------


def test_get_gmt_force_2_redownloads(tmp_path) -> None:
    """``force=2`` skips the cache check and calls ``download_datasets``."""
    cfg = _fake_script_config()
    dataset = "GTEx Tissue Gene Expression Profiles"

    # Pre-seed cache; force=2 should still redownload anyway.
    ds_dir = tmp_path / dataset
    ds_dir.mkdir(parents=True)
    (ds_dir / "gene_set_library_crisp.gmt.gz").write_bytes(gzip.compress(_gmt_body().encode()))

    download_called: dict[str, int] = {"n": 0}

    def fake_download(**kwargs):
        download_called["n"] += 1
        # Write a fresh file as if download succeeded.
        f = Path(kwargs["cache_dir"]) / dataset / "gene_set_library_crisp.gmt.gz"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(gzip.compress(_gmt_body().encode()))
        return {"gene_set_library_crisp.gmt.gz": str(f)}

    with (
        mock.patch.object(harmonizome, "_load_config", return_value=cfg),
        mock.patch.object(
            harmonizome,
            "list_datasets",
            return_value=pd.DataFrame({"name": [dataset], "href": ["/x"]}),
        ),
        mock.patch.object(
            harmonizome, "get_dataset_metadata", return_value=_fake_dataset_metadata()
        ),
        mock.patch.object(harmonizome, "download_datasets", side_effect=fake_download),
    ):
        out = harmonizome.get_gmt(
            datasets=[dataset],
            cache_dir=tmp_path,
            save_path=False,
            force=2,
            verbose=0,
        )
    assert download_called["n"] == 1
    assert not out.empty


def test_get_gmt_falls_back_on_corrupt_cached_parquet(tmp_path, caplog) -> None:
    """When the cached merged parquet is unreadable, we rebuild rather than crash."""
    cfg = _fake_script_config()
    dataset = "GTEx Tissue Gene Expression Profiles"

    # Seed a corrupt parquet so the fast-path read raises.
    bad_parquet = tmp_path / "gmt_merged.parquet"
    bad_parquet.write_bytes(b"not a parquet file")

    # Seed a usable cached GMT so the rebuild succeeds.
    ds_dir = tmp_path / dataset
    ds_dir.mkdir(parents=True)
    (ds_dir / "gene_set_library_crisp.gmt.gz").write_bytes(gzip.compress(_gmt_body().encode()))

    with (
        mock.patch.object(harmonizome, "_load_config", return_value=cfg),
        mock.patch.object(
            harmonizome,
            "list_datasets",
            return_value=pd.DataFrame({"name": [dataset], "href": ["/x"]}),
        ),
        mock.patch.object(
            harmonizome, "get_dataset_metadata", return_value=_fake_dataset_metadata()
        ),
    ):
        out = harmonizome.get_gmt(
            datasets=[dataset],
            cache_dir=tmp_path,
            save_path=str(bad_parquet),
            verbose=2,
        )
    assert not out.empty


# ---------------------------------------------------------------------------
# load_gene_attribute_matrix — gz path + tissue metadata
# ---------------------------------------------------------------------------


def test_load_gene_attribute_matrix_gz_path(tmp_path) -> None:
    """File only present as ``.gz``; loader should gunzip and parse."""
    dataset = "demo"
    ds_dir = tmp_path / dataset
    ds_dir.mkdir(parents=True)
    gz = ds_dir / "gene_attribute_matrix.txt.gz"
    gz.write_bytes(gzip.compress(_gene_attr_matrix_text().encode("utf-8")))
    df, _, _ = harmonizome.load_gene_attribute_matrix(dataset_name=dataset, cache_dir=tmp_path)
    assert set(df["GeneSym"]) == {"BRCA1", "TP53"}


def test_load_gene_attribute_matrix_column_metadata_from_header_comments(tmp_path) -> None:
    """Commented metadata rows BEFORE the header are collected into ``column_metadata``."""
    body = (
        "# desc\tcomment\tCellLine\tCL1\tCL2\n"
        "# desc\tcomment\tTissue\tbrain\tliver\n"
        "GeneSym\tNA\tGeneID\tBRAIN\tLIVER\n"
        "BRCA1\tna\t672\t0.1\t0.4\n"
    )
    dataset = "demo"
    ds_dir = tmp_path / dataset
    ds_dir.mkdir(parents=True)
    (ds_dir / "gene_attribute_matrix.txt").write_text(body)
    df, _, col_meta = harmonizome.load_gene_attribute_matrix(
        dataset_name=dataset, cache_dir=tmp_path, include_col_metadata=True
    )
    assert col_meta is not None
    assert set(col_meta.columns) >= {"CellLine", "Tissue"}


# ---------------------------------------------------------------------------
# _detect_attribute_start — edge cases
# ---------------------------------------------------------------------------


def test_detect_attribute_start_short_header() -> None:
    """A header too short to satisfy the lookahead falls back to 3."""
    assert harmonizome._detect_attribute_start(["GeneSym", "NA"]) == 3


# ---------------------------------------------------------------------------
# Live integration tests — RUN BY DEFAULT in CI.
#
# Maayan-Lab's Harmonizome API is fast (~1 s) and the per-dataset metadata
# files are tiny (a few KB). The full ``get_gmt()`` flow downloads many MB
# so we don't run it here; instead we exercise the smaller live surface
# (config endpoint + dataset listing + per-dataset metadata).
# ---------------------------------------------------------------------------


def test_config_lazy_load_against_live_api() -> None:
    """The /dark/script_config endpoint returns the canonical
    ``downloads`` + ``datasets`` mappings."""
    downloads = harmonizome.DOWNLOADS
    assert isinstance(downloads, list)
    assert any("gmt" in d for d in downloads), (
        f"No GMT-shaped downloadables in DOWNLOADS — got {downloads[:5]}"
    )

    dataset_to_path = harmonizome.DATASET_TO_PATH
    assert isinstance(dataset_to_path, dict)
    assert len(dataset_to_path) > 50, (
        f"Only {len(dataset_to_path)} datasets — Harmonizome catalog is ~114."
    )


def test_list_datasets_returns_real_catalog() -> None:
    """The /api/1.0/dataset paginated endpoint returns the full catalog."""
    df = harmonizome.list_datasets(as_df=True)
    assert isinstance(df, pd.DataFrame)
    assert {"name", "href"}.issubset(df.columns)
    assert len(df) > 50, f"Got {len(df)} datasets — pagination broken?"
    # The catalog should include well-known datasets.
    names = set(df["name"])
    assert any("GTEx" in n for n in names), "GTEx not in Harmonizome catalog?"


def test_get_dataset_metadata_against_real_dataset() -> None:
    """Fetch metadata for a well-known dataset and verify the schema."""
    # CCLE Cell Line Gene Expression Profiles is a stable dataset in
    # Harmonizome since the original publication.
    metadata = harmonizome.get_dataset_metadata("GTEx Tissue Gene Expression Profiles")
    assert isinstance(metadata, dict)
    assert set(metadata.keys()) == {
        "description",
        "measurement",
        "association",
        "category",
        "resource",
        "citations",
        "last_updated",
        "stats",
    }
    # At least the description should be populated for a real public dataset.
    assert metadata["description"] != "", (
        "GTEx metadata returned an empty description — fetch probably failed silently."
    )


def test_download_single_gmt_file_from_live_server(tmp_path) -> None:
    """End-to-end: download ONE real ``.gmt.gz`` file from a Harmonizome
    dataset and verify the file is a non-empty gene-set library.

    Try a few (dataset, downloadable) combos and stop on the first
    successful fetch. Not every dataset hosts every downloadable, so the
    test stays resilient to upstream changes in which combos exist —
    but it still has to fetch at least one real file.
    """
    cfg = harmonizome._load_config()
    dataset_to_path = cfg.get("datasets", {})
    gmt_filenames = [d for d in cfg.get("downloads", []) if d.endswith(".gmt.gz")]
    assert gmt_filenames, "No .gmt.gz downloadables advertised by Harmonizome config."

    # Try a handful of historically stable datasets with the GMT files
    # most commonly hosted across Harmonizome.
    candidate_datasets = [
        "Achilles Cell Line Gene Essentiality Profiles",
        "CCLE Cell Line Gene CNV Profiles",
        "CCLE Cell Line Gene Expression Profiles",
        "GTEx Tissue Gene Expression Profiles",
        "HPA Cell Line Gene Expression Profiles",
    ]
    candidate_files = [
        "gene_set_library_crisp.gmt.gz",
        "gene_set_library_up_crisp.gmt.gz",
        "gene_set_library_dn_crisp.gmt.gz",
        "attribute_set_library_crisp.gmt.gz",
    ]

    successes: dict[str, str] = {}
    tried: list[str] = []
    for ds_name in candidate_datasets:
        if ds_name not in dataset_to_path:
            continue
        for fname in candidate_files:
            tried.append(f"({ds_name}, {fname})")
            paths = harmonizome.download_datasets(
                selected_datasets=[(ds_name, dataset_to_path[ds_name])],
                selected_downloads=[fname],
                cache_dir=tmp_path,
                verbose=False,
            )
            if paths:
                successes = paths
                break
        if successes:
            break

    assert successes, (
        "Could not download any GMT from Harmonizome across all candidates: "
        + ", ".join(tried[:8])
        + ". Either the catalog changed or upstream is down."
    )
    fname = next(iter(successes))
    path = Path(successes[fname])
    assert path.exists()
    assert path.stat().st_size > 100
