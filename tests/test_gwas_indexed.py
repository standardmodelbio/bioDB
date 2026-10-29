"""Bounded indexed Catalog retrieval: strict identity, coverage and fallback."""

import sys
from types import SimpleNamespace

import pytest
import requests

from biodb import gwas_catalog
from biodb._acquisition import IngestError

HEADER = (
    "#chromosome\tbase_pair_location\teffect_allele\tother_allele\tbeta\tstandard_error\tp_value"
)
ROW = "1\t100\tG\tA\t0.123456789123456\tNA\t0"


def remote():
    return {
        "source": "gwas_catalog",
        "study_id": "GCST1",
        "file_id": "remote",
        "url": "https://source.test/GCST1.h.tsv.gz",
    }


def test_indexed_ranges_preserve_raw_and_never_download_whole_source(tmp_path, monkeypatch):
    queries = []

    class Tabix:
        header = [HEADER]
        contigs = ["1"]

        def __init__(self, url, index):
            assert url == remote()["url"]
            assert index == url + ".tbi"

        def fetch(self, chrom, start, end):
            queries.append((chrom, start, end))
            yield ROW

        def close(self):
            pass

    def head(url, **kwargs):
        response = requests.Response()
        response._content = b""
        response._content_consumed = True
        response.status_code = 200
        response.headers = {
            "ETag": '"index-v1"' if url.endswith(".tbi") else '"source-v1"',
            "Content-Length": "1000",
        }
        return response

    monkeypatch.setitem(sys.modules, "pysam", SimpleNamespace(TabixFile=Tabix))
    monkeypatch.setattr(requests, "head", head)
    monkeypatch.setattr(requests, "get", lambda *_a, **_k: pytest.fail("Whole source GET"))
    result = gwas_catalog.ingest_sumstats(
        [remote()],
        tmp_path / "run",
        scope="regions",
        regions=["1:90-110", "1:100-200"],
        cache_dir=tmp_path / "cache",
    )
    assert queries == [("1", 89, 200)]
    assert result.rows == 1
    frame = result.scan().collect()
    assert frame["beta"][0] == 0.123456789123456
    assert frame["source_sha256"][0] is None
    assert frame["region_sha256"][0]
    assert frame["source_row"][0] is None
    assert result.files[0]["transport"] == "tabix"
    assert result.files[0]["source_etag"] == '"source-v1"'


def test_index_probe_transient_failure_does_not_fallback(tmp_path, monkeypatch):
    def head(*_a, **_k):
        response = requests.Response()
        response._content = b""
        response._content_consumed = True
        response.status_code = 503
        return response

    monkeypatch.setattr(requests, "head", head)
    monkeypatch.setattr(
        requests,
        "get",
        lambda *_a, **_k: pytest.fail("Transient index failure must not GET source"),
    )
    with pytest.raises(IngestError) as error:
        gwas_catalog.ingest_sumstats(
            [remote()], tmp_path / "run", scope="regions", regions=["1:1-200"]
        )
    assert error.value.result.failed == 1


def test_source_change_during_tabix_fetch_fails_without_output(tmp_path, monkeypatch):
    state = {"changed": False}

    class Tabix:
        header = [HEADER]
        contigs = ["1"]

        def __init__(self, *_a, **_k):
            pass

        def fetch(self, *_a):
            yield ROW
            state["changed"] = True

        def close(self):
            pass

    def head(url, **kwargs):
        response = requests.Response()
        response._content = b""
        response._content_consumed = True
        response.status_code = 200
        response.headers = {
            "ETag": '"v2"' if state["changed"] else '"v1"',
            "Content-Length": "1000",
        }
        return response

    monkeypatch.setitem(sys.modules, "pysam", SimpleNamespace(TabixFile=Tabix))
    monkeypatch.setattr(requests, "head", head)
    monkeypatch.setattr(requests, "get", lambda *_a, **_k: pytest.fail("Whole source GET"))
    with pytest.raises(IngestError) as error:
        gwas_catalog.ingest_sumstats(
            [remote()],
            tmp_path / "run",
            scope="regions",
            regions=["1:1-200"],
            cache_dir=tmp_path / "cache",
        )
    assert error.value.result.rows == 0
    assert "identity changed" in error.value.result.files[0]["detail"]
