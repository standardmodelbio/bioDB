"""Tests for :mod:`biodb.pubmed` — NCBI E-utilities (API) + bulk XML.

Mixed mocked + live, same pattern as ``test_ols.py`` and
``test_gprofiler.py``:

* **Mocked unit tests** — XML parser on synthetic fixtures, URL shape,
  rate-limit retry-on-429, MD5 verification (both success and
  mismatch), bulk directory listing.
* **Live integration tests** — small E-utilities round-trip on the
  ``pubmed`` database; HEAD probe on a real baseline ``.xml.gz`` URL
  (no download — the smallest baseline shard is ~20 MB).
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import requests
import responses

from biodb import pubmed

# ---------------------------------------------------------------------------
# Synthetic XML fixtures
# ---------------------------------------------------------------------------

_MINIMAL_ARTICLE_XML = b"""<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID Version="1">99999999</PMID>
      <Article>
        <Journal>
          <ISOAbbreviation>J Test</ISOAbbreviation>
          <JournalIssue>
            <PubDate><Year>2026</Year></PubDate>
          </JournalIssue>
        </Journal>
        <ArticleTitle>A tiny synthetic test article.</ArticleTitle>
        <Abstract>
          <AbstractText Label="AIM">Verify the parser.</AbstractText>
          <AbstractText Label="RESULTS">It works.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author><LastName>Doe</LastName><Initials>J</Initials></Author>
          <Author><LastName>Smith</LastName><Initials>A</Initials></Author>
          <Author><CollectiveName>The Test Consortium</CollectiveName></Author>
        </AuthorList>
      </Article>
      <MeshHeadingList>
        <MeshHeading><DescriptorName>Apoptosis</DescriptorName></MeshHeading>
        <MeshHeading><DescriptorName>Humans</DescriptorName></MeshHeading>
      </MeshHeadingList>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">99999999</ArticleId>
        <ArticleId IdType="doi">10.1234/test</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
"""


def _esearch_payload(pmids: list[str], total: int | None = None) -> dict:
    return {
        "esearchresult": {
            "count": str(total if total is not None else len(pmids)),
            "retmax": str(len(pmids)),
            "retstart": "0",
            "idlist": pmids,
            "querytranslation": "test query translation",
        }
    }


def _esummary_payload(records: list[dict]) -> dict:
    """Build the nested ``{result: {uids: [...], <uid>: {...}}}`` shape."""
    uids = [r["uid"] for r in records]
    payload: dict = {"result": {"uids": uids}}
    for r in records:
        payload["result"][r["uid"]] = r
    return payload


# ---------------------------------------------------------------------------
# Mocked unit tests — XML parser
# ---------------------------------------------------------------------------


def test_parse_pubmed_xml_extracts_documented_fields(tmp_path) -> None:
    xml_path = tmp_path / "tiny.xml"
    xml_path.write_bytes(_MINIMAL_ARTICLE_XML)
    df = pubmed.parse_pubmed_xml(xml_path)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["pmid"] == "99999999"
    assert row["title"].startswith("A tiny synthetic")
    assert "AIM: Verify the parser." in row["abstract"]
    assert "RESULTS: It works." in row["abstract"]
    assert row["authors"] == ["Doe J", "Smith A", "The Test Consortium"]
    assert row["journal"] == "J Test"
    assert row["pub_year"] == "2026"
    assert row["doi"] == "10.1234/test"
    assert row["mesh_terms"] == ["Apoptosis", "Humans"]


def test_parse_pubmed_xml_handles_gzip_input(tmp_path) -> None:
    """The parser must transparently open ``.xml.gz`` shards — that's the
    whole shape of the bulk-download corpus."""
    gz_path = tmp_path / "tiny.xml.gz"
    with gzip.open(gz_path, "wb") as f:
        f.write(_MINIMAL_ARTICLE_XML)
    df = pubmed.parse_pubmed_xml(gz_path)
    assert len(df) == 1
    assert df.iloc[0]["pmid"] == "99999999"


def test_parse_pubmed_xml_handles_multi_article_stream(tmp_path) -> None:
    """``iterparse`` should yield one row per ``PubmedArticle`` element."""
    multi = (
        b"<?xml version='1.0'?><PubmedArticleSet>"
        + (
            b"<PubmedArticle><MedlineCitation><PMID>1</PMID>"
            b"<Article><ArticleTitle>One</ArticleTitle></Article></MedlineCitation></PubmedArticle>"
        )
        + (
            b"<PubmedArticle><MedlineCitation><PMID>2</PMID>"
            b"<Article><ArticleTitle>Two</ArticleTitle></Article></MedlineCitation></PubmedArticle>"
        )
        + b"</PubmedArticleSet>"
    )
    path = tmp_path / "two.xml"
    path.write_bytes(multi)
    df = pubmed.parse_pubmed_xml(path)
    assert len(df) == 2
    assert list(df["pmid"]) == ["1", "2"]
    assert list(df["title"]) == ["One", "Two"]


def test_parse_pubmed_xml_raises_on_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        pubmed.parse_pubmed_xml(tmp_path / "does_not_exist.xml")


# ---------------------------------------------------------------------------
# Mocked unit tests — E-utilities transport
# ---------------------------------------------------------------------------


def test_search_pings_esearch_with_pubmed_db() -> None:
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(
            responses.GET,
            f"{pubmed.NCBI_EUTILS_BASE_URL}/esearch.fcgi",
            json=_esearch_payload(["42147998", "42147479"], total=25507),
            status=200,
        )
        result = pubmed.search("BRCA1", retmax=2)
        sent_url = mock_resp.calls[0].request.url
    assert result == {
        "pmids": ["42147998", "42147479"],
        "total": 25507,
        "query_translation": "test query translation",
    }
    assert "db=pubmed" in sent_url
    assert "term=BRCA1" in sent_url
    assert "retmax=2" in sent_url
    assert "retmode=json" in sent_url


def test_search_threads_api_key_when_provided() -> None:
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(
            responses.GET,
            f"{pubmed.NCBI_EUTILS_BASE_URL}/esearch.fcgi",
            json=_esearch_payload([]),
            status=200,
        )
        pubmed.search("foo", api_key="secret-key")
        sent_url = mock_resp.calls[0].request.url
    assert "api_key=secret-key" in sent_url


def test_eutils_get_retries_once_on_429(monkeypatch) -> None:
    """The wrapper should sleep + retry exactly once on a 429."""
    sleeps: list[float] = []
    monkeypatch.setattr(pubmed.time, "sleep", lambda s: sleeps.append(s))
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(
            responses.GET,
            f"{pubmed.NCBI_EUTILS_BASE_URL}/esearch.fcgi",
            json={"error": "rate-limited"},
            status=429,
        )
        mock_resp.add(
            responses.GET,
            f"{pubmed.NCBI_EUTILS_BASE_URL}/esearch.fcgi",
            json=_esearch_payload(["1"]),
            status=200,
        )
        result = pubmed.search("foo")
        assert len(mock_resp.calls) == 2
    assert result["pmids"] == ["1"]
    # First sleep is the 429-backoff (1.0s); second is the polite rate-limit sleep.
    assert sleeps[0] >= 1.0
    assert pubmed._RATE_LIMIT_SLEEP_S in sleeps


def test_query_summaries_normalizes_doi_from_articleids() -> None:
    """``esummary`` returns the DOI inside the ``articleids`` array — we
    flatten it into a top-level column."""
    record = {
        "uid": "30700171",
        "title": "Test article",
        "authors": [{"name": "Doe J"}, {"name": "Roe X"}],
        "source": "J Test",
        "pubdate": "2019",
        "epubdate": "2019 Jan 31",
        "articleids": [
            {"idtype": "pubmed", "value": "30700171"},
            {"idtype": "doi", "value": "10.5/test"},
        ],
    }
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(
            responses.GET,
            f"{pubmed.NCBI_EUTILS_BASE_URL}/esummary.fcgi",
            json=_esummary_payload([record]),
            status=200,
        )
        df = pubmed.query_summaries("30700171")
    assert df.shape == (1, 7)
    row = df.iloc[0]
    assert row["pmid"] == "30700171"
    assert row["title"] == "Test article"
    assert row["authors"] == "Doe J, Roe X"
    assert row["doi"] == "10.5/test"


def test_query_summaries_accepts_list_input() -> None:
    """Passing a list of PMIDs should comma-join them in the ``id=`` param."""
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(
            responses.GET,
            f"{pubmed.NCBI_EUTILS_BASE_URL}/esummary.fcgi",
            json=_esummary_payload([]),
            status=200,
        )
        pubmed.query_summaries(["1", "2", "3"])
        sent_url = mock_resp.calls[0].request.url
    assert "id=1%2C2%2C3" in sent_url or "id=1,2,3" in sent_url


def test_query_summaries_returns_empty_frame_for_empty_input() -> None:
    """No PMIDs → no HTTP call, just the documented zero-row DataFrame."""
    df = pubmed.query_summaries([])
    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert {"pmid", "title", "doi"} <= set(df.columns)


def test_query_pmid_raises_for_unknown(monkeypatch) -> None:
    """When ``esummary`` returns no records, ``query_pmid`` should
    surface a ``KeyError`` (not a silent empty)."""
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(
            responses.GET,
            f"{pubmed.NCBI_EUTILS_BASE_URL}/esummary.fcgi",
            json={"result": {"uids": []}},
            status=200,
        )
        with pytest.raises(KeyError, match="not found"):
            pubmed.query_pmid("99999999999")


def test_query_abstract_parses_efetch_xml() -> None:
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(
            responses.GET,
            f"{pubmed.NCBI_EUTILS_BASE_URL}/efetch.fcgi",
            body=_MINIMAL_ARTICLE_XML,
            status=200,
            content_type="application/xml",
        )
        record = pubmed.query_abstract("99999999")
    assert record["pmid"] == "99999999"
    assert record["doi"] == "10.1234/test"
    assert "AIM: Verify the parser." in record["abstract"]


def test_query_abstract_raises_on_empty_response() -> None:
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(
            responses.GET,
            f"{pubmed.NCBI_EUTILS_BASE_URL}/efetch.fcgi",
            body=b"<?xml version='1.0'?><PubmedArticleSet/>",
            status=200,
            content_type="application/xml",
        )
        with pytest.raises(KeyError, match="not found"):
            pubmed.query_abstract("99999999999")


def test_query_abstracts_empty_input_skips_http() -> None:
    """No PMIDs in → zero HTTP calls and an empty list out. Same shape
    as ``query_summaries([])``."""
    out = pubmed.query_abstracts([])
    assert out == []


def test_query_abstracts_chunks_by_batch_size() -> None:
    """Three PMIDs at ``batch_size=2`` → two efetch calls with comma-joined
    ids in each request URL."""
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(
            responses.GET,
            f"{pubmed.NCBI_EUTILS_BASE_URL}/efetch.fcgi",
            body=_MINIMAL_ARTICLE_XML,
            status=200,
            content_type="application/xml",
        )
        mock_resp.add(
            responses.GET,
            f"{pubmed.NCBI_EUTILS_BASE_URL}/efetch.fcgi",
            body=_MINIMAL_ARTICLE_XML,
            status=200,
            content_type="application/xml",
        )
        records = pubmed.query_abstracts(["1", "2", "3"], batch_size=2)
        assert len(mock_resp.calls) == 2
        first_url = mock_resp.calls[0].request.url
        second_url = mock_resp.calls[1].request.url
    # First batch: PMIDs 1+2 comma-joined.
    assert "id=1%2C2" in first_url or "id=1,2" in first_url
    # Second batch: just PMID 3.
    assert "id=3" in second_url
    # Two articles total (one per mocked response — each fixture has one record).
    assert len(records) == 2
    assert all(isinstance(r, dict) and r.get("pmid") for r in records)


def test_query_abstracts_threads_api_key() -> None:
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(
            responses.GET,
            f"{pubmed.NCBI_EUTILS_BASE_URL}/efetch.fcgi",
            body=_MINIMAL_ARTICLE_XML,
            status=200,
            content_type="application/xml",
        )
        pubmed.query_abstracts(["1"], api_key="DEADBEEF")
        url = mock_resp.calls[0].request.url
    assert "api_key=DEADBEEF" in url


# ---------------------------------------------------------------------------
# Mocked unit tests — bulk listing + MD5 verification
# ---------------------------------------------------------------------------


def test_list_baseline_files_scrapes_directory_index() -> None:
    """Apache-style listing → we extract every ``.xml.gz`` href."""
    html = """
    <html><body>
      <a href="../">Parent Directory</a>
      <a href="README.txt">README.txt</a>
      <a href="pubmed26n0001.xml.gz">pubmed26n0001.xml.gz</a>
      <a href="pubmed26n0001.xml.gz.md5">pubmed26n0001.xml.gz.md5</a>
      <a href="pubmed26n0002.xml.gz">pubmed26n0002.xml.gz</a>
    </body></html>
    """
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(
            responses.GET,
            f"{pubmed.PUBMED_FTP_BASE_URL}/baseline/",
            body=html,
            status=200,
        )
        files = pubmed.list_baseline_files()
    assert files == ["pubmed26n0001.xml.gz", "pubmed26n0002.xml.gz"]


def test_download_baseline_file_rejects_non_gzip_filename(tmp_path) -> None:
    with pytest.raises(ValueError, match="doesn't look like"):
        pubmed.download_baseline_file("readme.txt", cache_dir=tmp_path)


def test_download_baseline_file_caches_and_skips_redownload(tmp_path) -> None:
    """Second call with the same filename should NOT re-hit HTTP."""
    cached = tmp_path / "pubmed26n0001.xml.gz"
    cached.write_bytes(_MINIMAL_ARTICLE_XML)  # pretend we already have it
    with responses.RequestsMock() as mock_resp:
        path = pubmed.download_baseline_file(
            "pubmed26n0001.xml.gz",
            cache_dir=tmp_path,
            verify_md5=False,
        )
        assert len(mock_resp.calls) == 0
    assert path == cached


def test_download_baseline_file_verifies_md5_when_correct(tmp_path) -> None:
    """Happy-path MD5: the downloaded bytes' digest matches the sibling
    ``.md5`` file's content."""
    payload = b"hello pubmed shard"
    md5_hex = hashlib.md5(payload).hexdigest()  # noqa: S324
    md5_body = f"MD5(pubmed26n9999.xml.gz)= {md5_hex}\n"
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(
            responses.GET,
            f"{pubmed.PUBMED_FTP_BASE_URL}/baseline/pubmed26n9999.xml.gz",
            body=payload,
            status=200,
        )
        mock_resp.add(
            responses.GET,
            f"{pubmed.PUBMED_FTP_BASE_URL}/baseline/pubmed26n9999.xml.gz.md5",
            body=md5_body,
            status=200,
        )
        path = pubmed.download_baseline_file(
            "pubmed26n9999.xml.gz", cache_dir=tmp_path, verify_md5=True
        )
    assert path.read_bytes() == payload


def test_download_baseline_file_raises_on_md5_mismatch(tmp_path) -> None:
    """Corrupt download → loud error, not silent acceptance."""
    payload = b"corrupt bytes"
    wrong_md5 = "0" * 32
    md5_body = f"MD5(pubmed26n9999.xml.gz)= {wrong_md5}\n"
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(
            responses.GET,
            f"{pubmed.PUBMED_FTP_BASE_URL}/baseline/pubmed26n9999.xml.gz",
            body=payload,
            status=200,
        )
        mock_resp.add(
            responses.GET,
            f"{pubmed.PUBMED_FTP_BASE_URL}/baseline/pubmed26n9999.xml.gz.md5",
            body=md5_body,
            status=200,
        )
        with pytest.raises(RuntimeError, match="MD5 mismatch"):
            pubmed.download_baseline_file(
                "pubmed26n9999.xml.gz", cache_dir=tmp_path, verify_md5=True
            )


# ---------------------------------------------------------------------------
# Live integration tests — small payloads, real upstreams
# ---------------------------------------------------------------------------


def test_pubmed_search_brca1_finds_thousands_of_hits() -> None:
    """``BRCA1`` is canonical — it should always return well over 10k hits."""
    result = pubmed.search("BRCA1", retmax=5)
    assert len(result["pmids"]) == 5
    assert result["total"] > 10_000
    assert all(p.isdigit() for p in result["pmids"])


def test_pubmed_query_pmid_round_trip() -> None:
    """A known stable PMID round-trips through esummary."""
    record = pubmed.query_pmid(30700171)
    assert record["pmid"] == "30700171"
    assert record["doi"] == "10.1080/02656736.2018.1558289"
    assert "Mouratidis" in record["authors"]


def test_pubmed_query_abstract_extracts_real_abstract() -> None:
    """Full efetch XML round-trip: title + abstract + authors + MeSH."""
    record = pubmed.query_abstract(30700171)
    assert record["pmid"] == "30700171"
    assert "thermal" in record["title"].lower()
    assert len(record["abstract"]) > 500  # the real abstract is ~1.7 KB
    assert any("Mouratidis" in a for a in record["authors"])
    assert "Humans" in record["mesh_terms"]


def test_pubmed_list_baseline_files_returns_thousand_plus_shards() -> None:
    """The annual baseline is ~1,300 ``pubmed{YY}n####.xml.gz`` shards."""
    files = pubmed.list_baseline_files()
    assert len(files) > 1000
    assert files[0].startswith("pubmed") and files[0].endswith(".xml.gz")


def test_pubmed_baseline_shard_url_is_alive() -> None:
    """HEAD a real baseline shard URL — verify the URL pattern still
    works without pulling the 20 MB body."""
    url = f"{pubmed.PUBMED_FTP_BASE_URL}/baseline/pubmed26n0001.xml.gz"
    response = requests.head(url, timeout=15, allow_redirects=True)
    assert response.status_code == 200
    if "content-length" in response.headers:
        size = int(response.headers["content-length"])
        # Baseline shards are tens of MB; anything < 1 MB is an error page.
        assert size > 1_000_000


# ---------------------------------------------------------------------------
# Smoke: ``json`` and ``Path`` are imported via the module surface so
# users following the README examples don't trip over a NameError. Free
# coverage; no real runtime check.
# ---------------------------------------------------------------------------


def test_module_surface_exports_expected_names() -> None:
    for name in (
        "search",
        "query_summaries",
        "query_pmid",
        "query_abstract",
        "list_baseline_files",
        "list_update_files",
        "download_baseline_file",
        "download_update_file",
        "parse_pubmed_xml",
        "CACHE_DIR",
        "PUBMED_FTP_BASE_URL",
        "NCBI_EUTILS_BASE_URL",
    ):
        assert hasattr(pubmed, name), f"biodb.pubmed.{name} should be exported"
    # The CACHE_DIR auto-mkdirs; verify it's a Path on disk.
    assert isinstance(pubmed.CACHE_DIR, Path)


# Make sure the optional `json` import in fixture builders doesn't drift —
# we import it at module top so the synthetic-record helpers don't
# silently start raising NameError if a test imports them in isolation.
_ = json  # noqa: B018 -- ensures the import isn't dead
