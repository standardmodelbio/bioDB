"""Tests for :mod:`biodb.msigdb`."""

from __future__ import annotations

import pandas as pd
import pytest
import requests
import responses

from biodb import msigdb

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_GMT_BODY = (
    "HALLMARK_APOPTOSIS\thttp://example/apoptosis\tBAX\tBCL2\tCASP3\n"
    "HALLMARK_HYPOXIA\thttp://example/hypoxia\tHIF1A\tVEGFA\n"
)


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_module_imports_offline() -> None:
    assert msigdb.__name__ == "biodb.msigdb"


def test_constants_present() -> None:
    assert msigdb.MSIGDB_BASE_URL.endswith("/release")
    assert msigdb.DEFAULT_VERSION.endswith("Hs")
    assert "h.all" in msigdb.KNOWN_COLLECTIONS
    assert "msigdb" in msigdb.KNOWN_COLLECTIONS
    assert msigdb.CACHE_DIR.exists()


def test_public_api_signatures_stable() -> None:
    for name in ("download_gmt", "load_gmt"):
        assert hasattr(msigdb, name)


# ---------------------------------------------------------------------------
# URL/filename builders — pure functions
# ---------------------------------------------------------------------------


def test_gmt_filename_format() -> None:
    fn = msigdb._gmt_filename("msigdb", "2025.1.Hs", "symbols")
    assert fn == "msigdb.v2025.1.Hs.symbols.gmt"


def test_gmt_url_format() -> None:
    url = msigdb._gmt_url("h.all", "2025.1.Hs", "symbols")
    assert url.endswith("/2025.1.Hs/h.all.v2025.1.Hs.symbols.gmt")
    assert url.startswith(msigdb.MSIGDB_BASE_URL + "/")


def test_gmt_filename_entrez_variant() -> None:
    fn = msigdb._gmt_filename("c2.cp", "2024.1.Hs", "entrez")
    assert fn == "c2.cp.v2024.1.Hs.entrez.gmt"


# ---------------------------------------------------------------------------
# download_gmt — happy path, cache, force, error
# ---------------------------------------------------------------------------


def test_download_gmt_writes_file(tmp_path) -> None:
    url = msigdb._gmt_url("msigdb", "2025.1.Hs", "symbols")
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(responses.GET, url, body=_GMT_BODY, status=200)
        path = msigdb.download_gmt(cache_dir=tmp_path)
    assert path.name == "msigdb.v2025.1.Hs.symbols.gmt"
    assert path.read_text() == _GMT_BODY


def test_download_gmt_cache_hit_skips_http(tmp_path) -> None:
    cached = tmp_path / "msigdb.v2025.1.Hs.symbols.gmt"
    cached.write_text(_GMT_BODY)
    with responses.RequestsMock() as mock_resp:
        path = msigdb.download_gmt(cache_dir=tmp_path, force=False)
        assert len(mock_resp.calls) == 0
    assert path == cached


def test_download_gmt_force_redownloads(tmp_path) -> None:
    cached = tmp_path / "msigdb.v2025.1.Hs.symbols.gmt"
    cached.write_text("STALE")
    url = msigdb._gmt_url("msigdb", "2025.1.Hs", "symbols")
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(responses.GET, url, body=_GMT_BODY, status=200)
        path = msigdb.download_gmt(cache_dir=tmp_path, force=True)
    assert path.read_text() == _GMT_BODY


def test_download_gmt_propagates_404(tmp_path) -> None:
    url = msigdb._gmt_url("bogus.collection", "2025.1.Hs", "symbols")
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(responses.GET, url, status=404)
        with pytest.raises(requests.HTTPError):
            msigdb.download_gmt(collection="bogus.collection", cache_dir=tmp_path)


def test_download_gmt_collection_threads_to_url(tmp_path) -> None:
    """The chosen collection name must end up in the request URL."""
    url = msigdb._gmt_url("h.all", "2025.1.Hs", "symbols")
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(responses.GET, url, body=_GMT_BODY, status=200)
        msigdb.download_gmt(collection="h.all", cache_dir=tmp_path)
        assert "h.all.v2025.1.Hs.symbols.gmt" in mock_resp.calls[0].request.url


def test_download_gmt_version_id_type_thread_to_url(tmp_path) -> None:
    url = msigdb._gmt_url("c5.all", "2024.1.Hs", "entrez")
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(responses.GET, url, body=_GMT_BODY, status=200)
        msigdb.download_gmt(
            collection="c5.all", version="2024.1.Hs", id_type="entrez", cache_dir=tmp_path
        )


def test_download_gmt_falls_back_to_module_cache(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(msigdb, "CACHE_DIR", tmp_path)
    url = msigdb._gmt_url("msigdb", "2025.1.Hs", "symbols")
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(responses.GET, url, body=_GMT_BODY, status=200)
        path = msigdb.download_gmt()
    assert path.parent == tmp_path


# ---------------------------------------------------------------------------
# load_gmt — pandas + dict
# ---------------------------------------------------------------------------


def test_load_gmt_pandas(tmp_path) -> None:
    cached = tmp_path / "msigdb.v2025.1.Hs.symbols.gmt"
    cached.write_text(_GMT_BODY)
    df = msigdb.load_gmt(cache_dir=tmp_path, return_format="pandas")
    assert isinstance(df, pd.DataFrame)
    # 3 + 2 genes across two pathways.
    assert len(df) == 5


def test_load_gmt_dict(tmp_path) -> None:
    cached = tmp_path / "msigdb.v2025.1.Hs.symbols.gmt"
    cached.write_text(_GMT_BODY)
    d = msigdb.load_gmt(cache_dir=tmp_path, return_format="dict")
    assert isinstance(d, dict)
    assert len(d) == 2
    apoptosis_key = next(k for k in d if "APOPTOSIS" in k[0])
    assert d[apoptosis_key] == ["BAX", "BCL2", "CASP3"]


# ---------------------------------------------------------------------------
# Live integration tests — RUN BY DEFAULT.
#
# Hallmark (``h.all``) is the smallest MSigDB collection (50 sets, ~50 KB
# unzipped) so it's a cheap "is the downloader still working?" probe.
# ---------------------------------------------------------------------------


def test_download_hallmark_gmt_from_live_server(tmp_path) -> None:
    """Download the real Hallmark GMT and verify the file isn't empty / a
    redirect / an HTML error page."""
    path = msigdb.download_gmt(collection="h.all", cache_dir=tmp_path)
    assert path.exists()
    # Hallmark is ~50 KB; anything under 1 KB is almost certainly an error page.
    size = path.stat().st_size
    assert size > 1024, f"Hallmark GMT was only {size} bytes — looks like an error response."
    # Should start with a gene-set name (Hallmark sets begin with ``HALLMARK_``).
    head = path.read_text(encoding="utf-8")[:80]
    assert head.startswith("HALLMARK_"), f"Unexpected first 80 bytes of hallmark GMT: {head!r}"


def test_load_hallmark_gmt_parses_to_dataframe(tmp_path) -> None:
    """Download + parse Hallmark and check the documented schema. Hallmark
    has exactly 50 gene sets — anything wildly off means upstream changed."""
    df = msigdb.load_gmt(collection="h.all", cache_dir=tmp_path, return_format="pandas")
    assert set(df.columns) == {"id", "label", "gene"}
    n_sets = df["id"].nunique()
    assert 20 < n_sets < 200, f"Hallmark has 50 sets normally; got {n_sets}. Upstream changed?"
    # All Hallmark set IDs start with ``HALLMARK_``.
    assert df["id"].str.startswith("HALLMARK_").all()
