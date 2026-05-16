"""Smoke tests for :mod:`biodb.msigdb`."""

from __future__ import annotations

import pytest

from biodb import msigdb


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


def test_gmt_filename_format() -> None:
    fn = msigdb._gmt_filename("msigdb", "2025.1.Hs", "symbols")
    assert fn == "msigdb.v2025.1.Hs.symbols.gmt"


def test_gmt_url_format() -> None:
    url = msigdb._gmt_url("h.all", "2025.1.Hs", "symbols")
    assert url.endswith("/2025.1.Hs/h.all.v2025.1.Hs.symbols.gmt")


@pytest.mark.network
def test_download_hallmark_gmt(tmp_path) -> None:
    path = msigdb.download_gmt(collection="h.all", cache_dir=tmp_path)
    assert path.exists()
    assert path.stat().st_size > 100
