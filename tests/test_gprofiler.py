"""Smoke tests for :mod:`biodb.gprofiler`."""

from __future__ import annotations

import inspect

import pytest

from biodb import gprofiler


def test_module_imports_offline() -> None:
    assert gprofiler.__name__ == "biodb.gprofiler"


def test_constants_present() -> None:
    assert "biit.cs.ut.ee" in gprofiler.GPROFILER_GMT_URL_TEMPLATE
    assert "biit.cs.ut.ee" in gprofiler.GPROFILER_REST_API
    assert gprofiler.CACHE_DIR.exists()


def test_public_api_signatures_stable() -> None:
    for name in ("download_gmt", "load_gmt", "gost"):
        assert hasattr(gprofiler, name)


def test_download_gmt_organism_kwarg() -> None:
    sig = inspect.signature(gprofiler.download_gmt)
    assert sig.parameters["organism"].default == "hsapiens"


@pytest.mark.network
def test_download_gmt_hsapiens(tmp_path) -> None:
    path = gprofiler.download_gmt(organism="hsapiens", cache_dir=tmp_path)
    assert path.exists()
    assert path.stat().st_size > 1000
