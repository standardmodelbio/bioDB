"""Tests for :mod:`biodb.gprofiler`."""

from __future__ import annotations

from unittest import mock

import pandas as pd
import pytest
import requests
import responses

from biodb import gprofiler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gmt_text() -> str:
    """Tiny but realistic GMT body — two pathways, five genes total."""
    return (
        "GO:0006915\tapoptotic process\tBAX\tBCL2\tCASP3\n"
        "GO:0008283\tcell population proliferation\tMKI67\tPCNA\n"
    )


def _gmt_bytes() -> bytes:
    """Plain ``.gmt`` body — gProfiler stopped zipping the combined file
    when they migrated the static-asset URL pattern in 2026."""
    return _gmt_text().encode("utf-8")


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_module_imports_offline() -> None:
    assert gprofiler.__name__ == "biodb.gprofiler"


def test_constants_present() -> None:
    assert gprofiler.GPROFILER_GMT_URL_TEMPLATE.endswith(".gmt")
    assert "{organism}" in gprofiler.GPROFILER_GMT_URL_TEMPLATE
    assert gprofiler.GPROFILER_REST_API.endswith("/gost/profile/")
    assert gprofiler.CACHE_DIR.exists()


def test_public_api_signatures_stable() -> None:
    for name in ("download_gmt", "load_gmt", "gost"):
        assert hasattr(gprofiler, name)


# ---------------------------------------------------------------------------
# download_gmt — happy path, cache, force, errors
# ---------------------------------------------------------------------------


def test_download_gmt_writes_unzipped_file(tmp_path) -> None:
    url = gprofiler.GPROFILER_GMT_URL_TEMPLATE.format(organism="hsapiens")
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(responses.GET, url, body=_gmt_bytes(), status=200)
        path = gprofiler.download_gmt(organism="hsapiens", cache_dir=tmp_path)
    assert path.exists()
    assert path.name == "gprofiler_full_hsapiens.name.gmt"
    assert path.read_text().startswith("GO:")


def test_download_gmt_returns_cached_path(tmp_path) -> None:
    cached = tmp_path / "gprofiler_full_hsapiens.name.gmt"
    cached.write_text(_gmt_text())
    with responses.RequestsMock() as mock_resp:
        path = gprofiler.download_gmt(organism="hsapiens", cache_dir=tmp_path, force=False)
    assert path == cached
    assert len(mock_resp.calls) == 0


def test_download_gmt_force_overrides_cache(tmp_path) -> None:
    cached = tmp_path / "gprofiler_full_hsapiens.name.gmt"
    cached.write_text("OLD CONTENTS")
    url = gprofiler.GPROFILER_GMT_URL_TEMPLATE.format(organism="hsapiens")
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(responses.GET, url, body=_gmt_bytes(), status=200)
        path = gprofiler.download_gmt(organism="hsapiens", cache_dir=tmp_path, force=True)
    assert path.read_text().startswith("GO:")


def test_download_gmt_propagates_http_error(tmp_path) -> None:
    url = gprofiler.GPROFILER_GMT_URL_TEMPLATE.format(organism="hsapiens")
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(responses.GET, url, status=404)
        with pytest.raises(requests.HTTPError):
            gprofiler.download_gmt(organism="hsapiens", cache_dir=tmp_path)


def test_download_gmt_organism_argument_threads_through(tmp_path) -> None:
    url = gprofiler.GPROFILER_GMT_URL_TEMPLATE.format(organism="mmusculus")
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(responses.GET, url, body=_gmt_bytes(), status=200)
        path = gprofiler.download_gmt(organism="mmusculus", cache_dir=tmp_path)
    assert path.name == "gprofiler_full_mmusculus.name.gmt"


def test_download_gmt_falls_back_to_default_cache(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(gprofiler, "CACHE_DIR", tmp_path)
    url = gprofiler.GPROFILER_GMT_URL_TEMPLATE.format(organism="hsapiens")
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(responses.GET, url, body=_gmt_bytes(), status=200)
        path = gprofiler.download_gmt(organism="hsapiens")
    assert path.parent == tmp_path


# ---------------------------------------------------------------------------
# load_gmt — exercise both pandas + dict return formats
# ---------------------------------------------------------------------------


def test_load_gmt_pandas(tmp_path) -> None:
    cached = tmp_path / "gprofiler_full_hsapiens.name.gmt"
    cached.write_text(_gmt_text())
    df = gprofiler.load_gmt(cache_dir=tmp_path, return_format="pandas")
    assert isinstance(df, pd.DataFrame)
    assert {"id", "label", "gene"} <= set(df.columns)
    # 3 + 2 genes across two pathways.
    assert len(df) == 5


def test_load_gmt_dict(tmp_path) -> None:
    cached = tmp_path / "gprofiler_full_hsapiens.name.gmt"
    cached.write_text(_gmt_text())
    d = gprofiler.load_gmt(cache_dir=tmp_path, return_format="dict")
    assert isinstance(d, dict)
    assert ("GO:0006915", "apoptotic process") in d
    assert d[("GO:0008283", "cell population proliferation")] == ["MKI67", "PCNA"]


# ---------------------------------------------------------------------------
# gost — REST fallback path (no gprofiler-official installed in CI)
# ---------------------------------------------------------------------------


def _force_no_gprofiler(monkeypatch) -> None:
    """Force ``import gprofiler`` to raise ``ImportError``.

    ``setitem(sys.modules, "gprofiler", None)`` is the canonical recipe — the
    import system treats a ``None`` entry as a poisoned name and re-raises
    ``ImportError`` for every subsequent ``import gprofiler``.
    """
    import sys

    monkeypatch.setitem(sys.modules, "gprofiler", None)


def test_gost_rest_fallback_returns_dataframe(monkeypatch) -> None:
    """When ``gprofiler-official`` isn't installed, fall back to a plain POST."""
    _force_no_gprofiler(monkeypatch)
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(
            responses.POST,
            gprofiler.GPROFILER_REST_API,
            json={"result": [{"source": "GO:BP", "name": "apoptosis", "p_value": 1e-9}]},
            status=200,
        )
        df = gprofiler.gost(query=["BAX", "BCL2"], sources=["GO:BP"], extra_kwarg="forwarded")
        # ``sources`` and ``extra_kwarg`` should be folded into the JSON body.
        # Read the call body before the context exits and clears it.
        sent = mock_resp.calls[0].request.body
        if isinstance(sent, (bytes, bytearray)):
            sent = sent.decode("utf-8")
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, 3)
    assert "BAX" in sent
    assert "GO:BP" in sent
    assert "extra_kwarg" in sent


def test_gost_rest_fallback_empty_result(monkeypatch) -> None:
    _force_no_gprofiler(monkeypatch)
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(responses.POST, gprofiler.GPROFILER_REST_API, json={}, status=200)
        df = gprofiler.gost(query=["foo"])
    assert df.empty


def test_gost_uses_gprofiler_official_when_available(monkeypatch) -> None:
    """When the official client is importable, we delegate to it."""
    expected = pd.DataFrame([{"source": "REAC", "name": "via package"}])
    fake_client = mock.MagicMock()
    fake_client.profile.return_value = expected
    fake_module = mock.MagicMock(GProfiler=mock.MagicMock(return_value=fake_client))
    import sys

    monkeypatch.setitem(sys.modules, "gprofiler", fake_module)
    df = gprofiler.gost(query=["FOO"], organism="hsapiens", sources=["REAC"])
    fake_client.profile.assert_called_once()
    pd.testing.assert_frame_equal(df, expected)


# ---------------------------------------------------------------------------
# Live integration tests — RUN BY DEFAULT in CI.
# ---------------------------------------------------------------------------


def test_gost_against_live_server() -> None:
    """Hit g:Profiler's live ``/gost`` endpoint with a small gene list and
    verify we get back an enrichment DataFrame with the documented schema.

    ``gost`` is a tiny POST + JSON response (~10 KB), so it's the cheapest
    live probe and verifies the functional-enrichment surface works."""
    pytest.importorskip("gprofiler")
    # BRCA1/BRCA2/TP53 are well-studied tumour suppressors. They reliably
    # produce significant enrichment hits.
    result = gprofiler.gost(query=["BRCA1", "BRCA2", "TP53"], organism="hsapiens")
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0, (
        "g:Profiler returned 0 hits for BRCA1/BRCA2/TP53 — upstream likely changed."
    )
    for col in ("name", "p_value", "source"):
        assert col in result.columns, (
            f"Documented column {col!r} missing from {list(result.columns)}"
        )
    # We set ``significant=True`` by default → every row should be < 0.05.
    assert (result["p_value"] < 0.05).all()


def test_download_gmt_live(tmp_path) -> None:
    """Download the real combined per-organism ``.gmt`` from g:Profiler's
    new (post-2026 SPA migration) URL pattern.

    hsapiens is ~41 MB; we accept the per-CI cost because this is the
    only test that proves the download flow still works end-to-end.
    """
    path = gprofiler.download_gmt(organism="hsapiens", cache_dir=tmp_path, force=True)
    assert path.exists()
    size = path.stat().st_size
    # Real combined GMT is tens of MB; anything < 1 MB is an error page.
    assert size > 1_000_000, (
        f"GMT file is only {size} bytes — upstream probably returned an error page."
    )
    assert path.suffix == ".gmt"
    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    assert "\t" in first_line, f"GMT first line doesn't look tab-separated: {first_line!r}"
