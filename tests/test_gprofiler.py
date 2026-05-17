"""Tests for :mod:`biodb.gprofiler`."""

from __future__ import annotations

import io
import zipfile
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


def _gmt_zip_bytes() -> bytes:
    """A zip archive containing exactly one .gmt entry — mirrors upstream."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("gprofiler_full_hsapiens.name.gmt", _gmt_text())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_module_imports_offline() -> None:
    assert gprofiler.__name__ == "biodb.gprofiler"


def test_constants_present() -> None:
    assert gprofiler.GPROFILER_GMT_URL_TEMPLATE.endswith(".gmt.zip")
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
        mock_resp.add(responses.GET, url, body=_gmt_zip_bytes(), status=200)
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
        mock_resp.add(responses.GET, url, body=_gmt_zip_bytes(), status=200)
        path = gprofiler.download_gmt(organism="hsapiens", cache_dir=tmp_path, force=True)
    assert path.read_text().startswith("GO:")


def test_download_gmt_propagates_http_error(tmp_path) -> None:
    url = gprofiler.GPROFILER_GMT_URL_TEMPLATE.format(organism="hsapiens")
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(responses.GET, url, status=404)
        with pytest.raises(requests.HTTPError):
            gprofiler.download_gmt(organism="hsapiens", cache_dir=tmp_path)


def test_download_gmt_raises_when_archive_has_no_gmt(tmp_path) -> None:
    """Mirror an upstream archive shape change — no ``.gmt`` entry inside the zip."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("readme.txt", "no gmt here")
    url = gprofiler.GPROFILER_GMT_URL_TEMPLATE.format(organism="hsapiens")
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(responses.GET, url, body=buf.getvalue(), status=200)
        with pytest.raises(RuntimeError, match="No .gmt entry"):
            gprofiler.download_gmt(organism="hsapiens", cache_dir=tmp_path)


def test_download_gmt_organism_argument_threads_through(tmp_path) -> None:
    url = gprofiler.GPROFILER_GMT_URL_TEMPLATE.format(organism="mmusculus")
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(responses.GET, url, body=_gmt_zip_bytes(), status=200)
        path = gprofiler.download_gmt(organism="mmusculus", cache_dir=tmp_path)
    assert path.name == "gprofiler_full_mmusculus.name.gmt"


def test_download_gmt_falls_back_to_default_cache(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(gprofiler, "CACHE_DIR", tmp_path)
    url = gprofiler.GPROFILER_GMT_URL_TEMPLATE.format(organism="hsapiens")
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(responses.GET, url, body=_gmt_zip_bytes(), status=200)
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


@pytest.mark.xfail(
    reason=(
        "The legacy gprofiler GMT URL pattern "
        "``biit.cs.ut.ee/gprofiler/static/gprofiler_full_<organism>.name.gmt.zip`` "
        "now returns 404 — gProfiler migrated to a Vue SPA in 2026 and "
        "the bulk-download path needs rediscovery. Tracked as a follow-up; "
        "the test stays so we notice the day they restore (or we fix) the URL."
    ),
    strict=False,
)
def test_download_gmt_live(tmp_path) -> None:
    """Download the per-organism GMT and verify it parses. **Currently xfailed**
    because the upstream URL pattern is dead — see the marker above."""
    path = gprofiler.download_gmt(organism="hsapiens", cache_dir=tmp_path, force=True)
    assert path.exists()
    size = path.stat().st_size
    assert size > 100_000, (
        f"GMT file is only {size} bytes — upstream probably returned an error page."
    )
    assert path.suffix == ".gmt"
    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    assert "\t" in first_line
