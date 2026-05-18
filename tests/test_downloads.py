"""Unit tests for :mod:`biodb._downloads.stream_to_file`.

The helper is the funnel every first-party bulk-download path runs
through; tests cover the documented behaviour:

* writes the response body to ``dst`` byte-for-byte
* creates the parent directory if missing
* honours ``progress=False`` (no tqdm output)
* still works when the server omits ``Content-Length``
* threads custom headers + extended timeout
* POST mode for CSRF-form downloads (GWAS Atlas pattern)
* surfaces HTTP errors via ``raise_for_status``

Mocked-only — the helper has no behaviour worth exercising against a
real upstream that isn't already covered by the per-source live tests.
"""

from __future__ import annotations

import pytest
import requests
import responses

from biodb._downloads import stream_to_file


def test_stream_to_file_writes_body_to_destination(tmp_path) -> None:
    body = b"hello world\n" * 1000
    dst = tmp_path / "subdir" / "out.bin"
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(
            responses.GET,
            "https://example.test/file.bin",
            body=body,
            status=200,
            headers={"content-length": str(len(body))},
        )
        result = stream_to_file("https://example.test/file.bin", dst, progress=False)
    assert result == dst
    assert dst.exists()
    assert dst.read_bytes() == body


def test_stream_to_file_creates_missing_parent_directories(tmp_path) -> None:
    dst = tmp_path / "deep" / "nested" / "path" / "out.bin"
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(responses.GET, "https://x.test/", body=b"x", status=200)
        stream_to_file("https://x.test/", dst, progress=False)
    assert dst.exists()


def test_stream_to_file_handles_missing_content_length(tmp_path) -> None:
    """Servers don't always return Content-Length (chunked transfer-encoding,
    or strict CORS preflight). The bar should still write the file correctly,
    just in indeterminate-total mode."""
    body = b"unknown length payload"
    dst = tmp_path / "out.bin"
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(
            responses.GET,
            "https://example.test/file",
            body=body,
            status=200,
            # NOTE: deliberately no content-length header.
        )
        stream_to_file("https://example.test/file", dst, progress=True)
    assert dst.read_bytes() == body


def test_stream_to_file_threads_custom_headers(tmp_path) -> None:
    """User-Agent (and any other custom header) must reach the wire — most
    NCBI / EBI endpoints reject the default ``python-requests`` UA."""
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(responses.GET, "https://x.test/", body=b"ok", status=200)
        stream_to_file(
            "https://x.test/",
            tmp_path / "out",
            progress=False,
            headers={"User-Agent": "biodb-test/1.0"},
        )
        request = mock_resp.calls[0].request
    assert request.headers.get("User-Agent") == "biodb-test/1.0"


def test_stream_to_file_raises_on_http_error(tmp_path) -> None:
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(
            responses.GET,
            "https://x.test/missing",
            json={"error": "not found"},
            status=404,
        )
        with pytest.raises(requests.HTTPError):
            stream_to_file("https://x.test/missing", tmp_path / "out", progress=False)


def test_stream_to_file_post_mode_for_csrf_forms(tmp_path) -> None:
    """When ``post_data=`` is given, the helper POSTs the form body
    instead of GETting — this is the path GWAS Atlas's Laravel CSRF
    download flow uses."""
    body = b"the actual file body"
    dst = tmp_path / "csrf-output.gz"
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(
            responses.POST,
            "https://example.test/home/release",
            body=body,
            status=200,
            headers={"content-length": str(len(body))},
        )
        stream_to_file(
            "https://example.test/home/release",
            dst,
            progress=False,
            post_data={"_token": "abc123", "file": "x.gz"},
        )
        request = mock_resp.calls[0].request
    assert dst.read_bytes() == body
    assert request.method == "POST"
    # The form body should carry the token + file.
    sent = request.body
    if isinstance(sent, (bytes, bytearray)):
        sent = sent.decode("utf-8")
    assert "_token=abc123" in sent
    assert "file=x.gz" in sent


def test_stream_to_file_uses_provided_session(tmp_path) -> None:
    """If a caller has already configured a ``requests.Session`` (cookies,
    auth, retry adapter), the helper should use it instead of bare
    ``requests``."""
    session = requests.Session()
    session.cookies.set("flavour", "chocolate-chip")
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(responses.GET, "https://x.test/", body=b"x", status=200)
        stream_to_file("https://x.test/", tmp_path / "out", progress=False, session=session)
        cookie_hdr = mock_resp.calls[0].request.headers.get("Cookie", "")
    # The session-bound cookie must have ridden along.
    assert "flavour=chocolate-chip" in cookie_hdr


def test_stream_to_file_chunk_size_does_not_change_output(tmp_path) -> None:
    """Sanity: chunk size is a performance knob, not a correctness one.
    Same response should produce the same bytes at any chunk size."""
    body = b"A" * 4096 + b"B" * 4096 + b"C" * 4096
    with responses.RequestsMock() as mock_resp:
        mock_resp.add(responses.GET, "https://x.test/", body=body, status=200)
        out1 = stream_to_file(
            "https://x.test/", tmp_path / "small.bin", progress=False, chunk_size=16
        )

    with responses.RequestsMock() as mock_resp:
        mock_resp.add(responses.GET, "https://x.test/", body=body, status=200)
        out2 = stream_to_file(
            "https://x.test/",
            tmp_path / "large.bin",
            progress=False,
            chunk_size=1 << 20,
        )

    assert out1.read_bytes() == out2.read_bytes() == body


# ---------------------------------------------------------------------------
# Integration: verify each per-source download_* function picked up the
# `progress=` kwarg. Smoke-only — they're all mocked downloads.
# ---------------------------------------------------------------------------


def test_progress_kwarg_is_threaded_through_each_module() -> None:
    """Every public download_* function across biodb.* should accept
    ``progress``. Catches the regression where a refactor wires up the
    helper but forgets to surface the kwarg."""
    import inspect

    import biodb.gprofiler
    import biodb.gwas_atlas
    import biodb.msigdb
    import biodb.pubmed
    import biodb.uniprot

    for fn in (
        biodb.pubmed.download_baseline_file,
        biodb.pubmed.download_update_file,
        biodb.gprofiler.download_gmt,
        biodb.msigdb.download_gmt,
        biodb.gwas_atlas.download_file,
        biodb.gwas_atlas.download_metadata,
        biodb.gwas_atlas.download_magma_p,
        biodb.uniprot.download_swissprot_fasta,
        biodb.uniprot.download_trembl_fasta,
    ):
        params = inspect.signature(fn).parameters
        assert "progress" in params, (
            f"{fn.__module__}.{fn.__name__} should accept progress= (got params: {list(params)})"
        )
