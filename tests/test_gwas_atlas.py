"""Tests for :mod:`biodb.gwas_atlas`.

The module's download path is a Laravel CSRF-protected form POST. We
exercise the full handshake (``GET / → scrape _token → POST /home/release``)
against mocked HTTP via :mod:`responses` so CI never touches the network.
The two ``@pytest.mark.network`` smoke tests at the bottom verify the
real upstream still behaves as we mock it.
"""

from __future__ import annotations

import gzip
import io

import pandas as pd
import pytest
import requests
import responses

from biodb import gwas_atlas

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

# Minimal homepage HTML containing the form Laravel emits.
_FAKE_HOMEPAGE = """
<html>
  <body>
    <form method="post" action="/home/release">
      <input type="hidden" name="_token" value="TESTTOKEN0123456789abcdef" />
      <input type="hidden" name="file" id="release_file" val="" />
    </form>
  </body>
</html>
"""


def _gzipped(text: str) -> bytes:
    """Gzip-encode ``text`` so pandas.read_csv(..., compression="gzip") can ingest it."""
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(text.encode("utf-8"))
    return buf.getvalue()


@pytest.fixture
def fake_homepage():
    """Mock the homepage GET so ``_session()`` finds a valid ``_token``."""
    with responses.RequestsMock() as mock:
        mock.add(responses.GET, "https://atlas.ctglab.nl/", body=_FAKE_HOMEPAGE, status=200)
        yield mock


# ---------------------------------------------------------------------------
# Module surface tests
# ---------------------------------------------------------------------------


def test_module_imports_offline() -> None:
    assert gwas_atlas.__name__ == "biodb.gwas_atlas"


def test_constants_present() -> None:
    # The base URL is the site root; downloads go through the CSRF-form endpoint.
    assert gwas_atlas.GWAS_ATLAS_BASE_URL == "https://atlas.ctglab.nl"
    assert gwas_atlas.GWAS_ATLAS_RELEASE_ENDPOINT.endswith("/home/release")
    assert gwas_atlas.DEFAULT_VERSION == "20191115"
    assert gwas_atlas.CACHE_DIR.exists()


def test_public_api_signatures_stable() -> None:
    for name in (
        "download_file",
        "download_metadata",
        "download_magma_p",
        "load_metadata",
        "load_magma_p",
        "melt_magma_p",
    ):
        assert hasattr(gwas_atlas, name)


# ---------------------------------------------------------------------------
# _session() — CSRF handshake
# ---------------------------------------------------------------------------


def test_session_scrapes_token_from_homepage(fake_homepage) -> None:
    session, token = gwas_atlas._session(timeout=5)
    assert token == "TESTTOKEN0123456789abcdef"
    # Headers are set on the session.
    assert "biodb" in session.headers["User-Agent"].lower()


def test_session_raises_when_token_missing() -> None:
    """If the page layout changes and the regex misses, raise a descriptive error."""
    with responses.RequestsMock() as mock:
        mock.add(
            responses.GET,
            "https://atlas.ctglab.nl/",
            body="<html><body>no token here</body></html>",
            status=200,
        )
        with pytest.raises(RuntimeError, match="_token field"):
            gwas_atlas._session(timeout=5)


def test_session_raises_on_http_error() -> None:
    """A 5xx on the homepage propagates as ``HTTPError`` so callers can retry."""
    with responses.RequestsMock() as mock:
        mock.add(responses.GET, "https://atlas.ctglab.nl/", status=502)
        with pytest.raises(requests.HTTPError):
            gwas_atlas._session(timeout=5)


# ---------------------------------------------------------------------------
# _download() — form POST + stream-to-disk
# ---------------------------------------------------------------------------


def test_download_writes_response_body_to_dst(fake_homepage, tmp_path) -> None:
    payload = b"file-contents-bytes"
    fake_homepage.add(
        responses.POST, gwas_atlas.GWAS_ATLAS_RELEASE_ENDPOINT, body=payload, status=200
    )

    dst = tmp_path / "out.bin"
    result = gwas_atlas._download("anyfile.txt", dst)

    assert result == dst
    assert dst.read_bytes() == payload


def test_download_posts_token_and_filename(fake_homepage, tmp_path) -> None:
    fake_homepage.add(
        responses.POST, gwas_atlas.GWAS_ATLAS_RELEASE_ENDPOINT, body=b"ok", status=200
    )

    gwas_atlas._download("gwasATLAS_v20191115.readme", tmp_path / "readme.txt")

    # The second call is the form POST; inspect its body.
    post = fake_homepage.calls[1]
    assert post.request.method == "POST"
    assert "_token=TESTTOKEN0123456789abcdef" in post.request.body
    assert "file=gwasATLAS_v20191115.readme" in post.request.body


def test_download_creates_missing_parent_directories(fake_homepage, tmp_path) -> None:
    fake_homepage.add(responses.POST, gwas_atlas.GWAS_ATLAS_RELEASE_ENDPOINT, body=b"x", status=200)

    nested = tmp_path / "a" / "b" / "c" / "file.bin"
    gwas_atlas._download("any.txt", nested)
    assert nested.exists()


def test_download_propagates_post_errors(fake_homepage, tmp_path) -> None:
    fake_homepage.add(responses.POST, gwas_atlas.GWAS_ATLAS_RELEASE_ENDPOINT, status=503)
    with pytest.raises(requests.HTTPError):
        gwas_atlas._download("any.txt", tmp_path / "nope.bin")


# ---------------------------------------------------------------------------
# download_file() — cache layer
# ---------------------------------------------------------------------------


def test_download_file_returns_cached_path_when_present(tmp_path) -> None:
    """No HTTP at all when the destination already exists."""
    target = tmp_path / "gwasATLAS_v20191115.readme"
    target.write_text("preexisting")
    with responses.RequestsMock() as mock:  # no .add(); any call would fail
        result = gwas_atlas.download_file(
            "gwasATLAS_v20191115.readme", cache_dir=tmp_path, force=False
        )
        assert result == target
        assert len(mock.calls) == 0


def test_download_file_force_redownloads(fake_homepage, tmp_path) -> None:
    """``force=True`` overwrites the cached file even if it exists."""
    target = tmp_path / "gwasATLAS_v20191115.readme"
    target.write_bytes(b"OLD")
    fake_homepage.add(
        responses.POST, gwas_atlas.GWAS_ATLAS_RELEASE_ENDPOINT, body=b"NEW", status=200
    )
    result = gwas_atlas.download_file("gwasATLAS_v20191115.readme", cache_dir=tmp_path, force=True)
    assert result.read_bytes() == b"NEW"


def test_download_file_cache_dir_accepts_str(fake_homepage, tmp_path) -> None:
    """``cache_dir`` accepts ``str`` or ``Path``; tilde-expansion works."""
    fake_homepage.add(
        responses.POST, gwas_atlas.GWAS_ATLAS_RELEASE_ENDPOINT, body=b"hi", status=200
    )
    result = gwas_atlas.download_file("any.txt", cache_dir=str(tmp_path))
    assert result.parent == tmp_path


def test_download_file_falls_back_to_default_cache(fake_homepage, monkeypatch, tmp_path) -> None:
    """When ``cache_dir`` is ``None``, the module-level CACHE_DIR is used."""
    monkeypatch.setattr(gwas_atlas, "CACHE_DIR", tmp_path)
    fake_homepage.add(
        responses.POST, gwas_atlas.GWAS_ATLAS_RELEASE_ENDPOINT, body=b"hi", status=200
    )
    result = gwas_atlas.download_file("file.txt")
    assert result == tmp_path / "file.txt"


# ---------------------------------------------------------------------------
# download_metadata / download_magma_p convenience wrappers
# ---------------------------------------------------------------------------


def test_download_metadata_uses_correct_filename(fake_homepage, tmp_path) -> None:
    fake_homepage.add(
        responses.POST, gwas_atlas.GWAS_ATLAS_RELEASE_ENDPOINT, body=b"meta", status=200
    )
    result = gwas_atlas.download_metadata(version="20191115", cache_dir=tmp_path)
    assert result.name == "gwasATLAS_v20191115.txt.gz"


def test_download_magma_p_uses_correct_filename(fake_homepage, tmp_path) -> None:
    fake_homepage.add(
        responses.POST, gwas_atlas.GWAS_ATLAS_RELEASE_ENDPOINT, body=b"magma", status=200
    )
    result = gwas_atlas.download_magma_p(version="20191115", cache_dir=tmp_path)
    assert result.name == "gwasATLAS_v20191115_magma_P.txt.gz"


def test_download_metadata_honours_version_argument(fake_homepage, tmp_path) -> None:
    fake_homepage.add(
        responses.POST, gwas_atlas.GWAS_ATLAS_RELEASE_ENDPOINT, body=b"meta", status=200
    )
    result = gwas_atlas.download_metadata(version="20240101", cache_dir=tmp_path)
    assert result.name == "gwasATLAS_v20240101.txt.gz"
    # Confirm the POST asked for the right file.
    post_body = fake_homepage.calls[1].request.body
    assert "file=gwasATLAS_v20240101.txt.gz" in post_body


# ---------------------------------------------------------------------------
# load_metadata / load_magma_p — pandas integration
# ---------------------------------------------------------------------------


def test_load_metadata_parses_tsv(fake_homepage, tmp_path) -> None:
    body = _gzipped("id\tTrait\tPMID\n1\tBMI\t123\n2\tHeight\t456\n")
    fake_homepage.add(responses.POST, gwas_atlas.GWAS_ATLAS_RELEASE_ENDPOINT, body=body, status=200)
    df = gwas_atlas.load_metadata(cache_dir=tmp_path)
    assert list(df.columns) == ["id", "Trait", "PMID"]
    assert df.shape == (2, 3)


def test_load_magma_p_uses_index_col(fake_homepage, tmp_path) -> None:
    body = _gzipped("gene\tstudy_A\tstudy_B\nENSG_1\t3.2\t1.1\nENSG_2\t0.5\t4.0\n")
    fake_homepage.add(responses.POST, gwas_atlas.GWAS_ATLAS_RELEASE_ENDPOINT, body=body, status=200)
    df = gwas_atlas.load_magma_p(cache_dir=tmp_path)
    assert df.index.name == "gene"
    assert list(df.columns) == ["study_A", "study_B"]
    assert df.loc["ENSG_1", "study_A"] == 3.2


def test_load_metadata_passes_through_read_kwargs(fake_homepage, tmp_path) -> None:
    body = _gzipped("a\tb\tc\n1\t2\t3\n4\t5\t6\n7\t8\t9\n")
    fake_homepage.add(responses.POST, gwas_atlas.GWAS_ATLAS_RELEASE_ENDPOINT, body=body, status=200)
    # ``nrows`` is a pandas-only kwarg, so this verifies forwarding cleanly.
    df = gwas_atlas.load_metadata(cache_dir=tmp_path, nrows=2)
    assert len(df) == 2


# ---------------------------------------------------------------------------
# melt_magma_p — pure pandas
# ---------------------------------------------------------------------------


def test_melt_magma_p_shape() -> None:
    """Pivot a tiny wide frame and verify the long output schema."""
    wide = pd.DataFrame(
        {"study_A": [3.2, 1.1, None], "study_B": [0.5, None, 4.0]},
        index=pd.Index(["ENSG_X", "ENSG_Y", "ENSG_Z"], name="gene_id"),
    )
    long = gwas_atlas.melt_magma_p(wide, p_col="score")
    assert set(long.columns) == {"targetId", "sourceId", "score"}
    # 6 wide cells minus 2 NaNs = 4 rows.
    assert len(long) == 4
    assert set(long["sourceId"]) == {"study_A", "study_B"}


def test_melt_magma_p_uses_custom_p_col_name() -> None:
    wide = pd.DataFrame({"s1": [1.0, 2.0]}, index=pd.Index(["g1", "g2"], name="g"))
    long = gwas_atlas.melt_magma_p(wide, p_col="neg_log10_p")
    assert "neg_log10_p" in long.columns
    assert "score" not in long.columns


def test_melt_magma_p_handles_unnamed_index() -> None:
    """The melt should still produce sourceId/targetId even when the index has no name."""
    wide = pd.DataFrame({"s1": [1.0], "s2": [2.0]})
    long = gwas_atlas.melt_magma_p(wide)
    assert set(long.columns) >= {"sourceId", "score"}


# ---------------------------------------------------------------------------
# Live network smoke tests (skipped in CI)
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_session_csrf_handshake_live() -> None:
    """Hit the live homepage and confirm the CSRF token + cookies come back."""
    session, token = gwas_atlas._session(timeout=30)
    assert isinstance(token, str) and len(token) > 20
    assert "atlas_session" in session.cookies
    assert "XSRF-TOKEN" in session.cookies


@pytest.mark.network
def test_download_readme_live(tmp_path) -> None:
    """End-to-end smoke: fetch the small readme via the form-POST flow."""
    path = gwas_atlas.download_file("gwasATLAS_v20191115.readme", cache_dir=tmp_path, force=True)
    body = path.read_text()
    # Distinctive header line from the upstream readme.
    assert "GWAS ATLAS release v20191115" in body
