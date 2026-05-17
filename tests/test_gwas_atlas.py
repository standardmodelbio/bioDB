"""Smoke tests for :mod:`biodb.gwas_atlas`."""

from __future__ import annotations

import pandas as pd
import pytest

from biodb import gwas_atlas


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


def test_melt_magma_p_shape() -> None:
    """Pivot a tiny wide frame and verify the long output schema."""
    wide = pd.DataFrame(
        {"study_A": [3.2, 1.1, None], "study_B": [0.5, None, 4.0]},
        index=pd.Index(["ENSG_X", "ENSG_Y", "ENSG_Z"], name="gene_id"),
    )
    long = gwas_atlas.melt_magma_p(wide, p_col="score")
    assert set(long.columns) == {"targetId", "sourceId", "score"}
    # 6 wide cells minus 2 NaNs = 4 rows
    assert len(long) == 4
    assert set(long["sourceId"]) == {"study_A", "study_B"}


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
