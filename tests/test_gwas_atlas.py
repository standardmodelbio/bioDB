"""Smoke tests for :mod:`biodb.gwas_atlas`."""

from __future__ import annotations

import pandas as pd

from biodb import gwas_atlas


def test_module_imports_offline() -> None:
    assert gwas_atlas.__name__ == "biodb.gwas_atlas"


def test_constants_present() -> None:
    assert gwas_atlas.GWAS_ATLAS_BASE_URL.endswith("ukb2_sumstats")
    assert gwas_atlas.DEFAULT_VERSION == "20191115"
    assert gwas_atlas.CACHE_DIR.exists()


def test_public_api_signatures_stable() -> None:
    for name in (
        "download_metadata",
        "download_magma_p",
        "load_metadata",
        "load_magma_p",
        "melt_magma_p",
    ):
        assert hasattr(gwas_atlas, name)


def test_url_format() -> None:
    assert gwas_atlas._metadata_url("20191115").endswith("gwasATLAS_v20191115.txt.gz")
    assert gwas_atlas._magma_p_url("20191115").endswith("gwasATLAS_v20191115_magma_P.txt.gz")


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
