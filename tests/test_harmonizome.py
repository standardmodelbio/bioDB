"""Smoke tests for :mod:`biodb.harmonizome`.

Live REST calls (``DOWNLOADS`` / ``DATASET_TO_PATH`` access, ``list_datasets``,
``get_gmt``) hit the Maayan-Lab API and are gated behind
``@pytest.mark.network`` so CI skips them. The default suite exercises:

* the module imports without touching the network (the critical bug fixed
  in this port — the AoU original fetched config at import time);
* pure GMT-parse helpers on synthetic input;
* signature shape of every public function.
"""

from __future__ import annotations

import inspect

import pandas as pd
import pytest

from biodb import harmonizome


def test_module_imports_offline() -> None:
    """Importing must not require network access.

    The AoU original fetched ``/dark/script_config`` at import time;
    we lazy-load via ``__getattr__`` so this is fast and offline-safe.
    """
    assert harmonizome.__name__ == "biodb.harmonizome"
    assert harmonizome.VERSION == "1.0"


def test_constants_present() -> None:
    assert harmonizome.API_URL.endswith("/Harmonizome/api")
    assert harmonizome.DOWNLOAD_URL.endswith("/data")
    assert harmonizome.CACHE_DIR.exists()


def test_public_api_signatures_stable() -> None:
    expected = {
        "list_datasets",
        "download_datasets",
        "get_gmt",
        "load_gene_attribute_matrix",
        "get_dataset_metadata",
        "json_from_url",
        "Entity",
        "Harmonizome",
    }
    missing = [name for name in expected if not hasattr(harmonizome, name)]
    assert not missing, f"missing public symbols: {missing}"


def test_get_gmt_signature_uses_string_defaults() -> None:
    """``file_types`` defaults to a glob string, not a list-as-sentinel."""
    sig = inspect.signature(harmonizome.get_gmt)
    assert sig.parameters["file_types"].default == "gene_set*.gmt.gz"
    assert sig.parameters["force"].default == 0
    assert sig.parameters["verbose"].default == 1


def test_looks_like_gene_set_name() -> None:
    """The missing-newline-detection helper used by ``_read_gmt``."""
    assert harmonizome._looks_like_gene_set_name("*Marker")
    assert harmonizome._looks_like_gene_set_name("__special")
    assert harmonizome._looks_like_gene_set_name("001N03")
    assert harmonizome._looks_like_gene_set_name("001N03_INTEGRIN")
    assert not harmonizome._looks_like_gene_set_name("BRCA1")
    assert not harmonizome._looks_like_gene_set_name("TP53")
    assert not harmonizome._looks_like_gene_set_name("")
    assert not harmonizome._looks_like_gene_set_name(None)  # type: ignore[arg-type]


def test_reverse_excel_date_conversion() -> None:
    """Excel mangles e.g. ``MARCH1`` → ``1-MAR``; this undoes it."""
    assert harmonizome._reverse_excel_date("1-MAR") == "MARCH1"
    assert harmonizome._reverse_excel_date("2-SEP") == "SEPT2"
    assert harmonizome._reverse_excel_date("10-DEC") == "DEC10"
    assert harmonizome._reverse_excel_date("BRCA1") == "BRCA1"
    assert harmonizome._reverse_excel_date("not-a-date") == "not-a-date"


def test_matches_file_type_glob() -> None:
    assert harmonizome._matches_file_type("gene_set_library_crisp.gmt", "gene_set*.gmt")
    assert harmonizome._matches_file_type("gene_set_library_crisp.gmt.gz", "gene_set*.gmt")
    assert not harmonizome._matches_file_type("attribute_set_library.gmt", "gene_set*.gmt")


def test_matches_file_type_list() -> None:
    types = ["gene_set_library_up_crisp.gmt", "attribute_set_library_dn_crisp.gmt"]
    assert harmonizome._matches_file_type("gene_set_library_up_crisp.gmt", types)
    assert harmonizome._matches_file_type("gene_set_library_up_crisp.gmt.gz", types)
    assert not harmonizome._matches_file_type("gene_set_library_crisp.gmt", types)


def test_matches_file_type_none_default() -> None:
    """``None`` accepts every ``.gmt`` and ``.gmt.gz`` file."""
    assert harmonizome._matches_file_type("anything.gmt", None)
    assert harmonizome._matches_file_type("anything.gmt.gz", None)
    assert not harmonizome._matches_file_type("anything.txt", None)


def test_matches_file_type_rejects_bad_type() -> None:
    with pytest.raises(TypeError, match="file_types must be"):
        harmonizome._matches_file_type("x.gmt", 123)  # type: ignore[arg-type]


def test_read_gmt_pandas(tmp_path) -> None:
    """Synthetic GMT round-trip — DataFrame shape + values."""
    gmt = tmp_path / "tiny.gmt"
    gmt.write_text(
        "DEMENTIA\tlist of dementia genes\tAPP\tPSEN1\tMAPT\nALZHEIMER\tAD-associated\tAPP\tAPOE\n"
    )
    df = harmonizome._read_gmt(gmt, return_format="pandas", suppress_stats=True)
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"id", "label", "gene"}
    assert set(df["id"].unique()) == {"DEMENTIA", "ALZHEIMER"}
    assert "APP" in set(df["gene"])


def test_read_gmt_dict(tmp_path) -> None:
    gmt = tmp_path / "tiny.gmt"
    gmt.write_text("SET1\tdesc1\tA\tB\tC\n")
    result = harmonizome._read_gmt(gmt, return_format="dict")
    assert isinstance(result, dict)
    assert ("SET1", "desc1") in result
    assert result[("SET1", "desc1")] == ["A", "B", "C"]


def test_read_gmt_rejects_bad_return_format(tmp_path) -> None:
    gmt = tmp_path / "tiny.gmt"
    gmt.write_text("SET\tdesc\tA\n")
    with pytest.raises(ValueError, match="return_format must be"):
        harmonizome._read_gmt(gmt, return_format="bogus")


def test_read_gmt_reverses_excel_dates(tmp_path) -> None:
    """Genes mangled by Excel (``MARCH1`` → ``1-MAR``) are recovered."""
    gmt = tmp_path / "tiny.gmt"
    gmt.write_text("SET\tdesc\t1-MAR\t2-SEP\tBRCA1\n")
    df = harmonizome._read_gmt(gmt, return_format="pandas", suppress_stats=True)
    assert set(df["gene"]) == {"MARCH1", "SEPT2", "BRCA1"}


@pytest.mark.network
def test_config_lazy_load() -> None:
    """``DOWNLOADS`` / ``DATASET_TO_PATH`` are populated on first access."""
    downloads = harmonizome.DOWNLOADS
    assert isinstance(downloads, list)
    assert any("gmt" in d for d in downloads)
    dataset_to_path = harmonizome.DATASET_TO_PATH
    assert isinstance(dataset_to_path, dict)
    assert len(dataset_to_path) > 10


@pytest.mark.network
def test_list_datasets_returns_dataframe() -> None:
    df = harmonizome.list_datasets(as_df=True)
    assert isinstance(df, pd.DataFrame)
    assert {"name", "href"}.issubset(df.columns)
    assert len(df) > 50
