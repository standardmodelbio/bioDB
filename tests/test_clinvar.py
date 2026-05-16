"""Smoke tests for ``biodb.clinvar``.

The module is vendored and most paths require live NCBI downloads or
``genoray`` + ``pooch``. We only assert the import surface and the few
pure-Python helpers that operate on in-memory Polars frames.
"""

from __future__ import annotations

import inspect

import polars as pl
import pytest

from biodb import clinvar

# --------------------------------------------------------------------------
# Import surface
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "INFO_COLS_SELECT",
        "bed_to_sites",
        "count_sites_per_gene",
        "df_to_bed",
        "df_to_sites",
        "download_vcf",
        "filter_df",
        "read_bed",
        "simplify_annotations",
        "vcf_to_df",
    ],
)
def test_public_symbol_exists(name: str) -> None:
    assert hasattr(clinvar, name), f"clinvar.{name} should be public"


def test_info_cols_select_is_non_empty_list_of_strings() -> None:
    cols = clinvar.INFO_COLS_SELECT
    assert isinstance(cols, list) and len(cols) > 0
    assert all(isinstance(c, str) for c in cols)
    # Spot-check a couple of canonical entries.
    assert "CLNSIG" in cols
    assert "GENEINFO" in cols


def test_functions_have_docstrings() -> None:
    for name in (
        "download_vcf",
        "vcf_to_df",
        "simplify_annotations",
        "df_to_bed",
        "df_to_sites",
        "read_bed",
        "filter_df",
        "count_sites_per_gene",
    ):
        fn = getattr(clinvar, name)
        assert inspect.isfunction(fn)
        assert (fn.__doc__ or "").strip(), f"{name} should have a docstring"


# --------------------------------------------------------------------------
# Pure helpers exercised in-memory
# --------------------------------------------------------------------------


def test_simplify_annotations_polars_default_maps() -> None:
    """Long-tail CLNSIG strings collapse to 6 + 4-class buckets."""
    df = pl.DataFrame(
        {
            "CLNSIG": ["Benign", "Pathogenic", "Likely_benign", "Uncertain_significance"],
            "GENEINFO": ["BRCA1:672", "TP53:7157", "EGFR:1956", "MYC:4609"],
        }
    )
    out = clinvar.simplify_annotations(df, verbose=False)
    assert "CLNSIG_simple" in out.columns
    assert "CLNSIG_super_simple" in out.columns
    assert "GENE" in out.columns  # GENEINFO → GENE shortcut
    assert out["CLNSIG_simple"].to_list() == [
        "benign",
        "path",
        "likely_benign",
        "other",
    ]


def test_simplify_annotations_pandas_in_pandas_out() -> None:
    """Pandas → pandas roundtrip. Needs ``pyarrow`` for the polars
    interop step on string columns, so skip when it's unavailable."""
    pytest.importorskip("pyarrow")
    import pandas as pd

    df = pd.DataFrame({"CLNSIG": ["Benign", "Pathogenic"]})
    out = clinvar.simplify_annotations(df, verbose=False)
    assert isinstance(out, pd.DataFrame)
    assert list(out["CLNSIG_simple"]) == ["benign", "path"]


def test_filter_df_skips_missing_columns() -> None:
    """Filter keys that aren't in the DataFrame are silently skipped."""
    df = pl.DataFrame({"CHROM": ["1", "2"], "POS": [100, 200], "CLNDN": ["a", "b"]})
    out = clinvar.filter_df(df, filters={"NOT_THERE": "x"}, verbose=False)
    # No conditions → returns the input untouched
    assert out.shape == df.shape


def test_filter_df_empty_filters_returns_input() -> None:
    df = pl.DataFrame({"CHROM": ["1"], "POS": [100], "CLNDN": ["a"]})
    out = clinvar.filter_df(df, filters={}, verbose=False)
    assert out.equals(df)


def test_filter_df_revstat_score_threshold() -> None:
    """CLNREVSTAT_score is treated as a >= threshold filter."""
    df = pl.DataFrame(
        {
            "CHROM": ["1", "1", "1"],
            "POS": [10, 20, 30],
            "CLNDN": ["a", "b", "c"],
            "CLNREVSTAT_score": [0, 2, 4],
        }
    )
    out = clinvar.filter_df(df, filters={"CLNREVSTAT_score": 2}, verbose=False)
    assert out.shape[0] == 2
    assert set(out["CLNREVSTAT_score"].to_list()) == {2, 4}
