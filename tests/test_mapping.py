"""Tests for :mod:`biodb.mapping`."""

from __future__ import annotations

import inspect
import sys
from unittest import mock

import pandas as pd
import pytest

from biodb import mapping

# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_module_imports_offline() -> None:
    """Importing must not require gprofiler — it's lazy."""
    assert mapping.__name__ == "biodb.mapping"
    assert hasattr(mapping, "map_gene_ids")


def test_map_gene_ids_signature() -> None:
    sig = inspect.signature(mapping.map_gene_ids)
    params = sig.parameters
    assert "df" in params
    assert params["target_id_col"].default == "targetId"
    assert params["target_namespace"].default == "HGNC"
    assert params["organism"].default == "hsapiens"


# ---------------------------------------------------------------------------
# Input validation — covered without any network
# ---------------------------------------------------------------------------


def test_map_gene_ids_rejects_missing_column() -> None:
    df = pd.DataFrame({"foo": ["a", "b"]})
    with pytest.raises(ValueError, match="not found in DataFrame"):
        mapping.map_gene_ids(df, target_id_col="missing", verbose=False)


def test_map_gene_ids_empty_df_short_circuits() -> None:
    """Empty input short-circuits before touching gprofiler — never imports it."""
    out = mapping.map_gene_ids(pd.DataFrame({"targetId": []}), verbose=False)
    assert out.empty
    assert "targetId" in out.columns


def test_map_gene_ids_missing_gprofiler_raises_import_error(monkeypatch) -> None:
    """When ``gprofiler-official`` isn't installed, raise a guided ImportError."""
    monkeypatch.setitem(sys.modules, "gprofiler", None)
    df = pd.DataFrame({"targetId": ["ENSG00000012048"]})
    with pytest.raises(ImportError, match="biodb\\[mapping\\]"):
        mapping.map_gene_ids(df, verbose=False)


# ---------------------------------------------------------------------------
# Happy path via a stubbed GProfiler client
# ---------------------------------------------------------------------------


def _stub_gprofiler_module(monkeypatch, convert_result: pd.DataFrame) -> mock.MagicMock:
    """Install a fake ``gprofiler`` module whose ``GProfiler().convert()`` returns ``convert_result``."""
    fake_client = mock.MagicMock()
    fake_client.convert.return_value = convert_result
    fake_module = mock.MagicMock(GProfiler=mock.MagicMock(return_value=fake_client))
    monkeypatch.setitem(sys.modules, "gprofiler", fake_module)
    return fake_client


def test_map_gene_ids_happy_path(monkeypatch) -> None:
    convert_df = pd.DataFrame(
        [
            {"incoming": "ENSG00000012048", "converted": "BRCA1", "HGNC": "BRCA1"},
            {"incoming": "ENSG00000141510", "converted": "TP53", "HGNC": "TP53"},
        ]
    )
    fake_client = _stub_gprofiler_module(monkeypatch, convert_df)

    df = pd.DataFrame({"targetId": ["ENSG00000012048", "ENSG00000141510"], "score": [0.5, 0.7]})
    out = mapping.map_gene_ids(df, target_namespace="HGNC", verbose=False)
    fake_client.convert.assert_called_once()
    assert "HGNC" in out.columns
    assert out.loc[0, "HGNC"] == "BRCA1"
    assert out.loc[1, "HGNC"] == "TP53"
    # Score column preserved untouched.
    assert (out["score"] == df["score"]).all()


def test_map_gene_ids_unmapped_falls_back_to_incoming(monkeypatch) -> None:
    """gProfiler returns string ``"None"`` for unmapped IDs — flip back to source ID."""
    convert_df = pd.DataFrame(
        [
            {"incoming": "ENSG_REAL", "converted": "BRCA1", "HGNC": "BRCA1"},
            {"incoming": "ENSG_FAKE", "converted": "None", "HGNC": "None"},
        ]
    )
    _stub_gprofiler_module(monkeypatch, convert_df)

    df = pd.DataFrame({"targetId": ["ENSG_REAL", "ENSG_FAKE"]})
    out = mapping.map_gene_ids(df, target_namespace="HGNC", verbose=False)
    assert out.loc[0, "HGNC"] == "BRCA1"
    # Unmapped row falls back to the source ID, not to the literal "None" string.
    assert out.loc[1, "HGNC"] == "ENSG_FAKE"


def test_map_gene_ids_verbose_logs(monkeypatch, caplog) -> None:
    """``verbose=True`` should emit info-level log lines (count + mapping rate)."""
    convert_df = pd.DataFrame([{"incoming": "ENSG_X", "converted": "X", "HGNC": "X"}])
    _stub_gprofiler_module(monkeypatch, convert_df)
    df = pd.DataFrame({"targetId": ["ENSG_X"]})
    with caplog.at_level("INFO", logger="biodb.mapping"):
        mapping.map_gene_ids(df, verbose=True)
    messages = " ".join(record.message for record in caplog.records)
    assert "Mapping gene IDs" in messages or "Mapped" in messages


# ---------------------------------------------------------------------------
# Live integration test — RUN BY DEFAULT.
#
# gProfiler's ``/convert`` REST API is public + free + fast (<1 s for a
# handful of genes). Proves the mapper actually works against the real
# upstream — the mocked tests above only verify our reshaping of the
# response, not that g:Profiler still returns the response we expect.
# ---------------------------------------------------------------------------


def test_map_gene_ids_ensembl_to_hgnc_live() -> None:
    """Convert three well-known Ensembl IDs to HGNC symbols via the real
    g:Profiler API. Pinning the expected symbols would catch silent
    upstream-mapping changes (genes do occasionally get renamed)."""
    pytest.importorskip("gprofiler")
    df = pd.DataFrame(
        {
            "targetId": [
                "ENSG00000012048",  # BRCA1
                "ENSG00000141510",  # TP53
                "ENSG00000146648",  # EGFR
            ],
            "score": [0.1, 0.2, 0.3],
        }
    )
    out = mapping.map_gene_ids(
        df,
        target_id_col="targetId",
        target_namespace="HGNC",
        verbose=False,
    )

    assert "HGNC" in out.columns
    # Score column preserved; row count preserved.
    assert len(out) == 3
    assert list(out["score"]) == [0.1, 0.2, 0.3]
    # The three canonical genes should map to their well-known symbols.
    by_ensembl = dict(zip(out["targetId"], out["HGNC"], strict=True))
    assert by_ensembl["ENSG00000012048"] == "BRCA1"
    assert by_ensembl["ENSG00000141510"] == "TP53"
    assert by_ensembl["ENSG00000146648"] == "EGFR"
