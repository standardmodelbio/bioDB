"""Tests for :mod:`biodb.uniprot`.

The module wraps the UniProt REST endpoint + Biopython's ``SeqIO``
parser. We mock ``_fetch_records`` directly so the tests neither hit
the network nor require ``biopython``. The two ``@pytest.mark.network``
smoke tests at the bottom exercise the live path on demand.
"""

from __future__ import annotations

import inspect
from unittest import mock

import pandas as pd
import pytest
import requests
import responses

from biodb import uniprot

# ---------------------------------------------------------------------------
# Tiny Biopython-record stand-ins. Real ``SeqRecord`` objects exist but
# pulling in Biopython for unit tests is heavyweight; we mirror the
# attribute surface the module actually reads.
# ---------------------------------------------------------------------------


class _FakeLocation:
    def __init__(self, start: int, end: int) -> None:
        self.start = start
        self.end = end


class _FakeFeature:
    def __init__(self, ftype: str, start: int, end: int, **qualifiers) -> None:
        self.id = qualifiers.pop("id", f"feat_{start}_{end}")
        self.type = ftype
        self.location = _FakeLocation(start, end)
        self.qualifiers = qualifiers


class _FakeSeq(str):
    """Stand-in for Bio.Seq — strings already quack like sequences."""


class _FakeRecord:
    def __init__(
        self,
        rid: str,
        name: str = "",
        description: str = "",
        seq: str = "MVLSPAD",
        features: list[_FakeFeature] | None = None,
        dbxrefs: list[str] | None = None,
    ) -> None:
        self.id = rid
        self.name = name
        self.description = description
        self.seq = _FakeSeq(seq)
        self.features = features or []
        self.dbxrefs = dbxrefs or []


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_module_imports() -> None:
    assert uniprot.__name__ == "biodb.uniprot"


def test_endpoint_constant() -> None:
    assert uniprot.UNIPROT_REST_API == "https://rest.uniprot.org/uniprotkb"


def test_public_api_signatures_stable() -> None:
    for name in ("query_protein", "get_sequences", "get_features", "get_dbxrefs"):
        assert hasattr(uniprot, name)


def test_query_protein_signature_uses_keyword_only_options() -> None:
    """``fmt``, ``timeout_s``, ``verbose`` must be keyword-only (no positional foot-guns)."""
    sig = inspect.signature(uniprot.query_protein)
    params = sig.parameters
    assert params["uniprot_id"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert params["fmt"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["timeout_s"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["verbose"].kind is inspect.Parameter.KEYWORD_ONLY


# ---------------------------------------------------------------------------
# _fetch_records — HTTP + parsing
# ---------------------------------------------------------------------------


def test_fetch_records_propagates_http_error() -> None:
    """A 404 from UniProt must raise ``HTTPError`` rather than silently
    returning an empty list."""
    fake_bio = mock.MagicMock(SeqIO=mock.MagicMock())
    with (
        responses.RequestsMock() as mock_resp,
        mock.patch.dict("sys.modules", {"Bio": fake_bio}),
        pytest.raises(requests.HTTPError),
    ):
        mock_resp.add(responses.GET, f"{uniprot.UNIPROT_REST_API}/UNKNOWN.xml", status=404)
        uniprot._fetch_records("UNKNOWN")


def test_fetch_records_uses_uniprot_xml_parser() -> None:
    """``fmt="xml"`` must map to Biopython's ``uniprot-xml`` parser
    (not the literal string ``"xml"`` which would parse to nothing)."""
    fake_seqio = mock.MagicMock()
    fake_seqio.parse.return_value = iter([_FakeRecord("P12345")])
    fake_bio = mock.MagicMock(SeqIO=fake_seqio)

    with responses.RequestsMock() as mock_resp:
        mock_resp.add(
            responses.GET,
            f"{uniprot.UNIPROT_REST_API}/P12345.xml",
            body="<xml></xml>",
            status=200,
        )
        with mock.patch.dict("sys.modules", {"Bio": fake_bio}):
            records = uniprot._fetch_records("P12345", fmt="xml")

    assert len(records) == 1
    fake_seqio.parse.assert_called_once()
    assert fake_seqio.parse.call_args.args[1] == "uniprot-xml"


def test_fetch_records_returns_materialized_list() -> None:
    """``SeqIO.parse`` returns an iterator; the module must materialize it
    so callers can re-iterate."""
    fake_seqio = mock.MagicMock()
    # Iterator yields once, then exhausts — if we returned it directly the
    # second ``len(records)`` would be 0.
    fake_seqio.parse.return_value = iter([_FakeRecord("P1"), _FakeRecord("P2")])
    fake_bio = mock.MagicMock(SeqIO=fake_seqio)

    with responses.RequestsMock() as mock_resp:
        mock_resp.add(
            responses.GET,
            f"{uniprot.UNIPROT_REST_API}/X.xml",
            body="<xml></xml>",
            status=200,
        )
        with mock.patch.dict("sys.modules", {"Bio": fake_bio}):
            records = uniprot._fetch_records("X")

    # If parse() returned an iterator straight, the second access would be empty.
    assert len(records) == 2
    assert [r.id for r in records] == ["P1", "P2"]


# ---------------------------------------------------------------------------
# Higher-level wrappers — get *records / sequences / features / dbxrefs*
# ---------------------------------------------------------------------------


def _patch_fetch(monkeypatch, records: list[_FakeRecord]) -> None:
    monkeypatch.setattr(uniprot, "_fetch_records", lambda *a, **kw: records)


def test_query_protein_returns_records(monkeypatch) -> None:
    _patch_fetch(monkeypatch, [_FakeRecord("P12345")])
    records = uniprot.query_protein("P12345")
    assert isinstance(records, list)
    assert records[0].id == "P12345"


def test_query_protein_verbose_logs(monkeypatch, caplog) -> None:
    _patch_fetch(
        monkeypatch,
        [_FakeRecord("P12345", name="EX_NAME", description="example protein")],
    )
    with caplog.at_level("INFO", logger="biodb.uniprot"):
        uniprot.query_protein("P12345", verbose=True)
    assert any("P12345" in r.message and "EX_NAME" in r.message for r in caplog.records)


def test_get_sequences(monkeypatch) -> None:
    _patch_fetch(
        monkeypatch,
        [_FakeRecord("P1", seq="MVLSPAD"), _FakeRecord("P2", seq="ACGT")],
    )
    seqs = uniprot.get_sequences("any")
    assert seqs == ["MVLSPAD", "ACGT"]


def test_get_features_populates_dataframe(monkeypatch) -> None:
    record = _FakeRecord(
        "P12345",
        features=[
            _FakeFeature("DOMAIN", 1, 100, description="kinase"),
            _FakeFeature("SIGNAL", 1, 24),
        ],
    )
    _patch_fetch(monkeypatch, [record])
    df = uniprot.get_features("P12345")
    assert {"id", "type", "start", "end", "length"}.issubset(df.columns)
    assert df.loc[df["type"] == "DOMAIN", "length"].iloc[0] == 99
    # qualifier columns are unpacked into the row dict.
    assert "description" in df.columns


def test_get_features_empty_record(monkeypatch) -> None:
    _patch_fetch(monkeypatch, [_FakeRecord("P12345")])  # no features
    df = uniprot.get_features("P12345")
    assert df.empty


def test_get_dbxrefs_splits_db_id(monkeypatch) -> None:
    record = _FakeRecord("P12345", dbxrefs=["GO:0008150", "PDB:1ABC"])
    _patch_fetch(monkeypatch, [record])
    df = uniprot.get_dbxrefs("P12345")
    assert {"dbxref", "db", "id"}.issubset(df.columns)
    assert set(df["db"]) == {"GO", "PDB"}
    assert set(df["id"]) == {"0008150", "1ABC"}


def test_get_dbxrefs_empty(monkeypatch) -> None:
    _patch_fetch(monkeypatch, [_FakeRecord("P12345")])
    df = uniprot.get_dbxrefs("P12345")
    assert df.empty


# ---------------------------------------------------------------------------
# Live network smoke tests (CI skips)
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_query_protein_returns_materialized_list_live() -> None:
    pytest.importorskip("Bio")
    records = uniprot.query_protein("P12345")
    assert isinstance(records, list)
    # If query_protein returned a raw SeqIO.parse() iterator, the first
    # iteration below would exhaust it and the second would yield 0.
    first_pass = sum(1 for _ in records)
    second_pass = sum(1 for _ in records)
    assert first_pass == second_pass > 0


@pytest.mark.network
def test_get_features_live() -> None:
    pytest.importorskip("Bio")
    df = uniprot.get_features("P12345")
    assert isinstance(df, pd.DataFrame)
