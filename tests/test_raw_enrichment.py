import pytest

from biodb.europepmc import fetch_metadata
from biodb.protein_structure import fetch_structure_for_accession


class Response:
    def __init__(self, data, status=200):
        self.data, self.status_code = data, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(str(self.status_code))

    def json(self):
        return self.data


class Session:
    def __init__(self, replies):
        self.replies, self.calls = iter(replies), []

    def get(self, url, **kwargs):
        self.calls.append(url)
        r = next(self.replies)
        if isinstance(r, Exception):
            raise r
        return r


def test_europepmc_retries_failed_batch_only(tmp_path):
    first = Session(
        [
            Response({"resultList": {"result": [{"id": "1", "title": "one"}]}}),
            RuntimeError("offline"),
        ]
    )
    with pytest.raises(RuntimeError):
        fetch_metadata(["1", "2"], cache_dir=tmp_path, batch_size=1, max_retries=1, session=first)
    second = Session([Response({"resultList": {"result": []}})])
    assert fetch_metadata(["1", "2"], cache_dir=tmp_path, batch_size=1, session=second) == {
        "1": {"id": "1", "title": "one"}
    }
    assert len(second.calls) == 1
    assert fetch_metadata(["2"], cache_dir=tmp_path, session=Session([])) == {}


def test_protein_retries_failed_provider_and_retains_raw(tmp_path):
    af = [{"entryId": "AF-P1", "extra": 7}]
    with pytest.raises(RuntimeError):
        fetch_structure_for_accession(
            "P1", cache_dir=tmp_path, session=Session([Response(af), RuntimeError("offline")])
        )
    session = Session([Response([], 404)])
    out = fetch_structure_for_accession("P1", cache_dir=tmp_path, session=session)
    assert out["alphafold"] == af
    assert out["rcsb"] == []
    assert len(session.calls) == 1


def test_provider_500_never_becomes_completed_blank(tmp_path):
    out = fetch_structure_for_accession(
        "P1",
        cache_dir=tmp_path,
        allow_partial=True,
        session=Session([Response([], 500), Response([], 404)]),
    )
    assert "alphafold" not in out
    session = Session([Response([{"entryId": "AF"}])])
    assert (
        fetch_structure_for_accession("P1", cache_dir=tmp_path, session=session)["alphafold"][0][
            "entryId"
        ]
        == "AF"
    )


def test_literature_reads_full_raw_rows_and_requires_public_cache(monkeypatch, tmp_path):
    import polars as pl

    import biodb.opentargets as ot
    from biodb.opentargets.literature import fetch_literature_matches

    path = tmp_path / "raw.parquet"
    pl.DataFrame(
        [
            {
                "type": "CD",
                "keywordId": "D",
                "pmid": "1",
                "text": "hello",
                "section": "title",
                "unmodeled": "preserved",
            }
        ]
    ).write_parquet(path)
    captured = {}

    def shards(*args, **kwargs):
        captured.update(kwargs)
        return [path]

    monkeypatch.setattr(ot, "ensure_cached_shards", shards)
    rows = fetch_literature_matches("25.12", {("D", "1")}, session="injected")
    assert rows[0]["unmodeled"] == "preserved"
    assert "/intermediate/literature_match/" in captured["remote_url"]
    assert captured["session"] == "injected"


def test_constants_fetch_retains_provenance_and_reuses_validated_source(tmp_path):
    import base64

    from biodb.opentargets.constants import fetch_source

    session = Session(
        [Response({"content": base64.b64encode(b"const x = 1").decode(), "sha": "abc"})]
    )
    assert fetch_source("variant.ts", cache_dir=tmp_path, session=session) == ("const x = 1", "abc")
    assert fetch_source("variant.ts", cache_dir=tmp_path, session=Session([])) == (
        "const x = 1",
        "abc",
    )


def test_sequence_ontology_uses_bulk_source(monkeypatch):
    import polars as pl

    import biodb.opentargets as ot
    from biodb.opentargets.constants import fetch_sequence_ontology_labels

    monkeypatch.setattr(
        ot, "get_dataset", lambda *a, **k: pl.DataFrame({"id": ["SO:1"], "label": ["region"]})
    )
    assert fetch_sequence_ontology_labels() == {"SO_1": "region"}
