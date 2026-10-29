import pytest

from biodb.ols_text import text_record


def test_v2_preserves_all_labels_synonyms_and_definitions():
    row = text_record(
        {
            "curie": "SNOMED:1",
            "iri": "http://snomed.info/id/1",
            "label": ["Preferred (finding)", "Preferred"],
            "http://www.w3.org/2004/02/skos/core#prefLabel": "Preferred",
            "http://www.w3.org/2004/02/skos/core#altLabel": ["Alternative", "Second alias"],
            "http://purl.obolibrary.org/obo/IAO_0000115": ["First definition", "Second definition"],
            "linkedEntities": {
                "http://snomed.info/id/2": {"label": ["Do not import neighboring labels"]}
            },
        }
    )
    assert row["label"] == "Preferred"
    assert row["synonyms"] == ["Preferred (finding)", "Alternative", "Second alias"]
    assert row["description"] == ["First definition", "Second definition"]
    assert "Do not import neighboring labels" not in row["synonyms"]


def test_empty_source_fields_stay_empty_and_reified_literals_work():
    row = text_record(
        {
            "curie": "SNOMED:2",
            "label": ["A"],
            "http://www.w3.org/2004/02/skos/core#altLabel": [
                {"type": ["reification"], "value": "Alias", "axioms": []}
            ],
        }
    )
    assert row["description"] == []
    assert row["synonyms"] == ["Alias"]


@pytest.mark.parametrize("changed", [False, True])
def test_cache_audits_source_version_before_publication(tmp_path, monkeypatch, changed):
    from types import SimpleNamespace

    import pyarrow.parquet as pq

    from biodb import ols_text

    metadata_calls = 0

    def get(url, params=None, timeout=None):
        nonlocal metadata_calls
        if url.endswith("/classes"):
            payload = {
                "totalPages": 1,
                "totalElements": 1,
                "elements": [{"curie": "SNOMED:1", "label": ["One"]}],
            }
        else:
            metadata_calls += 1
            payload = {
                "ontologyId": "snomed",
                "loaded": "snapshot-b" if changed and metadata_calls > 1 else "snapshot-a",
            }
        return SimpleNamespace(json=lambda: payload, raise_for_status=lambda: None)

    monkeypatch.setattr(ols_text.requests, "get", get)
    destination = tmp_path / "concepts.parquet"
    kwargs = dict(page_cache=tmp_path / "pages", expected_ids=["SNOMED:1"])
    if changed:
        with pytest.raises(ValueError, match="metadata changed"):
            ols_text.download_text_cache("snomed", destination, **kwargs)
        assert not destination.exists()
    else:
        result = ols_text.download_text_cache("snomed", destination, **kwargs)
        assert result["source_metadata"]["loaded"] == "snapshot-a"
        assert pq.read_table(destination).num_rows == 1
