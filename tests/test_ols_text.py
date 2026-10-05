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
