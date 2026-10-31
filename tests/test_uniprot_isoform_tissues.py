import copy
import json
from pathlib import Path


def entry():
    return json.loads(
        (Path(__file__).parent / "fixtures/transcripts/uniprot_Q07817.json").read_text()
    )


def test_mixed_tissue_paragraph_is_split_by_explicit_isoform_names():
    from biodb.uniprot import iter_isoform_annotations

    rows = list(iter_isoform_annotations(entry(), release="2026_03"))
    for isoform, expected, excluded in [
        ("Q07817-1", "adult brain", "developing lymphocytes"),
        ("Q07817-2", "developing lymphocytes", "adult brain"),
    ]:
        selected = [r for r in rows if r["isoform_id"] == isoform]
        assert selected
        for row in selected:
            tissue = [c for c in row["comments"] if c["commentType"] == "TISSUE SPECIFICITY"]
            assert tissue
            assert expected in json.dumps(tissue)
            assert excluded not in json.dumps([c["texts"] for c in tissue])
            assert all(c["scope_basis"] == "explicit_named_isoform_sentence" for c in tissue)


def test_statement_comparing_two_isoforms_is_not_assigned_to_one():
    from biodb.uniprot import iter_isoform_annotations

    data = copy.deepcopy(entry())
    data["comments"] = [
        {
            "commentType": "TISSUE SPECIFICITY",
            "texts": [{"value": "Bcl-X(L) and Bcl-X(S) differ between these tissues."}],
        }
    ] + [c for c in data["comments"] if c["commentType"] == "ALTERNATIVE PRODUCTS"]
    rows = list(iter_isoform_annotations(data, release="2026_03"))
    assert all(not r["comments"] for r in rows)
    assert all(r["gene_context"] for r in rows)


def test_ambiguous_form_synonym_stays_unassigned():
    from biodb.uniprot import iter_isoform_annotations

    data = {
        "primaryAccession": "P1",
        "comments": [
            {
                "commentType": "ALTERNATIVE PRODUCTS",
                "isoforms": [
                    {
                        "name": {"value": "A"},
                        "synonyms": [{"value": "Shared"}],
                        "isoformIds": ["P1-1"],
                    },
                    {
                        "name": {"value": "B"},
                        "synonyms": [{"value": "Shared"}],
                        "isoformIds": ["P1-2"],
                    },
                ],
            },
            {
                "commentType": "TISSUE SPECIFICITY",
                "texts": [{"value": "Shared is expressed in brain."}],
            },
            {
                "commentType": "TISSUE SPECIFICITY",
                "texts": [{"value": "A subset of cells expresses this protein."}],
            },
        ],
        "uniProtKBCrossReferences": [
            {"database": "Ensembl", "id": "ENST1.1", "isoformId": "P1-1"},
            {"database": "Ensembl", "id": "ENST2.1", "isoformId": "P1-2"},
        ],
    }
    rows = list(iter_isoform_annotations(data, release="fixture"))
    assert all(not r["comments"] for r in rows)


def test_common_form_name_does_not_match_general_tissue_language():
    from biodb.uniprot import iter_isoform_annotations

    data = entry()
    data["comments"] = [
        {
            "commentType": "ALTERNATIVE PRODUCTS",
            "isoforms": [{"name": {"value": "Long"}, "isoformIds": ["Q07817-1"]}],
        },
        {
            "commentType": "TISSUE SPECIFICITY",
            "texts": [{"value": "Long bones and liver show expression."}],
        },
    ]
    assert all(not row["comments"] for row in iter_isoform_annotations(data, release="fixture"))


def test_mapt_comparative_tissue_clause_is_not_assigned_to_the_wrong_form():
    from biodb.uniprot import iter_isoform_annotations

    path = Path(__file__).parent / "fixtures/transcripts/uniprot_P10636.json"
    rows = [
        r
        for r in iter_isoform_annotations(json.loads(path.read_text()), release="2026_03")
        if r["isoform_id"] == "P10636-1"
    ]
    tissue = [
        c for row in rows for c in row["comments"] if c["commentType"] == "TISSUE SPECIFICITY"
    ]
    assert tissue
    assert "peripheral" in json.dumps([c["texts"] for c in tissue])
    assert "central nervous system" not in json.dumps([c["texts"] for c in tissue])
