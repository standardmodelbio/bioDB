"""Source-content omissions reproduced by the fresh paired-repository review."""

import json
from pathlib import Path


def test_recorded_mapt_synonyms_are_preserved_with_the_protein_form():
    from biodb.uniprot import iter_isoform_annotations

    path = Path(__file__).parent / "fixtures" / "transcripts" / "uniprot_P10636.json"
    rows = [
        r
        for r in iter_isoform_annotations(json.loads(path.read_text()), release="2026_03")
        if r["isoform_id"] == "P10636-2"
    ]
    assert rows and all("0N3R" in r["synonyms"] for r in rows)


def test_single_product_entry_retains_canonical_sequence_features():
    from biodb.uniprot import iter_isoform_annotations

    entry = {
        "primaryAccession": "P1",
        "sequence": {"value": "MK"},
        "uniProtKBCrossReferences": [{"database": "Ensembl", "id": "ENST1.1"}],
        "features": [{"type": "Chain", "location": {"start": {"value": 1}, "end": {"value": 2}}}],
        "comments": [{"commentType": "FUNCTION", "texts": [{"value": "General function"}]}],
    }
    row = next(iter_isoform_annotations(entry, release="fixture"))
    assert row["protein_sequence"] == "MK"
    assert row["features"][0]["annotation_scope"] == "canonical_protein_sequence"
    assert row["comments"] == []
    assert row["gene_context"]
