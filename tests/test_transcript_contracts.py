"""Recorded public source contracts; live checks require an explicit opt-in."""

import hashlib
import json
import os
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "transcripts"


def test_actual_uniprot_isoform_sequence_status_and_molecule_scope():
    from biodb.uniprot import iter_isoform_annotations

    entry = json.loads((FIXTURES / "uniprot_P04637.json").read_text())
    rows = list(iter_isoform_annotations(entry, release="2026_03"))
    displayed = [r for r in rows if r["isoform_id"] == "P04637-1"]
    assert displayed
    assert all(r["protein_sequence"] == entry["sequence"]["value"] for r in displayed)
    assert all(r["features"] for r in displayed)
    cytoplasmic = [r for r in rows if r["isoform_id"] == "P04637-9"]
    assert cytoplasmic
    assert all(any(c.get("molecule") == "Isoform 9" for c in r["comments"]) for r in cytoplasmic)
    assert all(c.get("molecule") != "Isoform 9" for r in displayed for c in r["comments"])


def test_appris_downloads_method_annotations_rather_than_the_backbone(monkeypatch, tmp_path):
    from biodb import appris

    seen = []

    def download(url, destination, **kwargs):
        seen.append(url)
        return destination

    monkeypatch.setattr(appris, "download_source", download)
    paths = appris.download_features(cache_dir=tmp_path)
    assert {"spade", "firestar", "matador3d", "corsair", "thump", "crash", "proteo"} <= paths.keys()
    assert all("appris_method." in url for url in seen)
    assert not any("appris_data.annot" in url for url in seen)


def test_recorded_appris_scores_and_spade_method_attributes():
    from biodb.appris import iter_features, iter_scores

    scores = list(iter_scores(FIXTURES / "appris_data.appris.txt"))
    assert scores[0]["scores"]["Trifid Score"] is None
    assert scores[-1]["annotation"] == "ALTERNATIVE:1"
    assert scores[-1]["scores"]["Trifid Score"] == 0.9847
    features = list(iter_features(FIXTURES / "appris_method_features.gtf"))
    assert features[0]["method"] == "SPADE"
    assert features[0]["feature_type"] == "domain_wrong"
    assert "PF00520.30" in features[0]["attributes"]["note"][0]
    assert features[0]["start"] == 26191343


def test_explicit_isoform_text_prefix_is_scoped_without_copying_general_function():
    from biodb.uniprot import iter_isoform_annotations

    entry = json.loads((FIXTURES / "uniprot_Q07817.json").read_text())
    rows = list(iter_isoform_annotations(entry, release="2026_03"))
    short = [r for r in rows if r["isoform_id"] == "Q07817-2"]
    long = [r for r in rows if r["isoform_id"] == "Q07817-1"]
    assert short and long
    assert all("promotes apoptosis" in json.dumps(r["comments"]) for r in short)
    assert all("promotes apoptosis" not in json.dumps(r["comments"]) for r in long)
    assert all("Potent inhibitor of cell death" not in json.dumps(r["comments"]) for r in rows)


def test_recorded_refseq_noncoding_end_variants_keep_scoped_descriptions():
    from biodb.refseq import iter_transcripts

    rows = list(iter_transcripts(FIXTURES / "refseq_neat1.gb", release="recorded exact accessions"))
    assert [r["transcript_id"] for r in rows] == ["NR_028272.1", "NR_131012.1"]
    assert "shorter 3' end" in rows[0]["variant_description"]
    assert "longer transcript" in rows[1]["variant_description"]
    assert len(rows[0]["sequence"]) < len(rows[1]["sequence"])
    assert "paraspeckles" not in rows[0]["variant_description"]


def test_recorded_contract_files_have_auditable_hashes():
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    for entry in manifest:
        path = FIXTURES / entry["filename"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]


@pytest.mark.network
@pytest.mark.skipif(
    os.environ.get("ISODOCS_NETWORK_TESTS") != "1", reason="bounded live contracts are opt-in"
)
def test_live_uniprot_transcript_qualifiers():
    from biodb.uniprot import query_protein_json

    entry = query_protein_json("P04637")
    assert any(
        x.get("database") == "Ensembl" and x.get("isoformId")
        for x in entry["uniProtKBCrossReferences"]
    )
