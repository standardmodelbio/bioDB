"""Biological annotation must survive acquisition without widening its evidence."""

import json

import pytest
from Bio.Seq import Seq
from Bio.SeqFeature import BeforePosition, CompoundLocation, SeqFeature, SimpleLocation
from Bio.SeqRecord import SeqRecord


def test_refseq_adjacent_labels_do_not_leak_variant_or_publication_into_summary():
    from biodb.refseq import parse_record

    record = SeqRecord(Seq("A" * 100), id="NR_1.1")
    record.annotations["comment"] = (
        "REVIEWED REFSEQ: reviewed.\nSummary: Shared nuclear RNA function. "
        "Transcript Variant: The longer RNA has a stabilizing triple helix. "
        "Publication Note: See the gene record. COMPLETENESS: complete on the 3' end."
    )
    row = parse_record(record, release="test")
    assert row["gene_context"] == "Shared nuclear RNA function."
    assert row["variant_description"] == "The longer RNA has a stabilizing triple helix."


def test_refseq_rna_and_regulatory_features_keep_native_coordinates_and_evidence():
    from biodb.refseq import parse_record

    record = SeqRecord(Seq("A" * 100), id="NR_1.1")
    record.features = [
        SeqFeature(
            SimpleLocation(80, 90, strand=1),
            type="regulatory",
            qualifiers={
                "regulatory_class": ["polyA_signal_sequence"],
                "note": ["hexamer: AATAAA"],
                "experiment": ["DESCRIPTION:reporter assay"],
            },
        ),
        SeqFeature(
            SimpleLocation(90, 100),
            type="misc_feature",
            qualifiers={
                "note": ["triple helical structure"],
                "inference": ["similar to RNA sequence"],
            },
        ),
        SeqFeature(
            SimpleLocation(30, 40),
            type="misc_feature",
            qualifiers={
                "note": ["propagated from UniProtKB; phosphorylation site; ECO:0000269"],
            },
        ),
        SeqFeature(SimpleLocation(0, 100), type="gene", qualifiers={"gene": ["NC"]}),
    ]
    annotations = parse_record(record, release="test")["biological_annotations"]
    assert len(annotations) == 3
    a, b, protein = annotations
    assert (a["kind"], a["scope"], a["evidence_type"]) == ("RG", "transcript", "experimental")
    assert a["location"]["reference"] == "NR_1.1"
    assert a["location"]["coordinate_system"] == "0-based-half-open RNA"
    assert a["location"]["parts"] == [
        {"start": 80, "end": 90, "strand": 1, "start_exact": True, "end_exact": True}
    ]
    assert b["kind"] == "RN" and b["evidence_type"] == "inferred"
    assert protein["kind"] == "PF" and protein["evidence_type"] == "propagated"
    assert protein["qualifiers"]["note"][0].endswith("ECO:0000269")


def test_refseq_fuzzy_compound_locations_are_not_rendered_as_exact_contiguous_features():
    from biodb.refseq import parse_record

    record = SeqRecord(Seq("A" * 100), id="NR_1.1")
    record.features = [
        SeqFeature(
            CompoundLocation([SimpleLocation(BeforePosition(2), 10), SimpleLocation(30, 40)]),
            type="stem_loop",
            qualifiers={"note": ["annotated RNA structure"]},
        )
    ]
    a = parse_record(record)["biological_annotations"][0]
    assert a["location"]["operator"] == "join"
    assert a["location"]["parts"][0]["start_exact"] is False
    assert a["location"]["parts"][1]["start"] == 30


def test_gtex_distribution_support_is_sample_based_not_ratio_of_medians(tmp_path):
    from biodb.gtex import iter_expression_summaries

    tpm = tmp_path / "tpm.tsv"
    tpm.write_text(
        "transcript_id\tgene_id\ts1\ts2\ts3\ts4\nENST1.1\tG\t0\t2\t6\t8\nENST2.1\tG\t0\t8\t4\t2\n"
    )
    samples = tmp_path / "samples.tsv"
    samples.write_text("SAMPID\tSMTSD\ns1\tT\ns2\tT\ns3\tT\ns4\tT\n")
    rows = list(
        iter_expression_summaries(
            tpm, samples, work_dir=tmp_path, transcript_to_gene={"ENST1.1": "G", "ENST2.1": "G"}
        )
    )
    a = rows[0]
    assert a["n_detected"] == 3
    assert a["tpm_q25"] == 1.5 and a["tpm_q75"] == 6.5
    assert a["usage_q25"] == pytest.approx(0.4)
    assert a["usage_q75"] == pytest.approx(0.7)
    assert a["median_usage"] == pytest.approx(0.6)
    assert a["n_gene_detected"] == 3
    assert a["n_quantified_siblings"] == 2


def test_normalized_annotations_preserve_context_and_reject_unscoped_or_unproven_claims(tmp_path):
    from biodb.transcript_annotations import iter_annotations

    p = tmp_path / "annotations.jsonl"
    row = {
        "transcript_id": "ENST1.1",
        "release": "study-1",
        "scope": "transcript",
        "annotations": [
            {
                "kind": "BC",
                "text": "Preferential use in activated T cells.",
                "scope": "transcript",
                "evidence_type": "measured",
                "source_url": "https://example.org/study",
                "publications": ["123"],
                "conditions": {"cell_type": "T cell", "state": "activated"},
            }
        ],
    }
    p.write_text(json.dumps(row) + "\n")
    result = list(iter_annotations(p, release="study-1"))[0]
    assert result["annotations"][0]["conditions"]["state"] == "activated"
    for change, match in [
        ({"scope": "unspecified"}, "scope"),
        ({"source_url": ""}, "provenance"),
        ({"evidence_type": "proved"}, "evidence"),
    ]:
        invalid = {**row, "annotations": [{**row["annotations"][0], **change}]}
        p.write_text(json.dumps(invalid) + "\n")
        with pytest.raises(ValueError, match=match):
            list(iter_annotations(p, release="study-1"))


def test_refseq_structured_attributes_retain_gene_product_scope_and_inference():
    from biodb.refseq import parse_record

    record = SeqRecord(Seq("A" * 100), id="NM_1.1")
    record.annotations["structured_comment"] = {
        "RefSeq-Attributes": {"gene product(s) localized to mito.": "inferred from homology"},
        "Evidence-Data": {"Transcript exon combination": "read1 [ECO:0000332]"},
    }
    row = parse_record(record)
    assert row["structured_comments"]["Evidence-Data"]["Transcript exon combination"].startswith(
        "read1"
    )
    a = row["biological_annotations"][0]
    assert a["scope"] == "gene" and a["kind"] == "LO" and a["evidence_type"] == "inferred"


def test_study_interchange_rejects_mixed_molecular_scopes_and_missing_protein_sequence(tmp_path):
    from biodb.transcript_annotations import iter_annotations

    row = {
        "transcript_id": "ENST1.2",
        "release": "study1",
        "scope": "transcript",
        "protein_sequence": "AAAA",
        "annotations": [
            {
                "kind": "FN",
                "scope": "protein",
                "evidence_type": "measured",
                "text": "Binds X.",
                "source_url": "https://example.org/study",
            }
        ],
    }
    path = tmp_path / "annotations.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="protein.*scope"):
        list(iter_annotations(path, release="study1"))
    row["scope"] = "protein"
    row.pop("protein_sequence")
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="protein.*sequence"):
        list(iter_annotations(path, release="study1"))
