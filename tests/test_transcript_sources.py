"""Offline transcript contracts; fixtures are intentionally tiny."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest
import responses


def _gtf(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.gtf.gz"
    rows = [
        'chr1\tHAVANA\tgene\t101\t500\t.\t-\t.\tgene_id "ENSG1.2"; gene_name "G";',
        'chr1\tHAVANA\ttranscript\t101\t500\t.\t-\t.\tgene_id "ENSG1.2"; transcript_id "ENST1.3"; gene_name "G"; transcript_type "protein_coding"; tag "MANE_Select"; tag "basic"; protein_id "ENSP1.2";',
        'chr1\tHAVANA\texon\t401\t500\t.\t-\t.\tgene_id "ENSG1.2"; transcript_id "ENST1.3"; exon_id "ENSE1.1"; exon_number "1";',
        'chr1\tHAVANA\tUTR\t481\t500\t.\t-\t.\tgene_id "ENSG1.2"; transcript_id "ENST1.3";',
        'chr1\tHAVANA\tCDS\t401\t480\t.\t-\t1\tgene_id "ENSG1.2"; transcript_id "ENST1.3";',
        'chr1\tHAVANA\texon\t101\t200\t.\t-\t.\tgene_id "ENSG1.2"; transcript_id "ENST1.3"; exon_number "2";',
        'chr1\tHAVANA\tCDS\t151\t200\t.\t-\t2\tgene_id "ENSG1.2"; transcript_id "ENST1.3";',
        'chr1\tHAVANA\tUTR\t101\t150\t.\t-\t.\tgene_id "ENSG1.2"; transcript_id "ENST1.3";',
        'chr1\tHAVANA\ttranscript\t101\t490\t.\t-\t.\tgene_id "ENSG1.2"; transcript_id "ENST2.1"; gene_name "G"; transcript_type "lncRNA";',
        'chr1\tHAVANA\texon\t101\t490\t.\t-\t.\tgene_id "ENSG1.2"; transcript_id "ENST2.1"; exon_number "1";',
        'chrY\tHAVANA\tgene\t1\t6\t.\t+\t.\tgene_id "ENSG3.2_PAR_Y"; gene_name "PAR";',
        'chrY\tHAVANA\ttranscript\t1\t6\t.\t+\t.\tgene_id "ENSG3.2_PAR_Y"; transcript_id "ENST3.2_PAR_Y"; transcript_type "lncRNA";',
        'chrY\tHAVANA\texon\t1\t6\t.\t+\t.\tgene_id "ENSG3.2_PAR_Y"; transcript_id "ENST3.2_PAR_Y";',
        'GL0001\tHAVANA\tgene\t1\t5\t.\t+\t.\tgene_id "ENSG4.1";',
        'GL0001\tHAVANA\ttranscript\t1\t5\t.\t+\t.\tgene_id "ENSG4.1"; transcript_id "ENST4.1"; transcript_type "lncRNA";',
    ]
    with gzip.open(path, "wt") as fh:
        fh.write("\n".join(rows) + "\n")
    return path


def test_gencode_all_transcripts_keep_structure_phase_and_noncanonical(tmp_path):
    from biodb.gencode import iter_transcripts

    rows = list(iter_transcripts(_gtf(tmp_path), release="50"))
    assert [r["transcript_id"] for r in rows] == ["ENST1.3", "ENST2.1", "ENST3.2_PAR_Y"]
    first = rows[0]
    assert first["gene_id"] == "ENSG1"
    assert first["tags"] == ["MANE_Select", "basic"]
    assert [(e["start"], e["end"]) for e in first["exons"]] == [(400, 500), (100, 200)]
    assert [c["phase"] for c in first["cds"]] == [1, 2]
    assert first["transcript_length"] == 200
    assert first["utr5_length"] == 20
    assert first["utr3_length"] == 50
    assert rows[1]["protein_id"] is None
    assert rows[2]["stable_id"] == "ENST3_PAR_Y"
    assert rows[2]["transcript_version"] == "2"


def test_gencode_accounts_for_reference_and_patch_only_transcripts(tmp_path):
    from biodb.gencode import iter_transcripts

    accounting = {}
    rows = list(iter_transcripts(_gtf(tmp_path), accounting=accounting))
    assert accounting["input_transcript_features"] == 4
    assert accounting["excluded_nonreference_transcripts"] == 1
    assert accounting["selected_transcripts"] == len(rows) == 3


def test_gencode_sequence_index_uses_native_ids_and_protein_header(tmp_path):
    from biodb.gencode import iter_transcripts
    from biodb.transcript_io import SequenceStore

    rna = tmp_path / "rna.fa"
    rna.write_text(
        ">ENST1.3|ENSG1.2|x|y|G|200|protein_coding|\n" + "ACGT" * 50 + "\n"
        ">ENST3.2_PAR_Y|ENSG3.2_PAR_Y|\nAAAAAA\n"
    )
    pep = tmp_path / "pep.fa"
    pep.write_text(">ENSP1.2|ENST1.3|ENSG1.2|x|y|G|2|\nMK\n")
    with SequenceStore.create(tmp_path / "seq.sqlite") as store:
        store.add_fasta(rna, kind="rna")
        store.add_fasta(pep, kind="protein")
        rows = list(iter_transcripts(_gtf(tmp_path), release="50", sequences=store))
        assert rows[0]["sequence"] == "ACGT" * 50
        assert rows[0]["protein_sequence"] == "MK"
        assert rows[0]["sequence_hash"] != rows[2]["sequence_hash"]
        assert store.get("ENST1.9", kind="rna") is None


def test_reject_noncontiguous_gene_blocks_instead_of_duplicate_records(tmp_path):
    from biodb.gencode import iter_transcripts

    path = _gtf(tmp_path)
    with gzip.open(path, "at") as fh:
        fh.write(
            'chr1\tHAVANA\ttranscript\t1\t5\t.\t+\t.\tgene_id "ENSG1.2"; transcript_id "ENST9.1";\n'
        )
    with pytest.raises(ValueError, match="contiguous"):
        list(iter_transcripts(path, release="50"))


@responses.activate
def test_download_manifest_cache_and_checksum(tmp_path):
    from biodb.transcript_io import download_source

    url = "https://example.org/release50/test.txt"
    responses.get(url, body="annotation")
    path = download_source(url, tmp_path / "test.txt", source="test", release="50")
    manifest = json.loads(path.with_name(path.name + ".source.json").read_text())
    assert manifest["release"] == "50"
    assert len(manifest["sha256"]) == 64
    assert download_source(url, path, source="test", release="50") == path
    assert len(responses.calls) == 1
    path.write_text("corrupt")
    with pytest.raises(ValueError, match="checksum"):
        download_source(url, path, source="test", release="50")


def test_appris_parses_missing_scores_as_missing_and_preserves_annotation(tmp_path):
    from biodb.appris import iter_features, iter_scores

    scores = tmp_path / "scores.txt"
    scores.write_text(
        "Ensembl Gene ID\tTranscript ID\tTranslation ID\tTrifid Score\tAPPRIS Annotation\n"
        "ENSG1\tENST1\tENSP1\t0.91\tPRINCIPAL:1\n"
        "ENSG1\tENST2\t-\t-\tMINOR\n"
    )
    records = list(iter_scores(scores, dataset="e116v50"))
    assert records[0]["scores"]["Trifid Score"] == 0.91
    assert records[0]["annotation_release"] == "50"
    assert records[1]["scores"]["Trifid Score"] is None
    assert records[1]["protein_id"] is None
    feature = tmp_path / "spade.gtf"
    feature.write_text(
        'chr1\tSPADE\tdomain_damaged\t111\t130\t2.5\t+\t.\ttranscript_id "ENST1"; note "PF00001";\n'
    )
    row = next(iter_features(feature, dataset="e116v50"))
    assert row["start"] == 110
    assert row["feature_type"] == "domain_damaged"
    assert row["evidence_type"] == "predicted"


def test_mane_retains_exact_refseq_and_ensembl_versions(tmp_path):
    from biodb.mane import iter_mappings

    path = tmp_path / "mane.txt"
    path.write_text(
        "#NCBI_GeneID\tsymbol\tRefSeq_nuc\tRefSeq_prot\tEnsembl_nuc\tEnsembl_prot\tMANE_status\tGRCh38_chr\n"
        "1\tG\tNM_1.2\tNP_1.3\tENST1.3\tENSP1.2\tMANE Plus Clinical\tNC_000001.11\n"
    )
    row = next(iter_mappings(path, release="1.5"))
    assert row["transcript_id"] == "ENST1.3"
    assert row["refseq_transcript_id"] == "NM_1.2"
    assert row["status"] == "MANE Plus Clinical"
    assert row["mapping_basis"] == "mane_exact"


def test_refseq_variant_comment_is_separate_from_gene_summary():
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

    from biodb.refseq import parse_record

    record = SeqRecord(Seq("ATGAAA"), id="NM_1.2", description="G transcript variant 2")
    record.annotations["comment"] = (
        "REVIEWED REFSEQ: curated.\nSummary: Gene-level general description.\n\n"
        "Transcript Variant: This variant lacks exon 3 and encodes the shorter isoform.\n\n"
        "Publication Note: Other publications exist."
    )
    row = parse_record(record)
    assert "shorter isoform" in row["variant_description"]
    assert "Gene-level" not in row["variant_description"]
    assert "Gene-level" in row["gene_context"]
    assert row["transcript_id"] == "NM_1.2"


def test_uniprot_isoform_qualifiers_and_many_to_many_transcripts():
    from biodb.uniprot import iter_isoform_annotations

    entry = {
        "primaryAccession": "P1",
        "comments": [
            {
                "commentType": "ALTERNATIVE PRODUCTS",
                "isoforms": [
                    {
                        "name": {"value": "Long"},
                        "isoformIds": ["P1-1"],
                        "sequenceStatus": "Displayed",
                    },
                    {
                        "name": {"value": "Short"},
                        "isoformIds": ["P1-2"],
                        "sequenceStatus": "Described",
                    },
                ],
            },
            {
                "commentType": "FUNCTION",
                "molecule": "Isoform Long",
                "texts": [{"value": "Inhibits apoptosis"}],
            },
            {"commentType": "FUNCTION", "texts": [{"value": "Gene context only"}]},
        ],
        "uniProtKBCrossReferences": [
            {"database": "Ensembl", "id": "ENST1.2", "isoformId": "P1-1"},
            {"database": "Ensembl", "id": "ENST2.1", "isoformId": "P1-1"},
        ],
        "sequence": {"value": "MK"},
    }
    rows = list(iter_isoform_annotations(entry, release="2026_04"))
    assert {r["transcript_id"] for r in rows} == {"ENST1.2", "ENST2.1"}
    assert all(r["isoform_id"] == "P1-1" for r in rows)
    assert rows[0]["comments"][0]["texts"][0]["value"] == "Inhibits apoptosis"
    assert rows[0]["gene_context"][0]["texts"][0]["value"] == "Gene context only"


def test_gencode_rejects_sequence_length_mismatch(tmp_path):
    from biodb.gencode import iter_transcripts
    from biodb.transcript_io import SequenceStore

    fasta = tmp_path / "bad.fa"
    fasta.write_text(">ENST1.3\nAC\n")
    with SequenceStore.create(tmp_path / "bad.sqlite") as store:
        store.add_fasta(fasta, kind="rna")
        with pytest.raises(ValueError, match="length"):
            list(iter_transcripts(_gtf(tmp_path), release="50", sequences=store))


@responses.activate
def test_uniprot_bulk_rejects_upstream_release_drift(tmp_path):
    from biodb.uniprot import download_human_entries

    responses.get(
        "https://rest.uniprot.org/uniprotkb/stream",
        body=b"wrong snapshot",
        headers={"X-UniProt-Release": "2026_99"},
    )
    with pytest.raises(ValueError, match="release header"):
        download_human_entries(release="2026_03", cache_dir=tmp_path)
    assert not list(tmp_path.rglob("*.source.json"))


@responses.activate
def test_refseq_pinned_default_bulk_url(tmp_path):
    from biodb.refseq import get_dataset

    url = (
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/annotation_releases/9606/"
        "GCF_000001405.40-RS_2025_08/GCF_000001405.40_GRCh38.p14_rna.gbff.gz"
    )
    responses.get(url, body=b"source")
    path = get_dataset(cache_dir=tmp_path)
    assert path.read_bytes() == b"source"
    assert (
        json.loads(path.with_name(path.name + ".source.json").read_text())["release"]
        == "GCF_000001405.40-RS_2025_08"
    )


@responses.activate
def test_gtex_targeted_queries_page_and_pin_release():
    from biodb.gtex import query_expression

    url = "https://gtexportal.org/api/v2/expression/medianTranscriptExpression"
    for page, tx in enumerate(["ENST1.2", "ENST2.1"]):
        responses.get(
            url,
            json={
                "data": [{"transcriptId": tx, "datasetId": "gtex_v11", "median": 0.0}],
                "paging_info": {"page": page, "numberOfPages": 2},
            },
        )
    rows = query_expression("ENSG1.2", dataset="gtex_v11", items_per_page=1)
    assert len(rows) == 2
    assert all("datasetId=gtex_v11" in c.request.url for c in responses.calls)


def test_gtex_usage_is_median_of_sample_fractions_not_ratio_of_medians(tmp_path):
    from biodb.gtex import iter_expression_summaries

    matrix = tmp_path / "tpm.txt"
    matrix.write_text("transcript_id\tS1\tS2\tS3\nENST1.1\t9\t0\t0\nENST2.1\t1\t1\t0\n")
    samples = tmp_path / "samples.txt"
    samples.write_text("SAMPID\tSMTSD\nS1\tBrain\nS2\tBrain\nS3\tBrain\n")
    rows = list(
        iter_expression_summaries(
            matrix,
            samples,
            transcript_to_gene={"ENST1.1": "ENSG1", "ENST2.1": "ENSG1"},
            work_dir=tmp_path,
            dataset="gtex_v11",
        )
    )
    a = next(r for r in rows if r["transcript_id"] == "ENST1.1")
    assert a["median_tpm"] == 0.0
    assert a["median_usage"] == pytest.approx(0.45)
    assert a["n_usage_samples"] == 2
    assert a["n_samples"] == 3
    assert a["detection_fraction"] == pytest.approx(1 / 3)


def test_gtex_unknown_source_transcript_fails_instead_of_biased_denominator(tmp_path):
    from biodb.gtex import iter_expression_summaries

    matrix = tmp_path / "tpm.txt"
    matrix.write_text("transcript_id\tS1\nENST_MISSING.1\t9\n")
    samples = tmp_path / "samples.txt"
    samples.write_text("SAMPID\tSMTSD\nS1\tBrain\n")
    with pytest.raises(ValueError, match="mapping"):
        list(iter_expression_summaries(matrix, samples, transcript_to_gene={}, work_dir=tmp_path))
