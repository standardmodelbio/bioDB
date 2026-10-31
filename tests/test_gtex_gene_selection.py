import pytest


def test_missing_quantified_sibling_fails_before_a_denominator_can_be_biased(tmp_path):
    from biodb.gtex import select_gene_transcript_matrix

    original = tmp_path / "original.tsv"
    original.write_text("transcript_id\tgene_id\tS1\nENST1.1\tENSG1.2\t1\nENST2.1\tENSG1.2\t9\n")
    out = tmp_path / "selected.tsv"
    with pytest.raises(ValueError, match="missing.*mapping"):
        select_gene_transcript_matrix(
            original, out, transcript_to_gene={"ENST1.1": "ENSG1"}, gene_ids={"ENSG1"}
        )
    assert not out.exists()


def test_complete_gene_selection_preserves_per_sample_usage(tmp_path):
    from biodb.gtex import iter_expression_summaries, select_gene_transcript_matrix

    original = tmp_path / "original.tsv"
    original.write_text(
        "transcript_id\tgene_id\tS1\nENST1.1\tENSG1.2\t1\nENST2.1\tENSG1.2\t9\nOTHER.1\tOTHERGENE.1\t8\n"
    )
    mapping = {"ENST1.1": "ENSG1", "ENST2.1": "ENSG1"}
    out = tmp_path / "selected.tsv"
    accounting = select_gene_transcript_matrix(
        original, out, transcript_to_gene=mapping, gene_ids={"ENSG1"}
    )
    samples = tmp_path / "samples.tsv"
    samples.write_text("SAMPID\tSMTSD\nS1\tBrain\n")
    rows = list(
        iter_expression_summaries(out, samples, transcript_to_gene=mapping, work_dir=tmp_path)
    )
    assert rows[0]["median_usage"] == pytest.approx(0.1)
    assert accounting["quantified_transcripts"] == ["ENST1.1", "ENST2.1"]
    assert "OTHER" not in out.read_text()
