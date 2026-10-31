import pytest


@pytest.mark.parametrize("score,phase", [("-", "."), (".", "-")])
def test_appris_method_gtf_dash_means_missing_not_zero(tmp_path, score, phase):
    from biodb.appris import iter_features

    path = tmp_path / "method.gtf"
    path.write_text(
        f'1\tAPPRIS\tprincipal_isoform\t10\t20\t{score}\t+\t{phase}\ttranscript_id "ENST1";\n'
    )
    row = next(iter_features(path))
    assert row["score"] is None
    assert row["phase"] is None
