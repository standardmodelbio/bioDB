"""Harvest provider-specific manifest, local-root, schema and precision fixtures."""

import gzip

import pytest

from biodb import gwas_harvest

HEADER = "SNP\tCHR\tBP\tA1\tA2\tFRQ\tBETA\tSE\tZ\tP\tN\n"
ROW = "rs1\tX\t100\tA\tG\t0.0545003\t-3.07292\t\t-1.35001010447145\t0\t40943\n"


def test_harvest_manifest_identity_local_root_and_precision(tmp_path):
    stem = "21833088-GCST001198-EFO_0003885_buildGRCh37"
    manifest = tmp_path / "manifest"
    url = f"https://source.test/dl/{stem}/{stem}.tsv.gz"
    manifest.write_text(url + "\n")
    files = gwas_harvest.list_datasets(manifest)
    assert files["study_id"][0] == stem
    assert files["gcst_id"][0] == "GCST001198"
    directory = tmp_path / "raw" / stem
    directory.mkdir(parents=True)
    path = directory / f"{stem}.tsv.gz"
    path.write_bytes(
        gzip.compress((HEADER + ROW).encode()) + gzip.compress(ROW.replace("rs1", "rs2").encode())
    )
    result = gwas_harvest.ingest_sumstats(files, tmp_path / "run", local_root=tmp_path / "raw")
    frame = result.scan().collect()
    assert result.rows == 2
    assert frame["chrom"].to_list() == ["X", "X"]
    assert frame["z"][0] == -1.35001010447145
    assert frame["se"][0] is None
    assert frame["p"][0] == 0.0
    assert frame["effect_allele"][0] == "G"
    assert gwas_harvest.load_ingest_result(tmp_path / "run").rows == 2
    assert gwas_harvest.resolve_local(url, tmp_path / "raw") == path
    assert gwas_harvest.list_datasets(manifest, study_ids=["GCST001198"]).height == 1


def test_harvest_requires_observed_schema_and_valid_eof(tmp_path):
    path = tmp_path / "GCST1.tsv.gz"
    path.write_bytes(gzip.compress(b"CHROM\tPOS\tPVAL\n1\t100\t0\n"))
    with pytest.raises(ValueError, match="Missing columns"):
        list(gwas_harvest.iter_sumstats(path))
    path.write_bytes(gzip.compress((HEADER + ROW).encode())[:-6])
    with pytest.raises(EOFError):
        list(gwas_harvest.iter_sumstats(path))


def test_harvest_rejects_error_page_before_manifest_cache_publish(tmp_path, monkeypatch):
    import requests

    from tests.test_downloads_resume import Response

    manifest_url = "https://source.test/manifest"
    valid = b"https://source.test/dl/GCST1/GCST1.tsv.gz\n"
    replies = iter(
        [Response(b"<html>error</html>"), Response(valid), Response(b"<html>error</html>")]
    )
    monkeypatch.setattr(requests, "get", lambda *_a, **_k: next(replies))
    with pytest.raises(ValueError):
        gwas_harvest.list_datasets(manifest_url, cache_dir=tmp_path)
    assert not list(tmp_path.rglob("*.txt"))
    assert gwas_harvest.list_datasets(manifest_url, cache_dir=tmp_path).height == 1
    with pytest.raises(ValueError):
        gwas_harvest.list_datasets(manifest_url, cache_dir=tmp_path, force=True)
    assert gwas_harvest.list_datasets(manifest_url, cache_dir=tmp_path).height == 1
