"""Raw GWAS acquisition and durable per-file completion fixtures."""

import gzip
import json

import polars as pl
import pytest

from biodb import gwas_catalog
from biodb._acquisition import IngestError, file_record

HEADER = "SNP\tCHR\tBP\tA1\tA2\tFRQ\tBETA\tSE\tZ\tP\tN\n"
ROW = "rs1\tX\t100\tA\tG\t0.0545003\t-3.07292\t\t-1.35001010447145\t0\t40943\n"


def harvest(tmp_path, name="GCST1", body=ROW):
    path = tmp_path / f"{name}.tsv.gz"
    path.write_bytes(gzip.compress((HEADER + body).encode()))
    manifest = tmp_path / f"{name}.manifest"
    manifest.write_text(path.as_uri() + "\n")
    return pl.DataFrame([file_record(path.as_uri(), name, "gwas_catalog")]), path


def test_exact_catalog_manifest_keeps_all_files(tmp_path):
    manifest = tmp_path / "manifest"
    manifest.write_text(
        "./GCST1-GCST1000/GCST1/harmonised/alias.h.tsv.gz\n"
        "./GCST1-GCST1000/GCST1/harmonised/GCST1.h.tsv.gz\n"
        "./GCST1-GCST1000/GCST1/harmonised/GCST1.h.tsv.gz.tbi\n"
    )
    files = gwas_catalog.list_datasets(manifest)
    assert isinstance(files, pl.DataFrame)
    assert files.height == 2
    assert files["study_id"].to_list() == ["GCST1", "GCST1"]
    assert files["file_id"].n_unique() == 2
    assert files["url"][0].endswith("/alias.h.tsv.gz")
    assert gwas_catalog.list_datasets(manifest, study_ids=["GCST404"]).is_empty()


def test_raw_precision_missingness_zero_and_concat(tmp_path):
    files, path = harvest(tmp_path)
    path.write_bytes(path.read_bytes() + gzip.compress(ROW.replace("rs1", "rs2").encode()))
    frame = pl.concat(gwas_catalog.iter_sumstats(files.row(0, named=True), batch_size=1))
    assert frame.height == 2
    assert frame["chrom"].to_list() == ["X", "X"]
    assert frame["p"].to_list() == [0.0, 0.0]
    assert frame["se"].null_count() == 2
    assert frame["z"][0] == -1.35001010447145
    assert frame["beta"][0] == -3.07292
    assert frame["n"][0] == 40943.0
    assert frame["effect_allele"][0] == "G"
    assert {v["name"]: v["value"] for v in frame["raw_fields"][0]}["P"] == "0"


def test_catalog_header_aware_ssf_and_legacy(tmp_path):
    new = tmp_path / "new.tsv"
    new.write_text(
        "p_value\tbase_pair_location\tchromosome\teffect_allele\tother_allele\tbeta\tstandard_error\n"
        "0.4\t9\tMT\tA\tC\t0.123456789123456\tNA\n"
    )
    frame = pl.concat(gwas_catalog.iter_sumstats(new))
    assert frame["p"][0] == 0.4  # unfiltered by default
    assert frame["beta"][0] == 0.123456789123456
    assert frame["se"][0] is None
    old = tmp_path / "old.tsv"
    old.write_text(
        "hm_chrom\thm_pos\thm_other_allele\thm_effect_allele\tneg_log_10_p_value\tbeta\n"
        "2\t10\tA\tT\t400\tNA\n"
    )
    frame = pl.concat(gwas_catalog.iter_sumstats(old, p_max=5e-8))
    assert frame.height == 1
    assert frame["neg_log_10_p_value"][0] == 400.0
    assert frame["beta"][0] is None


def test_truncated_gzip_never_commits_success(tmp_path):
    files, path = harvest(tmp_path)
    path.write_bytes(path.read_bytes()[:-6])
    with pytest.raises(IngestError) as error:
        gwas_catalog.ingest_sumstats(files, tmp_path / "run")
    assert error.value.result.failed == 1
    assert error.value.result.rows == 0
    assert error.value.result.artifacts == []


def test_receipts_zero_rows_and_resume_only_failed(tmp_path):
    files, path = harvest(tmp_path, body=ROW.replace("\t0\t40943", "\t0.8\t40943"))
    good = files.row(0, named=True)
    bad = dict(good, file_id="missing", url=(tmp_path / "missing.tsv.gz").as_uri())
    with pytest.raises(IngestError) as error:
        gwas_catalog.ingest_sumstats([good, bad], tmp_path / "run", p_max=5e-8)
    result = error.value.result
    assert (result.succeeded, result.failed, result.rows) == (1, 1, 0)
    receipts = list((tmp_path / "run" / "receipts").glob("*.json"))
    success = [p for p in receipts if json.loads(p.read_text())["status"] == "ok"][0]
    before = success.stat().st_mtime_ns
    with pytest.raises(IngestError):
        gwas_catalog.ingest_sumstats([good, bad], tmp_path / "run", p_max=5e-8, workers=1)
    assert success.stat().st_mtime_ns == before
    with pytest.raises(ValueError, match="configuration|identity|membership"):
        gwas_catalog.ingest_sumstats([good, bad], tmp_path / "run", p_max=0.1)
    with pytest.raises(ValueError, match="configuration|identity|membership"):
        gwas_catalog.ingest_sumstats([good], tmp_path / "run", p_max=5e-8)


def test_regions_explicit_and_raw_chrom_preserved(tmp_path):
    files, _ = harvest(tmp_path)
    with pytest.raises(ValueError, match="scope"):
        list(gwas_catalog.iter_sumstats(files.row(0, named=True), regions=["X:1-200"]))
    frame = pl.concat(
        gwas_catalog.iter_sumstats(
            files.row(0, named=True), scope="regions", regions=["chrX:1-200"]
        )
    )
    assert frame["chrom"].to_list() == ["X"]


def test_verified_artifacts_only_and_input_changes_rejected(tmp_path):
    files, source = harvest(tmp_path)
    result = gwas_catalog.ingest_sumstats(files, tmp_path / "run", batch_size=1)
    (tmp_path / "run" / "artifacts" / "uncommitted.parquet").write_text("untrusted")
    assert result.scan().collect().height == 1
    artifact = result.artifacts[0]
    artifact.write_bytes(b"corrupt")
    repaired = gwas_catalog.ingest_sumstats(files, tmp_path / "run")
    assert repaired.scan().collect()["z"][0] == -1.35001010447145
    source.write_bytes(gzip.compress((HEADER + ROW.replace("rs1", "rs9")).encode()))
    with pytest.raises(ValueError, match="identity"):
        gwas_catalog.ingest_sumstats(files, tmp_path / "run")


def test_subprocess_kill_retains_committed_file_and_retries_uncommitted(tmp_path):
    import os
    import subprocess
    import sys
    import time

    a, _ = harvest(tmp_path, name="GCST1")
    b, _ = harvest(tmp_path, name="GCST2")
    rows = [dict(a.row(0, named=True), file_id="01"), dict(b.row(0, named=True), file_id="02")]
    input_path = tmp_path / "inputs.json"
    input_path.write_text(json.dumps(rows))
    marker = tmp_path / "second-started"
    script = """import json, sys, time
from pathlib import Path
from biodb import gwas_catalog
from biodb import _acquisition as engine
original = engine.iter_file
def blocked(path, **kwargs):
    for frame in original(path, **kwargs):
        if kwargs['file_id'] == '02':
            Path(sys.argv[3]).write_text('started')
            time.sleep(120)
        yield frame
engine.iter_file = blocked
gwas_catalog.ingest_sumstats(json.loads(Path(sys.argv[1]).read_text()), sys.argv[2], workers=1)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(input_path), str(tmp_path / "run"), str(marker)],
        env={
            **os.environ,
            "PYTHONPATH": str(__import__("pathlib").Path(__file__).parents[1] / "src"),
        },
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 40
        while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert marker.exists(), f"child failed before pause: {process.poll()}"
        receipt = next((tmp_path / "run" / "receipts").glob("*.json"))
        before = receipt.read_bytes(), receipt.stat().st_mtime_ns
        process.kill()
        process.wait(timeout=5)
        result = gwas_catalog.ingest_sumstats(rows, tmp_path / "run", workers=2)
        assert result.rows == 2
        assert result.scan().collect()["source_file_id"].sort().to_list() == ["01", "02"]
        assert (receipt.read_bytes(), receipt.stat().st_mtime_ns) == before
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_per_study_regions_are_frozen_and_not_shared(tmp_path):
    one, _ = harvest(tmp_path, name="GCST1")
    two, _ = harvest(tmp_path, name="GCST2")
    files = pl.concat([one, two])
    result = gwas_catalog.ingest_sumstats(
        files,
        tmp_path / "run",
        scope="regions",
        regions={"GCST1": ["X:1-200"], "GCST2": ["2:1-200"]},
    )
    assert result.scan().collect()["study_id"].to_list() == ["GCST1"]
    assert sorted(item["rows"] for item in result.files) == [0, 1]
    with pytest.raises(ValueError, match="region|study"):
        gwas_catalog.ingest_sumstats(
            files, tmp_path / "other", scope="regions", regions={"GCST1": ["X:1-200"]}
        )


def test_load_result_verifies_only_manifest_receipts(tmp_path):
    files, _ = harvest(tmp_path)
    result = gwas_catalog.ingest_sumstats(files, tmp_path / "run")
    loaded = gwas_catalog.load_ingest_result(tmp_path / "run")
    assert loaded.scan().collect().equals(result.scan().collect())
    result.artifacts[0].write_bytes(b"bad")
    with pytest.raises(IngestError):
        gwas_catalog.load_ingest_result(tmp_path / "run")


def test_corrupt_receipt_is_retried_and_wrong_schema_not_adopted(tmp_path):
    files, _ = harvest(tmp_path)
    gwas_catalog.ingest_sumstats(files, tmp_path / "run")
    receipt = next((tmp_path / "run" / "receipts").glob("*.json"))
    receipt.write_text("{incomplete")
    fixed = gwas_catalog.ingest_sumstats(files, tmp_path / "run")
    assert fixed.rows == 1
    state = json.loads(receipt.read_text())
    state["rows"] = 999
    receipt.write_text(json.dumps(state))
    assert gwas_catalog.ingest_sumstats(files, tmp_path / "run").rows == 1


def test_failed_remote_input_is_bound_to_first_server_identity(tmp_path, monkeypatch):
    import requests

    from tests.test_downloads_resume import Response

    payload = gzip.compress((HEADER + ROW).encode())
    replies = iter(
        [
            Response(
                payload[:10],
                headers={"ETag": '"v1"', "Content-Length": str(len(payload))},
                fail=True,
            ),
            Response(payload, headers={"ETag": '"v2"', "Content-Length": str(len(payload))}),
        ]
    )
    monkeypatch.setattr(requests, "get", lambda *_args, **_kwargs: next(replies))
    row = {
        "source": "gwas_catalog",
        "study_id": "GCST1",
        "file_id": "remote",
        "url": "https://source.test/GCST1.tsv.gz",
    }
    options = dict(cache_dir=tmp_path / "cache", workers=1)
    with pytest.raises(IngestError):
        gwas_catalog.ingest_sumstats([row], tmp_path / "run", **options)
    with pytest.raises(IngestError) as error:
        gwas_catalog.ingest_sumstats([row], tmp_path / "run", **options)
    assert "identity changed" in error.value.result.files[0]["detail"]
    assert error.value.result.rows == 0
    assert error.value.result.artifacts == []


@pytest.mark.parametrize("text", ["<html>temporary error</html>", "schema_changed\nunknown_data"])
def test_catalog_manifest_rejects_non_catalog_response(tmp_path, text):
    manifest = tmp_path / "manifest"
    manifest.write_text(text)
    with pytest.raises(ValueError, match="manifest"):
        gwas_catalog.list_datasets(manifest, study_ids=["absent"])


@pytest.mark.parametrize("bad_kind", ["truncated", "bad_crc", "html"])
def test_corrupt_gzip_download_discards_part_and_retries_repaired_source(
    tmp_path, monkeypatch, bad_kind
):
    import requests

    from tests.test_downloads_resume import Response

    payload = gzip.compress((HEADER + ROW).encode())
    bad = {
        "truncated": payload[:-6],
        "bad_crc": payload[:-8] + bytes([payload[-8] ^ 1]) + payload[-7:],
        "html": b"<html>error</html>",
    }[bad_kind]
    replies = iter(
        [
            Response(bad, headers={"ETag": '"v1"', "Content-Length": str(len(bad))}),
            Response(payload, headers={"ETag": '"v2"', "Content-Length": str(len(payload))}),
        ]
    )
    monkeypatch.setattr(requests, "get", lambda *_a, **_k: next(replies))
    row = file_record("https://source.test/stats.tsv.gz", "GCST1", "gwas_catalog")
    with pytest.raises(ValueError):
        gwas_catalog.download_dataset(row, cache_dir=tmp_path, progress=False)
    assert not list(tmp_path.rglob("*.part"))
    assert not list(tmp_path.rglob("*.verified.json"))
    path = gwas_catalog.download_dataset(row, cache_dir=tmp_path, progress=False)
    assert path.read_bytes() == payload


def test_shared_cache_refresh_cannot_replace_frozen_failed_input(tmp_path, monkeypatch):
    import requests

    from biodb._acquisition import download
    from tests.test_downloads_resume import Response

    payload = gzip.compress((HEADER + ROW).encode(), compresslevel=0, mtime=0)
    replacement = gzip.compress(
        (HEADER + ROW.replace("rs1", "rs2")).encode(), compresslevel=0, mtime=0
    )
    assert len(payload) == len(replacement)
    replies = iter(
        [
            Response(
                payload[:10],
                headers={"ETag": '"v1"', "Content-Length": str(len(payload))},
                fail=True,
            ),
            Response(
                replacement, headers={"ETag": '"v2"', "Content-Length": str(len(replacement))}
            ),
        ]
    )
    monkeypatch.setattr(requests, "get", lambda *_a, **_k: next(replies))
    row = file_record("https://source.test/stats.tsv.gz", "GCST1", "gwas_catalog")
    frozen = tmp_path / "identity.json"
    with pytest.raises(requests.ConnectionError):
        download(
            row,
            source="gwas_catalog",
            cache_dir=tmp_path / "cache",
            identity_path=frozen,
            progress=False,
        )
    cached = gwas_catalog.download_dataset(
        row, cache_dir=tmp_path / "cache", force=True, progress=False
    )
    receipt = json.loads(cached.with_name(cached.name + ".verified.json").read_text())
    assert receipt["etag"] == '"v2"'
    assert receipt["total"] == len(replacement)
    assert receipt["sha256"] == __import__("hashlib").sha256(replacement).hexdigest()
    with pytest.raises(ValueError, match="identity changed"):
        download(
            row,
            source="gwas_catalog",
            cache_dir=tmp_path / "cache",
            identity_path=frozen,
            progress=False,
        )


def test_catalog_manifest_validates_unselected_entries_and_http_error_page(tmp_path, monkeypatch):
    import requests

    from tests.test_downloads_resume import Response

    manifest = tmp_path / "manifest"
    manifest.write_text("./GCST1-GCST1000/GCST1/harmonised/exact.h.tsv.gz\n<html>error</html>")
    with pytest.raises(ValueError, match="manifest"):
        gwas_catalog.list_datasets(manifest, study_ids=["GCST404"])
    monkeypatch.setattr(requests, "get", lambda *_a, **_k: Response(b"<html>error</html>"))
    with pytest.raises(ValueError, match="manifest"):
        gwas_catalog.list_datasets("https://source.test/manifest", cache_dir=tmp_path / "cache")


def test_catalog_download_allows_explicit_local_plain_fixture(tmp_path):
    fixture = tmp_path / "plain.tsv"
    fixture.write_text(HEADER + ROW)
    row = file_record(fixture.as_uri(), "GCST1", "gwas_catalog")
    assert gwas_catalog.download_dataset(row) == fixture
    assert pl.concat(gwas_catalog.iter_sumstats(row)).height == 1


def test_catalog_rejects_error_page_before_manifest_cache_publish(tmp_path, monkeypatch):
    import requests

    from tests.test_downloads_resume import Response

    manifest_url = "https://source.test/manifest"
    valid = b"./GCST1-GCST1000/GCST1/harmonised/exact.h.tsv.gz\n"
    replies = iter(
        [Response(b"<html>error</html>"), Response(valid), Response(b"<html>error</html>")]
    )
    monkeypatch.setattr(requests, "get", lambda *_a, **_k: next(replies))
    with pytest.raises(ValueError, match="manifest"):
        gwas_catalog.list_datasets(manifest_url, cache_dir=tmp_path)
    assert not list(tmp_path.rglob("*.txt"))
    assert gwas_catalog.list_datasets(manifest_url, cache_dir=tmp_path).height == 1
    with pytest.raises(ValueError, match="manifest"):
        gwas_catalog.list_datasets(manifest_url, cache_dir=tmp_path, force=True)
    assert gwas_catalog.list_datasets(manifest_url, cache_dir=tmp_path).height == 1
