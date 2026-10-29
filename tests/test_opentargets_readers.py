"""Raw readers preserve data while leaving projection in the Parquet scan."""

import shutil
from pathlib import Path

import polars as pl

from biodb import opentargets as ot
from biodb.opentargets import _bulk


def test_scan_dataset_full_after_preview_and_convenience_datasets(tmp_path, monkeypatch):
    remote = tmp_path / "source.parquet"
    pl.DataFrame({"id": ["ID1"], "name": ["raw name"], "nested": [[{"x": 1}]]}).write_parquet(
        remote
    )
    monkeypatch.setattr(_bulk, "_list_parquet_files", lambda url: [url + "/part.parquet"])
    sources = []

    def download(url, cache_dir, force=False):
        sources.append(url.split("/")[-2])
        path = cache_dir / Path(url).name
        shutil.copyfile(remote, path)
        return path

    monkeypatch.setattr(_bulk, "_download_to_cache", download)
    scan = ot.scan_dataset("disease", cache_dir=tmp_path / "cache", columns=["id"])
    assert isinstance(scan, pl.LazyFrame)
    assert "PROJECT 1/3 COLUMNS" in scan.explain()
    assert scan.collect().to_dicts() == [{"id": "ID1"}]
    assert ot.get_diseases(cache_dir=tmp_path / "cache")["nested"].to_list() == [[{"x": 1}]]
    assert ot.get_traits(cache_dir=tmp_path / "cache")["name"].to_list() == ["raw name"]
    assert ot.get_drugs(cache_dir=tmp_path / "cache")["name"].to_list() == ["raw name"]
    assert sources == ["disease", "drug_molecule"]


def test_remote_scan_uses_listing_and_projection_without_bulk_download(tmp_path, monkeypatch):
    source = tmp_path / "source.parquet"
    pl.DataFrame({"variantId": ["v1"], "large": [[1, 2, 3]]}).write_parquet(source)
    monkeypatch.setattr(_bulk, "_list_parquet_files", lambda url: [str(source)])
    monkeypatch.setattr(
        _bulk,
        "_download_to_cache",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("bulk download")),
    )
    lazy = ot.scan_variant_annotations(remote=True, columns=["variantId"])
    assert "PROJECT 1/2 COLUMNS" in lazy.explain()
    assert lazy.collect().to_dicts() == [{"variantId": "v1"}]
