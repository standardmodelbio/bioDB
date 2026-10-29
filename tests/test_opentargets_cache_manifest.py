"""Completeness receipts use tiny real Parquet files and no public cache."""

import shutil
from pathlib import Path

import polars as pl
import pytest

from biodb.opentargets import _bulk


@pytest.fixture
def source(tmp_path, monkeypatch):
    remote = tmp_path / "remote"
    remote.mkdir()
    for name, value in [("a.parquet", 1), ("b.parquet", 2)]:
        pl.DataFrame({"value": [value]}).write_parquet(remote / name)
    urls = [f"https://example.org/source/{n}" for n in ["a.parquet", "b.parquet"]]
    monkeypatch.setattr(_bulk, "_list_parquet_files", lambda url: urls)

    def download(url, cache_dir, force=False):
        path = cache_dir / Path(url).name
        if force or not path.exists():
            shutil.copyfile(remote / path.name, path)
        return path

    monkeypatch.setattr(_bulk, "_download_to_cache", download)
    return tmp_path / "cache", remote


def test_preview_then_full_fetches_remaining_shards(source):
    cache, _ = source
    assert len(_bulk.ensure_cached_shards("variant", cache_dir=cache, limit_files=1)) == 1
    files = _bulk.ensure_cached_shards("variant", cache_dir=cache)
    assert [pl.read_parquet(p)["value"][0] for p in files] == [1, 2]


def test_manifest_exact_membership_repairs_and_works_offline(source, monkeypatch):
    cache, _ = source
    files = _bulk.ensure_cached_shards("variant", cache_dir=cache)
    pl.DataFrame({"value": [99]}).write_parquet(files[0].parent / "extra.parquet")
    monkeypatch.setattr(_bulk, "_list_parquet_files", lambda url: pytest.fail("offline listing"))
    assert _bulk.ensure_cached_shards("variant", cache_dir=cache) == files
    files[0].write_bytes(b"truncated")
    files[1].unlink()
    repaired = _bulk.ensure_cached_shards("variant", cache_dir=cache)
    assert [pl.read_parquet(p)["value"][0] for p in repaired] == [1, 2]


def test_legacy_cache_requires_listing_before_adoption(source, monkeypatch):
    cache, remote = source
    legacy = cache / _bulk.DEFAULT_VERSION / "variant"
    legacy.mkdir(parents=True)
    shutil.copyfile(remote / "a.parquet", legacy / "a.parquet")
    monkeypatch.setattr(
        _bulk, "_list_parquet_files", lambda url: (_ for _ in ()).throw(OSError("offline"))
    )
    with pytest.raises(OSError, match="offline"):
        _bulk.ensure_cached_shards("variant", cache_dir=cache)


def test_custom_remote_sources_with_same_tail_never_share_cache(source, monkeypatch):
    cache, _ = source
    monkeypatch.setattr(_bulk, "_list_parquet_files", lambda url: [url + "/a.parquet"])
    one = _bulk.get_dataset(
        remote_url="https://one.example/variant", cache_dir=cache, output_format="polars"
    )

    # Changing source produces a fresh acquisition even with identical shard names.
    def download(url, cache_dir, force=False):
        path = cache_dir / "a.parquet"
        pl.DataFrame({"value": [2 if "two.example" in url else 1]}).write_parquet(path)
        return path

    monkeypatch.setattr(_bulk, "_download_to_cache", download)
    two = _bulk.get_dataset(
        remote_url="https://two.example/variant", cache_dir=cache, output_format="polars"
    )
    assert one["value"].to_list() == [1]
    assert two["value"].to_list() == [2]


@pytest.mark.parametrize("limit", [0, -1])
def test_invalid_preview_limit_never_claims_completeness(source, limit):
    cache, _ = source
    with pytest.raises(ValueError, match="limit_files"):
        _bulk.ensure_cached_shards("variant", cache_dir=cache, limit_files=limit)


def test_interrupted_shard_resumes_and_failed_refresh_preserves_final(tmp_path):
    import requests

    parquet = tmp_path / "remote.parquet"
    pl.DataFrame({"value": list(range(300_000))}).write_parquet(parquet, compression="uncompressed")
    body = parquet.read_bytes()
    prefix = 1 << 20
    assert len(body) > prefix

    class Response:
        def __init__(self, status, headers, data, fail=False):
            self.status_code, self.headers, self.data, self.fail = status, headers, data, fail
            self.text = '<a href="part.parquet">part</a>'

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError("upstream failed")

        def iter_content(self, chunk_size):
            yield self.data
            if self.fail:
                raise requests.ConnectionError("interrupted")

    class Session:
        def __init__(self):
            self.phase, self.requests = 0, []

        def get(self, url, **kwargs):
            if url.endswith("/"):
                return Response(200, {}, b"")
            self.requests.append(kwargs.get("headers", {}))
            self.phase += 1
            if self.phase == 1:
                return Response(
                    200,
                    {"Content-Length": str(len(body)), "ETag": '"v1"'},
                    body[:prefix],
                    fail=True,
                )
            if self.phase == 2:
                return Response(
                    206,
                    {
                        "Content-Length": str(len(body) - prefix),
                        "Content-Range": f"bytes {prefix}-{len(body) - 1}/{len(body)}",
                        "ETag": '"v1"',
                    },
                    body[prefix:],
                )
            return Response(200, {"Content-Length": "7", "ETag": '"v2"'}, b"invalid")

    session = Session()
    kwargs = dict(
        remote_url="https://example.org/custom", cache_dir=tmp_path / "cache", session=session
    )
    with pytest.raises(requests.ConnectionError, match="interrupted"):
        _bulk.ensure_cached_shards("custom", **kwargs)
    files = _bulk.ensure_cached_shards("custom", **kwargs)
    assert files[0].read_bytes() == body
    assert session.requests[1]["Range"] == f"bytes={prefix}-"
    with pytest.raises(ValueError):
        _bulk.ensure_cached_shards("custom", force=True, **kwargs)
    assert files[0].read_bytes() == body


def test_failed_force_refresh_is_retried_on_next_normal_run(source, monkeypatch):
    cache, remote = source
    old_files = _bulk.ensure_cached_shards("variant", cache_dir=cache)
    pl.DataFrame({"value": [100]}).write_parquet(remote / "a.parquet")
    pl.DataFrame({"value": [200]}).write_parquet(remote / "b.parquet")
    download = _bulk._download_to_cache
    failed = False
    calls = []

    def fail_second_once(url, cache_dir, force=False):
        nonlocal failed
        calls.append(Path(url).name)
        if url.endswith("b.parquet") and not failed:
            failed = True
            raise OSError("refresh interrupted")
        return download(url, cache_dir, force=force)

    monkeypatch.setattr(_bulk, "_download_to_cache", fail_second_once)
    with pytest.raises(OSError, match="refresh interrupted"):
        _bulk.ensure_cached_shards("variant", cache_dir=cache, force=True)
    assert pl.read_parquet(old_files[1])["value"][0] == 2
    calls.clear()
    files = _bulk.ensure_cached_shards("variant", cache_dir=cache)
    assert [pl.read_parquet(p)["value"][0] for p in files] == [100, 200]
    assert calls == ["b.parquet"]
