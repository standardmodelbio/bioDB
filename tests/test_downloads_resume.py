from pathlib import Path

import pytest
import requests

from biodb._downloads import stream_to_file


class Response:
    def __init__(self, body, *, status=200, headers=None, fail=False):
        self.body, self.status_code = body, status
        self.headers = headers or {}
        self.fail = fail

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def iter_content(self, chunk_size):
        yield self.body
        if self.fail:
            raise requests.ConnectionError("interrupted")


class Session:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.requests = []

    def get(self, url, **kwargs):
        self.requests.append(kwargs)
        return next(self.responses)


def test_interruption_preserves_final_and_resumes_verified_prefix(tmp_path):
    target = tmp_path / "file"
    target.write_bytes(b"old")
    first = Session(Response(b"abc", headers={"ETag": '"v1"', "Content-Length": "6"}, fail=True))
    with pytest.raises(requests.ConnectionError):
        stream_to_file("https://example.test/raw", target, session=first, progress=False)
    assert target.read_bytes() == b"old"
    second = Session(
        Response(
            b"def",
            status=206,
            headers={"ETag": '"v1"', "Content-Range": "bytes 3-5/6", "Content-Length": "3"},
        )
    )
    stream_to_file("https://example.test/raw", target, session=second, progress=False)
    assert target.read_bytes() == b"abcdef"
    assert second.requests[0]["headers"]["Range"] == "bytes=3-"


def test_ignored_range_restarts_without_duplicate_prefix(tmp_path):
    target = tmp_path / "file"
    session = Session(
        Response(b"abc", headers={"ETag": '"v1"', "Content-Length": "6"}, fail=True),
        Response(b"abcdef", headers={"ETag": '"v1"', "Content-Length": "6"}),
    )
    with pytest.raises(requests.ConnectionError):
        stream_to_file("https://example.test/raw", target, session=session, progress=False)
    stream_to_file("https://example.test/raw", target, session=session, progress=False)
    assert target.read_bytes() == b"abcdef"


def test_invalid_payload_never_replaces_valid_destination(tmp_path):
    target = tmp_path / "file"
    target.write_bytes(b"valid")

    def validate(path: Path):
        raise ValueError("invalid format")

    with pytest.raises(ValueError, match="invalid format"):
        stream_to_file(
            "https://example.test/raw",
            target,
            session=Session(Response(b"invalid")),
            validate=validate,
            progress=False,
        )
    assert target.read_bytes() == b"valid"
    assert not Path(str(target) + ".part").exists()


def test_malformed_range_is_retried_fresh(tmp_path):
    target = tmp_path / "file"
    first = Session(Response(b"abc", headers={"ETag": '"v1"', "Content-Length": "6"}, fail=True))
    with pytest.raises(requests.ConnectionError):
        stream_to_file("https://example.test/raw", target, session=first, progress=False)
    second = Session(
        Response(b"xxx", status=206, headers={"ETag": '"changed"', "Content-Range": "bytes 2-4/6"}),
        Response(b"UVWXYZ", headers={"ETag": '"changed"', "Content-Length": "6"}),
    )
    stream_to_file("https://example.test/raw", target, session=second, progress=False)
    assert target.read_bytes() == b"UVWXYZ"
    assert "Range" not in second.requests[1]["headers"]


def test_process_death_releases_lock_and_resumes_real_http(tmp_path):
    import os
    import subprocess
    import sys
    import threading
    import time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    payload = bytes(range(256)) * 1024
    offsets = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            start = int(self.headers.get("Range", "bytes=0-").split("=")[1].split("-")[0])
            offsets.append(start)
            self.send_response(206 if start else 200)
            self.send_header("ETag", '"stable"')
            self.send_header("Content-Length", str(len(payload) - start))
            if start:
                self.send_header(
                    "Content-Range", f"bytes {start}-{len(payload) - 1}/{len(payload)}"
                )
            self.end_headers()
            try:
                for index in range(start, len(payload), 4096):
                    self.wfile.write(payload[index : index + 4096])
                    self.wfile.flush()
                    time.sleep(0.002)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    dst = tmp_path / "payload"
    url = f"http://127.0.0.1:{server.server_port}/payload"
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from biodb._downloads import stream_to_file; import sys; stream_to_file(sys.argv[1],sys.argv[2],progress=False,chunk_size=4096)",
            url,
            str(dst),
        ],
        env=os.environ.copy(),
    )
    try:
        deadline = time.monotonic() + 20
        part = Path(str(dst) + ".part")
        while time.monotonic() < deadline:
            if part.exists() and part.stat().st_size >= 8192:
                break
            if process.poll() is not None:
                pytest.fail("worker completed before interruption")
            time.sleep(0.01)
        else:
            pytest.fail("worker failed to persist progress")
        process.kill()
        process.wait(timeout=5)
        offset = part.stat().st_size
        assert offset < len(payload)
        assert not dst.exists()
        stream_to_file(url, dst, progress=False, chunk_size=4096)
        assert dst.read_bytes() == payload
        assert offsets[-1] == offset > 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
