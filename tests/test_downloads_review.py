"""Independent review regressions for durable download receipts."""

from pathlib import Path

import pytest

from biodb._downloads import stream_to_file
from tests.test_downloads_resume import Response, Session


def test_finished_receipt_does_not_publish_truncated_part(tmp_path):
    target = tmp_path / "file"
    target.write_bytes(b"old")

    def unavailable_validator(path):
        raise RuntimeError("validator not available")

    with pytest.raises(RuntimeError):
        stream_to_file(
            "https://example.test/file",
            target,
            progress=False,
            validate=unavailable_validator,
            session=Session(Response(b"abcdef", headers={"ETag": '"v1"', "Content-Length": "6"})),
        )
    Path(str(target) + ".part").write_bytes(b"abc")
    session = Session(
        Response(
            b"def",
            status=206,
            headers={"ETag": '"v1"', "Content-Length": "3", "Content-Range": "bytes 3-5/6"},
        )
    )
    stream_to_file("https://example.test/file", target, progress=False, session=session)
    assert target.read_bytes() == b"abcdef"
    assert len(session.requests) == 1


def test_nonobject_receipt_restarts_instead_of_crashing(tmp_path):
    target = tmp_path / "file"
    Path(str(target) + ".part").write_bytes(b"partial")
    Path(str(target) + ".part.json").write_text("[]")
    session = Session(Response(b"complete", headers={"Content-Length": "8"}))
    stream_to_file("https://example.test/file", target, progress=False, session=session)
    assert target.read_bytes() == b"complete"
    assert "Range" not in session.requests[0]["headers"]
