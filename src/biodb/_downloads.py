"""Atomic downloads with durable, identity-checked HTTP byte resumption.

Only validated files are published at the destination. Interrupted GETs retain
private ``.part`` bytes and a credential-free receipt; callers own retry policy.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path

import requests
from filelock import FileLock
from tqdm import tqdm

_DEFAULT_CHUNK_SIZE = 1 << 16
_DEFAULT_TIMEOUT = 300


def _atomic_json(path: Path, value: dict) -> None:
    fd, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def stream_to_file(
    url: str,
    dst: str | Path,
    *,
    headers: dict | None = None,
    timeout: int = _DEFAULT_TIMEOUT,
    progress: bool = True,
    desc: str | None = None,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    session: requests.Session | None = None,
    post_data: dict | None = None,
    allow_redirects: bool = True,
    resume: bool = True,
    validate: Callable[[Path], None] | None = None,
    on_identity: Callable[[dict], None] | None = None,
) -> Path:
    """Download, validate, then atomically replace ``dst``.

    Existing final files survive failed refreshes. GET prefixes resume only
    with a strong ETag and a matching, identity-encoded Content-Range. Servers
    ignoring ranges restart the single file. POST requests always restart.
    ``resume=False`` discards a previous prefix. No retry loop is implicit.
    The optional validator runs locally before publication; ValueError denotes
    invalid bytes, while other validator failures retain bytes for revalidation.
    Authentication and request values are hashed, never written to receipts.
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = Path(str(dst) + ".part")
    receipt = Path(str(dst) + ".part.json")
    http = session or requests
    request_headers = dict(headers or {})
    # Include session representation/auth context in the digest, without saving
    # cookies or credentials themselves. No secret-bearing request is logged.
    context = {
        "method": "POST" if post_data is not None else "GET",
        "url": url,
        "body": post_data,
        "headers": request_headers,
        "session_headers": dict(getattr(session, "headers", {}) or {}),
        "cookies": getattr(getattr(session, "cookies", None), "get_dict", lambda: {})(),
        "auth": repr(getattr(session, "auth", None)),
    }
    fingerprint = hashlib.sha256(
        json.dumps(context, sort_keys=True, default=str).encode()
    ).hexdigest()
    with FileLock(str(dst) + ".lock"):

        def discard() -> None:
            receipt.unlink(missing_ok=True)
            part.unlink(missing_ok=True)

        saved = {}
        if resume and post_data is None and part.exists():
            with suppress(OSError, ValueError):
                saved = json.loads(receipt.read_text())
        if not isinstance(saved, dict):
            saved = {}
        saved_total = saved.get("total")
        saved_etag = saved.get("etag")
        if (
            (saved_total is not None and (type(saved_total) is not int or saved_total < 0))
            or (saved_etag is not None and not isinstance(saved_etag, str))
            or ("finished" in saved and type(saved["finished"]) is not bool)
        ):
            saved = {}
        if saved.get("fingerprint") != fingerprint:
            discard()
            saved = {}
        offset = part.stat().st_size if part.exists() else 0
        total = saved.get("total")
        if offset and total is not None and offset > total:
            discard()
            saved, offset = {}, 0

        def publish() -> Path:
            if on_identity is not None:
                on_identity({"etag": saved.get("etag"), "total": saved.get("total")})
            if validate is not None:
                try:
                    validate(part)
                except ValueError:
                    discard()
                    raise
            os.replace(part, dst)
            receipt.unlink(missing_ok=True)
            return dst

        # A process may have died after its final write but before rename.
        if saved.get("finished") and part.exists():
            if total is None or offset == total:
                return publish()
            # The completed receipt survived but local bytes were truncated.
            # Resume the verified prefix (or restart below), never publish it.
            saved["finished"] = False
        if offset and total == offset and saved.get("etag"):
            return publish()
        etag = saved.get("etag")
        resumable = bool(offset and etag and not etag.startswith("W/") and post_data is None)
        if offset and not resumable:
            discard()
            offset, saved = 0, {}
        # At most one protocol-recovery restart; transport retries belong to caller.
        for _protocol_attempt in range(2):
            outgoing = dict(request_headers)
            if post_data is None:
                outgoing.setdefault("Accept-Encoding", "identity")
            if offset:
                outgoing.update({"Range": f"bytes={offset}-", "If-Range": saved["etag"]})
            kwargs = dict(
                stream=True, timeout=timeout, headers=outgoing, allow_redirects=allow_redirects
            )
            response = (
                http.post(url, data=post_data, **kwargs)
                if post_data is not None
                else http.get(url, **kwargs)
            )
            with response:
                encoded = response.headers.get(
                    "Content-Encoding", response.headers.get("content-encoding", "identity")
                ) not in ("identity", "")
                current_etag = response.headers.get("ETag", response.headers.get("etag"))
                length = response.headers.get(
                    "Content-Length", response.headers.get("content-length")
                )
                expected_response = int(length) if length is not None and not encoded else None
                if offset and response.status_code == 206:
                    match = re.fullmatch(
                        r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", "")
                    )
                    valid = bool(match and not encoded and current_etag == saved["etag"])
                    if valid:
                        start, end, full = map(int, match.groups())
                        valid = (
                            start == offset
                            and end >= start
                            and end < full
                            and full == saved.get("total")
                        )
                        if valid:
                            expected_response = end - start + 1
                            total = full
                    if not valid:
                        discard()
                        offset, saved = 0, {}
                        continue
                elif offset and response.status_code == 416:
                    discard()
                    offset, saved = 0, {}
                    continue
                else:
                    response.raise_for_status()
                    if response.status_code == 206:
                        raise requests.HTTPError(
                            "Unsolicited partial response cannot establish a complete file"
                        )
                    # A 200 response to Range is a complete replacement, never appended.
                    receipt.unlink(missing_ok=True)
                    offset = 0
                    total = expected_response
                response.raise_for_status()
                state = {
                    "fingerprint": fingerprint,
                    "etag": current_etag if not encoded else None,
                    "total": total,
                    "finished": False,
                }
                if state["etag"] and state["etag"].startswith("W/"):
                    state["etag"] = None
                if on_identity is not None:
                    on_identity({"etag": state["etag"], "total": state["total"]})
                saved = state
                if not offset:
                    # Invalidate old receipt before truncating; next receipt is durable
                    # before any new bytes are accepted as a resumable prefix.
                    receipt.unlink(missing_ok=True)
                    part.write_bytes(b"")
                _atomic_json(receipt, state)
                received = 0
                with (
                    part.open("ab") as handle,
                    tqdm(
                        total=total,
                        initial=offset,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=desc or dst.name,
                        disable=not progress,
                        leave=False,
                    ) as bar,
                ):
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            handle.write(chunk)
                            received += len(chunk)
                            bar.update(len(chunk))
                    handle.flush()
                    os.fsync(handle.fileno())
                if expected_response is not None and received != expected_response:
                    raise requests.ConnectionError(
                        f"Incomplete response: received {received} of {expected_response} bytes"
                    )
                if total is not None and part.stat().st_size != total:
                    raise requests.ConnectionError("Incomplete downloaded representation")
                state["finished"] = True
                _atomic_json(receipt, state)
                return publish()
        raise requests.HTTPError("Server returned inconsistent byte ranges twice")


__all__ = ["stream_to_file"]
