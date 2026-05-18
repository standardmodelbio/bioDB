"""Shared bulk-download helper — chunked stream + tqdm progress bar.

Internal module (underscore-prefixed). Every first-party bulk-download
function in :mod:`biodb` should funnel through :func:`stream_to_file`
so users get consistent progress feedback regardless of which source
they're pulling.

Why this exists
---------------
Each source module originally had its own near-identical "GET stream
+ iter_content + write to disk" loop. None of them showed progress, so
a 90 MB Swiss-Prot pull or a 311 MB Monarch KG bundle looked like a
hung script. Pulling them through one helper means:

* Users always see a tqdm bar (or a clean fallback when the server
  omits ``Content-Length``).
* The chunk size, retry/timeout posture, and parent-directory creation
  are written down in exactly one place.
* New source modules pick up the same behaviour by default.

Callers can opt out with ``progress=False`` — useful in test
environments or when a higher layer is already managing progress.
"""

from __future__ import annotations

from pathlib import Path

import requests
from tqdm import tqdm

_DEFAULT_CHUNK_SIZE = 1 << 16  # 64 KiB — same as the prior in-module loops
_DEFAULT_TIMEOUT = 300


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
) -> Path:
    """Stream ``url`` to ``dst`` in chunks, with a tqdm progress bar.

    Parameters
    ----------
    url, dst
        Source URL and destination file path. The parent directory of
        ``dst`` is created if missing.
    headers
        Optional HTTP headers (e.g. ``{"User-Agent": "..."}``).
    timeout
        Per-request timeout in seconds. Defaults to 300 — large enough
        for a 100-MB-class artifact on a normal connection.
    progress
        If True (default), show a tqdm bar in bytes. If the server
        omits ``Content-Length``, the bar runs in indeterminate-total
        mode (still reports MB transferred).
    desc
        Optional tqdm bar description; defaults to the destination
        filename.
    chunk_size
        Bytes per read+write cycle. 64 KiB is the project default.
    session
        Optional ``requests.Session`` — pass one in if you've already
        configured retries / cookies / a CSRF flow (e.g. GWAS Atlas).
    post_data
        If given, sends a POST with this form body instead of a GET.
        Lets the same helper power Laravel-form CSRF-protected
        downloads.
    allow_redirects
        Forwarded to ``requests``. Default True.

    Returns
    -------
    pathlib.Path
        The destination path (cast to ``Path``).
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    http = session or requests
    method_kwargs: dict = {
        "stream": True,
        "timeout": timeout,
        "headers": headers,
        "allow_redirects": allow_redirects,
    }
    if post_data is not None:
        response = http.post(url, data=post_data, **method_kwargs)
    else:
        response = http.get(url, **method_kwargs)
    with response:
        response.raise_for_status()
        total = int(response.headers.get("content-length") or 0) or None
        bar_desc = desc or dst.name
        with (
            open(dst, "wb") as f,
            tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=bar_desc,
                disable=not progress,
                leave=False,
            ) as bar,
        ):
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                bar.update(len(chunk))
    return dst


__all__ = ["stream_to_file"]
