"""Identity-pinned tabix regional views for exact GWAS Catalog source files.

Only a confirmed absent index permits full-file fallback. HTTP failures, missing
validators, schema failures and incomplete iterators remain failures. A view hash
identifies retrieved regional text, never the bytes of the full source file.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path

import requests

from biodb._sumstats import normalize_regions


def probe(url):
    with requests.head(
        url, headers={"Accept-Encoding": "identity"}, timeout=60, allow_redirects=True
    ) as response:
        if response.status_code in {404, 410}:
            return None
        response.raise_for_status()
        etag = response.headers.get("ETag")
        if not etag or etag.startswith("W/"):
            raise ValueError(f"Indexed source requires a strong ETag: {url}")
        if response.headers.get("Content-Encoding", "identity") not in {"identity", ""}:
            raise ValueError("Encoded indexed source cannot establish byte identity")
        length = response.headers.get("Content-Length")
        return {"etag": etag, "total": int(length) if length is not None else None}


def merged_regions(regions):
    merged = []
    for chrom, start, end in normalize_regions("regions", regions):
        if merged and chrom == merged[-1][0] and start <= merged[-1][2] + 1:
            merged[-1] = (chrom, merged[-1][1], max(end, merged[-1][2]))
        else:
            merged.append((chrom, start, end))
    return merged


def acquire(dataset, *, regions, cache_dir, bind_identity, backend="auto"):
    """Return (regional TSV path or None for known no-index fallback, metadata)."""
    url = dataset["url"]
    index_url = dataset.get("index_url") or url + ".tbi"
    index_identity = probe(index_url)
    if index_identity is None:
        if backend == "tabix":
            raise FileNotFoundError(f"Explicit tabix backend index unavailable: {index_url}")
        return None, {
            "transport": "full",
            "fallback_reason": "index_missing",
            "index_url": index_url,
        }
    source_identity = probe(url)
    if source_identity is None:
        raise FileNotFoundError(f"Indexed source unavailable: {url}")
    identity = {
        **source_identity,
        "index_etag": index_identity["etag"],
        "index_total": index_identity["total"],
        "index_url": index_url,
    }
    bind_identity(identity)
    try:
        import pysam
    except ImportError as exc:
        raise ImportError(
            "Indexed Catalog regions require pysam; install biodb[gwas] "
            'or explicitly choose backend="full"'
        ) from exc
    normalized = merged_regions(regions)
    key = hashlib.sha256(
        json.dumps([url, identity, normalized], sort_keys=True).encode()
    ).hexdigest()
    root = Path(cache_dir).expanduser() / "regional_views" / key
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "view.tsv"
    temporary = root / (uuid.uuid4().hex + ".tmp")
    tabix = None
    try:
        tabix = pysam.TabixFile(url, index=index_url)
        headers = list(tabix.header)
        if not headers:
            raise ValueError("Indexed source has no readable header; no positional parser fallback")
        header = next(
            (line for line in reversed(headers) if "\t" in line and not line.startswith("##")), None
        )
        if header is None:
            raise ValueError("Indexed source has no tabular header")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(header + "\n")
            contigs = set(tabix.contigs)
            for chrom, start, end in normalized:
                # Absent indexed contigs are successful empty regions. This is
                # different from a failed range fetch, whose exception propagates.
                candidates = [token for token in (chrom, "chr" + chrom) if token in contigs]
                if len(candidates) > 1:
                    raise ValueError(f"Ambiguous indexed chromosome naming: {chrom}")
                if not candidates:
                    continue
                for line in tabix.fetch(candidates[0], max(0, start - 1), end):
                    handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        # A changed representation mid-query invalidates every row from this job.
        if probe(url) != source_identity or probe(index_url) != index_identity:
            raise ValueError("Indexed source/index identity changed during range acquisition")
        os.replace(temporary, destination)
    finally:
        if tabix is not None:
            tabix.close()
        temporary.unlink(missing_ok=True)
    return destination, {
        "transport": "tabix",
        "source_etag": source_identity["etag"],
        "source_size": source_identity["total"],
        "index_url": index_url,
        "index_etag": index_identity["etag"],
        "index_size": index_identity["total"],
        "source_sha256_scope": "regional_view",
    }
