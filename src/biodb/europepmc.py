"""EuropePMC raw bibliographic acquisition with per-PMID completion receipts."""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import requests

from biodb import _response_cache as cache

SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def fetch_metadata(
    pmids: Iterable[str],
    *,
    batch_size: int = 80,
    session: Any = None,
    timeout: float = 30.0,
    max_retries: int = 3,
    cache_dir: str | Path | None = None,
    allow_partial: bool = False,
    force: bool = False,
) -> dict[str, dict[str, Any]]:
    """Return original MED records keyed by PMID; successful zero matches persist.

    Failed requests never produce completion receipts. Each validated successful
    batch commits its per-ID results before the next request, permitting restart.
    """
    if batch_size < 1 or max_retries < 1:
        raise ValueError("batch_size and max_retries must be positive")
    ids = list(dict.fromkeys(str(p).strip() for p in pmids if str(p).strip().isdigit()))
    out = {}
    sess = session or requests.Session()
    # Serialize this provider's batch planning as well as per-ID updates.
    with cache.locked(cache.cache_path(cache_dir, "europepmc", "batch-lock")):
        needed = []
        for pmid in ids:
            identity = f"{SEARCH_URL}|MED|core|{pmid}"
            try:
                if force:
                    raise KeyError(pmid)
                record = cache.read(cache.cache_path(cache_dir, "europepmc", identity), identity)
                if record is not None:
                    out[pmid] = record
            except KeyError:
                needed.append(pmid)
        for start in range(0, len(needed), batch_size):
            batch = needed[start : start + batch_size]
            params = {
                "query": "(" + " OR ".join(f"EXT_ID:{p}" for p in batch) + ") AND SRC:MED",
                "format": "json",
                "resultType": "core",
                "pageSize": str(len(batch)),
            }
            error = None
            for _ in range(max_retries):
                try:
                    response = sess.get(SEARCH_URL, params=params, timeout=timeout)
                    response.raise_for_status()
                    data = response.json()
                    records = data["resultList"]["result"]
                    if not isinstance(records, list) or any(
                        not isinstance(r, dict) for r in records
                    ):
                        raise ValueError("Malformed EuropePMC resultList")
                    found = {str(r.get("pmid") or r.get("id")): r for r in records}
                    # A truncated response cannot prove absence for the batch.
                    if int(data.get("hitCount", len(records))) > len(records):
                        raise ValueError("EuropePMC response was truncated")
                    for pmid in batch:
                        identity = f"{SEARCH_URL}|MED|core|{pmid}"
                        record = found.get(pmid)
                        cache.write(
                            cache.cache_path(cache_dir, "europepmc", identity), identity, record
                        )
                        if record is not None:
                            out[pmid] = record
                    error = None
                    break
                except Exception as exc:
                    error = exc
            if error is not None:
                if not allow_partial:
                    raise error
                warnings.warn(
                    f"Partial EuropePMC acquisition: batch {batch} failed: {error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
    return out


fetch_europepmc_metadata = fetch_metadata
