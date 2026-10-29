"""Raw AlphaFold and RCSB metadata, cached independently per provider/accession."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

from biodb import _response_cache as cache

ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"
RCSB_HOLDINGS_API = "https://data.rcsb.org/rest/v1/holdings/uniprot/{accession}"


def fetch_provider(
    accession: str,
    provider: str,
    *,
    timeout: float = 30.0,
    session: Any = None,
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> Any:
    """Return the complete provider JSON. A 404 is a completed empty result."""
    template = {"alphafold": ALPHAFOLD_API, "rcsb": RCSB_HOLDINGS_API}[provider]
    url = template.format(accession=quote(accession, safe=""))
    path = cache.cache_path(cache_dir, "protein_structure/" + provider, url)
    with cache.locked(path):
        if not force:
            try:
                return cache.read(path, url)
            except KeyError:
                pass
        response = (session or requests).get(url, timeout=timeout)
        if response.status_code == 404:
            data = []
        else:
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, (dict, list)):
                raise ValueError(f"Malformed {provider} response")
        cache.write(path, url, data)
        return data


def fetch_structure_for_accession(
    accession: str,
    *,
    timeout: float = 30.0,
    session: Any = None,
    cache_dir: str | Path | None = None,
    allow_partial: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Return raw responses; retry only missing/failed providers on restart."""
    out = {"uniprot_accession": accession}
    errors = []
    for provider in ("alphafold", "rcsb"):
        try:
            out[provider] = fetch_provider(
                accession,
                provider,
                timeout=timeout,
                session=session,
                cache_dir=cache_dir,
                force=force,
            )
        except Exception as exc:
            errors.append(exc)
    if errors:
        if not allow_partial:
            raise errors[0]
        warnings.warn(
            f"Partial protein structure acquisition: {len(errors)} failure(s): {errors[0]}",
            RuntimeWarning,
            stacklevel=2,
        )
    return out


def build_structure_cache(
    accessions: list[str],
    cache_path: str | Path,
    *,
    concurrency: int = 8,
    fetch_fn: Any = None,
    allow_partial: bool = False,
) -> dict[str, dict[str, Any]]:
    """Persist each successful accession immediately and retry failures on restart.

    The parquet snapshot is an export; completion is established by per-input
    receipts, never by accession presence in an older (possibly failed) export.
    A supplied fetch_fn is caller-owned normalization and remains injectable.
    """
    import os
    import tempfile
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import polars as pl

    path = Path(cache_path)
    root = path.with_suffix(path.suffix + ".responses")
    fetch = fetch_fn or (lambda acc: fetch_structure_for_accession(acc, cache_dir=root))
    identity_prefix = "structure-v1|" + ("injected" if fetch_fn else "raw") + "|"
    result = {}
    errors = []

    def one(acc: str) -> dict[str, Any]:
        identity = identity_prefix + acc
        receipt = cache.cache_path(root, "accessions", identity)
        with cache.locked(receipt):
            try:
                return cache.read(receipt, identity)
            except KeyError:
                pass
            row = fetch(acc)
            if not isinstance(row, dict) or row.get("uniprot_accession") != acc:
                raise ValueError(f"Malformed structure record for {acc}")
            cache.write(receipt, identity, row)
            return row

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(one, acc): acc for acc in dict.fromkeys(accessions) if acc}
        for future in as_completed(futures):
            try:
                result[futures[future]] = future.result()
            except Exception as exc:
                errors.append(exc)
    if errors:
        if not allow_partial:
            raise errors[0]
        warnings.warn(
            f"Partial protein structure acquisition: {len(errors)} failure(s): {errors[0]}",
            RuntimeWarning,
            stacklevel=2,
        )
    if result:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
        os.close(fd)
        try:
            pl.from_dicts(list(result.values()), infer_schema_length=None).write_parquet(temporary)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return result
