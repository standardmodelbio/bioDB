"""Raw OpenTargets intermediate literature-match rows."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

INTERMEDIATE_TEMPLATE = "https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/{version}/intermediate/literature_match/"


def fetch_literature_matches(
    version: str,
    wanted_pairs: set[tuple[str, str]],
    *,
    cache_dir: str | Path | None = None,
    limit_files: int | None = None,
    session: Any = None,
    timeout: float = 60.0,
    allow_partial: bool = False,
    entity_type: str = "CD",
) -> list[dict[str, Any]]:
    """Return full matching rows without document-specific ranking or grouping.

    Shard identity/completion and validation are delegated to the public OT
    cache API, including intermediate paths. Partial acquisition is opt-in.
    """
    if not wanted_pairs:
        return []
    from biodb.opentargets import ensure_cached_shards

    try:
        shards = ensure_cached_shards(
            "literature_match",
            version=version,
            cache_dir=cache_dir,
            limit_files=limit_files,
            remote_url=INTERMEDIATE_TEMPLATE.format(version=version),
            session=session,
            timeout=timeout,
        )
    except Exception as exc:
        if not allow_partial:
            raise
        warnings.warn(f"Partial literature acquisition: {exc}", RuntimeWarning, stacklevel=2)
        return []
    rows = []
    for shard in shards:
        try:
            for batch in pq.ParquetFile(shard).iter_batches(batch_size=100_000):
                for row in batch.to_pylist():
                    if (
                        row.get("type") == entity_type
                        and (str(row.get("keywordId")), str(row.get("pmid"))) in wanted_pairs
                    ):
                        rows.append(row)
        except Exception as exc:
            if not allow_partial:
                raise
            warnings.warn(
                f"Partial literature acquisition: shard {shard} failed: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
    return rows
