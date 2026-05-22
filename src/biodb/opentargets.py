"""
OpenTargets Platform API integration for querying gene information.

This module provides functions to:
1. Get all gene names available in OpenTargets Platform
2. Get detailed information for each gene, especially associated phenotypes

This module uses gget (https://www.gget.bio) as the backend for querying OpenTargets Platform.

References:
- gget: https://pachterlab.github.io/gget/en/opentargets.html
- OpenTargets Platform: https://platform-docs.opentargets.org/
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import requests
from tqdm import tqdm

from biodb.utils import filter_adaptive

logger = logging.getLogger(__name__)

# ─── Bulk-download (FTP-mode) configuration ────────────────────────────────
DEFAULT_VERSION = "25.12"
"""Default OT Platform release version. Bump after testing against new release."""

RELEASES_BASE_URL = "http://ftp.ebi.ac.uk/pub/databases/opentargets/platform"
"""Root of the OT Platform release directory listing on the EBI FTP server."""

DOWNLOADS_BASE_URL_TEMPLATE = RELEASES_BASE_URL + "/{version}/output"
"""URL template for the per-release ``output`` directory containing Parquet datasets."""

# Backwards-compat alias for the pinned-version base URL used by older code paths
# (kept ending in ``/`` to match the previous module behaviour).
DOWNLOADS_BASE_URL = DOWNLOADS_BASE_URL_TEMPLATE.format(version=DEFAULT_VERSION) + "/"

DEFAULT_CACHE_DIR = Path("~/.cache/biodb/opentargets").expanduser()
"""Local cache root. Per-(version, dataset) subdirectories are created here."""

# Legacy single-flat-dir cache; kept readable for back-compat.
CACHE_DIR = Path("~/.cache/opentargets").expanduser()
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SCORE = 0.5

# OT release directory names look like ``25.12/``, ``26.03/``, etc.
_VERSION_RE = re.compile(r"^\d{2}\.\d{2}$")


def _version_sort_key(version: str) -> tuple[int, int]:
    """Return a ``(year, month)`` int-tuple sort key for ``YY.MM`` release strings."""
    year, month = version.split(".")
    return (int(year), int(month))


def list_available_versions(base_url: str = RELEASES_BASE_URL) -> list[str]:
    """List published OpenTargets Platform release versions on the FTP server.

    Returns versions in chronological order; the last element is the newest.
    Filters to ``YY.MM`` directory names so we ignore unrelated listings.

    Parameters
    ----------
    base_url : str
        Root URL of the OT Platform releases (defaults to :data:`RELEASES_BASE_URL`).

    Returns
    -------
    list[str]
        Version strings like ``["24.06", "24.09", "25.12"]``.

    Examples
    --------
    >>> versions = list_available_versions()  # doctest: +SKIP
    >>> versions[-1]  # latest  # doctest: +SKIP
    '25.12'
    """
    html = _http_get(base_url.rstrip("/") + "/")
    candidates: set[str] = set()
    for raw in re.findall(r'<a[^>]+href=["\']([^"\']+/)[^"\']*["\']', html, re.IGNORECASE):
        from urllib.parse import unquote

        name = unquote(raw.rstrip("/"))
        if _VERSION_RE.match(name):
            candidates.add(name)
    return sorted(candidates, key=_version_sort_key)


def _http_get(url: str, timeout: int = 30) -> str:
    """GET ``url`` and return its text body, raising on HTTP errors.

    Parameters
    ----------
    url : str
    timeout : int

    Returns
    -------
    str

    Examples
    --------
    >>> # html = _http_get("http://ftp.ebi.ac.uk/pub/databases/opentargets/platform/")  # doctest: +SKIP
    """
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def _is_parent_link(name: str, parent_path_prefix: str | None) -> bool:
    """Return True if a directory listing entry is just a parent-dir back-link."""
    if not name or name in {"..", ".", "Parent Directory", "../", "./"}:
        return True
    if name.startswith(("../", "/pub/")):
        return True
    return bool(parent_path_prefix and name.startswith(parent_path_prefix))


def list_datasets(base_url: str | None = None) -> dict[str, str]:
    """List top-level dataset directories under an OpenTargets release.

    Parameters
    ----------
    base_url : str, optional
        Override the OpenTargets release URL. Defaults to the release pinned
        by :data:`DEFAULT_VERSION`.

    Returns
    -------
    dict
        Mapping of dataset name -> full directory URL.

    Examples
    --------
    >>> from biodb.opentargets import list_datasets
    >>> ds = list_datasets()  # doctest: +SKIP
    >>> "target" in ds  # doctest: +SKIP
    True
    """
    from urllib.parse import unquote, urlparse

    if base_url is None:
        base_url = DOWNLOADS_BASE_URL_TEMPLATE.format(version=DEFAULT_VERSION)
    base_url = base_url.rstrip("/")

    parent_path_prefix: str | None = None
    parsed = urlparse(base_url)
    parts = parsed.path.strip("/").split("/")
    if "platform" in parts:
        idx = parts.index("platform")
        parent_path_prefix = "/" + "/".join(parts[: idx + 1])

    html = _http_get(base_url + "/")
    link_pattern = r'<a[^>]+href=["\']([^"\']+/)[^"\']*["\']'
    seen: list[str] = []
    for raw_match in re.findall(link_pattern, html, re.IGNORECASE):
        name = unquote(raw_match.rstrip("/"))
        if _is_parent_link(name, parent_path_prefix):
            continue
        if name not in seen:
            seen.append(name)
    return {name: f"{base_url}/{name}" for name in sorted(seen)}


def _list_parquet_files(dataset_url: str) -> list[str]:
    """List parquet file URLs under a dataset directory."""
    from urllib.parse import unquote

    dataset_url = dataset_url.rstrip("/")
    html = _http_get(dataset_url + "/")
    link_pattern = r'<a[^>]+href=["\']([^"\']+\.parquet)["\'][^>]*>'
    files = sorted({unquote(m) for m in re.findall(link_pattern, html, re.IGNORECASE)})
    return [f"{dataset_url}/{f}" for f in files]


def _download_to_cache(url: str, cache_dir: Path, force: bool = False) -> Path:
    """Download ``url`` into ``cache_dir`` if not already cached."""
    from urllib.parse import urlparse

    from biodb._downloads import stream_to_file

    cache_dir.mkdir(parents=True, exist_ok=True)
    fname = Path(urlparse(url).path).name
    out = cache_dir / fname
    if out.exists() and not force:
        return out
    logger.info("Downloading %s -> %s", url, out)
    return stream_to_file(url, out, timeout=60, chunk_size=1 << 20)


def ensure_cached_shards(
    dataset: str,
    *,
    version: str = DEFAULT_VERSION,
    cache_dir: str | Path | None = None,
    limit_files: int | None = None,
    force: bool = False,
) -> list[Path]:
    """Ensure every Parquet shard for ``dataset`` is on disk; return their paths.

    Splits cache-vs-download out of :func:`get_dataset` so the same logic
    can feed callers that want lazy / streamed reads (e.g. via
    ``pyarrow.dataset`` or ``polars.scan_parquet``) without materialising the
    full table.

    Parameters
    ----------
    dataset : str
        OT dataset directory name (e.g. ``"target"``, ``"association_overall_direct"``).
    version : str, default ``DEFAULT_VERSION``
        OT release version. Versioned cache layout: shards land under
        ``<cache_dir>/<version>/<dataset>/*.parquet``.
    cache_dir : str | Path, optional
        Cache root (defaults to :data:`DEFAULT_CACHE_DIR`).
    limit_files : int, optional
        If given, return only the first ``N`` shards (useful for smoke tests).
    force : bool, default False
        Re-download every shard even if already cached.

    Returns
    -------
    list[pathlib.Path]
        Local paths to the Parquet shards (sorted, deterministic).

    Raises
    ------
    FileNotFoundError
        If no parquet shards are listed for the dataset at ``version``.

    Examples
    --------
    >>> from biodb.opentargets import ensure_cached_shards
    >>> shards = ensure_cached_shards("target")  # doctest: +SKIP
    >>> len(shards) > 0  # doctest: +SKIP
    True
    """
    base_url = DOWNLOADS_BASE_URL_TEMPLATE.format(version=version)
    dataset_url = f"{base_url}/{dataset}"
    cache_root = (
        (Path(cache_dir).expanduser() if cache_dir else DEFAULT_CACHE_DIR) / version / dataset
    )
    cache_root.mkdir(parents=True, exist_ok=True)

    cached = sorted(cache_root.glob("*.parquet"))
    if cached and not force:
        return cached if limit_files is None else cached[:limit_files]
    parquet_urls = _list_parquet_files(dataset_url)
    if limit_files is not None:
        parquet_urls = parquet_urls[:limit_files]
    if not parquet_urls:
        raise FileNotFoundError(f"No parquet files found under {dataset_url}")
    from tqdm.auto import tqdm

    # Per-file ``_downloads.stream_to_file`` already shows a byte-level
    # bar; add an outer bar over shards so multi-shard datasets like
    # ``variant`` (25 shards in OT 25.12) report meaningful progress.
    return [
        _download_to_cache(url, cache_root, force=force)
        for url in tqdm(parquet_urls, desc=f"{dataset} shards", unit="shard", leave=False)
    ]


def get_dataset(
    dataset: str | None = None,
    *,
    remote_url: str | None = None,
    version: str = DEFAULT_VERSION,
    cache_dir: str | Path | None = None,
    limit_files: int | None = None,
    force: bool = False,
    output_format: str = "pandas",
    parse: bool = False,
    verbose: int = 1,
    limit: int | None = None,
) -> pd.DataFrame | pl.DataFrame:
    """Download and concatenate all Parquet shards for one OpenTargets dataset.

    The whole-dataset side of bioDB's dual-mode API. For one-target-at-a-time
    GraphQL lookups, use :mod:`biodb.opentargets_graphql` instead.

    Parameters
    ----------
    dataset : str, optional
        Dataset directory name (e.g. ``"target"``, ``"association_overall_direct"``).
        Use :func:`list_datasets` to enumerate.
    remote_url : str, optional
        Override the URL — takes precedence over (dataset, version) when set.
    version : str
        OT release version. Defaults to :data:`DEFAULT_VERSION`.
    cache_dir : str | Path, optional
        Local cache root (versioned subdirectory created automatically).
        Defaults to :data:`DEFAULT_CACHE_DIR`.
    limit_files : int, optional
        Limit to first ``N`` Parquet shards (smoke-test option).
    force : bool, default False
        Re-download even if cached.
    output_format : "pandas" | "polars"
        Concatenated DataFrame backend.
    parse : bool, default True
        For datasets with nested struct columns (``target_essentiality``,
        ``expression``), apply :func:`parse_gene_essentiality` / :func:`parse_expression`
        to flatten before returning.
    verbose : int
        0 = silent, 1 = progress, 2 = debug.
    limit : int, optional
        Cap the row count of the concatenated output.

    Returns
    -------
    pandas.DataFrame | polars.DataFrame

    Raises
    ------
    FileNotFoundError
        If no shards are listed at the resolved URL.

    Examples
    --------
    >>> from biodb.opentargets import get_dataset
    >>> targets = get_dataset("target", limit_files=1)  # doctest: +SKIP
    """
    # remote_url overrides resolved URL
    if remote_url:
        # Derive dataset name from URL tail for cache layout
        dataset_name = remote_url.rstrip("/").rsplit("/", 1)[-1]
        cache_root = (
            (Path(cache_dir).expanduser() if cache_dir else DEFAULT_CACHE_DIR)
            / version
            / dataset_name
        )
        cache_root.mkdir(parents=True, exist_ok=True)
        cached = sorted(cache_root.glob("*.parquet"))
        if cached and not force:
            local_files = cached if limit_files is None else cached[:limit_files]
        else:
            parquet_urls = _list_parquet_files(remote_url)
            if limit_files is not None:
                parquet_urls = parquet_urls[:limit_files]
            local_files = [_download_to_cache(u, cache_root, force=force) for u in parquet_urls]
    else:
        if dataset is None:
            dataset = "association_overall_direct"
        local_files = ensure_cached_shards(
            dataset,
            version=version,
            cache_dir=cache_dir,
            limit_files=limit_files,
            force=force,
        )

    if not local_files:
        raise FileNotFoundError(f"No parquet files cached for {dataset or remote_url}")

    if verbose >= 1:
        logger.info("Loading %d parquet shard(s) for %s", len(local_files), dataset or remote_url)

    if output_format == "polars":
        df_pl = pl.concat([pl.read_parquet(str(f)) for f in local_files], how="vertical_relaxed")
        if limit is not None:
            df_pl = df_pl.head(limit)
        return df_pl

    df_pd = pd.concat([pd.read_parquet(f) for f in local_files], ignore_index=True)
    if limit is not None:
        df_pd = df_pd.head(limit)

    if parse and dataset == "target_essentiality":
        df_pd = parse_gene_essentiality(df_pd)
    elif parse and dataset == "expression":
        df_pd = parse_expression(df_pd)

    return df_pd


def read_for_target(
    dataset: str,
    target_id: str,
    *,
    key_column: str = "targetId",
    version: str = DEFAULT_VERSION,
    cache_dir: str | Path | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Filtered read of one dataset's shards for a single target ID.

    Uses ``pyarrow.dataset`` to push the filter down to the Parquet scan so
    only the rows for ``target_id`` materialise into memory. Useful for
    per-gene lookups on the very large association / interaction tables.

    Parameters
    ----------
    dataset : str
    target_id : str
        Exact value to match in ``key_column``.
    key_column : str, default "targetId"
        Column to filter on. Some tables use different keys
        (``mouse_phenotype``: ``targetFromSourceId``; ``interaction``: ``targetA``;
        ``expression`` / ``target_essentiality``: ``id``).
    version : str
    cache_dir : str | Path, optional
    columns : list[str], optional
        Projection to a subset of columns.

    Returns
    -------
    pandas.DataFrame
        Empty if the gene has no entries.

    Examples
    --------
    >>> from biodb.opentargets import read_for_target
    >>> rows = read_for_target("known_drug", "ENSG00000012048")  # doctest: +SKIP
    """
    import pyarrow.dataset as ds

    paths = ensure_cached_shards(dataset, version=version, cache_dir=cache_dir)
    if not paths:
        return pd.DataFrame()
    dataset_obj = ds.dataset([str(p) for p in paths], format="parquet")
    if key_column not in dataset_obj.schema.names:
        return pd.DataFrame()
    table = dataset_obj.to_table(
        filter=ds.field(key_column) == target_id,
        columns=columns,
    )
    return table.to_pandas()


def variants_for_target(
    target_id: str,
    *,
    version: str = DEFAULT_VERSION,
    cache_dir: str | Path | None = None,
    columns: list[str] | None = None,
    progress: bool = True,
) -> "pl.DataFrame":
    """All variants whose ``transcriptConsequences`` include ``target_id``.

    OT's ``variant`` parquet table doesn't expose a top-level
    ``targetId`` column; gene linkage is recorded per-transcript inside
    the nested ``transcriptConsequences`` list-of-struct column. A
    plain :func:`read_for_target` won't reach it because pyarrow's
    pushdown filter can't traverse nested arrays. This helper does the
    list-of-struct filter via explode+dedupe so callers don't have to
    repeat the unnest plumbing on every consumer.

    Returns one row per variant (not per transcript-consequence) where
    at least one transcript consequence's ``targetId`` matches.

    Parameters
    ----------
    target_id : str
        Ensembl gene ID (e.g. ``"ENSG00000130164"`` for LDLR).
    version : str, default ``DEFAULT_VERSION``
        OT release tag.
    cache_dir : str | Path, optional
        Override for the shard cache root (defaults to
        :data:`DEFAULT_CACHE_DIR`).
    columns : list[str], optional
        Projection to a subset of top-level columns. ``None`` returns
        every column the parquet ships.
    progress : bool, default True
        Show a tqdm progress bar over the shard iteration. The full
        OT variant table is 25 shards in v25.12 and each one is a
        scan + explode + dedupe pass, so the bar gives meaningful
        feedback on multi-minute runs. Disable for noise-free batch
        callers.

    Returns
    -------
    polars.DataFrame
        Empty if the gene has no variant rows (or no shards on disk).

    Examples
    --------
    >>> from biodb.opentargets import variants_for_target
    >>> df = variants_for_target("ENSG00000130164")  # doctest: +SKIP
    >>> df.columns  # doctest: +SKIP
    ['variantId', 'chromosome', 'position', ...]
    """
    import polars as pl

    paths = ensure_cached_shards("variant", version=version, cache_dir=cache_dir)
    if not paths:
        return pl.DataFrame()

    from tqdm.auto import tqdm

    frames: list[pl.DataFrame] = []
    iterator = (
        tqdm(paths, desc=f"variants[{target_id}]", unit="shard", leave=False) if progress else paths
    )
    for shard in iterator:
        lf = pl.scan_parquet(str(shard))
        schema_names = lf.collect_schema().names()
        # Bail out cheaply if this shard's schema isn't what we expect
        # (older OT releases used a different layout for the nested
        # ``transcriptConsequences`` column).
        if "transcriptConsequences" not in schema_names:
            continue
        # The shard is filtered by exploding the list-of-struct column,
        # keeping rows whose exploded ``targetId`` matches, then
        # de-duplicating back to one row per source variant. This is
        # marginally more memory-intensive than a list-expression DSL
        # filter, but the per-shard row count is bounded (~290k for OT
        # 25.12) and the filter pass is dominated by the parquet read.
        matching = (
            lf.with_row_index("_var_row")
            .explode("transcriptConsequences")
            .filter(pl.col("transcriptConsequences").struct.field("targetId") == target_id)
            .select("_var_row")
            .unique()
        )
        out = lf.with_row_index("_var_row").join(matching, on="_var_row").drop("_var_row")
        if columns is not None:
            out = out.select(columns)
        frames.append(out.collect())
    return pl.concat(frames) if frames else pl.DataFrame()


def _preprocess_disease_to_gene(
    disease_to_gene: pd.DataFrame,
    target_ids: set[str] | None = None,
    limit: int | None = None,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Pre-process disease_to_gene DataFrame by grouping and aggregating per targetId.
    Returns a dictionary mapping targetId to pre-processed DataFrame.

    Parameters
    ----------
    disease_to_gene : pd.DataFrame
        DataFrame with disease-to-gene associations
    target_ids : set of str, optional
        If provided, only process these target IDs. Otherwise, process all unique target IDs.
    limit : int, optional
        Maximum number of associations per target. If provided, takes top N by score.
    verbose : bool, default True
        Whether to show progress
    """
    if disease_to_gene is None or len(disease_to_gene) == 0:
        return {}

    # Filter upfront if target_ids provided (much faster than filtering per target_id)
    if verbose:
        print(
            f"  Filtering disease associations for {len(target_ids) if target_ids else 'all'} target IDs..."
        )
    if target_ids is not None:
        disease_to_gene = disease_to_gene[disease_to_gene["targetId"].isin(target_ids)].copy()
        if len(disease_to_gene) == 0:
            return {}
        if verbose:
            print(f"  Filtered to {len(disease_to_gene)} disease associations")

    # Pre-process synonyms column to flatten it before groupby (much faster)
    # This avoids calling the aggregation function for every group
    if "synonyms" in disease_to_gene.columns and verbose:
        print("  Pre-processing synonyms column...")

    def flatten_synonyms(val):
        """Flatten synonyms dict/list to a set of strings."""
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return set()
        result = set()
        if isinstance(val, dict):
            for _k, v in val.items():
                if isinstance(v, (list, np.ndarray)):
                    result.update(str(x) for x in v if x is not None)
                elif v is not None:
                    result.add(str(v))
        elif isinstance(val, (list, np.ndarray)):
            result.update(str(x) for x in val if x is not None)
        else:
            result.add(str(val))
        return result

    # Pre-process synonyms to sets (much faster than doing it in groupby)
    if "synonyms" in disease_to_gene.columns:
        disease_to_gene = disease_to_gene.copy()  # Make sure we have a copy
        disease_to_gene["_synonyms_set"] = disease_to_gene["synonyms"].apply(flatten_synonyms)
    else:
        disease_to_gene["_synonyms_set"] = pd.Series(
            [set()] * len(disease_to_gene), index=disease_to_gene.index
        )

    # Optimize groupby by using categorical dtypes for groupby columns (if not too many unique values)
    if verbose:
        print("  Grouping and aggregating disease associations...")
        print(f"    DataFrame size: {len(disease_to_gene):,} rows")

    # Use more efficient aggregation - combine sets using union
    def combine_synonym_sets(srs):
        """Combine sets of synonyms."""
        result = set()
        for s in srs:
            if isinstance(s, set):
                result.update(s)
        return list(result) if result else []

    # Group by targetId and name, aggregate all at once
    # For very large DataFrames, chunk the groupby operation
    groupby_cols = ["targetId", "name"]
    chunk_size = 1_000_000  # Process in chunks of 1M rows if larger

    if len(disease_to_gene) > chunk_size:
        if verbose:
            print(
                f"    Large DataFrame detected ({len(disease_to_gene):,} rows), processing in chunks..."
            )
        # Process in chunks
        chunks = []
        num_chunks = (len(disease_to_gene) + chunk_size - 1) // chunk_size

        if verbose:
            from tqdm.auto import tqdm

            chunk_iterator = tqdm(
                range(0, len(disease_to_gene), chunk_size),
                desc="      Processing chunks",
                unit="chunk",
                total=num_chunks,
            )
        else:
            chunk_iterator = range(0, len(disease_to_gene), chunk_size)

        for i in chunk_iterator:
            chunk = disease_to_gene.iloc[i : i + chunk_size]
            chunk_grouped = chunk.groupby(groupby_cols, sort=False).agg(
                {
                    "diseaseId": "unique",
                    "score": "mean",
                    "evidenceCount": "sum",
                    "description": "first",
                    "_synonyms_set": combine_synonym_sets,
                }
            )
            chunks.append(chunk_grouped)

        # Combine chunks and re-aggregate
        if verbose:
            print(f"    Combining {len(chunks)} chunks...")
        combined = pd.concat(chunks)
        grouped = (
            combined.groupby(groupby_cols, sort=False)
            .agg(
                {
                    "diseaseId": lambda x: np.unique(
                        np.concatenate([arr if isinstance(arr, np.ndarray) else [arr] for arr in x])
                    ),
                    "score": "mean",
                    "evidenceCount": "sum",
                    "description": "first",
                    "_synonyms_set": combine_synonym_sets,
                }
            )
            .reset_index()
        )
    else:
        # Standard groupby for smaller DataFrames
        grouped = (
            disease_to_gene.groupby(groupby_cols, sort=False)
            .agg(
                {
                    "diseaseId": "unique",
                    "score": "mean",
                    "evidenceCount": "sum",
                    "description": "first",
                    "_synonyms_set": combine_synonym_sets,
                }
            )
            .reset_index()
        )

    # Rename the synonyms column back
    grouped = grouped.rename(columns={"_synonyms_set": "synonyms"})

    # Sort by score descending
    if verbose:
        print("  Sorting by score...")
    grouped = grouped.sort_values("score", ascending=False)

    # Split into dict by targetId (vectorized using groupby) and apply per-target limit
    if verbose:
        print("  Creating per-target dictionaries...")
        from tqdm import tqdm

        target_id_groups = list(grouped.groupby("targetId", sort=False))
        iterator = tqdm(
            target_id_groups, desc="  Processing targets", unit="target", disable=not verbose
        )
    else:
        iterator = grouped.groupby("targetId", sort=False)

    result = {}
    for target_id, group in iterator:
        # Apply per-target limit if specified
        if limit is not None:
            group = group.head(limit)
        # Drop targetId column to mark as preprocessed
        group_clean = group.drop(columns=["targetId"]).copy()
        result[target_id] = group_clean

    if verbose:
        print(
            f"  Pre-processed {len(result)} target IDs for diseases"
            + (f" (limited to {limit} per target)" if limit else "")
        )

    return result


def _preprocess_drug_to_gene(
    drug_to_gene: pd.DataFrame,
    target_ids: set[str] | None = None,
    limit: int | None = None,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Pre-process drug_to_gene DataFrame by deduplicating per targetId.
    Returns a dictionary mapping targetId to pre-processed DataFrame.

    Parameters
    ----------
    drug_to_gene : pd.DataFrame
        DataFrame with drug-to-gene associations
    target_ids : set of str, optional
        If provided, only process these target IDs. Otherwise, process all unique target IDs.
    limit : int, optional
        Maximum number of drugs per target. If provided, takes top N by phase.
    verbose : bool, default True
        Whether to show progress
    """
    if drug_to_gene is None or len(drug_to_gene) == 0:
        return {}

    # Filter upfront if target_ids provided (much faster than filtering per target_id)
    if verbose:
        print(
            f"  Filtering drug associations for {len(target_ids) if target_ids else 'all'} target IDs..."
        )
    if target_ids is not None:
        drug_to_gene = drug_to_gene[drug_to_gene["targetId"].isin(target_ids)].copy()
        if len(drug_to_gene) == 0:
            return {}
        if verbose:
            print(f"  Filtered to {len(drug_to_gene)} drug associations")

    # Vectorized: sort and deduplicate all at once, then split by targetId
    if "drugId" in drug_to_gene.columns:
        if verbose:
            print("  Sorting and deduplicating drugs...")
        # Sort by phase (descending, with NaN last) to get latest phase first
        if "phase" in drug_to_gene.columns:
            drug_to_gene = drug_to_gene.sort_values("phase", ascending=False, na_position="last")
        # Drop duplicates per (targetId, drugId) combination, keeping first (highest phase)
        drug_to_gene = drug_to_gene.drop_duplicates(subset=["targetId", "drugId"], keep="first")

    # Split into dict by targetId (vectorized using groupby) and apply per-target limit
    if verbose:
        print("  Creating per-target dictionaries...")
        from tqdm import tqdm

        target_id_groups = list(drug_to_gene.groupby("targetId", sort=False))
        iterator = tqdm(
            target_id_groups, desc="  Processing targets", unit="target", disable=not verbose
        )
    else:
        iterator = drug_to_gene.groupby("targetId", sort=False)

    result = {}
    for target_id, group in iterator:
        # Apply per-target limit if specified
        if limit is not None:
            group = group.head(limit)
        # Drop targetId column to mark as preprocessed
        if "targetId" in group.columns:
            group_clean = group.drop(columns=["targetId"]).copy()
        else:
            group_clean = group.copy()
        result[target_id] = group_clean

    if verbose:
        print(
            f"  Pre-processed {len(result)} target IDs for drugs"
            + (f" (limited to {limit} per target)" if limit else "")
        )

    return result


def _preprocess_pharmacogenomics(
    gene_to_pgx: pd.DataFrame,
    target_ids: set[str] | None = None,
    limit: int | None = None,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Pre-process gene_to_pharmacogenomics DataFrame by grouping per targetId.
    Returns a dictionary mapping targetId to pre-processed DataFrame.

    Parameters
    ----------
    gene_to_pgx : pd.DataFrame
        DataFrame with pharmacogenomics data
    target_ids : set of str, optional
        If provided, only process these target IDs. Otherwise, process all unique target IDs.
    limit : int, optional
        Maximum number of pharmacogenomics records per target. If provided, takes top N by evidenceLevel and isDirectTarget.
    verbose : bool, default True
        Whether to show progress
    """
    if gene_to_pgx is None or len(gene_to_pgx) == 0:
        return {}

    # Check which column name is used for target ID
    target_id_col = None
    for col in ["targetFromSourceId", "targetId", "id"]:
        if col in gene_to_pgx.columns:
            target_id_col = col
            break

    if target_id_col is None:
        return {}

    # Filter upfront if target_ids provided (much faster than filtering per target_id)
    if verbose:
        print(
            f"  Filtering pharmacogenomics for {len(target_ids) if target_ids else 'all'} target IDs..."
        )
    if target_ids is not None:
        gene_to_pgx = gene_to_pgx[gene_to_pgx[target_id_col].isin(target_ids)].copy()
        if len(gene_to_pgx) == 0:
            return {}
        if verbose:
            print(f"  Filtered to {len(gene_to_pgx)} pharmacogenomics records")

    # Sort by evidenceLevel and isDirectTarget if limit is specified (for per-target limiting)
    if limit is not None:
        if verbose:
            print("  Sorting pharmacogenomics records...")
        sort_cols = []
        if "evidenceLevel" in gene_to_pgx.columns:
            sort_cols.append("evidenceLevel")
        if "isDirectTarget" in gene_to_pgx.columns:
            sort_cols.append("isDirectTarget")
        if sort_cols:
            gene_to_pgx = gene_to_pgx.sort_values(sort_cols, ascending=False, na_position="last")

    # Split into dict by targetId (vectorized using groupby) and apply per-target limit
    if verbose:
        print("  Creating per-target dictionaries...")
        from tqdm import tqdm

        target_id_groups = list(gene_to_pgx.groupby(target_id_col, sort=False))
        iterator = tqdm(
            target_id_groups, desc="  Processing targets", unit="target", disable=not verbose
        )
    else:
        iterator = gene_to_pgx.groupby(target_id_col, sort=False)

    result = {}
    for target_id, group in iterator:
        # Apply per-target limit if specified
        if limit is not None:
            group = group.head(limit)
        # Drop targetId column to mark as preprocessed
        if target_id_col in group.columns:
            group_clean = group.drop(columns=[target_id_col]).copy()
        else:
            group_clean = group.copy()
        result[target_id] = group_clean

    if verbose:
        print(
            f"  Pre-processed {len(result)} target IDs for pharmacogenomics"
            + (f" (limited to {limit} per target)" if limit else "")
        )

    return result


def get_targets(
    *,
    save_path: str | None = None,
    force: bool = False,
    limit: int | None = None,
    verbose: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Load the OpenTargets ``target`` dataset as a DataFrame.

    Thin wrapper around :func:`get_dataset` that exists for backwards-compat
    with callers that used to receive an enriched (markdown-bearing) target
    table. Markdown generation moved to downstream consumers
    (e.g. `GeneDocs <https://github.com/bschilder/GeneDocs>`_); this function
    now returns the raw target dataset only.

    Parameters
    ----------
    save_path : str, optional
        If provided, save the resulting DataFrame to this parquet path. If the
        file already exists and ``force`` is False, it is loaded instead of
        re-downloaded.
    force : bool, default False
        Force re-download even if ``save_path`` exists.
    limit : int, optional
        Maximum number of target rows to return.
    verbose : bool, default True
        Print progress messages.
    **kwargs
        Forwarded to :func:`get_dataset`.

    Returns
    -------
    pd.DataFrame
        Raw target dataset rows (no enrichment).

    Examples
    --------
    >>> import biodb.opentargets as ot
    >>> targets = ot.get_targets(limit=100)  # doctest: +SKIP
    """
    if save_path is not None and not force:
        save_path_obj = Path(save_path)
        if save_path_obj.exists():
            if verbose:
                print(f"Loading existing targets DataFrame from: {save_path}")
            return pd.read_parquet(save_path)

    target = get_dataset(dataset="target", force=force, verbose=int(verbose), **kwargs)
    if limit is not None:
        target = target.head(limit)
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        target.to_parquet(save_path, index=False)
        if verbose:
            print(f"Saved targets DataFrame to: {save_path}")
    return target


def _prepare_disease_to_gene_associations(
    association_dataset: str = "association_by_datasource_direct",
    cache_dir: str | None = None,
    force: bool = False,
    output_format: str = "pandas",
    verbose: int = 1,
) -> pd.DataFrame:
    """
    Prepare disease-to-gene associations from OpenTargets association dataset.

    This function downloads the association dataset and disease metadata, merges them,
    and creates standardized columns for use with create_gene_association_matrix().

    Parameters
    ----------
    association_dataset : str, default "association_by_datasource_direct"
        Name of the association dataset to use.
    cache_dir : str, optional
        Local directory to cache downloaded files. If None, uses default cache directory.
    force : bool, default False
        If True, re-download files even if they already exist in cache.
    output_format : str, default "pandas"
        Output format: "pandas" or "polars".
    verbose : int, default 1
        Verbosity level:
        - 0: Minimal output
        - 1: Show progress information
        - 2: Show detailed information

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized columns:
        - database: "OpenTargets"
        - dataset: Name of the association dataset
        - sourceId: diseaseId.datatypeId.datasourceId (unique identifier for each association source)
        - targetId: targetId (gene identifier)
        - score: Association score
        - Additional columns from disease metadata (name, description, synonyms)
    """
    if verbose >= 1:
        logger.info(f"Preparing disease-to-gene associations from {association_dataset}")

    # Import disease-gene associations
    association_by_datasource_direct = get_dataset(
        dataset=association_dataset,
        cache_dir=cache_dir,
        force=force,
        output_format=output_format,
        verbose=verbose - 1 if verbose > 0 else 0,
    )

    # Import disease metadata
    disease = get_dataset(
        dataset="disease",
        cache_dir=cache_dir,
        force=force,
        output_format=output_format,
        verbose=verbose - 1 if verbose > 0 else 0,
    )

    # Merge with disease data
    disease_to_gene = association_by_datasource_direct.merge(
        disease[["id", "name", "description", "synonyms"]],
        left_on="diseaseId",
        right_on="id",
        how="left",
    )

    # Create sourceId column
    disease_to_gene["sourceId"] = (
        disease_to_gene["diseaseId"]
        + "."
        + disease_to_gene["datatypeId"]
        + "."
        + disease_to_gene["datasourceId"]
    )

    disease_to_gene["dataset"] = association_dataset
    disease_to_gene["database"] = "OpenTargets"

    # Add label column (from disease name)
    disease_to_gene["label"] = disease_to_gene["name"]

    if verbose >= 1:
        logger.info(f"Unique sourceIds: {disease_to_gene['sourceId'].nunique():,}")
        logger.info(f"DataFrame shape: {disease_to_gene.shape}")

    return disease_to_gene


def _prepare_known_drug_associations(
    cache_dir: str | None = None,
    force: bool = False,
    output_format: str = "pandas",
    default_score: float | None = None,
    verbose: int = 1,
) -> pd.DataFrame:
    """
    Prepare known drug associations from OpenTargets.

    This function downloads the known_drug dataset and creates standardized columns
    for use with create_gene_association_matrix().

    Parameters
    ----------
    cache_dir : str, optional
        Local directory to cache downloaded files. If None, uses default cache directory.
    force : bool, default False
        If True, re-download files even if they already exist in cache.
    output_format : str, default "pandas"
        Output format: "pandas" or "polars".
    default_score : float or None, default None
        Default score value to assign (known_drug dataset doesn't have scores).
        If None, fills the score column with NaN/NA values.
    verbose : int, default 1
        Verbosity level:
        - 0: Minimal output
        - 1: Show progress information
        - 2: Show detailed information

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized columns:
        - database: "OpenTargets"
        - dataset: "known_drug"
        - sourceId: drugId.diseaseId (unique identifier for each association source)
        - targetId: targetId (gene identifier)
        - score: Default score (0.5)
    """
    if verbose >= 1:
        logger.info("Preparing known drug associations")

    known_drug = get_dataset(
        dataset="known_drug",
        cache_dir=cache_dir,
        force=force,
        output_format=output_format,
        verbose=verbose - 1 if verbose > 0 else 0,
    )

    known_drug["sourceId"] = known_drug["drugId"] + "." + known_drug["diseaseId"]
    if default_score is None:
        known_drug["score"] = pd.NA
    else:
        known_drug["score"] = default_score
    known_drug["dataset"] = "known_drug"
    known_drug["database"] = "OpenTargets"

    # Add label column (from prefName - drug preferred name)
    known_drug["label"] = known_drug["prefName"]

    if verbose >= 1:
        logger.info(f"DataFrame shape: {known_drug.shape}")

    return known_drug


def _prepare_pharmacogenomics_associations(
    cache_dir: str | None = None,
    force: bool = False,
    output_format: str = "pandas",
    verbose: int = 1,
) -> pd.DataFrame:
    """
    Prepare pharmacogenomics associations from OpenTargets.

    This function downloads the pharmacogenomics dataset, aggregates by target and datasource,
    extracts drug information, and creates standardized columns for use with
    create_gene_association_matrix().

    Parameters
    ----------
    cache_dir : str, optional
        Local directory to cache downloaded files. If None, uses default cache directory.
    force : bool, default False
        If True, re-download files even if they already exist in cache.
    output_format : str, default "pandas"
        Output format: "pandas" or "polars".
    verbose : int, default 1
        Verbosity level:
        - 0: Minimal output
        - 1: Show progress information
        - 2: Show detailed information

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized columns:
        - database: "OpenTargets"
        - dataset: "pharmacogenomics"
        - sourceId: drugFromSource.datasourceId (unique identifier for each association source)
        - targetId: targetId (gene identifier, renamed from targetFromSourceId)
        - score: Normalized evidenceLevel (0-1 scale)
        - Additional columns: variantId (count), isDirectTarget, evidenceLevel
    """
    if verbose >= 1:
        logger.info("Preparing pharmacogenomics associations")

    pharmacogenomics = get_dataset(
        dataset="pharmacogenomics",
        cache_dir=cache_dir,
        force=force,
        output_format=output_format,
        verbose=verbose - 1 if verbose > 0 else 0,
    )

    # First, coerce non-numeric 'evidenceLevel' to NaN to avoid ValueError
    pharmacogenomics["evidenceLevel"] = pd.to_numeric(
        pharmacogenomics["evidenceLevel"], errors="coerce"
    )

    # Custom aggregation functions for each drug column separately
    def extract_drug_from_source(srs):
        drug_from_source_set = set()
        for arr in srs.dropna():
            drugs = arr if isinstance(arr, (list, np.ndarray)) else [arr]
            for drug in drugs:
                if (
                    isinstance(drug, dict)
                    and "drugFromSource" in drug
                    and drug["drugFromSource"] is not None
                ):
                    drug_from_source_set.add(drug["drugFromSource"])
        return list(drug_from_source_set)

    def extract_drug_id(srs):
        drug_id_set = set()
        for arr in srs.dropna():
            drugs = arr if isinstance(arr, (list, np.ndarray)) else [arr]
            for drug in drugs:
                if isinstance(drug, dict) and "drugId" in drug and drug["drugId"] is not None:
                    drug_id_set.add(drug["drugId"])
        return list(drug_id_set)

    # First do the standard aggregations
    gene_to_pharmacogenomics = (
        pharmacogenomics.groupby(["targetFromSourceId", "datasourceId"])
        .agg(
            {
                "variantId": "nunique",
                "isDirectTarget": "mean",
                "evidenceLevel": "mean",
            }
        )
        .reset_index()
    )

    # Extract drug columns separately using apply with reset_index(name="...")
    drug_from_source_col = (
        pharmacogenomics.groupby(["targetFromSourceId", "datasourceId"])["drugs"]
        .apply(extract_drug_from_source)
        .reset_index(name="drugFromSource")
    )
    drug_id_col = (
        pharmacogenomics.groupby(["targetFromSourceId", "datasourceId"])["drugs"]
        .apply(extract_drug_id)
        .reset_index(name="drugId")
    )

    # Merge both drug columns
    gene_to_pharmacogenomics = gene_to_pharmacogenomics.merge(
        drug_from_source_col, on=["targetFromSourceId", "datasourceId"], how="left"
    ).merge(drug_id_col, on=["targetFromSourceId", "datasourceId"], how="left")

    # Rename
    gene_to_pharmacogenomics = gene_to_pharmacogenomics.rename(
        columns={"targetFromSourceId": "targetId"}
    )

    # Sort first
    gene_to_pharmacogenomics = gene_to_pharmacogenomics.sort_values(
        ["targetId", "evidenceLevel", "isDirectTarget"], ascending=False
    )

    # Explode drugFromSource column - this will create one row per drug
    gene_to_pharmacogenomics = gene_to_pharmacogenomics.explode("drugFromSource")

    gene_to_pharmacogenomics["sourceId"] = (
        gene_to_pharmacogenomics["drugFromSource"] + "." + gene_to_pharmacogenomics["datasourceId"]
    )

    # Normalize score from evidenceLevel (0-1 scale)
    max_evidence = gene_to_pharmacogenomics["evidenceLevel"].max()
    if pd.notna(max_evidence) and max_evidence > 0:
        gene_to_pharmacogenomics["score"] = gene_to_pharmacogenomics["evidenceLevel"] / max_evidence
    else:
        gene_to_pharmacogenomics["score"] = 0.0

    gene_to_pharmacogenomics["dataset"] = "pharmacogenomics"
    gene_to_pharmacogenomics["database"] = "OpenTargets"

    # Add label column (from drugFromSource - drug name from source)
    gene_to_pharmacogenomics["label"] = gene_to_pharmacogenomics["drugFromSource"]

    if verbose >= 1:
        logger.info(f"DataFrame shape: {gene_to_pharmacogenomics.shape}")

    return gene_to_pharmacogenomics


def _prepare_mouse_phenotype_associations(
    cache_dir: str | None = None,
    force: bool = False,
    output_format: str = "pandas",
    default_score: float | None = None,
    verbose: int = 1,
) -> pd.DataFrame:
    """
    Prepare mouse phenotype associations from OpenTargets.

    This function downloads the mouse_phenotype dataset and creates standardized columns
    for use with create_gene_association_matrix().

    Parameters
    ----------
    cache_dir : str, optional
        Local directory to cache downloaded files. If None, uses default cache directory.
    force : bool, default False
        If True, re-download files even if they already exist in cache.
    output_format : str, default "pandas"
        Output format: "pandas" or "polars".
    default_score : float or None, default None
        Default score value to assign (mouse_phenotype dataset doesn't have scores).
        If None, fills the score column with NaN/NA values.
    verbose : int, default 1
        Verbosity level:
        - 0: Minimal output
        - 1: Show progress information
        - 2: Show detailed information

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized columns:
        - database: "OpenTargets"
        - dataset: "mouse_phenotype"
        - sourceId: modelPhenotypeId (unique identifier for each phenotype)
        - targetId: targetFromSourceId (human gene identifier)
        - score: DEFAULT_SCORE (0.5) for all rows
        - label: modelPhenotypeLabel (phenotype label)
    """
    if verbose >= 1:
        logger.info("Preparing mouse phenotype associations")

    mouse_phenotype = get_dataset(
        dataset="mouse_phenotype",
        cache_dir=cache_dir,
        force=force,
        output_format=output_format,
        verbose=verbose - 1 if verbose > 0 else 0,
    )

    # Rename targetFromSourceId to targetId (human gene ID)
    mouse_phenotype = mouse_phenotype.rename(columns={"targetFromSourceId": "targetId"})

    # Use modelPhenotypeId as sourceId
    mouse_phenotype["sourceId"] = mouse_phenotype["modelPhenotypeId"]

    # Add score column - always use DEFAULT_SCORE (0.5) for mouse_phenotype
    mouse_phenotype["score"] = DEFAULT_SCORE

    # Add dataset and database columns
    mouse_phenotype["dataset"] = "mouse_phenotype"
    mouse_phenotype["database"] = "OpenTargets"

    # Add label column (from modelPhenotypeLabel)
    mouse_phenotype["label"] = mouse_phenotype["modelPhenotypeLabel"]

    if verbose >= 1:
        logger.info(f"DataFrame shape: {mouse_phenotype.shape}")

    return mouse_phenotype


def _prepare_expression_associations(
    cache_dir: str | None = None,
    force: bool = False,
    output_format: str = "pandas",
    default_score: float | None = None,
    verbose: int = 1,
) -> pd.DataFrame:
    """
    Prepare expression associations from OpenTargets.

    This function downloads the expression dataset (which is automatically
    parsed by get_dataset), and creates standardized columns for use with
    create_gene_association_matrix().

    Parameters
    ----------
    cache_dir : str, optional
        Local directory to cache downloaded files. If None, uses default cache directory.
    force : bool, default False
        If True, re-download files even if they already exist in cache.
    output_format : str, default "pandas"
        Output format: "pandas" or "polars".
    default_score : float or None, default None
        Default score value to assign if rna_value is not available.
        If None, uses rna_value as score, or NaN if rna_value is missing.
    verbose : int, default 1
        Verbosity level:
        - 0: Minimal output
        - 1: Show progress information
        - 2: Show detailed information

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized columns:
        - database: "OpenTargets"
        - dataset: "expression"
        - sourceId: efo_code (tissue identifier)
        - targetId: id (gene identifier)
        - score: rna_value (expression value)
        - label: tissueLabel (tissue name)
    """
    if verbose >= 1:
        logger.info("Preparing expression associations")

    # get_dataset automatically parses expression
    expression = get_dataset(
        dataset="expression",
        cache_dir=cache_dir,
        force=force,
        output_format=output_format,
        verbose=verbose - 1 if verbose > 0 else 0,
    )

    # Rename id to targetId
    if "id" in expression.columns:
        expression = expression.rename(columns={"id": "targetId"})

    # Map efo_code to sourceId
    if "efo_code" in expression.columns:
        expression["sourceId"] = expression["efo_code"]

    # Add score column (use rna_value, or default_score, or NaN)
    if "rna_value" in expression.columns:
        # Use rna_value directly as score
        expression["score"] = expression["rna_value"]
        # Replace NaN with default_score if provided
        if default_score is not None:
            expression["score"] = expression["score"].fillna(default_score)
    else:
        # No rna_value column, use default_score or NaN
        if default_score is None:
            expression["score"] = pd.NA
        else:
            expression["score"] = default_score

    # Add dataset and database columns
    expression["dataset"] = "expression"
    expression["database"] = "OpenTargets"

    # Map tissueLabel to label
    if "tissueLabel" in expression.columns:
        expression["label"] = expression["tissueLabel"]
    else:
        expression["label"] = pd.NA

    if verbose >= 1:
        logger.info(f"DataFrame shape: {expression.shape}")

    return expression


def _prepare_target_essentiality_associations(
    cache_dir: str | None = None,
    force: bool = False,
    output_format: str = "pandas",
    default_score: float | None = None,
    verbose: int = 1,
) -> pd.DataFrame:
    """
    Prepare target essentiality associations from OpenTargets.

    This function downloads the target_essentiality dataset (which is automatically
    parsed by get_dataset), and creates standardized columns for use with
    create_gene_association_matrix().

    Parameters
    ----------
    cache_dir : str, optional
        Local directory to cache downloaded files. If None, uses default cache directory.
    force : bool, default False
        If True, re-download files even if they already exist in cache.
    output_format : str, default "pandas"
        Output format: "pandas" or "polars".
    default_score : float or None, default None
        Default score value to assign if geneEffect is not available.
        If None, uses geneEffect (absolute value) as score, or NaN if geneEffect is missing.
    verbose : int, default 1
        Verbosity level:
        - 0: Minimal output
        - 1: Show progress information
        - 2: Show detailed information

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized columns:
        - database: "OpenTargets"
        - dataset: "target_essentiality"
        - sourceId: tissueId.depmapId.diseaseCellLineId.mutation (unique identifier)
        - targetId: geneId (gene identifier)
        - score: geneEffect or default_score
        - label: tissueName + cellLineName + diseaseFromSource + mutation (if not None)
    """
    if verbose >= 1:
        logger.info("Preparing target essentiality associations")

    # get_dataset automatically parses target_essentiality
    target_essentiality = get_dataset(
        dataset="target_essentiality",
        cache_dir=cache_dir,
        force=force,
        output_format=output_format,
        verbose=verbose - 1 if verbose > 0 else 0,
    )

    # Rename geneId to targetId
    if "geneId" in target_essentiality.columns:
        target_essentiality = target_essentiality.rename(columns={"geneId": "targetId"})

    # Add score column (use geneEffect directly, or default_score, or NaN)
    if "geneEffect" in target_essentiality.columns:
        # Use geneEffect directly as score
        target_essentiality["score"] = target_essentiality["geneEffect"]
        # Replace NaN with default_score if provided
        if default_score is not None:
            target_essentiality["score"] = target_essentiality["score"].fillna(default_score)
    else:
        # No geneEffect column, use default_score or NaN
        if default_score is None:
            target_essentiality["score"] = pd.NA
        else:
            target_essentiality["score"] = default_score

    # Add dataset and database columns
    target_essentiality["dataset"] = "target_essentiality"
    target_essentiality["database"] = "OpenTargets"

    # Add label column: tissueName + cellLineName + diseaseFromSource + mutation (if not None)
    def combine_label(row):
        parts = []
        if "tissueName" in row.index and pd.notna(row["tissueName"]) and row["tissueName"]:
            parts.append(str(row["tissueName"]))
        if "cellLineName" in row.index and pd.notna(row["cellLineName"]) and row["cellLineName"]:
            parts.append(str(row["cellLineName"]))
        if (
            "diseaseFromSource" in row.index
            and pd.notna(row["diseaseFromSource"])
            and row["diseaseFromSource"]
        ):
            parts.append(str(row["diseaseFromSource"]))
        if "mutation" in row.index and pd.notna(row["mutation"]) and row["mutation"]:
            parts.append(str(row["mutation"]))
        return " ".join(parts) if parts else pd.NA

    target_essentiality["label"] = target_essentiality.apply(combine_label, axis=1)

    if verbose >= 1:
        logger.info(f"DataFrame shape: {target_essentiality.shape}")

    return target_essentiality


def get_gene_associations(
    datasets: list | None = None,
    association_dataset: str = "association_by_datasource_direct",
    cache_dir: str | None = None,
    force: int = 0,
    output_format: str = "pandas",
    default_score: float | None = None,
    verbose: int = 1,
    save_path: str | Path | None = None,
    filter_adaptive_kwargs: dict[str, Any] | dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """
    Prepare gene association matrix from multiple OpenTargets datasets.

    This function downloads and processes multiple OpenTargets datasets, standardizes
    their format, and combines them into a single DataFrame ready for use with
    create_gene_association_matrix().

    Parameters
    ----------
    datasets : list of str, optional
        List of dataset names to include. If None, uses default selection:
        - "disease-to-gene"
        - "known_drug"
        - "pharmacogenomics"
        - "mouse_phenotype"
        - "target_essentiality"
        - "expression"
    association_dataset : str, default "association_by_datasource_direct"
        Name of the association dataset to use for disease-to-gene associations.
    cache_dir : str, optional
        Local directory to cache downloaded files. If None, uses default cache directory.
    force : bool, default False
        If True, re-download files even if they already exist in cache.
    output_format : str, default "pandas"
        Output format: "pandas" or "polars".
    default_score : float or None, default None
        Default score value for known_drug dataset (which doesn't have scores).
        If None, fills the score column with NaN/NA values.
    verbose : int, default 1
        Verbosity level:
        - 0: Minimal output
        - 1: Show progress information
        - 2: Show detailed information
    save_path : str or Path, optional
        Path to save the final combined DataFrame as a parquet file.
        Can be a string or Path object (including PosixPath).
        If provided, the DataFrame will be saved before returning.
        If None, the DataFrame is not saved.
    force : int, default 0
        Controls caching behavior:
        - 0: Use cached merged file if save_path exists, otherwise process from cached individual chunks
        - 1: Recreate merged DataFrame from cached individual chunks and resave (don't re-download chunks)
        - 2: Re-download individual parquet chunks and recreate merged DataFrame
    filter_adaptive_kwargs : dict, optional
        Keyword arguments to pass to filter_adaptive for each dataset.
        If a single dict, applies the same arguments to all datasets.
        If a dict of dicts, keys are dataset names and values are kwargs for each dataset.
        E.g., {"target_essentiality": {"min_genes": 3, "percentile": 0.99},
               "expression": {"min_genes": 3, "percentile": 0.99}}
        Common kwargs: percentile, min_genes, max_genes, sort_by, etc.
        If None, no filtering is applied.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with standardized columns:
        - database: "OpenTargets" for all rows
        - dataset: Name of the source dataset
        - sourceId: Unique identifier for each association source
        - targetId: Gene identifier
        - score: Association score
        - label: Human-readable label (disease name, drug name, etc. depending on dataset)
        - All other columns from the original datasets

    Examples
    --------
    >>> import biodb.opentargets as opentargets
    >>>
    >>> # Use all datasets (default)
    >>> associations = opentargets.get_gene_associations()
    >>>
    >>> # Include only disease-to-gene and known_drug
    >>> associations = opentargets.get_gene_associations(
    ...     datasets=["disease-to-gene", "known_drug"]
    ... )
    >>>
    >>> # Use with create_gene_association_matrix
    >>> import biodb.utils as utils
    >>> associations = opentargets.get_gene_associations()
    >>> X, metadata = utils.create_gene_association_matrix(associations)
    """
    # Check save_path at the beginning - if it exists and force is False, load and return it
    if save_path is not None:
        # Convert to Path if it's a string, otherwise use as-is (already a Path/PosixPath)
        save_path_obj = Path(save_path) if isinstance(save_path, str) else save_path

        # If file exists and force is 0, load and return it
        if save_path_obj.exists() and force == 0:
            if verbose >= 1:
                logger.info(f"Loading existing file from: {save_path_obj}")
                print(f"Loading existing file from: {save_path_obj}")
            return pd.read_parquet(save_path_obj)

    # Default datasets if not provided
    if datasets is None:
        datasets = [
            "disease-to-gene",
            "known_drug",
            "pharmacogenomics",
            "mouse_phenotype",
            "target_essentiality",
            "expression",
        ]

    if verbose >= 1:
        msg1 = "Preparing gene associations from OpenTargets datasets"
        msg2 = f"Including {len(datasets)} datasets: {', '.join(datasets)}"
        logger.info(msg1)
        logger.info(msg2)
        # Also print to stdout for interactive usage (e.g., Jupyter)
        print(msg1)
        print(msg2)

    associations_list = []

    # Helper function to apply filter_adaptive if specified
    def apply_filter_adaptive(df, dataset_name):
        """Apply filter_adaptive to a dataset if filter_adaptive_kwargs is specified."""
        if filter_adaptive_kwargs is None:
            return df

        # Determine the kwargs for this dataset
        if isinstance(filter_adaptive_kwargs, dict):
            # Check if it's a dict of dicts (dataset-specific) or a single dict (apply to all)
            if dataset_name in filter_adaptive_kwargs and isinstance(
                filter_adaptive_kwargs[dataset_name], dict
            ):
                # Dataset-specific kwargs
                kwargs = filter_adaptive_kwargs[dataset_name].copy()
            elif all(isinstance(v, dict) for v in filter_adaptive_kwargs.values() if v is not None):
                # It's a dict of dicts but this dataset not specified
                kwargs = None
            else:
                # Single dict to apply to all datasets
                kwargs = filter_adaptive_kwargs.copy()
        else:
            kwargs = None

        # Apply filtering if kwargs are specified for this dataset
        if kwargs is not None:
            if verbose >= 1:
                kwargs_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
                logger.info(f"Applying filter_adaptive to {dataset_name} with: {kwargs_str}")
                print(f"Applying filter_adaptive to {dataset_name} with: {kwargs_str}")

            # Set default sort_by column based on dataset if not specified
            if "sort_by" not in kwargs:
                if dataset_name == "target_essentiality" and "geneEffect" in df.columns:
                    kwargs["sort_by"] = "geneEffect"
                else:
                    kwargs["sort_by"] = "score"

            # Set default verbose if not specified
            if "verbose" not in kwargs:
                kwargs["verbose"] = verbose >= 2

            df = filter_adaptive(
                df=df, source_id_col="sourceId", target_id_col="targetId", **kwargs
            )

            if verbose >= 1:
                logger.info(f"After filtering: {len(df):,} rows")
                print(f"After filtering: {len(df):,} rows")

        return df

    # Prepare disease-to-gene associations
    if "disease-to-gene" in datasets:
        disease_to_gene = _prepare_disease_to_gene_associations(
            association_dataset=association_dataset,
            cache_dir=cache_dir,
            force=(force >= 2),  # Only re-download if force >= 2
            output_format=output_format,
            verbose=verbose,
        )
        disease_to_gene = apply_filter_adaptive(disease_to_gene, "disease-to-gene")
        associations_list.append(disease_to_gene)

    # Prepare known drug associations
    if "known_drug" in datasets:
        known_drug = _prepare_known_drug_associations(
            cache_dir=cache_dir,
            force=(force >= 2),  # Only re-download if force >= 2
            output_format=output_format,
            default_score=default_score,
            verbose=verbose,
        )
        known_drug = apply_filter_adaptive(known_drug, "known_drug")
        associations_list.append(known_drug)

    # Prepare pharmacogenomics associations
    if "pharmacogenomics" in datasets:
        gene_to_pharmacogenomics = _prepare_pharmacogenomics_associations(
            cache_dir=cache_dir,
            force=(force >= 2),  # Only re-download if force >= 2
            output_format=output_format,
            verbose=verbose,
        )
        gene_to_pharmacogenomics = apply_filter_adaptive(
            gene_to_pharmacogenomics, "pharmacogenomics"
        )
        associations_list.append(gene_to_pharmacogenomics)

    # Prepare mouse phenotype associations
    if "mouse_phenotype" in datasets:
        mouse_phenotype = _prepare_mouse_phenotype_associations(
            cache_dir=cache_dir,
            force=(force >= 2),  # Only re-download if force >= 2
            output_format=output_format,
            default_score=default_score,
            verbose=verbose,
        )
        mouse_phenotype = apply_filter_adaptive(mouse_phenotype, "mouse_phenotype")
        associations_list.append(mouse_phenotype)

    # Prepare target essentiality associations
    if "target_essentiality" in datasets:
        target_essentiality = _prepare_target_essentiality_associations(
            cache_dir=cache_dir,
            force=(force >= 2),  # Only re-download if force >= 2
            output_format=output_format,
            default_score=default_score,
            verbose=verbose,
        )
        target_essentiality = apply_filter_adaptive(target_essentiality, "target_essentiality")
        associations_list.append(target_essentiality)

    # Prepare expression associations
    if "expression" in datasets:
        expression = _prepare_expression_associations(
            cache_dir=cache_dir,
            force=(force >= 2),  # Only re-download if force >= 2
            output_format=output_format,
            default_score=default_score,
            verbose=verbose,
        )
        expression = apply_filter_adaptive(expression, "expression")
        associations_list.append(expression)

    if len(associations_list) == 0:
        raise ValueError("At least one dataset must be included")

    # Select common columns and concatenate
    select_cols = ["database", "dataset", "sourceId", "targetId", "score", "label"]

    # Get all available columns from all dataframes
    all_cols = set()
    for df in associations_list:
        all_cols.update(df.columns)

    # Select columns that exist in all dataframes
    common_cols = [col for col in select_cols if all(col in df.columns for df in associations_list)]

    # Also include any additional columns that are in all dataframes
    for col in all_cols:
        if col not in common_cols and all(col in df.columns for df in associations_list):
            common_cols.append(col)

    if verbose >= 1:
        logger.info(f"Combining {len(associations_list)} datasets...")
        logger.info(f"Using columns: {common_cols}")

    # Concatenate all associations
    # Filter out empty DataFrames and ensure all have the same columns to avoid FutureWarning
    dfs_to_concat = []
    for df in associations_list:
        if not df.empty:
            # Select only common columns that exist in this DataFrame
            df_subset = df[[col for col in common_cols if col in df.columns]].copy()
            # Ensure all common_cols are present (fill missing with NaN)
            for col in common_cols:
                if col not in df_subset.columns:
                    df_subset[col] = pd.NA
            # Reorder columns to match common_cols
            df_subset = df_subset[common_cols]
            dfs_to_concat.append(df_subset)

    if dfs_to_concat:
        opentargets_associations = pd.concat(dfs_to_concat, ignore_index=True)
    else:
        # Return empty DataFrame with correct columns if all are empty
        opentargets_associations = pd.DataFrame(columns=common_cols)

    if verbose >= 1:
        logger.info(f"Final DataFrame shape: {opentargets_associations.shape}")
        logger.info("Unique counts for categorical/string columns:")
        categorical_cols = ["database", "dataset", "sourceId", "targetId", "label"]
        for col in categorical_cols:
            if col in opentargets_associations.columns:
                n_unique = opentargets_associations[col].nunique()
                logger.info(f"  {col:15s}: {n_unique:,} unique values")
        # Also print to stdout for interactive usage (e.g., Jupyter)
        print(f"Final DataFrame shape: {opentargets_associations.shape}")
        print("Unique counts for categorical/string columns:")
        for col in categorical_cols:
            if col in opentargets_associations.columns:
                n_unique = opentargets_associations[col].nunique()
                print(f"  {col:15s}: {n_unique:,} unique values")

    # Save to parquet if save_path is provided
    if save_path is not None:
        # save_path_obj was already created and checked at the beginning of the function
        save_path_obj = Path(save_path) if isinstance(save_path, str) else save_path

        if verbose >= 1:
            logger.info(f"Saving DataFrame to: {save_path_obj}")
            print(f"Saving DataFrame to: {save_path_obj}")

        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        opentargets_associations.to_parquet(save_path_obj, index=False)

        if verbose >= 1:
            logger.info(f"Saved {len(opentargets_associations):,} rows to {save_path_obj}")
            print(f"Saved {len(opentargets_associations):,} rows to {save_path_obj}")

    return opentargets_associations


# Parse nested geneEssentiality column into flattened rows
def parse_gene_essentiality(df):
    """
    Flatten the nested geneEssentiality column into multiple rows.
    Each row will represent one screen (cell line) with all parent information.

    Structure:
    - geneEssentiality: list of dicts
      - isEssential: bool
      - depMapEssentiality: list of dicts (tissues)
        - tissueId: str
        - tissueName: str
        - screens: list of dicts (cell lines)
          - depmapId, cellLineName, diseaseFromSource, etc.
    """
    rows = []

    # Add progress bar for parsing
    iterator = tqdm(df.iterrows(), total=len(df), desc="Parsing geneEssentiality")
    for _idx, row in iterator:
        gene_id = row["id"]
        gene_essentiality = row["geneEssentiality"]

        # Handle case where geneEssentiality might be None or empty
        if pd.isna(gene_essentiality):
            continue

        # Convert numpy array to list if needed
        if isinstance(gene_essentiality, np.ndarray):
            gene_essentiality = gene_essentiality.tolist()

        if not gene_essentiality:
            continue

        # Iterate through each essentiality entry
        for ess_entry in gene_essentiality:
            # Handle dict access - convert numpy types if needed
            if isinstance(ess_entry, np.ndarray):
                ess_entry = ess_entry.item() if ess_entry.size == 1 else ess_entry.tolist()

            if not isinstance(ess_entry, dict):
                continue

            is_essential = ess_entry.get("isEssential", None)
            depmap_essentiality = ess_entry.get("depMapEssentiality", [])

            # Convert numpy array to list if needed
            if isinstance(depmap_essentiality, np.ndarray):
                depmap_essentiality = depmap_essentiality.tolist()

            # Iterate through each tissue
            for tissue_entry in depmap_essentiality:
                # Handle dict access
                if isinstance(tissue_entry, np.ndarray):
                    tissue_entry = (
                        tissue_entry.item() if tissue_entry.size == 1 else tissue_entry.tolist()
                    )

                if not isinstance(tissue_entry, dict):
                    continue

                tissue_id = tissue_entry.get("tissueId", None)
                tissue_name = tissue_entry.get("tissueName", None)
                screens = tissue_entry.get("screens", [])

                # Convert numpy array to list if needed
                if isinstance(screens, np.ndarray):
                    screens = screens.tolist()

                # Iterate through each screen (cell line)
                for screen in screens:
                    # Handle dict access
                    if isinstance(screen, np.ndarray):
                        screen = screen.item() if screen.size == 1 else screen.tolist()

                    if not isinstance(screen, dict):
                        continue

                    row_data = {
                        "geneId": gene_id,
                        "isEssential": is_essential,
                        "tissueId": tissue_id,
                        "tissueName": tissue_name,
                        "depmapId": screen.get("depmapId", None),
                        "cellLineName": screen.get("cellLineName", None),
                        "diseaseFromSource": screen.get("diseaseFromSource", None),
                        "diseaseCellLineId": screen.get("diseaseCellLineId", None),
                        "mutation": screen.get("mutation", None),
                        "geneEffect": screen.get("geneEffect", None),
                        "expression": screen.get("expression", None),
                    }
                    rows.append(row_data)
    df = pd.DataFrame(rows)
    cols = ["tissueId", "depmapId", "diseaseCellLineId", "mutation"]
    df["sourceId"] = df[cols].astype(str).agg(".".join, axis=1)
    return df


def parse_expression(df):
    """
    Flatten the nested expression column into multiple rows.
    Each row will represent one gene-tissue combination.

    Structure:
    - id: gene ID
    - tissues: list of dicts
      - efo_code: tissue identifier
      - label: tissue name
      - organs: array of organ names
      - anatomical_systems: array of anatomical system names
      - rna: dict with value, zscore, level, unit
      - protein: dict with reliability, level, cell_type

    Returns DataFrame with columns:
    - geneId (renamed to id): gene identifier
    - efo_code: tissue identifier
    - tissueLabel: tissue name
    - organs: list of organ names
    - anatomical_systems: list of anatomical system names
    - rna_value: RNA expression value
    - rna_zscore: RNA z-score
    - rna_level: RNA expression level
    - rna_unit: RNA unit
    - protein_reliability: protein reliability flag
    - protein_level: protein level
    - protein_cell_type: list of cell types
    """
    rows = []

    # Add progress bar for parsing
    iterator = tqdm(df.iterrows(), total=len(df), desc="Parsing expression")
    for _idx, row in iterator:
        gene_id = row["id"]
        tissues = row["tissues"]

        # Handle case where tissues might be None or empty
        # Check for numpy array first before using pd.isna()
        if isinstance(tissues, np.ndarray):
            if tissues.size == 0:
                continue
            tissues = tissues.tolist()
        elif tissues is None or pd.isna(tissues):
            continue

        # Check if empty after conversion (avoid boolean check on numpy arrays)
        try:
            if len(tissues) == 0:
                continue
        except (TypeError, ValueError):
            # If it's not a sequence, skip
            continue

        # Iterate through each tissue
        for tissue_entry in tissues:
            # Convert numpy array to dict if needed
            if isinstance(tissue_entry, np.ndarray):
                if tissue_entry.dtype == object and tissue_entry.size > 0:
                    tissue_entry = (
                        tissue_entry.item() if tissue_entry.size == 1 else tissue_entry.tolist()[0]
                    )
                else:
                    continue

            if not isinstance(tissue_entry, dict):
                continue

            efo_code = tissue_entry.get("efo_code", None)
            label = tissue_entry.get("label", None)

            # Extract organs (convert numpy array to list)
            organs = tissue_entry.get("organs", [])
            if isinstance(organs, np.ndarray):
                organs = organs.tolist()

            # Extract anatomical_systems (convert numpy array to list)
            anatomical_systems = tissue_entry.get("anatomical_systems", [])
            if isinstance(anatomical_systems, np.ndarray):
                anatomical_systems = anatomical_systems.tolist()

            # Extract RNA data
            rna = tissue_entry.get("rna", {})
            rna_value = rna.get("value", None) if isinstance(rna, dict) else None
            rna_zscore = rna.get("zscore", None) if isinstance(rna, dict) else None
            rna_level = rna.get("level", None) if isinstance(rna, dict) else None
            rna_unit = rna.get("unit", None) if isinstance(rna, dict) else None

            # Extract protein data
            protein = tissue_entry.get("protein", {})
            protein_reliability = (
                protein.get("reliability", None) if isinstance(protein, dict) else None
            )
            protein_level = protein.get("level", None) if isinstance(protein, dict) else None
            protein_cell_type = protein.get("cell_type", [])
            if isinstance(protein_cell_type, np.ndarray):
                protein_cell_type = protein_cell_type.tolist()

            row_data = {
                "geneId": gene_id,
                "efo_code": efo_code,
                "tissueLabel": label,
                "organs": organs,
                "anatomical_systems": anatomical_systems,
                "rna_value": rna_value,
                "rna_zscore": rna_zscore,
                "rna_level": rna_level,
                "rna_unit": rna_unit,
                "protein_reliability": protein_reliability,
                "protein_level": protein_level,
                "protein_cell_type": protein_cell_type,
            }
            rows.append(row_data)

    df = pd.DataFrame(rows)
    # Rename geneId to id to match original structure
    if "geneId" in df.columns:
        df = df.rename(columns={"geneId": "id"})

    return df


# # Default cache directory
# CACHE_DIR = Path.home() / ".cache" / "aou" / "opentargets"
# CACHE_DIR.mkdir(parents=True, exist_ok=True)

# # Available resource types in gene info results
# GENE_INFO_RESOURCE_TYPES = {
#     "associated_diseases": "Associated diseases/phenotypes",
#     "associated_drugs": "Associated drugs",
#     "tractability": "Druggability assessment data",
#     "pharmacogenetics": "Pharmacogenetic response data",
#     "expression": "Gene expression by tissues, organs, and anatomical systems",
#     "depmap": "DepMap gene→disease-effect data",
#     "interactions": "Protein⇄protein interactions",
# }

# # List of resource type keys (for convenience)
# GENE_INFO_RESOURCE_KEYS = list(GENE_INFO_RESOURCE_TYPES.keys())


# def get_all_genes(
#     output_format: str = "pandas",
#     cache_file: Optional[str] = None,
#     force: int = 0,
#     biotype_filter: Optional[str] = None,
#     species: str = "homo_sapiens",
# ) -> Union[pd.DataFrame, pl.DataFrame]:
#     """
#     Get all gene names available from Ensembl (used by OpenTargets Platform).

#     This function uses gget.ref() to fetch the Ensembl GTF file and extracts
#     all gene information. This provides a comprehensive list of all genes.


#     raw_cache_file = str(DEFAULT_CACHE_DIR / f"raw_gtf_{species}.gz")


#     # Check raw cache (force=1) or download fresh (force=2)
#     download_raw = force == 2 or not os.path.exists(raw_cache_file)

#     if download_raw:
#         logger.info(f"Fetching GTF file from Ensembl for {species}...")
#         try:
#             # Get reference data from Ensembl using gget
#             ref_data = gget.ref(species)


#             if not gtf_url:
#                 raise ValueError(f"Could not retrieve GTF file URL from gget.ref() for {species}. Returned: {ref_data}")

#             logger.info(f"Downloading GTF file from: {gtf_url}")

#             # Download and save raw GTF file
#             response = requests.get(gtf_url, stream=True)
#             response.raise_for_status()

#             # Save raw data to cache
#             with open(raw_cache_file, "wb") as f:
#                 for chunk in response.iter_content(chunk_size=8192):
#                     f.write(chunk)

#             logger.info(f"Cached raw GTF file to: {raw_cache_file}")

#         except Exception as e:
#             logger.error(f"Error fetching GTF file from Ensembl: {e}")
#             raise
#     else:
#         logger.info(f"Using cached raw GTF file: {raw_cache_file}")

#     # Parse GTF file from cache
#     logger.info("Reading and parsing GTF file...")
#     try:
#         # Read cached raw GTF file
#         if raw_cache_file.endswith(".gz"):
#             gtf_file = gzip.open(raw_cache_file, "rt")
#         else:
#             gtf_file = open(raw_cache_file, "rt")

#         # Read GTF file with pandas (skip comment lines)
#         gtf_df = pd.read_csv(
#             gtf_file,
#             sep="\t",
#             comment="#",
#             header=None,
#             names=["seqname", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"],
#             low_memory=False,
#         )

#         gtf_file.close()

#     except Exception as e:
#         logger.error(f"Error reading cached GTF file: {e}")
#         raise

#     # Filter for gene entries only
#     logger.info("Extracting gene information...")
#     genes_df = gtf_df[gtf_df["feature"] == "gene"].copy()

#     # Extract gene_id, gene_name, gene_biotype, and description from attributes column
#     # Attributes format: gene_id "ENSG00000157764"; gene_name "BRAF"; gene_biotype "protein_coding"; description "...";
#     def extract_attr(attr_str, attr_name):
#         """Extract attribute value from GTF attribute string."""
#         pattern = f'{attr_name} "([^"]+)"'
#         match = re.search(pattern, attr_str)
#         return match.group(1) if match else ""

#     genes_df["ensembl_id"] = genes_df["attribute"].apply(lambda x: extract_attr(x, "gene_id"))
#     genes_df["approved_symbol"] = genes_df["attribute"].apply(lambda x: extract_attr(x, "gene_name"))
#     genes_df["biotype"] = genes_df["attribute"].apply(lambda x: extract_attr(x, "gene_biotype"))
#     genes_df["approved_name"] = genes_df["attribute"].apply(lambda x: extract_attr(x, "description"))

#     # Apply biotype filter if specified
#     if biotype_filter:
#         genes_df = genes_df[genes_df["biotype"] == biotype_filter].copy()

#     # Select and rename columns
#     gene_details = genes_df[["ensembl_id", "approved_symbol", "approved_name", "biotype"]].copy()

#     # Remove duplicates (keep first occurrence)
#     gene_details = gene_details.drop_duplicates(subset=["ensembl_id"], keep="first")

#     logger.info(f"Found {len(gene_details)} genes")

#     # Create DataFrame
#     if output_format == "pandas":
#         df = pd.DataFrame(gene_details)
#     else:
#         df = pl.DataFrame(gene_details)

#     # Sort by gene symbol for easier browsing
#     if output_format == "pandas":
#         df = df.sort_values("approved_symbol").reset_index(drop=True)
#     else:
#         df = df.sort("approved_symbol")

#     # Save to cache as parquet
#     if cache_file:
#         if output_format == "pandas":
#             df.to_parquet(cache_file, index=False)
#         else:
#             df.write_parquet(cache_file)
#         logger.info(f"Cached {len(df)} genes to: {cache_file}")

#     return df


#     Each resource type becomes its own DataFrame with a gene_id column.

#     Parameters
#     ----------
#     results : list of dict
#         List of gene info dictionaries (one per gene)
#     include_phenotypes, include_drugs, etc. : bool
#         Flags indicating which resources were included


#     return output


#     This function queries the OpenTargets Platform via gget to retrieve comprehensive
#     gene information including basic gene metadata and associated diseases/phenotypes.

#     Parameters
#     ----------
#     gene_id : str or list of str
#         Gene identifier(s). Can be either:
#         - Ensembl ID (e.g., "ENSG00000157764" for BRAF)
#         - Gene symbol (e.g., "BRAF", "TP53", "EGFR")
#         - List of gene identifiers (mix of Ensembl IDs and symbols is allowed)
#           Gene symbols will be automatically resolved to Ensembl IDs using gget.search()

#     include_phenotypes : bool, default True
#         Whether to include associated diseases/phenotypes from OpenTargets Platform.

#     include_drugs : bool, default True
#         Whether to include associated drugs from OpenTargets Platform.

#     include_tractability : bool, default True
#         Whether to include tractability data (druggability assessment).

#     include_pharmacogenetics : bool, default True
#         Whether to include pharmacogenetic response data.

#     include_expression : bool, default True
#         Whether to include gene expression data by tissues, organs, and anatomical systems.

#     include_depmap : bool, default True
#         Whether to include DepMap gene→disease-effect data.

#     include_interactions : bool, default True
#         Whether to include protein⇄protein interactions.

#     phenotype_size : int, default 1000
#         Maximum number of phenotypes/diseases to retrieve. The OpenTargets Platform
#         returns results sorted by association score, so this limits the top N results.
#         Set to None to retrieve all available associations (may be slow for genes
#         with many associations).

#     phenotype_from : int, default 0
#         Offset for phenotype pagination. Use this to skip the first N results.
#         For example, phenotype_from=10 will skip the top 10 associations and return
#         results starting from the 11th.
#     force : int, default 0
#         Force refresh level:
#         - 0: Use cached file if exists, otherwise fetch and cache
#         - 1: Force new query, bypass cache

#     verbose : int, default 0
#         Verbosity level for logging:
#         - 0: Minimal logging (default for multiple genes: progress bar only)
#         - 1: Standard logging (default for single gene)
#         - 2: Full verbose logging (prints details for each query/resource)

#     cache_as : str, default "json"
#         Cache format:
#         - "json": Store as JSON.gz (default, preserves nested structures)
#         - "parquet": Store as parquet (more efficient for large datasets, but nested DataFrames are converted to JSON strings)


#     Returns
#     -------
#     dict, list of dict, or dict of pandas.DataFrame
#         If return_as="json":
#             - If gene_id is a string, returns a dictionary containing gene information.
#             - If gene_id is a list, returns a list of dictionaries (one per gene).
#         If return_as="pandas":
#             - Returns a dictionary of pandas DataFrames, where each key is a resource type
#               and the value is a DataFrame containing that resource's data.
#               All resource DataFrames include a "gene_id" column linking back to the gene.


#         # Convert to pandas if requested
#         if return_as == "pandas":
#             return _convert_results_to_dataframes(results, include_phenotypes, include_drugs,
#                                                    include_tractability, include_pharmacogenetics,
#                                                    include_expression, include_depmap, include_interactions)
#         else:
#             return results


#     # Convert to pandas if requested
#     if return_as == "pandas":
#         return _convert_results_to_dataframes([result], include_phenotypes, include_drugs,
#                                                include_tractability, include_pharmacogenetics,
#                                                include_expression, include_depmap, include_interactions)
#     else:
#         return result


#     This contains the core logic that was previously in get_gene_info.
#     Results are cached as JSON.gz (default) or parquet files per gene ID.
#     """
#     # Suppress gget INFO messages when verbose < 2
#     gget_logger = logging.getLogger("gget")
#     original_level = gget_logger.level
#     if verbose < 2:
#         gget_logger.setLevel(logging.WARNING)


#     ensembl_id = gene_id


#     if resource_flags:
#         cache_parts.append("_".join(resource_flags))
#     else:
#         cache_parts.append("basic_only")

#     if include_phenotypes and phenotype_size is not None:
#         cache_parts.append(f"size{phenotype_size}")
#     if include_phenotypes and phenotype_from > 0:
#         cache_parts.append(f"from{phenotype_from}")

#     # Determine cache file extension based on cache_as
#     if cache_as == "parquet":
#         cache_file = str(DEFAULT_CACHE_DIR / f"{'_'.join(cache_parts)}.parquet")
#     else:
#         cache_file = str(DEFAULT_CACHE_DIR / f"{'_'.join(cache_parts)}.json.gz")


#     # Get gene information using gget
#     result = {
#         "ensembl_id": ensembl_id,
#         "approved_symbol": "",
#         "approved_name": "",
#         "biotype": "",
#     }

#     # If we resolved from a symbol, use that as the approved_symbol (fallback if gget.info doesn't provide it)
#     if resolved_symbol and resolved_symbol != ensembl_id:
#         result["approved_symbol"] = resolved_symbol


#     # Try to get basic gene info using gget.info
#     try:
#         gene_info = gget.info(ensembl_id)
#         if gene_info is not None:
#             # Log what we received for debugging (only at verbose=2)
#             if verbose >= 2:
#                 logger.info(f"gget.info returned type: {type(gene_info)}")

#             # gget.info can return DataFrame or dict
#             if isinstance(gene_info, pd.DataFrame) and len(gene_info) > 0:
#                 # Log columns for debugging (only at verbose=2)
#                 if verbose >= 2:
#                     logger.info(f"gget.info DataFrame columns: {list(gene_info.columns)}")
#                     logger.info(f"gget.info DataFrame shape: {gene_info.shape}")
#                     logger.info(f"gget.info first row:\n{gene_info.iloc[0]}")

#                 # Extract from first row - try both .get() and direct access
#                 first_row = gene_info.iloc[0]


#                 extracted_symbol = get_from_series(first_row, "gene_name", "name", "gene_name_ensembl", "external_name", "hgnc_symbol", "symbol", "gene_symbol")
#                 if extracted_symbol:
#                     result["approved_symbol"] = extracted_symbol
#                 # Otherwise keep the resolved_symbol if we have it


#                 result["biotype"] = (
#                     get_from_series(first_row, "gene_biotype", "biotype", "biotype_ensembl") or
#                     ""
#                 )

#                 # Log all available columns and their values for debugging (only at verbose=2)
#                 if verbose >= 2:
#                     logger.info(f"All columns in gget.info result: {list(gene_info.columns)}")
#                     logger.info(f"All values in first row: {first_row.to_dict()}")
#                     logger.info(f"Extracted: symbol={result['approved_symbol']}, name={result['approved_name']}, biotype={result['biotype']}")

#             elif isinstance(gene_info, dict):
#                 # Log keys for debugging (only at verbose=2)
#                 if verbose >= 2:
#                     logger.info(f"gget.info dict keys: {list(gene_info.keys())}")
#                     logger.info(f"gget.info dict full content: {gene_info}")


#                 result["biotype"] = (
#                     gene_info.get("gene_biotype") or
#                     gene_info.get("biotype") or
#                     gene_info.get("biotype_ensembl") or
#                     ""
#                 )

#                 if verbose >= 2:
#                     logger.info(f"Extracted from dict: symbol={result['approved_symbol']}, name={result['approved_name']}, biotype={result['biotype']}")
#             else:
#                 logger.warning(f"Unexpected type from gget.info: {type(gene_info)}")

#     except Exception as e:
#         logger.warning(f"Error fetching gene info for {ensembl_id}: {e}")
#         import traceback
#         logger.debug(traceback.format_exc())

#     # Save to cache
#     try:
#         if cache_as == "parquet":
#             # Save as parquet
#             # Convert dict to DataFrame (single row)
#             df = pd.DataFrame([result])


#             serializable_result = make_json_serializable(result)


#     # Restore original gget logger level
#     gget_logger.setLevel(original_level)

#     return result


# def get_genes_info_batch(
#     gene_ids: List[str],
#     include_phenotypes: bool = True,
#     phenotype_size: int = 1000,
#     delay: float = 0.1,
#     output_format: str = "pandas",
# ) -> Union[pd.DataFrame, pl.DataFrame]:
#     """
#     Get detailed information for multiple genes in batch.


#     for gene_id in tqdm(gene_ids, desc="Fetching gene info"):
#         try:
#             info = get_gene_info(
#                 gene_id,
#                 include_phenotypes=include_phenotypes,
#                 phenotype_size=phenotype_size,
#             )

#             # Flatten diseases for DataFrame
#             if include_phenotypes and "associated_diseases" in info:
#                 diseases = info.pop("associated_diseases")
#                 if output_format == "pandas":
#                     info["associated_diseases"] = str(diseases)  # JSON string for pandas
#                 else:
#                     info["associated_diseases"] = diseases  # Keep as list for polars

#             results.append(info)


#         # Rate limiting
#         import time
#         time.sleep(delay)

#     # Create DataFrame
#     if output_format == "pandas":
#         df = pd.DataFrame(results)
#     else:
#         df = pl.DataFrame(results)

#     return df


# def get_gene_diseases(
#     gene_id: Union[str, List[str]],
#     limit: Optional[int] = None,
#     output_format: str = "pandas",
#     force: int = 0,
# ) -> Union[pd.DataFrame, pl.DataFrame]:
#     """
#     Get diseases associated with a specific gene or multiple genes using gget.

#     This is a convenience wrapper around gget.opentargets() for diseases.
#     Results are cached as parquet files to avoid repeated API calls.


#         # Concatenate all results
#         if len(all_dfs) == 0:
#             if output_format == "pandas":
#                 return pd.DataFrame()
#             else:
#                 return pl.DataFrame()

#         # Filter out empty DataFrames before concatenating
#         non_empty_dfs = [df for df in all_dfs if len(df) > 0]
#         if len(non_empty_dfs) == 0:
#             if output_format == "pandas":
#                 return pd.DataFrame(columns=["gene_id"])
#             else:
#                 return pl.DataFrame({"gene_id": []})

#         if output_format == "pandas":
#             result_df = pd.concat(non_empty_dfs, ignore_index=True)
#         else:
#             result_df = pl.concat(non_empty_dfs)

#         return result_df

#     # Single gene ID - delegate to helper function
#     return _get_gene_diseases_single(
#         gene_id,
#         limit=limit,
#         output_format=output_format,
#         force=force,
#     )


# def _get_gene_diseases_single(
#     gene_id: str,
#     limit: Optional[int] = None,
#     output_format: str = "pandas",
#     force: int = 0,
# ) -> Union[pd.DataFrame, pl.DataFrame]:
#     """
#     Internal helper function to get diseases for a single gene.


#     # Set up cache file path
#     cache_file = str(DEFAULT_CACHE_DIR / f"diseases_{ensembl_id}.parquet")
#     if limit is not None:
#         cache_file = str(DEFAULT_CACHE_DIR / f"diseases_{ensembl_id}_limit{limit}.parquet")


#     # Fetch from API
#     logger.info(f"Fetching diseases for {ensembl_id} from OpenTargets Platform...")
#     diseases_data = gget.opentargets(ensembl_id, resource="diseases", limit=limit)


#         # Convert list of dicts to DataFrame - this preserves ALL fields from raw JSON
#         df = pd.DataFrame(diseases_data)

#         # Log what columns we have after conversion
#         logger.info(f"DataFrame created with {len(df)} rows and columns: {list(df.columns)}")

#         if output_format == "polars":
#             df = pl.from_pandas(df)


#     return df


# def get_gene_drugs(
#     gene_id: Union[str, List[str]],
#     disease_id: Optional[str] = None,
#     limit: Optional[int] = None,
#     output_format: str = "pandas",
#     force: int = 0,
# ) -> Union[pd.DataFrame, pl.DataFrame]:
#     """
#     Get drugs associated with a specific gene or multiple genes using gget.

#     This is a convenience wrapper around gget.opentargets() for drugs.
#     Results are cached as parquet files to avoid repeated API calls.


#         # Concatenate all results
#         if len(all_dfs) == 0:
#             if output_format == "pandas":
#                 return pd.DataFrame()
#             else:
#                 return pl.DataFrame()

#         # Filter out empty DataFrames before concatenating
#         non_empty_dfs = [df for df in all_dfs if len(df) > 0]
#         if len(non_empty_dfs) == 0:
#             if output_format == "pandas":
#                 return pd.DataFrame(columns=["gene_id"])
#             else:
#                 return pl.DataFrame({"gene_id": []})

#         if output_format == "pandas":
#             result_df = pd.concat(non_empty_dfs, ignore_index=True)
#         else:
#             result_df = pl.concat(non_empty_dfs)

#         return result_df

#     # Single gene ID - delegate to helper function
#     return _get_gene_drugs_single(
#         gene_id,
#         disease_id=disease_id,
#         limit=limit,
#         output_format=output_format,
#         force=force,
#     )


#     # Set up cache file path
#     cache_file = str(DEFAULT_CACHE_DIR / f"drugs_{ensembl_id}.parquet")
#     if disease_id:
#         cache_file = str(DEFAULT_CACHE_DIR / f"drugs_{ensembl_id}_disease_{disease_id}.parquet")
#     if limit is not None:
#         cache_file = str(DEFAULT_CACHE_DIR / f"drugs_{ensembl_id}_limit{limit}.parquet")
#         if disease_id:
#             cache_file = str(DEFAULT_CACHE_DIR / f"drugs_{ensembl_id}_disease_{disease_id}_limit{limit}.parquet")


#     # Fetch from API - use filters parameter for disease_id
#     logger.info(f"Fetching drugs for {ensembl_id} from OpenTargets Platform...")
#     filters = {}
#     if disease_id:
#         filters["disease_id"] = [disease_id]

#     drugs_data = gget.opentargets(
#         ensembl_id,
#         resource="drugs",
#         limit=limit,
#         filters=filters if filters else None,
#     )

#     if drugs_data is None or len(drugs_data) == 0:
#         logger.warning(f"No drugs found for {ensembl_id}")
#         if output_format == "pandas":
#             df = pd.DataFrame()
#         else:
#             df = pl.DataFrame()
#     else:
#         # Convert list of dicts to DataFrame
#         df = pd.DataFrame(drugs_data)

#         if output_format == "polars":
#             df = pl.from_pandas(df)


#     return df


# def get_gene_tractability(
#     gene_id: Union[str, List[str]],
#     output_format: str = "pandas",
#     force: int = 0,
# ) -> Union[pd.DataFrame, pl.DataFrame]:
#     """
#     Get tractability data for a specific gene or multiple genes using gget.

#     Tractability data indicates how "druggable" a gene is based on various criteria.


# def get_gene_pharmacogenetics(
#     gene_id: Union[str, List[str]],
#     drug_id: Optional[str] = None,
#     limit: Optional[int] = None,
#     output_format: str = "pandas",
#     force: int = 0,
# ) -> Union[pd.DataFrame, pl.DataFrame]:
#     """
#     Get pharmacogenetic responses for a specific gene or multiple genes using gget.


# def get_gene_depmap(
#     gene_id: Union[str, List[str]],
#     tissue_id: Optional[str] = None,
#     output_format: str = "pandas",
#     force: int = 0,
# ) -> Union[pd.DataFrame, pl.DataFrame]:
#     """
#     Get DepMap gene→disease-effect data using gget.


#     This centralizes the logic for fetching, caching, and processing gget.opentargets results.
#     """
#     # Suppress gget INFO messages (default behavior - can be overridden by caller)
#     gget_logger = logging.getLogger("gget")
#     original_level = gget_logger.level
#     gget_logger.setLevel(logging.WARNING)


#         if len(all_dfs) == 0:
#             if output_format == "pandas":
#                 return pd.DataFrame()
#             else:
#                 return pl.DataFrame()

#         non_empty_dfs = [df for df in all_dfs if len(df) > 0]
#         if len(non_empty_dfs) == 0:
#             if output_format == "pandas":
#                 return pd.DataFrame(columns=["gene_id"])
#             else:
#                 return pl.DataFrame({"gene_id": []})

#         if output_format == "pandas":
#             result_df = pd.concat(non_empty_dfs, ignore_index=True)
#         else:
#             result_df = pl.concat(non_empty_dfs)
#         # Restore original gget logger level before returning
#         gget_logger.setLevel(original_level)
#         return result_df


#     cache_file = str(DEFAULT_CACHE_DIR / f"{'_'.join(cache_parts)}.parquet")


#                 return series.apply(convert_value)


#                 df_pd.to_parquet(cache_file, index=False)
#                 df = pl.read_parquet(cache_file)
#             logger.info(f"Cached {len(df)} {resource} results to: {cache_file}")
#         except Exception as e:
#             logger.warning(f"Error caching {resource} to {cache_file}: {e}")
#             import traceback
#             logger.debug(traceback.format_exc())

#     # Restore original gget logger level
#     gget_logger.setLevel(original_level)

#     return df


# def list_cache(
#     pattern: Optional[str] = None,
#     cache_dir: Optional[str] = None,
#     output_format: str = "pandas",
# ) -> Union[pd.DataFrame, pl.DataFrame]:
#     """
#     List all cached files in the OpenTargets cache directory.

#     Parameters
#     ----------
#     pattern : str, optional
#         Pattern to filter files (e.g., "gene_info", "pharmacogenetics", "ENSG00000157764").
#         If None, returns all cached files. The pattern is matched against the filename
#         (case-insensitive substring match).
#     cache_dir : str, optional
#         Cache directory to list. If None, uses DEFAULT_CACHE_DIR.
#     output_format : str, default "pandas"
#         Output format: "pandas" or "polars"


#     cache_dir = Path(cache_dir)


#     # Get all files in cache directory
#     all_files = []
#     for file_path in cache_dir.iterdir():
#         if file_path.is_file():
#             filename = file_path.name

#             # Apply pattern filter if provided
#             if pattern is not None and pattern.lower() not in filename.lower():
#                 continue

#             # Get file stats
#             stat = file_path.stat()
#             size_bytes = stat.st_size
#             size_mb = round(size_bytes / (1024 * 1024), 2)
#             modified_time = pd.Timestamp.fromtimestamp(stat.st_mtime)

#             # Determine file type
#             if filename.endswith('.json.gz'):
#                 file_type = "json.gz"
#             elif filename.endswith('.parquet'):
#                 file_type = "parquet"
#             elif filename.endswith('.gz'):
#                 file_type = "gz"
#             else:
#                 file_type = "other"

#             # Infer resource type from filename
#             resource_type = "unknown"
#             gene_id = None

#             # Extract gene ID (ENSG pattern)
#             import re
#             gene_match = re.search(r'ENSG\d+', filename)
#             if gene_match:
#                 gene_id = gene_match.group(0)


#             all_files.append({
#                 "filename": filename,
#                 "filepath": str(file_path),
#                 "size_bytes": size_bytes,
#                 "size_mb": size_mb,
#                 "modified_time": modified_time,
#                 "file_type": file_type,
#                 "resource_type": resource_type,
#                 "gene_id": gene_id,
#             })


def get_pathways(
    targets: pd.DataFrame,
    gene_symbol_col: str = "approvedSymbol",
    gene_id_col: str = "id",
    save_path: str | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Extract pathways from the nested pathways column and create a DataFrame with one row per pathway.

    The pathways column contains a list of dictionaries, where each dictionary
    has keys like 'pathway', 'pathwayId', 'name', 'label', 'id'. This function
    extracts pathway information and groups genes by pathway.

    Parameters
    ----------
    targets : pd.DataFrame
        DataFrame with a "pathways" column containing nested pathway data and
        a gene symbol column (default: "approvedSymbol").
        Each row's pathways column should be either:
        - None/NaN: no pathways
        - A list of dictionaries with pathway information
        - A list of strings (less common)
    gene_symbol_col : str, default "approvedSymbol"
        Column name containing gene symbols to associate with each pathway.
    gene_id_col : str, default "id"
        Column name containing gene IDs to associate with each pathway.
    save_path : str, optional
        If provided, save the resulting DataFrame to this parquet file path.
        The DataFrame will be saved with all columns including nested structures (lists).
        If None (default), the DataFrame is not saved.
    force : bool, default False
        If True, force regeneration even if save_path already exists.
        If False and save_path exists, load the existing parquet file instead of regenerating.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per unique pathway. Columns:
        - pathway: pathway name/identifier (str)
        - pathwayId: pathway ID if available (str, may be NaN)
        - gene_symbols: list of unique gene symbols associated with this pathway (list)
        - gene_ids: list of unique gene IDs associated with this pathway (list)
        - gene_count: number of unique genes associated with this pathway (int)
        - markdown: pathway-centric markdown description (str)
        - tokens: approximate token count in markdown (int)

    Examples
    --------
    >>> import biodb.opentargets as ot
    >>> targets = ot.get_targets(limit=100)
    >>> pathways_df = ot.get_pathways(targets)
    >>> # Get pathways with most genes
    >>> pathways_df.sort_values('gene_count', ascending=False).head()
    >>> # Get all gene symbols for a specific pathway
    >>> pathway_genes = pathways_df[pathways_df['pathway'] == 'Some Pathway']['gene_symbols'].iloc[0]
    >>> # Save to file
    >>> pathways_df = ot.get_pathways(targets, save_path="pathways.parquet")
    >>> # Force regeneration even if file exists
    >>> pathways_df = ot.get_pathways(targets, save_path="pathways.parquet", force=True)
    """
    # Check if file exists and force is False - if so, just load and return it
    if save_path is not None and not force:
        save_path_obj = Path(save_path)
        if save_path_obj.exists():
            print(f"Loading existing pathways DataFrame from: {save_path}")
            try:
                pathways_df = pd.read_parquet(save_path)
                print(f"  Loaded {len(pathways_df)} pathways from existing file")
                print(f"\nNumber of unique pathways: {len(pathways_df)}")
                return pathways_df
            except Exception as e:
                print(f"  Error loading existing file: {e}")
                print("  Will regenerate pathways DataFrame...")

    if "pathways" not in targets.columns:
        raise ValueError("DataFrame must have a 'pathways' column")

    if gene_symbol_col not in targets.columns:
        raise ValueError(f"DataFrame must have a '{gene_symbol_col}' column")

    include_gene_ids = gene_id_col in targets.columns

    def extract_pathway_info(item):
        """Extract pathway information from a single item (dict or string)."""
        if item is None:
            return None, None

        # If it's already a string, use it as pathway name
        if isinstance(item, str):
            return item, None

        # If it's a dict, extract pathway name and ID
        if isinstance(item, dict):
            pathway_name = None
            pathway_id = None

            # Try to extract pathway name (preferred keys in order)
            for key in ["pathway", "name", "label"]:
                if key in item and item[key] is not None:
                    val = item[key]
                    pathway_name = str(val) if not isinstance(val, str) else val
                    break

            # Try to extract pathway ID
            for key in ["pathwayId", "id"]:
                if key in item and item[key] is not None:
                    val = item[key]
                    pathway_id = str(val) if not isinstance(val, str) else val
                    break

            # If no pathway name found, use ID or string representation
            if pathway_name is None:
                pathway_name = pathway_id if pathway_id is not None else str(item)

            return pathway_name, pathway_id

        # For other types, convert to string
        return str(item), None

    def extract_pathways_from_row(pathways_value):
        """Extract all pathways from a single row's pathways value."""
        if pathways_value is None or (
            isinstance(pathways_value, float) and pd.isna(pathways_value)
        ):
            return []

        # Handle string representation of list/dict (JSON)
        if isinstance(pathways_value, str):
            try:
                pathways_value = json.loads(pathways_value)
            except (json.JSONDecodeError, TypeError):
                # If it's not JSON, treat as a single pathway string
                pathway_name, pathway_id = extract_pathway_info(pathways_value)
                return [(pathway_name, pathway_id)] if pathway_name is not None else []

        # Convert numpy array to list if needed
        if isinstance(pathways_value, np.ndarray):
            pathways_value = pathways_value.tolist()

        # If it's not a list, wrap it
        if not isinstance(pathways_value, list):
            pathway_name, pathway_id = extract_pathway_info(pathways_value)
            return [(pathway_name, pathway_id)] if pathway_name is not None else []

        # Extract pathways from list
        extracted = []
        for item in pathways_value:
            pathway_name, pathway_id = extract_pathway_info(item)
            if pathway_name is not None:
                extracted.append((pathway_name, pathway_id))

        return extracted

    # Build pathway -> genes mapping
    pathway_to_genes = {}

    for _idx, row in targets.iterrows():
        gene_symbol = row[gene_symbol_col]

        # Skip if gene symbol is missing
        if pd.isna(gene_symbol) or gene_symbol is None:
            continue

        # Convert to string
        gene_symbol = str(gene_symbol)

        # Get gene ID if available
        gene_id = None
        if include_gene_ids:
            gene_id_val = row[gene_id_col]
            if not (pd.isna(gene_id_val) or gene_id_val is None):
                gene_id = str(gene_id_val)

        # Extract pathways for this gene
        pathways_info = extract_pathways_from_row(row["pathways"])

        # Add gene to each pathway
        for pathway_name, pathway_id in pathways_info:
            if pathway_name not in pathway_to_genes:
                pathway_to_genes[pathway_name] = {
                    "pathwayId": pathway_id,
                    "gene_symbols": set(),
                    "gene_ids": set(),
                }
            pathway_to_genes[pathway_name]["gene_symbols"].add(gene_symbol)
            if gene_id is not None:
                pathway_to_genes[pathway_name]["gene_ids"].add(gene_id)

    # Convert to DataFrame
    rows = []
    for pathway_name, pathway_data in pathway_to_genes.items():
        gene_symbols_list = sorted(list(pathway_data["gene_symbols"]))  # Sort for consistency
        gene_ids_list = sorted(list(pathway_data["gene_ids"])) if include_gene_ids else []

        row_dict = {
            "pathway": pathway_name,
            "pathwayId": pathway_data["pathwayId"],
            "gene_symbols": gene_symbols_list,
            "gene_count": len(gene_symbols_list),
        }

        if include_gene_ids:
            row_dict["gene_ids"] = gene_ids_list

        rows.append(row_dict)

    pathways_df = pd.DataFrame(rows)

    # Sort by gene count (descending) then by pathway name for consistency
    pathways_df = pathways_df.sort_values(
        ["gene_count", "pathway"], ascending=[False, True]
    ).reset_index(drop=True)

    # Create pathway-centric markdown
    def pathway_to_markdown(row):
        """Create markdown for a pathway row."""
        lines = []

        # Pathway info
        lines.append("# Pathway Info")
        lines.append(f"Pathway: {row['pathway']}")
        if pd.notna(row["pathwayId"]):
            lines.append(f"Pathway ID: {row['pathwayId']}")
        lines.append(f"Number of genes: {row['gene_count']}")
        lines.append("")

        # Associated genes
        if row["gene_symbols"]:
            lines.append("## Associated Genes")
            lines.append(f"Gene symbols: {', '.join(row['gene_symbols'])}")
            if "gene_ids" in row and row["gene_ids"]:
                lines.append(f"Gene IDs: {', '.join(row['gene_ids'])}")
            lines.append("")

        return "\n".join(lines)

    # Add markdown column
    pathways_df["markdown"] = pathways_df.apply(pathway_to_markdown, axis=1)

    # Add tokens column
    try:
        from .utils import count_tokens

        pathways_df["tokens"] = pathways_df["markdown"].apply(
            lambda x: count_tokens(x, approximate=True)
        )
    except (ImportError, AttributeError):
        # Fallback if count_tokens is not available
        pathways_df["tokens"] = pathways_df["markdown"].apply(lambda x: len(x) // 4)

    pathways_df.index = pathways_df.pathwayId.tolist()

    # Save to parquet if save_path is provided
    if save_path is not None:
        # Ensure the directory exists
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Save to parquet (parquet format handles nested structures like lists automatically)
        pathways_df.to_parquet(
            save_path, index=True
        )  # Keep index=True since we set pathwayId as index

        print(f"\nSaved pathways DataFrame to: {save_path}")
        print(f"  Shape: {pathways_df.shape}")
        print(f"  Columns: {list(pathways_df.columns)}")

    print(f"\nNumber of unique pathways: {len(pathways_df)}")

    return pathways_df
