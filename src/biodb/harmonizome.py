"""Harmonizome client — bulk Maayan-Lab gene-set library downloader.

The [Harmonizome](https://maayanlab.cloud/Harmonizome/) is a Maayan-Lab
collection of curated gene-attribute association datasets (~114 datasets,
~30 GB if fully downloaded). This module wraps the public REST API
(``maayanlab.cloud/Harmonizome/api``) and the bulk-download tree at
``maayanlab.cloud/static/hdfs/harmonizome/data``.

Ported from `AoU.phenome.harmonizome <https://github.com/bschilder/AoU>`_
with these hygiene fixes:

* **Lazy config** — the original module fetched ``DOWNLOADS`` and
  ``DATASET_TO_PATH`` from the API at import time, which broke offline
  imports and made `pytest --collect-only` hit the network. Here those
  constants are exposed via a module-level ``__getattr__`` (PEP 562) so
  the fetch only happens on first access.
* Python-2 compat shims removed (``raw_input``, ``StringIO as BytesIO``,
  ``urllib2``). Python 3.10+ only.
* Dead helpers using removed APIs (``pd.SparseDataFrame``, ``np.object``)
  pruned — ``_parse``, ``_parse_df``, ``_read_as_dataframe`` family.
* The 150-line ``if __name__ == '__main__'`` example block dropped.
* ``read_gmt`` (and the ``looks_like_gene_set_name`` /
  ``reverse_excel_date_conversion`` helpers) inlined as private symbols
  here since :func:`get_gmt` is their only consumer.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from fnmatch import fnmatch
from functools import cache
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote_plus
from urllib.request import urlopen

import pandas as pd
import requests
from tqdm.auto import tqdm

from biodb.utils import (
    _is_skippable_gene,  # noqa: F401  -- back-compat alias for tests
    _looks_like_gene_set_name,  # noqa: F401  -- back-compat alias for tests
    _parse_gmt_line,  # noqa: F401  -- back-compat alias for tests
    _reverse_excel_date,  # noqa: F401  -- back-compat alias for tests
    read_gmt,
)

logger = logging.getLogger(__name__)

VERSION = "1.0"
API_URL = "https://maayanlab.cloud/Harmonizome/api"
DOWNLOAD_URL = "https://maayanlab.cloud/static/hdfs/harmonizome/data"

CACHE_DIR = Path("~/.cache/harmonizome").expanduser()
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ─── Lazy config (PEP 562 module-level __getattr__) ─────────────────────────
# The Harmonizome API exposes the canonical list of dataset paths and
# downloadable filenames at /dark/script_config. Fetching it at import
# time broke offline imports; we cache the response after the first
# successful call instead.


def json_from_url(url: str) -> Any:
    """GET ``url`` and decode the JSON body."""
    with urlopen(url) as response:
        return json.loads(response.read().decode("utf-8"))


@cache
def _load_config() -> dict[str, Any]:
    """Fetch ``/dark/script_config`` once; subsequent calls hit the cache."""
    return json_from_url(API_URL + "/dark/script_config")


def __getattr__(name: str) -> Any:
    if name == "DOWNLOADS":
        return list(_load_config().get("downloads", []))
    if name == "DATASET_TO_PATH":
        return _load_config().get("datasets", {})
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ─── Bulk download ──────────────────────────────────────────────────────────


def _download_file(response: requests.Response, filename: Path) -> None:
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)


def _download_and_decompress_file(response: requests.Response, filename: Path) -> None:
    """Stream a gzip response to disk, decompressed in-place."""
    filename.parent.mkdir(parents=True, exist_ok=True)
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    output_filename = filename.with_suffix("") if isinstance(filename, Path) else filename[:-3]
    with open(output_filename, "wb") as f:
        while True:
            chunk = response.raw.read(1024)
            if not chunk:
                break
            f.write(decompressor.decompress(chunk))


def download_datasets(
    selected_datasets: list[tuple[str, str]],
    selected_downloads: list[str],
    decompress: bool = False,
    cache_dir: str | Path | None = None,
    verbose: bool = False,
) -> dict[str, str]:
    """Download a set of Harmonizome dataset files.

    Parameters
    ----------
    selected_datasets : list[tuple[str, str]]
        ``(dataset_name, dataset_path)`` pairs. ``dataset_path`` is the
        directory under ``DOWNLOAD_URL`` (e.g. ``"achilles"``).
    selected_downloads : list[str]
        Filenames to fetch for each dataset (e.g.
        ``["gene_attribute_matrix.txt.gz"]``).
    decompress : bool, default False
        If True, gunzip ``.txt.gz`` files on the fly.
    cache_dir : str or Path, optional
        Cache root. Defaults to :data:`CACHE_DIR`.
    verbose : bool, default False

    Returns
    -------
    dict[str, str]
        Map of ``downloadable → absolute local path``. Files that
        couldn't be fetched (HTTP error, network failure) are omitted.
    """
    cache_path = Path(cache_dir).expanduser() if cache_dir else CACHE_DIR
    file_paths: dict[str, str] = {}

    for dataset, dataset_path in selected_datasets:
        dataset_dir = cache_path / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)

        for downloadable in selected_downloads:
            filename = dataset_dir / downloadable
            actual_file = (
                filename.with_suffix("") if decompress and "txt.gz" in str(filename) else filename
            )

            # Reuse cached version if present.
            if actual_file.exists():
                file_paths[downloadable] = str(actual_file.resolve())
                continue
            if filename.exists() and actual_file != filename:
                file_paths[downloadable] = str(filename.resolve())
                continue

            url = f"{DOWNLOAD_URL}/{dataset_path}/{downloadable}"
            try:
                response = requests.get(url, stream=True, timeout=60)
            except requests.RequestException:
                continue
            # Not every dataset has every downloadable.
            if response.status_code != 200:
                continue

            try:
                if decompress and "txt.gz" in str(filename):
                    _download_and_decompress_file(response, filename)
                    target = actual_file if actual_file.exists() else filename
                    file_paths[downloadable] = str(target.resolve())
                else:
                    _download_file(response, filename)
                    file_paths[downloadable] = str(filename.resolve())
            except OSError as exc:
                logger.warning("Failed to write %s: %s", filename, exc)

        if verbose:
            logger.info("%s downloaded", dataset)

    return file_paths


# ─── REST API ───────────────────────────────────────────────────────────────


def list_datasets(as_df: bool = True) -> pd.DataFrame | list[dict[str, Any]]:
    """Page through ``/api/<VERSION>/dataset`` and return every dataset entry.

    Parameters
    ----------
    as_df : bool, default True
        If True, return a DataFrame with columns ``name`` and ``href``.

    Returns
    -------
    pd.DataFrame or list[dict]
    """
    all_entities: list[dict[str, Any]] = []
    base_url = "https://maayanlab.cloud/Harmonizome"
    url: str | None = f"{base_url}/api/{VERSION}/dataset"

    while url:
        try:
            response = json_from_url(url)
        except (HTTPError, OSError):
            break
        all_entities.extend(response.get("entities", []))
        nxt = response.get("next")
        if not nxt:
            url = None
        elif nxt.startswith("/"):
            url = f"{base_url}{nxt}"
        elif nxt.startswith("http"):
            url = nxt
        else:
            url = f"{base_url}/api/{VERSION}/{nxt}"

    if as_df:
        return pd.DataFrame(all_entities or [], columns=["name", "href"])
    return all_entities


_EMPTY_METADATA = {
    "description": "",
    "measurement": "",
    "association": "",
    "category": "",
    "resource": "",
    "citations": "",
    "last_updated": "",
    "stats": "",
}


def get_dataset_metadata(dataset_name: str, verbose: int = 0) -> dict[str, str]:
    """Fetch metadata for one Harmonizome dataset, stringifying nested fields.

    All values are coerced to strings (citations joined with ``"; "``,
    nested dicts rendered as ``"k: v; …"``) so the result is parquet-safe.
    """
    try:
        encoded = quote_plus(dataset_name)
        metadata = json_from_url(f"{API_URL}/{VERSION}/dataset/{encoded}")
    except (HTTPError, OSError, ValueError) as exc:
        if verbose >= 2:
            logger.warning("Could not fetch metadata for %s: %s", dataset_name, exc)
        return _EMPTY_METADATA.copy()

    resource = metadata.get("resource", "")
    if not isinstance(resource, str):
        resource = str(resource) if resource else ""

    citations = metadata.get("citations", [])
    if isinstance(citations, list):
        citations = "; ".join(citations) if citations else ""

    stats = metadata.get("stats", {})
    if isinstance(stats, dict):
        stats = "; ".join(f"{k}: {v}" for k, v in stats.items())
    elif not isinstance(stats, str):
        stats = str(stats) if stats else ""

    return {
        "description": str(metadata.get("description", "")),
        "measurement": str(metadata.get("measurement", "")),
        "association": str(metadata.get("association", "")),
        "category": str(metadata.get("category", "")),
        "resource": resource,
        "citations": citations,
        "last_updated": str(metadata.get("lastUpdated", "")),
        "stats": stats,
    }


# ─── GMT reader (delegated to biodb.utils.read_gmt) ─────────────────────────
# The GMT-parse helpers moved to :mod:`biodb.utils` so MSigDB / gProfiler /
# Enrichr can reuse them. ``_read_gmt`` here is a back-compat alias for
# anything that still patches it by its old in-module name.

_read_gmt = read_gmt

# ─── get_gmt: parallel download + read + merge ──────────────────────────────


_GMT_COLUMNS = [
    "dataset",
    "dataset_href",
    "file",
    "description",
    "measurement",
    "association",
    "category",
    "resource",
    "citations",
    "last_updated",
    "stats",
    "id",
    "label",
    "gene",
]


def _decompress_inplace(file_path: Path) -> Path:
    """If ``file_path`` is gzip, write the decompressed version next to it."""
    if file_path.suffix != ".gz":
        return file_path
    decompressed = file_path.with_suffix("")
    if not decompressed.exists():
        with gzip.open(file_path, "rb") as f_in, open(decompressed, "wb") as f_out:
            f_out.write(f_in.read())
    return decompressed


def _read_one_gmt(
    dataset_name: str,
    dataset_href: str,
    file_path: str,
) -> pd.DataFrame | None:
    """Load one GMT file and tag it with its dataset metadata."""
    try:
        actual = _decompress_inplace(Path(file_path))
        gmt_df = _read_gmt(actual, return_format="pandas", suppress_stats=True)
        return pd.DataFrame(
            {
                "dataset": dataset_name,
                "dataset_href": dataset_href,
                "file": actual.name,
                "id": gmt_df["id"],
                "label": gmt_df["label"],
                "gene": gmt_df["gene"],
            }
        )
    except (OSError, ValueError) as exc:
        logger.warning("Error reading %s: %s", file_path, exc)
        return None


def _matches_file_type(filename: str, file_types: str | list[str] | None) -> bool:
    if file_types is None:
        return filename.endswith((".gmt", ".gmt.gz"))
    basename = filename[:-3] if filename.endswith(".gz") else filename
    if isinstance(file_types, list):
        return basename in file_types or filename in file_types
    if isinstance(file_types, str):
        return fnmatch(basename, file_types) or fnmatch(filename, file_types)
    raise TypeError(
        f"file_types must be None, list, or glob string; got {type(file_types).__name__}"
    )


def get_gmt(
    datasets: list[str] | pd.DataFrame | None = None,
    cache_dir: str | Path | None = None,
    verbose: int = 1,
    force: int = 0,
    max_workers: int | None = 1,
    save_path: str | Path | bool | None = None,
    limit: int | None = None,
    file_types: str | list[str] | None = "gene_set*.gmt.gz",
) -> pd.DataFrame:
    """Download + concatenate GMT files across Harmonizome datasets.

    Parameters
    ----------
    datasets : list[str], pd.DataFrame, or None
        ``None`` → every dataset from :func:`list_datasets`.
        ``list`` → dataset names. ``DataFrame`` → must have ``name`` /
        ``href`` columns.
    cache_dir : str or Path, optional
        Cache root. Defaults to :data:`CACHE_DIR`.
    verbose : {0, 1, 2}, default 1
        0 silent, 1 progress bars + summary, 2 debug logging.
    force : {0, 1, 2}, default 0
        0 reuse merged parquet, 1 reimport cached GMTs, 2 redownload all.
    max_workers : int or None, default 1
        Thread pool size for GMT read + metadata fetch. ``None`` →
        ``min(8, cpu_count(), n_tasks)``.
    save_path : str, Path, False, or None
        Where to cache the merged DataFrame. ``False`` disables caching.
    limit : int, optional
        Cap the number of datasets processed (useful for smoke tests).
    file_types : str, list[str], or None
        GMT filename filter — glob string or explicit list, applied to
        the basename with ``.gz`` stripped.

    Returns
    -------
    pd.DataFrame with columns: ``dataset``, ``dataset_href``, ``file``,
    plus dataset metadata fields (``description``, ``measurement``,
    ``association``, ``category``, ``resource``, ``citations``,
    ``last_updated``, ``stats``) and GMT fields (``id``, ``label``,
    ``gene``).
    """
    cache_path = Path(cache_dir).expanduser() if cache_dir else CACHE_DIR

    if save_path is None:
        merged_cache: Path | None = cache_path / "gmt_merged.parquet"
    elif save_path is False:
        merged_cache = None
    else:
        merged_cache = Path(save_path)

    # Fast path: cached merged parquet
    if force == 0 and merged_cache is not None and merged_cache.exists():
        try:
            return pd.read_parquet(merged_cache)
        except (OSError, ValueError) as exc:
            if verbose >= 2:
                logger.warning("Failed to read cached parquet %s: %s", merged_cache, exc)

    # Resolve dataset list
    if datasets is None:
        datasets_df = list_datasets(as_df=True)
    elif isinstance(datasets, pd.DataFrame):
        datasets_df = datasets.copy()
    else:
        datasets_df = pd.DataFrame({"name": list(datasets)})
        all_datasets = list_datasets(as_df=True)
        datasets_df = datasets_df.merge(all_datasets, on="name", how="left")

    dataset_to_path = _load_config().get("datasets", {})
    downloads_config = list(_load_config().get("downloads", []))
    gmt_downloads = [d for d in downloads_config if _matches_file_type(d, file_types)]

    if not gmt_downloads:
        return pd.DataFrame(columns=_GMT_COLUMNS)

    datasets_df = datasets_df[datasets_df["name"].isin(dataset_to_path.keys())].copy()
    if limit is not None and limit > 0:
        datasets_df = datasets_df.head(limit)
    if datasets_df.empty:
        return pd.DataFrame(columns=_GMT_COLUMNS)

    # Pre-resolve each dataset's cached files (or download if needed)
    all_tasks: list[tuple[str, str, str, str]] = []
    dataset_iter: Any = [(row["name"], row.get("href", "")) for _, row in datasets_df.iterrows()]
    if verbose == 1:
        dataset_iter = tqdm(dataset_iter, desc="Scanning datasets")

    for dataset_name, dataset_href in dataset_iter:
        dataset_dir = cache_path / dataset_name
        cached: dict[str, str] = {}
        need_download = force >= 2

        if not need_download:
            for dl in gmt_downloads:
                f = dataset_dir / dl
                if f.exists():
                    cached[dl] = str(f)
                elif dl.endswith(".gz") and (decomp := f.with_suffix("")).exists():
                    cached[dl] = str(decomp)
                else:
                    need_download = True
                    break
            if not cached and not need_download:
                continue

        if need_download:
            downloaded = download_datasets(
                selected_datasets=[(dataset_name, dataset_to_path[dataset_name])],
                selected_downloads=gmt_downloads,
                decompress=False,
                cache_dir=cache_path,
                verbose=(verbose >= 2),
            )
            cached.update(downloaded)

        all_tasks.extend((dataset_name, dataset_href, dl, fp) for dl, fp in cached.items())

    if not all_tasks:
        return pd.DataFrame(columns=_GMT_COLUMNS)

    # Parallel-read every GMT
    workers = max_workers or min(8, os.cpu_count() or 4, len(all_tasks))
    workers = min(workers, len(all_tasks))
    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_read_one_gmt, ds, href, fp): (ds, fp) for ds, href, _, fp in all_tasks
        }
        it: Any = as_completed(futures)
        if verbose == 1:
            it = tqdm(it, total=len(futures), desc="Reading GMT files")
        for future in it:
            result = future.result()
            if result is not None:
                frames.append(result)

    if not frames:
        return pd.DataFrame(columns=_GMT_COLUMNS)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    if verbose >= 1:
        logger.info(
            "Loaded %d rows; %d gene sets, %d labels, %d genes",
            len(combined),
            combined["id"].nunique(),
            combined["label"].nunique(),
            combined["gene"].nunique(),
        )

    # Fetch metadata for every unique dataset in parallel
    unique_datasets = combined["dataset"].unique().tolist()
    metadata_workers = min(10, len(unique_datasets)) or 1
    metadata_map: dict[str, dict[str, str]] = {}
    with ThreadPoolExecutor(max_workers=metadata_workers) as pool:
        futures = {pool.submit(get_dataset_metadata, ds, verbose): ds for ds in unique_datasets}
        it = as_completed(futures)
        if verbose == 1:
            it = tqdm(it, total=len(futures), desc="Fetching metadata")
        for future in it:
            ds = futures[future]
            try:
                metadata_map[ds] = future.result()
            except Exception:  # noqa: BLE001 -- network/parse failures should not abort the run
                metadata_map[ds] = _EMPTY_METADATA.copy()

    metadata_df = pd.DataFrame(
        [{"dataset": ds, **metadata_map.get(ds, _EMPTY_METADATA)} for ds in unique_datasets]
    )
    for col in metadata_df.columns:
        if col == "dataset":
            continue
        combined[col] = combined["dataset"].map(metadata_df.set_index("dataset")[col])

    # Reorder
    cols = [c for c in _GMT_COLUMNS if c in combined.columns]
    combined = combined[cols + [c for c in combined.columns if c not in cols]]

    if merged_cache is not None:
        try:
            merged_cache.parent.mkdir(parents=True, exist_ok=True)
            combined.to_parquet(merged_cache, index=False, compression="snappy")
        except (OSError, ValueError, ImportError) as exc:
            if verbose >= 1:
                logger.warning("Failed to cache merged DataFrame to %s: %s", merged_cache, exc)

    return combined


# ─── gene_attribute_matrix.txt loader ───────────────────────────────────────


def load_gene_attribute_matrix(
    dataset_name: str,
    filename: str = "gene_attribute_matrix.txt.gz",
    cache_dir: str | Path | None = None,
    include_col_metadata: bool = True,
) -> tuple[pd.DataFrame, dict[str, str] | None, pd.DataFrame | None]:
    """Load one Harmonizome ``gene_attribute_matrix`` file.

    File format
    -----------
    * Comment lines starting with ``#`` (tab-separated column metadata)
    * Header row: ``GeneSym``, ``NA``, ``GeneID``/``NA``, then per-column
      attribute (e.g. cell-line) names
    * Optional metadata row: ``#``, ``#``, ``CellLine``, *cell line names*,
      ``Tissue``, *tissue types*
    * Data rows: gene symbol, NA, gene id, then numeric values

    Returns
    -------
    (df, tissue_metadata, column_metadata) — ``tissue_metadata`` is a
    ``dict[col_name, tissue]`` (or ``None`` if missing); ``column_metadata``
    is a DataFrame collecting every commented metadata row keyed by
    column name.
    """
    cache_path = Path(cache_dir).expanduser() if cache_dir else CACHE_DIR
    filepath = cache_path / dataset_name / filename

    use_gzip = filename.endswith(".gz")
    if use_gzip and not filepath.exists():
        decompressed = cache_path / dataset_name / filename[:-3]
        if decompressed.exists():
            filepath = decompressed
            use_gzip = False
    if not filepath.exists():
        raise FileNotFoundError(
            f"File not found: {filepath}\nDownload it first with download_datasets("
            f"[('{dataset_name}', '<path>')], ['{filename}'])"
        )

    opener: Any = gzip.open if use_gzip else open
    with opener(filepath, "rt") as f:
        lines = f.readlines()

    # Collect leading commented metadata + locate header
    column_metadata_rows: list[list[str]] = []
    header_idx: int | None = None
    metadata_idx: int | None = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#"):
            inner = stripped[1:].strip()
            if inner:
                column_metadata_rows.append(inner.split("\t"))
        elif header_idx is None:
            header_idx = i
        elif metadata_idx is None and stripped.split("\t")[0] not in ("GeneSym", "na"):
            parts = stripped.split("\t")
            if len(parts) > 2 and parts[2] == "CellLine":
                metadata_idx = i
                break

    if header_idx is None:
        raise ValueError("Could not find header line in file")

    header_parts = lines[header_idx].strip().split("\t")
    if header_parts[0] != "GeneSym":
        raise ValueError(f"Expected first column to be 'GeneSym', got: {header_parts[0]}")

    cell_line_start = _detect_attribute_start(header_parts)
    cell_line_names = header_parts[cell_line_start:]

    tissue_metadata: dict[str, str] | None = None
    if include_col_metadata and metadata_idx is not None:
        meta_parts = lines[metadata_idx].strip().split("\t")
        try:
            tissue_idx = meta_parts.index("Tissue")
            slice_end = tissue_idx + 1 + len(cell_line_names)
            if slice_end <= len(meta_parts):
                tissue_metadata = dict(
                    zip(cell_line_names, meta_parts[tissue_idx + 1 : slice_end], strict=False)
                )
        except ValueError:
            pass

    data_start = (metadata_idx + 1) if metadata_idx is not None else (header_idx + 1)
    if data_start < len(lines):
        first = lines[data_start].strip().split("\t")
        if len(first) > 2 and first[1].lower() == "na" and first[2].lower() == "na":
            data_start += 1

    data_rows = []
    for i in range(data_start, len(lines)):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        gene_sym = parts[0]
        gene_id: str | None = None
        for j in range(1, min(4, len(parts))):
            val = parts[j]
            if val.lower() in ("na", "nan", ""):
                continue
            try:
                float(val)
                gene_id = val
                break
            except ValueError:
                if len(val) < 20 and not val.startswith("Gene"):
                    gene_id = val
                    break
        values = parts[cell_line_start : cell_line_start + len(cell_line_names)]
        numeric_values = [
            float("nan") if v.lower() in ("na", "nan", "") else _safe_float(v) for v in values
        ]
        data_rows.append([gene_sym, gene_id, *numeric_values])

    df = pd.DataFrame(data_rows, columns=["GeneSym", "GeneID", *cell_line_names])
    for col in cell_line_names:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["GeneID"] = pd.to_numeric(df["GeneID"], errors="coerce")

    column_metadata = _build_column_metadata(column_metadata_rows, cell_line_names)
    return df, tissue_metadata, column_metadata


def _safe_float(v: str) -> float:
    try:
        return float(v)
    except ValueError:
        return float("nan")


_IDENTIFIER_PATTERNS = {"GeneSym", "NA", "GeneID", "ProbesetID", "na", "GeneID/NA"}
_METADATA_PATTERNS = {"CellLine", "Tissue"}


def _detect_attribute_start(header_parts: list[str]) -> int:
    """Find the first column index where per-attribute names begin."""
    for i, col in enumerate(header_parts):
        if col in _IDENTIFIER_PATTERNS or col.lower() in ("na", "nan"):
            continue
        if col in _METADATA_PATTERNS or col.startswith("Gene"):
            continue
        if (
            2 <= len(col) <= 15
            and col.replace("_", "").replace("-", "").isalnum()
            and i + 2 < len(header_parts)
        ):
            nxt = header_parts[i + 1]
            if (
                nxt not in _IDENTIFIER_PATTERNS
                and nxt not in _METADATA_PATTERNS
                and not nxt.startswith("Gene")
                and 2 <= len(nxt) <= 15
            ):
                return i
    # Fallback: most files have 3 identifier columns
    return 3


def _build_column_metadata(
    rows: list[list[str]], cell_line_names: list[str]
) -> pd.DataFrame | None:
    """Assemble the commented ``#`` rows into a per-attribute DataFrame."""
    if not rows or not cell_line_names:
        return None
    metadata: dict[str, list[str]] = {}
    for row in rows:
        if len(row) < 3:
            continue
        kind = row[2].strip() if len(row) > 2 else ""
        if not kind:
            continue
        values = row[3:] if len(row) > 3 else []
        aligned = [values[i] if i < len(values) else "" for i in range(len(cell_line_names))]
        metadata[kind] = aligned
    return pd.DataFrame(metadata, index=cell_line_names) if metadata else None


# ─── Legacy Harmonizome class (back-compat) ─────────────────────────────────


class Entity:
    """Enumeration of entity types the Harmonizome REST API accepts."""

    DATASET = "dataset"
    GENE = "gene"
    GENE_SET = "gene_set"
    ATTRIBUTE = "attribute"
    GENE_FAMILY = "gene_family"
    NAMING_AUTHORITY = "naming_authority"
    PROTEIN = "protein"
    RESOURCE = "resource"


def _get_with_cursor(entity: str, start_at: int) -> Any:
    return json_from_url(f"{API_URL}/{VERSION}/{entity}?cursor={start_at}")


def _get_by_name(entity: str, name: str) -> Any:
    return json_from_url(f"{API_URL}/{VERSION}/{entity}/{name}")


def _get_entity(response: dict[str, Any]) -> str:
    return response["next"].split("?")[0].split("/")[3]


def _get_next(response: dict[str, Any]) -> int | None:
    nxt = response.get("next")
    return int(nxt.split("=")[1]) if nxt else None


class Harmonizome:
    """Back-compat wrapper around the Harmonizome REST API.

    Preserved from the AoU module for old call sites that look like
    ``Harmonizome.get(Entity.DATASET, name="GTEx Tissue Gene Expression Profiles")``.
    New code should prefer the module-level :func:`list_datasets` /
    :func:`get_dataset_metadata` / :func:`download_datasets`.
    """

    __version__ = VERSION

    @classmethod
    def get(
        cls,
        entity: str,
        name: str | None = None,
        start_at: int | None = None,
    ) -> Any:
        if name:
            return _get_by_name(entity, quote_plus(name))
        if start_at is not None and isinstance(start_at, int):
            return _get_with_cursor(entity, start_at)
        return json_from_url(f"{API_URL}/{VERSION}/{entity}")

    @classmethod
    def next(cls, response: dict[str, Any]) -> Any:
        return cls.get(entity=_get_entity(response), start_at=_get_next(response))

    @classmethod
    def download(
        cls,
        datasets: list[str] | None = None,
        what: list[str] | None = None,
    ) -> Any:
        """Yield filenames as datasets are downloaded.

        Refuses to download every dataset (~30GB) without an explicit list.
        """
        if datasets is None:
            raise ValueError(
                "Pass an explicit list of datasets — refusing to download "
                "the full ~30 GB Harmonizome catalog by default."
            )
        dataset_to_path = _load_config().get("datasets", {})
        downloads = what if what is not None else list(_load_config().get("downloads", []))
        for dataset in datasets:
            if dataset not in dataset_to_path:
                raise AttributeError(
                    f"{dataset!r} is not a valid dataset name. "
                    "Inspect DATASET_TO_PATH for the full list."
                )
            for dl in downloads:
                url = f"{DOWNLOAD_URL}/{dataset_to_path[dataset]}/{dl}"
                try:
                    response = urlopen(url)
                except HTTPError as exc:
                    if what is not None:
                        raise RuntimeError(f"Error downloading from {url}: {exc}") from exc
                    continue
                filename = Path(dataset) / dl.replace(".gz", "")
                filename.parent.mkdir(parents=True, exist_ok=True)
                if filename.exists():
                    logger.info("Using cached %s", filename)
                else:
                    logger.info("Downloading %s", filename)
                    _download_url_and_decompress(response, filename)
                yield str(filename)


def _download_url_and_decompress(response: Any, filename: Path) -> None:
    """Urlopen → gzip-decompressed file (used by :meth:`Harmonizome.download`)."""
    compressed = BytesIO(response.read())
    decompressed = gzip.GzipFile(fileobj=compressed)
    with open(filename, "wb") as out:
        out.write(decompressed.read())
