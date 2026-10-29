"""Durable, locked per-file acquisition ledger shared by raw GWAS providers.

The run freezes membership and semantic options. Each success receipt is an
atomic commit record for hashed immutable artifacts, including successful empty
files. Consolidation reads receipts only, never directory globs. Workers can die
at any boundary: uncommitted staging files are ignored and retry replaces only
that file. A run lock prevents two processes from publishing competing results.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

import polars as pl
from filelock import FileLock

from biodb._downloads import stream_to_file
from biodb._sumstats import SCHEMA, SCHEMA_VERSION, iter_file, normalize_regions


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def read_json(path):
    try:
        value = json.loads(Path(path).read_text())
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def valid_artifacts(root, receipt, config_hash, row):
    if (
        receipt.get("status") != "ok"
        or receipt.get("config_hash") != config_hash
        or receipt.get("schema") != SCHEMA_VERSION
        or any(receipt.get(key) != row[key] for key in ("file_id", "study_id", "url"))
        or not (receipt.get("source_sha256") or receipt.get("region_sha256"))
    ):
        return False
    try:
        total = 0
        for item in receipt["artifacts"]:
            path = (root / item["path"]).resolve()
            if not path.is_relative_to(root / "artifacts") or digest(path) != item["sha256"]:
                return False
            if pl.read_parquet_schema(path) != SCHEMA:
                return False
            count = pl.scan_parquet(path).select(pl.len()).collect().item()
            if count != item["rows"]:
                return False
            total += count
        return total == receipt["rows"]
    except (OSError, ValueError, KeyError, pl.exceptions.PolarsError):
        return False


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("w") as handle:
        handle.write(canonical(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def records(datasets):
    if isinstance(datasets, pl.DataFrame):
        datasets = datasets.to_dicts()
    elif isinstance(datasets, dict):
        datasets = [datasets]
    return [dict(row) for row in datasets]


def file_record(url, study_id, source, **metadata):
    file_id = hashlib.sha256(f"{source}\n{url}".encode()).hexdigest()
    return dict(source=source, study_id=study_id, file_id=file_id, url=url, **metadata)


def local_path(dataset, local_root=None):
    url = dataset["url"]
    parsed = urlparse(url)
    if local_root is not None:
        root = Path(local_root).expanduser().resolve()
        filename = Path(unquote(parsed.path)).name
        stem = filename.removesuffix(".tsv.gz")
        candidates = [root / stem / filename, root / filename]
        # Catalog manifests retain an explicit relative path; try it without
        # permitting traversal out of the caller's source directory.
        relative = dataset.get("manifest_path", "")
        if relative:
            candidate = (root / relative.lstrip("./")).resolve()
            if candidate.is_relative_to(root):
                candidates.insert(0, candidate)
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"{filename} not found under {root}")
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    if not parsed.scheme:
        return Path(url).expanduser()
    return None


def validate_file(path, *, expect_gzip=False):
    """Validate complete gzip members, including CRC/trailer/EOF; never truncate."""
    import gzip
    import zlib

    with Path(path).open("rb") as handle:
        compressed = handle.read(2) == b"\x1f\x8b"
    if (expect_gzip or str(path).endswith(".gz")) and not compressed:
        raise ValueError("Expected gzip content")
    if compressed:
        try:
            with gzip.open(path, "rb") as handle:
                for _ in iter(lambda: handle.read(1024 * 1024), b""):
                    pass
        except (EOFError, gzip.BadGzipFile, zlib.error) as exc:
            raise ValueError(f"Invalid/incomplete gzip source: {exc}") from exc


def bind_input_identity(identity_path, identity):
    if identity_path is None:
        return
    identity_path = Path(identity_path)
    old = read_json(identity_path)
    if identity_path.exists() and old is None:
        raise ValueError("Corrupt frozen source identity; use a new output_dir")
    if old is not None:
        for key in ("etag", "total", "sha256", "index_etag", "index_total", "index_url"):
            if old.get(key) is not None and key in identity and old[key] != identity[key]:
                raise ValueError("Frozen source input identity changed; use a new output_dir")
        identity = {**old, **identity}
    atomic_json(identity_path, identity)


def download(
    dataset,
    *,
    source,
    cache_dir=None,
    local_root=None,
    force=False,
    resume=True,
    progress=True,
    identity_path=None,
):
    dataset = dict(dataset)
    identity_path = Path(identity_path) if identity_path is not None else None
    transport_identity = {}

    def bind_identity(identity):
        transport_identity.update(identity)
        bind_input_identity(identity_path, identity)

    local = local_path(dataset, local_root)
    if local is not None:
        if not local.is_file():
            raise FileNotFoundError(local)
        validate_file(local)
        bind_identity({"sha256": digest(local), "total": local.stat().st_size})
        return local
    root = Path(cache_dir or f"~/.cache/biodb/{source}").expanduser()
    filename = Path(unquote(urlparse(dataset["url"]).path)).name
    path = root / dataset["file_id"] / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".provider.lock"):
        receipt_path = path.with_name(path.name + ".verified.json")
        if path.is_file() and receipt_path.exists() and not force:
            receipt = read_json(receipt_path) or {}
            if receipt.get("sha256") == digest(path) and receipt.get("url") == dataset["url"]:
                bind_identity(
                    {
                        "sha256": receipt["sha256"],
                        "total": receipt["size"],
                        "etag": receipt.get("etag"),
                    }
                )
                return path
        # Existing unverified/corrupt files must never masquerade as cache hits.
        stream_to_file(
            dataset["url"],
            path,
            resume=resume,
            validate=lambda part: validate_file(
                part, expect_gzip=urlparse(dataset["url"]).path.endswith(".gz")
            ),
            progress=progress,
            on_identity=bind_identity,
        )
        bind_identity({"sha256": digest(path), "total": path.stat().st_size})
        atomic_json(
            receipt_path,
            {
                **transport_identity,
                "url": dataset["url"],
                "sha256": digest(path),
                "size": path.stat().st_size,
            },
        )
    return path


def acquire_input(
    dataset,
    *,
    source,
    scope="genomewide",
    regions=None,
    backend="auto",
    cache_dir=None,
    local_root=None,
    identity_path=None,
):
    if backend not in {"auto", "tabix", "full"}:
        raise ValueError("backend must be auto, tabix or full")
    metadata = {
        "transport": "local" if local_path(dataset, local_root) is not None else "full",
        "source_sha256_scope": "full_file",
    }
    if (
        source == "gwas_catalog"
        and scope == "regions"
        and backend != "full"
        and local_path(dataset, local_root) is None
    ):
        from biodb._gwas_indexed import acquire

        path, metadata = acquire(
            dataset,
            regions=regions,
            cache_dir=cache_dir or "~/.cache/biodb/gwas_catalog",
            backend=backend,
            bind_identity=lambda identity: bind_input_identity(identity_path, identity),
        )
        if path is not None:
            return path, metadata
    path = download(
        dataset,
        source=source,
        cache_dir=cache_dir,
        local_root=local_root,
        progress=False,
        identity_path=identity_path,
    )
    return path, metadata


def iterate(dataset_or_path, *, source, cache_dir=None, local_root=None, backend="auto", **kwargs):
    if isinstance(dataset_or_path, (str, Path)):
        path = Path(dataset_or_path)
        dataset = file_record(path.resolve().as_uri(), path.name.removesuffix(".tsv.gz"), source)
        metadata = {"transport": "local"}
    else:
        dataset = dict(dataset_or_path)
        path, metadata = acquire_input(
            dataset,
            source=source,
            cache_dir=cache_dir,
            local_root=local_root,
            backend=backend,
            scope=kwargs.get("scope", "genomewide"),
            regions=kwargs.get("regions"),
        )
    yield from iter_file(
        path,
        source=source,
        study_id=dataset["study_id"],
        file_id=dataset["file_id"],
        url=dataset["url"],
        sha256=digest(path),
        source_metadata=metadata,
        **kwargs,
    )


@dataclass
class IngestResult:
    output_dir: Path
    files: list[dict]
    artifacts: list[Path]
    rows: int
    succeeded: int
    failed: int
    status: str

    def scan(self):
        """Lazy concatenation of committed successful raw output only."""
        return (
            pl.scan_parquet(self.artifacts)
            if self.artifacts
            else pl.DataFrame(schema=SCHEMA).lazy()
        )

    def consolidate(self, destination):
        """Atomically materialize only the committed file artifacts."""
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".tmp")
        self.scan().sink_parquet(temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        fsync_directory(destination.parent)
        return destination


class IngestError(RuntimeError):
    """Strict batch failure, retaining all durable successes in ``result``."""

    def __init__(self, result):
        self.result = result
        super().__init__(
            f"{result.failed}/{len(result.files)} source files failed; inspect error.result.files"
        )


def ingest(
    datasets,
    output_dir=None,
    *,
    source,
    out_dir=None,
    workers=4,
    resume=True,
    allow_partial=False,
    cache_dir=None,
    local_root=None,
    batch_size=100_000,
    batch_rows=None,
    backend="auto",
    required_study_ids=None,
    p_max=None,
    scope="genomewide",
    regions=None,
):
    """Run a frozen, resumable raw batch. Worker count and chunk size may change.

    A changed semantic filter, schema, membership, supplied metadata or local
    input bytes requires a new output directory. Source files remain cached;
    validated artifact checksums are mandatory before skipping a success.
    Failed files are retried. ``resume=False`` requires a fresh run directory.
    """
    if output_dir is None:
        output_dir = out_dir
    elif out_dir is not None and Path(output_dir) != Path(out_dir):
        raise ValueError("output_dir and out_dir disagree")
    if output_dir is None:
        raise ValueError("output_dir is required")
    if workers < 1:
        raise ValueError("workers must be positive")
    batch_size = batch_rows if batch_rows is not None else batch_size
    per_study_regions = isinstance(regions, dict)
    intervals = (
        {study: normalize_regions(scope, values) for study, values in sorted(regions.items())}
        if per_study_regions
        else normalize_regions(scope, regions)
    )
    if p_max is not None and not 0 < p_max <= 1:
        raise ValueError("p_max must satisfy 0 < p_max <= 1")
    inputs = records(datasets)
    available = {row["study_id"] for row in inputs}
    for study in sorted(set(required_study_ids or []) - available):
        inputs.append(
            {
                "source": source,
                "study_id": study,
                "file_id": "manifest-missing:" + study,
                "url": "",
                "_selection_missing": True,
            }
        )
    inputs.sort(key=lambda row: row["file_id"])
    if len({row["file_id"] for row in inputs}) != len(inputs):
        raise ValueError("Duplicate source file identity")
    for row in inputs:
        if per_study_regions and row["study_id"] not in regions:
            raise ValueError(f"Missing regions for study {row['study_id']}")
        if row.get("source", source) != source:
            raise ValueError("Mixed sources in run")
        # Local identity is frozen before any worker runs. A missing file has a
        # durable URL identity and can be repaired without changing membership.
        try:
            local = local_path(row, local_root)
            row["_local_sha256"] = digest(local) if local is not None and local.is_file() else None
        except FileNotFoundError:
            row["_local_sha256"] = None
    config = {
        "source": source,
        "schema": SCHEMA_VERSION,
        "scope": scope,
        "backend": backend,
        "regions": intervals,
        "p_max": p_max,
        "inputs": inputs,
    }
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / ".run.lock")):
        manifest = root / "run.json"
        if manifest.exists():
            if not resume:
                raise ValueError("Existing run; resume=False requires a new output_dir")
            previous = json.loads(manifest.read_text())
            # A previously unavailable local input may become available on retry;
            # once a source digest is known it may never silently change.
            previous_inputs = {row["file_id"]: row for row in previous["inputs"]}
            for row in inputs:
                old = previous_inputs.get(row["file_id"])
                if old and old.get("_local_sha256") is None:
                    row["_local_sha256"] = None
            if canonical(previous) != canonical(config):
                raise ValueError(
                    "Run configuration, membership or input identity changed; use a new output_dir"
                )
        else:
            atomic_json(manifest, config)
        config_hash = hashlib.sha256(canonical(config).encode()).hexdigest()
        for directory in ("receipts", "artifacts", "staging", "inputs"):
            (root / directory).mkdir(exist_ok=True)

        def receipt_path(row):
            return (
                root / "receipts" / (hashlib.sha256(row["file_id"].encode()).hexdigest() + ".json")
            )

        def process(row):
            receipt_file = receipt_path(row)
            previous = read_json(receipt_file)
            if previous is not None and valid_artifacts(root, previous, config_hash, row):
                try:
                    local = local_path(row, local_root)
                except FileNotFoundError:
                    local = None
                if (
                    local is not None
                    and local.is_file()
                    and digest(local) != previous["source_sha256"]
                ):
                    raise ValueError("Source input identity changed; use a new output_dir")
                return previous
            staging = root / "staging" / uuid.uuid4().hex
            staging.mkdir()
            file_regions = regions[row["study_id"]] if per_study_regions else regions
            file_intervals = intervals[row["study_id"]] if per_study_regions else intervals
            status = {
                "file_id": row["file_id"],
                "study_id": row["study_id"],
                "url": row["url"],
                "status": "failed",
                "rows": 0,
                "artifacts": [],
                "source_sha256": (previous or {}).get("source_sha256"),
                "config_hash": config_hash,
                "schema": SCHEMA_VERSION,
                "scope": scope,
                "regions": file_intervals,
                "p_max": p_max,
                "metadata": {key: value for key, value in row.items() if not key.startswith("_")},
            }
            try:
                if row.get("_selection_missing"):
                    status["failure_reason"] = "manifest_missing"
                    raise ValueError(
                        f"Requested study {row['study_id']} has no file in the source manifest"
                    )
                path, acquisition = acquire_input(
                    row,
                    scope=scope,
                    regions=file_regions,
                    backend=backend,
                    source=source,
                    cache_dir=cache_dir,
                    local_root=local_root,
                    identity_path=root
                    / "inputs"
                    / (hashlib.sha256(row["file_id"].encode()).hexdigest() + ".json"),
                )
                sha = digest(path)
                expected = (
                    row.get("_local_sha256")
                    or (previous or {}).get("source_sha256")
                    or (previous or {}).get("region_sha256")
                )
                if expected and sha != expected:
                    raise ValueError("Source input identity changed; use a new output_dir")
                indexed = acquisition.get("transport") == "tabix"
                status.update(acquisition)
                status["source_sha256"] = None if indexed else sha
                status["region_sha256"] = sha if indexed else None
                batches = []
                count = 0
                for index, frame in enumerate(
                    iter_file(
                        path,
                        source=source,
                        study_id=row["study_id"],
                        file_id=row["file_id"],
                        url=row["url"],
                        sha256=sha,
                        source_metadata=acquisition,
                        batch_size=batch_size,
                        p_max=p_max,
                        scope=scope,
                        regions=file_regions,
                    )
                ):
                    artifact = staging / f"{index:08d}.parquet"
                    frame.write_parquet(artifact)
                    with artifact.open("rb") as handle:
                        os.fsync(handle.fileno())
                    batches.append(
                        {"name": artifact.name, "sha256": digest(artifact), "rows": frame.height}
                    )
                    count += frame.height
                # Publish an immutable generation, then atomically commit its receipt.
                # Old/uncommitted generations are deliberately ignored by readers.
                target = root / "artifacts" / uuid.uuid4().hex
                os.replace(staging, target)
                fsync_directory(root / "artifacts")
                status.update(
                    status="ok",
                    rows=count,
                    artifacts=[
                        {
                            "path": str((target / item["name"]).relative_to(root)),
                            "sha256": item["sha256"],
                            "rows": item["rows"],
                        }
                        for item in batches
                    ],
                )
            except Exception as exc:
                status.update(status="failed", detail=f"{type(exc).__name__}: {exc}")
            finally:
                shutil.rmtree(staging, ignore_errors=True)
            atomic_json(receipt_file, status)
            return status

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process, row) for row in inputs]
            completed = [future.result() for future in as_completed(futures)]
        completed.sort(key=lambda row: row["file_id"])
        successes = [item for item in completed if item["status"] == "ok"]
        failed = len(completed) - len(successes)
        result = IngestResult(
            root,
            completed,
            [root / artifact["path"] for item in successes for artifact in item["artifacts"]],
            sum(item["rows"] for item in successes),
            len(successes),
            failed,
            "partial" if failed else "complete",
        )
        atomic_json(
            root / "result.json",
            {
                **asdict(result),
                "output_dir": str(root),
                "artifacts": [str(path) for path in result.artifacts],
            },
        )
        if failed and not allow_partial:
            raise IngestError(result)
        return result


def load_ingest_result(output_dir, *, allow_partial=False):
    """Read and verify the frozen membership's receipts without acquiring data.

    Missing/uncommitted/corrupt files are explicit failures. This is the only
    supported way to consolidate a prior run; stray parquet files are ignored.
    """
    root = Path(output_dir).expanduser().resolve()
    with FileLock(str(root / ".run.lock")):
        manifest = json.loads((root / "run.json").read_text())
        completed = []
        artifacts = []
        config_hash = hashlib.sha256(canonical(manifest).encode()).hexdigest()
        for row in manifest["inputs"]:
            path = (
                root / "receipts" / (hashlib.sha256(row["file_id"].encode()).hexdigest() + ".json")
            )
            receipt = read_json(path) or {
                "file_id": row["file_id"],
                "study_id": row["study_id"],
                "url": row["url"],
                "status": "failed",
                "detail": "Missing/corrupt completion receipt",
                "rows": 0,
                "artifacts": [],
            }
            if receipt.get("status") == "ok":
                if valid_artifacts(root, receipt, config_hash, row):
                    artifacts.extend(root / item["path"] for item in receipt["artifacts"])
                else:
                    receipt.update(
                        status="failed",
                        detail="Unverified artifact/receipt identity",
                        rows=0,
                        artifacts=[],
                    )
            completed.append(receipt)
        successes = [item for item in completed if item["status"] == "ok"]
        failed = len(completed) - len(successes)
        result = IngestResult(
            root,
            completed,
            artifacts,
            sum(item["rows"] for item in successes),
            len(successes),
            failed,
            "partial" if failed else "complete",
        )
        if failed and not allow_partial:
            raise IngestError(result)
        return result
