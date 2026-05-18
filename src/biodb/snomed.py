"""SNOMED CT vocabulary downloader and loader.

SNOMED CT is the canonical clinical-terminology vocabulary — used to
encode diagnoses, procedures, findings, body sites, etc. in modern
EHR systems. bioDB ships the OHDSI-flavoured ``CONCEPT.csv`` (the
concept dimension of the OMOP Common Data Model's standardized
vocabulary) as a GitHub Release asset on this repo, so downstream
packages get a fast, versioned, no-credentials download path.

Module surface:

* :func:`download_concept_csv` — GET the gzipped CSV from the bioDB
  release, decompress, cache. Returns the local path.
* :func:`load_concept_csv` — :func:`download_concept_csv` + a
  :func:`pandas.read_csv` call with the right ``dtype`` overrides.
* :func:`is_available` — fast cache-existence probe.
* :func:`get_concept_csv_path` — return the cache path, downloading
  on first call. Compatibility shim for the original ``synthlab``
  surface.

The asset lives at:

    https://github.com/bschilder/bioDB/releases/download/vocab-v1/CONCEPT.csv.gz

(Relocated from ``bschilder/synthlab`` on 2026-05-18 — same bytes,
same SHA-256.)

Authentication
--------------
For private mirrors of this release, the downloader tries three
strategies in order, mirroring the original synthlab flow:

1. ``gh`` CLI (if installed and authenticated — best for private repos)
2. ``GITHUB_TOKEN`` / ``GH_TOKEN`` environment variable
3. Public URL (default; works for this repo since it's public)

Examples
--------
>>> from biodb import snomed
>>> path = snomed.download_concept_csv()       # doctest: +SKIP
>>> df = snomed.load_concept_csv()             # doctest: +SKIP
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import pandas as pd
import requests

from biodb._downloads import stream_to_file

logger = logging.getLogger(__name__)

GITHUB_REPO = "bschilder/bioDB"
"""GitHub repo that hosts the release asset."""

GITHUB_RELEASE_TAG = "vocab-v1"
"""Release tag for the SNOMED vocabulary."""

GITHUB_ASSET_NAME = "CONCEPT.csv.gz"

SNOMED_RELEASE_URL = (
    f"https://github.com/{GITHUB_REPO}/releases/download/{GITHUB_RELEASE_TAG}/{GITHUB_ASSET_NAME}"
)
"""Public direct-download URL for the asset. Works for this (public) repo."""

CACHE_DIR = Path("~/.cache/biodb/snomed").expanduser()
"""On-disk cache for the decompressed ``CONCEPT.csv``."""


# ─── Auth helpers (lifted from synthlab; same behaviour) ───────────────────


def _get_github_token() -> str | None:
    """Return ``GITHUB_TOKEN`` or ``GH_TOKEN`` from the environment, or None."""
    return os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")


def _find_gh_cli() -> str | None:
    """Find a ``gh`` CLI binary on PATH or alongside the active Python."""
    candidates = ["gh", str(Path(sys.executable).parent / "gh")]
    for candidate in candidates:
        try:
            result = subprocess.run(
                [candidate, "--version"], capture_output=True, timeout=5, check=False
            )
            if result.returncode == 0:
                return candidate
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            continue
    return None


def _gh_cli_available() -> bool:
    """Check whether ``gh`` is on PATH **and** an authenticated session exists."""
    gh_path = _find_gh_cli()
    if not gh_path:
        return False
    try:
        result = subprocess.run(
            [gh_path, "auth", "status"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def _download_with_gh_cli(output_path: Path) -> bool:
    """Use ``gh release download`` to fetch the asset. Returns success."""
    gh_path = _find_gh_cli()
    if not gh_path:
        return False
    try:
        logger.info("SNOMED: using gh CLI for authenticated download")
        result = subprocess.run(
            [
                gh_path,
                "release",
                "download",
                GITHUB_RELEASE_TAG,
                "--repo",
                GITHUB_REPO,
                "--pattern",
                GITHUB_ASSET_NAME,
                "--dir",
                str(output_path.parent),
                "--clobber",
            ],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        if result.returncode == 0:
            return True
        logger.warning("SNOMED: gh CLI failed: %s", result.stderr.strip())
        return False
    except (subprocess.SubprocessError, FileNotFoundError) as exc:
        logger.warning("SNOMED: gh CLI error: %s", exc)
        return False


def _download_with_token(token: str, output_path: Path, *, progress: bool) -> bool:
    """Resolve the asset URL via the GitHub API, then GET with token auth."""
    api_url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/tags/{GITHUB_RELEASE_TAG}"
    try:
        logger.info("SNOMED: using token authentication")
        request = urllib.request.Request(api_url)
        request.add_header("Authorization", f"token {token}")
        request.add_header("Accept", "application/vnd.github.v3+json")
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            release_data = json.loads(response.read().decode())
        asset_url = next(
            (a["url"] for a in release_data.get("assets", []) if a["name"] == GITHUB_ASSET_NAME),
            None,
        )
        if not asset_url:
            logger.warning("SNOMED: asset %s not in release payload", GITHUB_ASSET_NAME)
            return False
        stream_to_file(
            asset_url,
            output_path,
            timeout=600,
            progress=progress,
            headers={
                "Authorization": f"token {token}",
                "Accept": "application/octet-stream",
            },
        )
        return True
    except (requests.HTTPError, OSError, ValueError) as exc:
        logger.warning("SNOMED: token auth failed: %s", exc)
        return False


def _download_public(output_path: Path, *, progress: bool) -> bool:
    """Plain public download — works for the bioDB repo since it's public."""
    try:
        logger.info("SNOMED: attempting public download from %s", SNOMED_RELEASE_URL)
        stream_to_file(
            SNOMED_RELEASE_URL,
            output_path,
            timeout=600,
            progress=progress,
        )
        return True
    except requests.HTTPError as exc:
        logger.warning("SNOMED: public download failed: %s", exc)
        return False


# ─── Public API ────────────────────────────────────────────────────────────


def get_snomed_data_dir() -> Path:
    """Return the SNOMED cache directory, creating it on first call."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def is_available() -> bool:
    """Cheap check: is ``CONCEPT.csv`` already cached on disk?"""
    return (CACHE_DIR / "CONCEPT.csv").exists()


def get_concept_csv_path(*, progress: bool = True) -> Path:
    """Return the local ``CONCEPT.csv`` path, downloading on first call."""
    data_dir = get_snomed_data_dir()
    concept_path = data_dir / "CONCEPT.csv"
    if not concept_path.exists():
        download_concept_csv(progress=progress)
    return concept_path


def download_concept_csv(
    output_dir: Path | str | None = None,
    *,
    force: bool = False,
    progress: bool = True,
) -> Path:
    """Download + decompress the bioDB SNOMED ``CONCEPT.csv`` asset.

    Parameters
    ----------
    output_dir
        Cache root. Defaults to :data:`CACHE_DIR`.
    force
        Re-download even if cached.
    progress
        Show a tqdm progress bar during the network transfer.

    Returns
    -------
    pathlib.Path
        Absolute path to the decompressed ``CONCEPT.csv``.

    Raises
    ------
    RuntimeError
        If all three download strategies fail.
    """
    if output_dir is None:
        output_dir = get_snomed_data_dir()
    else:
        output_dir = Path(output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)

    concept_path = output_dir / "CONCEPT.csv"
    compressed_path = output_dir / GITHUB_ASSET_NAME

    if concept_path.exists() and not force:
        logger.info("SNOMED CONCEPT.csv already cached at %s", concept_path)
        return concept_path

    logger.info("Downloading SNOMED vocabulary from %s", GITHUB_REPO)

    try:
        download_success = False

        if _gh_cli_available():
            download_success = _download_with_gh_cli(compressed_path)

        if not download_success:
            token = _get_github_token()
            if token:
                download_success = _download_with_token(token, compressed_path, progress=progress)

        if not download_success:
            download_success = _download_public(compressed_path, progress=progress)

        if not download_success:
            raise RuntimeError(
                "Failed to download SNOMED vocabulary.\n\n"
                "For private mirrors of the release, ensure one of:\n"
                "  1. gh CLI is installed and authenticated: `gh auth login`\n"
                "  2. GITHUB_TOKEN env var is set\n"
                "  3. GH_TOKEN env var is set\n\n"
                f"Source: {GITHUB_REPO} (release {GITHUB_RELEASE_TAG})"
            )

        size_mb = compressed_path.stat().st_size / (1024 * 1024)
        logger.info("SNOMED: downloaded %.1f MB", size_mb)

        with gzip.open(compressed_path, "rb") as f_in, open(concept_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

        compressed_path.unlink()

        size_mb = concept_path.stat().st_size / (1024 * 1024)
        logger.info("SNOMED CONCEPT.csv saved (%.1f MB) at %s", size_mb, concept_path)
        return concept_path

    except Exception:
        if compressed_path.exists():
            compressed_path.unlink()
        if concept_path.exists():
            concept_path.unlink()
        raise


def load_concept_csv(
    *,
    progress: bool = True,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Download + parse ``CONCEPT.csv`` into a DataFrame.

    Forwards ``**read_csv_kwargs`` to :func:`pandas.read_csv`. The
    OHDSI ``concept_id`` column is integer; the date columns are
    parsed automatically.
    """
    path = get_concept_csv_path(progress=progress)
    defaults: dict = {
        "sep": "\t",
        "dtype": {"concept_id": "Int64", "concept_code": "string"},
    }
    defaults.update(read_csv_kwargs)
    return pd.read_csv(path, **defaults)


__all__ = [
    "CACHE_DIR",
    "GITHUB_ASSET_NAME",
    "GITHUB_RELEASE_TAG",
    "GITHUB_REPO",
    "SNOMED_RELEASE_URL",
    "download_concept_csv",
    "get_concept_csv_path",
    "get_snomed_data_dir",
    "is_available",
    "load_concept_csv",
]
