"""SNOMED CT — per-concept lookups (via OLS4) + bulk CONCEPT.csv download.

SNOMED CT is the canonical clinical-terminology vocabulary — diagnoses,
procedures, findings, body sites, etc. ``biodb.snomed`` covers both
modes:

* **API mode** — per-concept lookups via EBI's `OLS4
  <https://www.ebi.ac.uk/ols4/>`_ (~376 k SNOMED terms indexed).
  See :func:`query_concept`, :func:`search_concepts`,
  :func:`get_descendants`, :func:`get_ancestors`,
  :func:`get_children`, :func:`get_parents`. Each accepts an ``int``,
  a CURIE (``"SNOMED:38341003"``), or a full IRI.

* **Bulk mode** — the OHDSI-flavoured ``CONCEPT.csv`` (concept dimension
  of the OMOP CDM) as a GitHub Release asset, so downstream packages
  get a fast, versioned, no-credentials download path:
  :func:`download_concept_csv`, :func:`load_concept_csv`,
  :func:`get_concept_csv_path`, :func:`is_available`.

The bulk asset lives at:

    https://github.com/bschilder/bioDB/releases/download/vocab-v1/CONCEPT.csv.gz

(Relocated from ``bschilder/synthlab`` on 2026-05-18 — same bytes,
same SHA-256.)

Authentication (bulk only)
--------------------------
For private mirrors of the release, the downloader tries three
strategies in order, mirroring the original synthlab flow:

1. ``gh`` CLI (if installed and authenticated — best for private repos)
2. ``GITHUB_TOKEN`` / ``GH_TOKEN`` environment variable
3. Public URL (default; works for this repo since it's public)

Examples
--------
>>> from biodb import snomed
>>> snomed.query_concept(38341003)["label"]    # doctest: +SKIP
'Hypertensive disorder'
>>> snomed.search_concepts("diabetes", rows=5) # doctest: +SKIP
>>> snomed.get_descendants(73211009)           # doctest: +SKIP
>>> # Bulk:
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


# ─── Per-concept lookups via OLS ───────────────────────────────────────────
# SNOMED CT is indexed on EBI's OLS4 (~376 k terms). For one-concept-at-a-time
# queries we go through OLS rather than the bulk CSV — it's a single HTTP call
# vs. parsing a 175 MB file. These wrappers add SNOMED-shaped argument handling
# (accept ``"38341003"``, ``"SNOMED:38341003"``, or the full
# ``http://snomed.info/id/38341003`` IRI) on top of the generic OLS client.

OLS_ONTOLOGY_SLUG = "snomed"
"""The OLS slug for SNOMED CT."""


def _normalize_concept_id(concept_id: str | int) -> str:
    """Turn ``int``, bare digit string, ``SNOMED:xxx``, or full IRI → CURIE."""
    if isinstance(concept_id, int):
        return f"SNOMED:{concept_id}"
    text = str(concept_id).strip()
    if text.startswith(("http://", "https://")):
        return text
    if ":" in text:
        return text
    return f"SNOMED:{text}"


def query_concept(concept_id: str | int, *, timeout: int = 30) -> dict:
    """Look up one SNOMED concept via OLS4.

    Accepts a bare concept ID (``38341003``), a SNOMED CURIE
    (``"SNOMED:38341003"``), or the full SNOMED IRI
    (``"http://snomed.info/id/38341003"``).

    Returns the OLS term record — ``label``, ``description``, ``synonyms``,
    ``obo_id``, ``iri``, ``is_obsolete``, ``has_children``, ``is_root``.
    """
    from biodb import ols

    return ols.get_term(OLS_ONTOLOGY_SLUG, _normalize_concept_id(concept_id), timeout=timeout)


def get_descendants(
    concept_id: str | int,
    *,
    size: int = 500,
    timeout: int = 30,
) -> pd.DataFrame:
    """Return every transitive descendant of ``concept_id`` (via OLS)."""
    from biodb import ols

    return ols.get_descendants(
        OLS_ONTOLOGY_SLUG, _normalize_concept_id(concept_id), size=size, timeout=timeout
    )


def get_ancestors(
    concept_id: str | int,
    *,
    size: int = 500,
    timeout: int = 30,
) -> pd.DataFrame:
    """Return every transitive ancestor of ``concept_id`` (via OLS)."""
    from biodb import ols

    return ols.get_ancestors(
        OLS_ONTOLOGY_SLUG, _normalize_concept_id(concept_id), size=size, timeout=timeout
    )


def get_children(
    concept_id: str | int,
    *,
    size: int = 500,
    timeout: int = 30,
) -> pd.DataFrame:
    """Return the direct (one-hop) children of ``concept_id`` (via OLS)."""
    from biodb import ols

    return ols.get_children(
        OLS_ONTOLOGY_SLUG, _normalize_concept_id(concept_id), size=size, timeout=timeout
    )


def get_parents(
    concept_id: str | int,
    *,
    size: int = 500,
    timeout: int = 30,
) -> pd.DataFrame:
    """Return the direct (one-hop) parents of ``concept_id`` (via OLS)."""
    from biodb import ols

    return ols.get_parents(
        OLS_ONTOLOGY_SLUG, _normalize_concept_id(concept_id), size=size, timeout=timeout
    )


def search_concepts(
    query: str,
    *,
    rows: int = 20,
    exact: bool = False,
    timeout: int = 30,
) -> pd.DataFrame:
    """Full-text search SNOMED concepts via OLS (label / synonym / definition)."""
    from biodb import ols

    return ols.search(query, ontology=OLS_ONTOLOGY_SLUG, rows=rows, exact=exact, timeout=timeout)


__all__ = [
    "CACHE_DIR",
    "GITHUB_ASSET_NAME",
    "GITHUB_RELEASE_TAG",
    "GITHUB_REPO",
    "OLS_ONTOLOGY_SLUG",
    "SNOMED_RELEASE_URL",
    "download_concept_csv",
    "get_ancestors",
    "get_children",
    "get_concept_csv_path",
    "get_descendants",
    "get_parents",
    "get_snomed_data_dir",
    "is_available",
    "load_concept_csv",
    "query_concept",
    "search_concepts",
]
