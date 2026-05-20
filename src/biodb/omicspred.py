"""OmicsPred client — INTERVAL-trained molecular-trait polygenic scores.

[OmicsPred](https://www.omicspred.org/) ([Xu et al., *Nature* 2023,
PMID 36991119](https://pubmed.ncbi.nlm.nih.gov/36991119/)) publishes
Bayesian-Ridge polygenic prediction models for biomolecular traits —
proteins (Olink + SomaScan), metabolites (Metabolon HD4 + Nightingale
NMR), gene expression (Illumina RNA-seq), and RNA-splicing — trained
on the [INTERVAL](http://www.intervalstudy.org.uk/) cohort
(~50k UK blood donors). As of this writing the catalog has grown well
past the 17,227 traits in the original paper.

This module exposes both modes :mod:`biodb` ships throughout:

* **REST API** (``rest.omicspred.org``) for per-record lookups:
  :func:`list_platforms`, :func:`list_datasets`, :func:`get_dataset`,
  :func:`get_score`, :func:`search_scores`, :func:`get_performance`,
  :func:`get_publication`.
* **Bulk download** for whole-dataset analyses:
  :func:`download_metadata_excel`, :func:`load_scores_metadata`,
  :func:`load_performances_metadata`, :func:`download_scoring_files`,
  :func:`read_scoring_file`.

The actual SNP-weight scoring files are hosted on **Box.com** (one zip
archive per dataset). bioDB downloads them and caches under
``~/.cache/biodb/omicspred/<version>/<opd_id>/...``.

Per-score metadata already includes the **cis gene** for proteins
(UniProt + Ensembl) and transcripts (Ensembl) — so the downstream
"SNP → gene" aggregation step that's load-bearing for naive GWAS PRS
ingest is mostly *unnecessary* here. Use :func:`load_scores_metadata`
and pull ``Gene ID(s)`` directly for the cis signal; only fall back to
:func:`read_scoring_file` when you need the full SNP-level model.

Examples
--------
>>> from biodb.omicspred import list_platforms, get_score   # doctest: +SKIP
>>> platforms = list_platforms()                             # doctest: +SKIP
>>> score = get_score("OPGS000001")                          # doctest: +SKIP
>>> score["genes"][0]["external_id"]                         # doctest: +SKIP
'ENSG00000172322'
"""

from __future__ import annotations

import json
import logging
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://www.omicspred.org"
"""Root of the OmicsPred web portal."""

API_URL = "https://rest.omicspred.org/api"
"""Root of the public unauthenticated REST API (Swagger UI at ``rest.omicspred.org/``)."""

DEFAULT_VERSION = "v1"
"""OmicsPred snapshot tag. Bump after testing against a new catalog refresh."""

CACHE_DIR = Path("~/.cache/biodb/omicspred").expanduser() / DEFAULT_VERSION
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PLATFORMS: tuple[str, ...] = (
    "Metabolon",
    "Nightingale",
    "Olink",
    "Somalogic",
    "RNAseq - Expression",
    "RNAseq - Splicing",
)
"""Platforms exposed by the OmicsPred REST API ``/api/platform/all`` endpoint.

The 17,227 number reported by the [Nature 2023 paper](https://pubmed.ncbi.nlm.nih.gov/36991119/)
described an earlier snapshot. The live catalog includes far more
RNA-seq Expression (~982k) and Splicing (~2.3M) models — when
materializing for downstream tools, filter aggressively by platform
and validation R².
"""

_USER_AGENT = "biodb/0.1 (+https://github.com/bschilder/bioDB)"
_DEFAULT_TIMEOUT = 60
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


# ─── HTTP plumbing ──────────────────────────────────────────────────────────


def _request_json(
    path: str,
    params: dict | None = None,
    *,
    timeout: float = _DEFAULT_TIMEOUT,
    max_retries: int = 5,
    base_url: str = API_URL,
    session: requests.Session | None = None,
) -> list | dict:
    """GET ``base_url + path``; return parsed JSON. Retries on 429 / 5xx."""
    url = f"{base_url}{path}"
    sess = session or requests.Session()
    headers = {"User-Agent": _USER_AGENT, "Accept": "application/json"}
    delay = 1.0
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = sess.get(url, params=params, headers=headers, timeout=timeout)
            if response.status_code in _RETRY_STATUSES:
                retry_after = response.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else delay
                logger.debug("HTTP %s on %s — retry in %.1fs", response.status_code, url, wait)
                time.sleep(wait)
                delay = min(delay * 2, 30)
                continue
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
            logger.debug("Request error on %s (attempt %d): %s", url, attempt + 1, exc)
            time.sleep(delay)
            delay = min(delay * 2, 30)
    raise RuntimeError(f"Failed to GET {url} after {max_retries} attempts") from last_exc


def _cache_path(name: str) -> Path:
    p = CACHE_DIR / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _stream_box_download(url: str, dst: Path, *, timeout: int = 600) -> Path:
    """Download a Box.com ``shared/static/<hash>`` URL with redirect following.

    Box requires ``allow_redirects=True``; HEAD requests return 404, so always
    use GET. The actual file lands at ``public.boxcloud.com``.
    """
    headers = {"User-Agent": _USER_AGENT}
    with requests.get(
        url, headers=headers, stream=True, allow_redirects=True, timeout=timeout
    ) as r:
        r.raise_for_status()
        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open("wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 16):
                if chunk:
                    fh.write(chunk)
    return dst


# ─── Small-payload metadata endpoints ───────────────────────────────────────


def list_platforms(force: bool = False) -> pd.DataFrame:
    """Return the platform table: name, full_name, technic, type, scores_count."""
    dst = _cache_path("platforms.parquet")
    if dst.exists() and not force:
        return pd.read_parquet(dst)
    payload = _request_json("/platform/all")
    df = pd.DataFrame(payload["results"])
    df.to_parquet(dst, index=False)
    return df


def list_datasets(force: bool = False) -> pd.DataFrame:
    """Return the dataset registry — one row per (platform × version × cohort)."""
    dst = _cache_path("datasets.parquet")
    if dst.exists() and not force:
        return pd.read_parquet(dst)
    payload = _request_json("/dataset/all")
    rows = [_flatten_dataset_summary(d) for d in payload["results"]]
    df = pd.DataFrame(rows)
    df.to_parquet(dst, index=False)
    return df


def _flatten_dataset_summary(d: dict) -> dict:
    """Flatten the nested ``/dataset/all`` row into a tabular shape."""
    return {
        "dataset_id": d["id"],
        "name": d["name"],
        "platform_name": d.get("platform", {}).get("name"),
        "platform_full_name": d.get("platform", {}).get("full_name"),
        "platform_type": d.get("platform", {}).get("type"),
        "platform_version": d.get("platform", {}).get("version"),
        "scores_count": d.get("scores_count"),
        "phewas_count": d.get("phewas_count"),
        "omics_count": d.get("omics_count"),
        "omics_type": d.get("omics_type"),
        "method_name": d.get("method_name"),
        "tissue_id": (d.get("tissue") or {}).get("id"),
        "tissue_label": (d.get("tissue") or {}).get("label"),
        "license": d.get("license"),
    }


def get_dataset(opd_id: str, force: bool = False) -> dict:
    """Return the full nested dataset record, including ``scoring_files_urls``."""
    dst = _cache_path(f"datasets/{opd_id}.json")
    if dst.exists() and not force:
        return json.loads(dst.read_text())
    payload = _request_json(f"/dataset/{opd_id}")
    dst.write_text(json.dumps(payload, indent=2))
    return payload


def get_score(opgs_id: str, force: bool = False) -> dict:
    """Return the full per-score metadata — cis gene/protein/metabolite, variant_number, license."""
    dst = _cache_path(f"scores/{opgs_id}.json")
    if dst.exists() and not force:
        return json.loads(dst.read_text())
    payload = _request_json(f"/score/{opgs_id}")
    dst.write_text(json.dumps(payload, indent=2))
    return payload


def search_scores(
    *,
    platform: str | None = None,
    dataset_id: str | None = None,
    limit: int | None = None,
    page_size: int = 250,
) -> pd.DataFrame:
    """Page through ``/api/score/search`` returning all matching scores as a DataFrame.

    Notes
    -----
    The search endpoint requires at least one filter — passing none returns
    zero rows. Use :func:`load_scores_metadata` if you want every score in a
    dataset; that pulls the much-faster bulk Excel.

    Parameters
    ----------
    platform : str, optional
        One of :data:`PLATFORMS`.
    dataset_id : str, optional
        Restrict to one OPD.
    limit : int, optional
        Cap the total number of returned rows (handy for smoke tests).
    page_size : int, default 250
        Per-page page size on the upstream paginator.
    """
    if platform is None and dataset_id is None:
        raise ValueError("search_scores requires at least one filter (platform or dataset_id).")
    params: dict[str, str | int] = {"limit": page_size}
    if platform is not None:
        params["platform"] = platform
    if dataset_id is not None:
        params["dataset_id"] = dataset_id

    rows: list[dict] = []
    offset = 0
    while True:
        params["offset"] = offset
        payload = _request_json("/score/search", params=params)
        rows.extend(payload.get("results", []))
        if payload.get("next") is None:
            break
        offset += page_size
        if limit is not None and len(rows) >= limit:
            rows = rows[:limit]
            break
    return pd.DataFrame(rows)


def get_performance(opgs_id: str, force: bool = False) -> dict:
    """Return the per-score performance record."""
    dst = _cache_path(f"performance/{opgs_id}.json")
    if dst.exists() and not force:
        return json.loads(dst.read_text())
    payload = _request_json(f"/score/performance/{opgs_id}")
    dst.write_text(json.dumps(payload, indent=2))
    return payload


def get_publication(opp_id: str = "OPP000001", force: bool = False) -> dict:
    """Return one OmicsPred publication record (default = the Nature 2023 paper)."""
    dst = _cache_path(f"publications/{opp_id}.json")
    if dst.exists() and not force:
        return json.loads(dst.read_text())
    payload = _request_json(f"/publication/{opp_id}")
    dst.write_text(json.dumps(payload, indent=2))
    return payload


# ─── Bulk-metadata Excel downloads ──────────────────────────────────────────
#
# Each dataset's REST record carries a ``scoring_files_urls`` dict with seven
# Box.com URLs. The ``metadata`` key points to a 5-sheet Excel
# (Publication, Dataset, Scores, Performances, Cohorts) — the cleanest
# bulk metadata source. Per-dataset Excels are typically 1–2 MB.


def download_metadata_excel(opd_id: str, force: bool = False) -> Path:
    """Download the per-dataset metadata Excel and return its local path.

    The Excel is hosted on Box.com. Cached under
    ``~/.cache/biodb/omicspred/<version>/metadata/<opd_id>.xlsx``.
    """
    dst = _cache_path(f"metadata/{opd_id}.xlsx")
    if dst.exists() and not force:
        return dst
    record = get_dataset(opd_id, force=force)
    urls = record.get("scoring_files_urls") or {}
    url = urls.get("metadata")
    if not url:
        raise ValueError(
            f"Dataset {opd_id} has no 'metadata' download URL in scoring_files_urls — "
            f"available keys: {sorted(urls)}"
        )
    logger.info("Downloading %s metadata Excel from %s", opd_id, url)
    return _stream_box_download(url, dst)


def load_scores_metadata(
    opd_id: str,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """Read the ``Scores`` sheet of a dataset's metadata Excel.

    Each row carries the OPGS id, score name, reported trait, the development
    method, the number of variants, and — critically — the cis gene/protein/
    metabolite cross-references that OmicsPred has already mapped for us.
    """
    path = download_metadata_excel(opd_id, force=force)
    return _read_xlsx_sheet(path, "Scores")


def load_performances_metadata(
    opd_id: str,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """Read the ``Performances`` sheet of a dataset's metadata Excel.

    One row per (OPGS × cohort × ancestry × study_stage) — i.e. each score
    has multiple performance rows. Use a groupby on ``OmicsPred ID`` and
    pick the best (or most-relevant) R² for downstream filtering.
    """
    path = download_metadata_excel(opd_id, force=force)
    return _read_xlsx_sheet(path, "Performances")


def _read_xlsx_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    """Read one sheet of an OmicsPred metadata Excel via openpyxl."""
    try:
        import openpyxl  # noqa: F401  — required for pandas.read_excel(engine='openpyxl')
    except ImportError as exc:  # pragma: no cover - exercised via skip marker
        raise ImportError(
            "OmicsPred metadata parsing needs openpyxl. "
            "Install with `pip install 'biodb[omicspred]'` or `pip install openpyxl`."
        ) from exc
    return pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")


# ─── Bulk-scoring-file downloads + parsing ──────────────────────────────────
#
# Each dataset publishes its full set of per-score scoring files as a zip
# archive on Box.com. Three formats are offered:
#
# * ``scoring_files``      — original release (GRCh37)
# * ``scoring_files_hm_38`` — harmonized to GRCh38
# * ``scoring_files_pgsc_calc`` — PGS Catalog scoring-format (pgsc_calc-compatible)
#
# Prefer ``scoring_files_pgsc_calc`` for downstream pipelines — that format
# is standardized at https://www.pgscatalog.org/downloads/#scoring_columns.


SCORING_FORMATS: tuple[str, ...] = (
    "scoring_files_pgsc_calc",
    "scoring_files_hm_38",
    "scoring_files",
)
"""Available scoring-file archive formats, in order of preference for downstream use."""


def download_scoring_files(
    opd_id: str,
    *,
    format: str = "scoring_files_pgsc_calc",
    force: bool = False,
    extract: bool = True,
) -> Path:
    """Download the dataset's scoring-files zip archive (and optionally unzip).

    Returns
    -------
    pathlib.Path
        The local path. If ``extract=True`` (default), returns the directory
        containing the extracted per-score files; otherwise returns the zip
        itself.
    """
    if format not in SCORING_FORMATS:
        raise ValueError(f"format={format!r} not in {SCORING_FORMATS}")
    record = get_dataset(opd_id, force=force)
    urls = record.get("scoring_files_urls") or {}
    url = urls.get(format)
    if not url:
        raise ValueError(
            f"Dataset {opd_id} has no {format!r} URL — available formats: {sorted(urls)}"
        )

    zip_dst = _cache_path(f"scoring/{opd_id}_{format}.zip")
    extracted_dir = _cache_path(f"scoring/{opd_id}_{format}/")
    if extract and extracted_dir.exists() and any(extracted_dir.iterdir()) and not force:
        return extracted_dir
    if not zip_dst.exists() or force:
        logger.info("Downloading %s %s archive from %s", opd_id, format, url)
        _stream_box_download(url, zip_dst)

    if not extract:
        return zip_dst

    extracted_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_dst) as z:
        z.extractall(extracted_dir)
    return extracted_dir


def read_scoring_file(path: str | Path) -> pd.DataFrame:
    """Parse one PGS Catalog-format scoring file (``.txt`` / ``.txt.gz``).

    PGS Catalog scoring files are tab-separated with a ``##``-prefixed
    multi-line YAML header carrying metadata, followed by columns including
    ``rsID``, ``chr_name``, ``chr_position``, ``effect_allele``,
    ``other_allele``, and ``effect_weight``. See
    https://www.pgscatalog.org/downloads/#scoring_columns for the full spec.
    """
    p = Path(path)
    # Find header end by sniffing the first non-``##`` line.
    opener = _maybe_gzip_opener(p)
    header_rows = 0
    with opener(p, "rt") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            header_rows += 1
    return pd.read_csv(p, sep="\t", skiprows=header_rows, comment="#")


def _maybe_gzip_opener(path: Path):
    """Return ``gzip.open`` if path ends in .gz, else builtin ``open``."""
    if path.suffix == ".gz":
        import gzip

        return gzip.open
    return open


# ─── Cross-cutting helpers — produce a long-form (sourceId, targetId, score) ─


STUDY_STAGES: tuple[str, ...] = ("Training", "External Validation")
"""The two study-stage values the OmicsPred ``Performances`` sheet exposes.

Training-stage R² is inflated (the model was tuned on the same cohort);
External Validation R² is the right one for filtering "usable" signatures
when materializing for downstream tools.
"""


def melt_scores_to_gene_table(
    scores: pd.DataFrame,
    performances: pd.DataFrame,
    *,
    study_stage: str = "External Validation",
    cohort: str | None = None,
    score_column: str = "R2",
) -> pd.DataFrame:
    """Reshape OmicsPred metadata into a long ``(sourceId=OPGS, targetId=gene, score=R²)`` frame.

    For proteins/transcripts the cis gene is read from the ``Gene ID(s)``
    column of the ``Scores`` sheet (already mapped by OmicsPred — we don't
    need a SNP→gene aggregation step for the cis signal). Metabolites
    have no single cis gene; those rows are dropped here. Callers that
    want trans signal should run :func:`read_scoring_file` and aggregate
    explicitly.

    Parameters
    ----------
    scores : pd.DataFrame
        Output of :func:`load_scores_metadata`.
    performances : pd.DataFrame
        Output of :func:`load_performances_metadata`. Multiple rows per
        OPGS are reduced by selecting one ``study_stage`` and (optionally)
        one cohort.
    study_stage : str, default ``"External Validation"``
        One of :data:`STUDY_STAGES`. Defaults to the external-validation
        R² because the training-stage R² is inflated.
    cohort : str, optional
        Filter to one validation cohort (e.g. ``"FENLAND"``,
        ``"Jackson Heart Study"``). ``None`` keeps all and picks the
        max R² per OPGS — useful when you want a *best-case* signal.
    score_column : str, default ``"R2"``
        Which performance column to use as the gene-weight magnitude.
        ``"Rho"`` is the Spearman correlation; ``"R2"`` is the squared
        Pearson — both are populated.
    """
    if study_stage not in STUDY_STAGES:
        raise ValueError(f"study_stage={study_stage!r} not in {STUDY_STAGES}")
    if "OmicsPred ID" not in scores.columns:
        raise KeyError("scores frame must contain 'OmicsPred ID' (from the 'Scores' sheet).")
    if "OmicsPred ID" not in performances.columns:
        raise KeyError(
            "performances frame must contain 'OmicsPred ID' (from the 'Performances' sheet)."
        )
    if score_column not in performances.columns:
        raise KeyError(f"score_column={score_column!r} not in performances frame.")

    perf = performances[performances["Study stage"] == study_stage]
    if cohort is not None:
        perf = perf[perf["Cohort(s)"] == cohort]
    perf_best = (
        perf.dropna(subset=[score_column])
        .sort_values(score_column, ascending=False)
        .groupby("OmicsPred ID", as_index=False)
        .first()[["OmicsPred ID", score_column]]
    )
    gene_col = "Gene ID(s)"
    sc = scores[["OmicsPred ID", "Reported Trait", gene_col]].dropna(subset=[gene_col])
    # ``Gene ID(s)`` is sometimes a comma-separated list — explode into rows.
    sc = sc.assign(**{gene_col: sc[gene_col].astype(str).str.split(r"[,;]\s*")}).explode(gene_col)
    long = sc.merge(perf_best, on="OmicsPred ID", how="inner").rename(
        columns={
            "OmicsPred ID": "sourceId",
            gene_col: "targetId",
            score_column: "score",
            "Reported Trait": "trait",
        }
    )
    long["targetId"] = long["targetId"].str.strip()
    return long.dropna(subset=["targetId"]).reset_index(drop=True)


__all__ = [
    "API_URL",
    "BASE_URL",
    "CACHE_DIR",
    "DEFAULT_VERSION",
    "PLATFORMS",
    "SCORING_FORMATS",
    "STUDY_STAGES",
    "download_metadata_excel",
    "download_scoring_files",
    "get_dataset",
    "get_performance",
    "get_publication",
    "get_score",
    "list_datasets",
    "list_platforms",
    "load_performances_metadata",
    "load_scores_metadata",
    "melt_scores_to_gene_table",
    "read_scoring_file",
    "search_scores",
]
