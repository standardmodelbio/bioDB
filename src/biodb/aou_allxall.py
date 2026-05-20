"""All of Us *All-by-All* PheWAS client.

[All-by-All](https://allbyall.researchallofus.org/) (released April 2024,
updated to CDR v8 February 2025) is the *All of Us* program's
publicly-browsable PheWAS atlas across **~3,600 phenotypes** × **~414k
short-read-WGS participants**. Phenotypes span six categories: physical
measurements, lab measurements, phecodes, phecodeX, personal/family
health history (PFHH) surveys, and EHR-sourced drug exposures.

The public browser at ``allbyall.researchallofus.org`` is a thin React
frontend over a Rust + ClickHouse backend (Broad source:
https://github.com/broadinstitute/all-by-all-aou-browser). The browser
itself has no bulk-export button, but the backing HTTP API is **open
and unauthenticated** for the same data the browser displays
(genome-wide-significant variants and the full gene-burden test
matrix). Raw participant-level data and the full non-significant
variant table remain Controlled-Tier-only at
``gs://fc-aou-datasets-controlled/AllxAll/v1/`` — see issue
``bschilder/GenForge#6`` for the Researcher-Workbench route.

This module wraps the public API:

* :func:`list_analyses` — the 3,602 (analysis × META) phenotype rows
  with sample sizes, trait types, and category tags.
* :func:`list_assets` — the underlying GCS URIs of the per-(phenotype,
  ancestry, asset-type) Hail Tables (private bucket; URIs are
  informational unless you have Workbench access).
* :func:`list_categories`, :func:`get_config` — small-payload metadata
  helpers.
* :func:`get_gene_burden` — per-phenotype gene-burden result table,
  ~22 MB JSON × ~6k rows / phenotype across the three burden masks
  (pLoF, missenseLC, synonymous).
* :func:`download_all_gene_burden` — concurrent bulk pull across all
  phenotypes, writing one consolidated Parquet (~500 MB after compression).
* :func:`load_gene_burden`, :func:`melt_gene_burden` — read the
  consolidated Parquet and reshape to a long
  ``(sourceId=analysis_id, targetId=gene_id, score)`` frame suitable
  for :func:`biodb.transform.create_gene_association_matrix`.

All artifacts are cached under ``~/.cache/biodb/aou_allxall/<version>/``.

Examples
--------
>>> from biodb.aou_allxall import list_analyses, get_gene_burden
>>> analyses = list_analyses()                                   # doctest: +SKIP
>>> analyses.shape                                                # doctest: +SKIP
(3602, 15)
>>> burden = get_gene_burden(analyses.iloc[0]["analysis_id"])    # doctest: +SKIP
>>> burden[["gene_symbol", "annotation", "neg_log10_p_burden"]].head()  # doctest: +SKIP
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import polars as pl
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

BASE_URL = "https://allbyall.researchallofus.org"
"""Root of the public All-by-All browser. Source: https://github.com/broadinstitute/all-by-all-aou-browser."""

API_URL = f"{BASE_URL}/api"
"""Root of the unauthenticated JSON API the browser frontend consumes."""

DEFAULT_VERSION = "v1"
"""All-by-All release. ``v1`` = CDR v8 (~414k participants, Feb 2025)."""

CACHE_DIR = Path("~/.cache/biodb/aou_allxall").expanduser() / DEFAULT_VERSION
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ANCESTRY_CODES: tuple[str, ...] = ("afr", "amr", "eas", "eur", "mid", "sas", "meta")
"""Computed-genetic-ancestry groups + meta-analysis. ``meta`` is the default consumer-facing track."""

BURDEN_SETS: tuple[str, ...] = ("pLoF", "missenseLC", "pLoF;missenseLC", "synonymous")
"""Annotation labels returned by the API.

The browser UI surfaces three buttons (``pLoF`` / ``Missense`` / ``Syn``), but
the underlying data also includes a combined ``pLoF;missenseLC`` joint mask.
``synonymous`` is the negative-control mask.
"""

MAF_THRESHOLDS: tuple[float, ...] = (0.01, 0.001, 0.0001)
"""Three minor-allele-frequency buckets the gene-burden tests are computed at: 1%, 0.1%, 0.01%."""

BURDEN_TESTS: tuple[str, ...] = ("burden", "skat", "skato")
"""The three burden test types.

Each row of a ``/api/phenotype/.../genes`` response carries all three p-values
in parallel columns: ``pvalue_burden`` (Burden), ``pvalue_skat`` (SKAT), and
``pvalue`` (SKAT-O — the combined / consensus statistic).
"""

_TEST_TO_NEG_LOG10_P_COL: dict[str, str] = {
    "burden": "neg_log10_p_burden",
    "skat": "neg_log10_p_skat",
    "skato": "neg_log10_p",  # SKAT-O is reported under the bare "pvalue"/"neg_log10_p" columns
}
"""Per-test mapping from the user-facing test label to the response column carrying the −log10 p-value."""

_TEST_TO_PVALUE_COL: dict[str, str] = {
    "burden": "pvalue_burden",
    "skat": "pvalue_skat",
    "skato": "pvalue",
}
"""Per-test mapping from the user-facing test label to the response column carrying the raw p-value."""

_USER_AGENT = "biodb/0.1 (+https://github.com/bschilder/bioDB)"
_DEFAULT_TIMEOUT = 60
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})

_API_ROW_LIMIT = 50_000
"""Hard row cap baked into the upstream Rust server's gene-burden endpoint.

See ``broadinstitute/all-by-all-aou-browser`` ``axaou-server/src/api.rs``:
``let limit = params.limit.unwrap_or(50000) as u64;``. A response that
returns *exactly* this many rows is almost certainly truncated — high-N
phenotypes (lab measurements, common phecodes) routinely hit it. We
warn so silent truncation doesn't slip through to downstream pipelines.
"""

EXPECTED_GENE_BURDEN_COLUMNS: frozenset[str] = frozenset(
    {
        "gene_id",
        "gene_symbol",
        "annotation",
        "max_maf",
        "analysis_id",
        "ancestry_group",
        "pvalue",
        "neg_log10_p",
        "pvalue_burden",
        "neg_log10_p_burden",
        "pvalue_skat",
        "neg_log10_p_skat",
        "beta_burden",
        "mac",
        "contig",
        "gene_start_position",
    }
)
"""Columns this module assumes the gene-burden endpoint returns.

Used by :func:`get_gene_burden` for upstream-schema-drift detection
and by the live integration tests as the canonical column set —
keep in sync with the fixture in ``tests/test_aou_allxall.py``.
"""


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
    """GET ``base_url + path`` with retry/backoff; return parsed JSON.

    Retries on transient HTTP statuses (429, 5xx) with exponential
    backoff. Honors the server's ``Retry-After`` header when present.
    """
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
                logger.debug("HTTP %s on %s — retrying in %.1fs", response.status_code, url, wait)
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
    """Resolve ``CACHE_DIR / name``, creating parent dirs."""
    p = CACHE_DIR / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ─── Small-payload metadata endpoints ───────────────────────────────────────


def get_config(force: bool = False) -> dict:
    """Return the browser config payload (ancestry codes, burden sets, reference genome, …).

    Cached as ``config.json`` under :data:`CACHE_DIR`.
    """
    dst = _cache_path("config.json")
    if dst.exists() and not force:
        return json.loads(dst.read_text())
    payload = _request_json("/config")
    dst.write_text(json.dumps(payload, indent=2))
    return payload


def list_categories(force: bool = False) -> pd.DataFrame:
    """Return the phenotype-category table — ``(category, color, analyses)``.

    The ``analyses`` column is a list of analysis IDs in that category.
    """
    dst = _cache_path("categories.parquet")
    if dst.exists() and not force:
        return pd.read_parquet(dst)
    payload = _request_json("/categories")
    df = pd.DataFrame(payload)
    df.to_parquet(dst, index=False)
    return df


def list_analyses(
    ancestry: str = "meta",
    *,
    force: bool = False,
) -> pd.DataFrame:
    """Return the per-phenotype analysis metadata for one ancestry group.

    Parameters
    ----------
    ancestry : str, default ``"meta"``
        One of :data:`ANCESTRY_CODES`. ``"meta"`` is the cross-ancestry
        meta-analysis and is the default consumer track.
    force : bool, default False
        Re-download even if cached.

    Returns
    -------
    pandas.DataFrame
        ~3,600 rows × 15 columns including ``analysis_id``,
        ``ancestry_group``, ``category``, ``description``, ``trait_type``
        (``binary``/``continuous``), ``n_cases``, ``n_controls``, and
        per-test ``lambda_gc_*`` genomic-control values.
    """
    if ancestry not in ANCESTRY_CODES:
        raise ValueError(f"ancestry {ancestry!r} not in {ANCESTRY_CODES}")
    dst = _cache_path(f"analyses_{ancestry}.parquet")
    if dst.exists() and not force:
        return pd.read_parquet(dst)
    payload = _request_json("/analyses", params={"ancestry_group": ancestry})
    df = pd.DataFrame(payload)
    df.to_parquet(dst, index=False)
    logger.info("Cached %d analyses (ancestry=%s) → %s", len(df), ancestry, dst)
    return df


def list_assets(
    ancestry: str | None = None,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """Return the asset table — GCS URIs of the per-phenotype Hail Tables.

    The bucket ``gs://aou_results/`` is **not publicly readable** — these
    URIs are informational unless you have *All of Us* Researcher
    Workbench Controlled-Tier access. The columns are stable enough to
    use as a cross-reference for the asset universe.

    Parameters
    ----------
    ancestry : str, optional
        If given, filter to one ancestry. Otherwise the API returns
        all 7 ancestries (~75k rows).
    force : bool, default False
        Re-download even if cached.
    """
    key = "all" if ancestry is None else ancestry
    if ancestry is not None and ancestry not in ANCESTRY_CODES:
        raise ValueError(f"ancestry {ancestry!r} not in {ANCESTRY_CODES}")
    dst = _cache_path(f"assets_{key}.parquet")
    if dst.exists() and not force:
        return pd.read_parquet(dst)
    params = {"ancestry_group": ancestry} if ancestry else None
    payload = _request_json("/assets", params=params)
    df = pd.DataFrame(payload)
    df.to_parquet(dst, index=False)
    return df


def get_assets_summary() -> dict:
    """Return the GCS-asset-count summary — ``total_assets``, ``total_phenotypes``, by_ancestry, …."""
    return _request_json("/assets/summary")  # type: ignore[return-value]


# ─── Per-phenotype gene-burden endpoint ─────────────────────────────────────


def get_gene_burden(
    analysis_id: str | int,
    ancestry: str | None = None,
    max_maf: float = 0.001,
    *,
    force: bool = False,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch the gene-burden test table for one phenotype at one MAF threshold.

    Each row is a ``(gene_id, annotation)`` result at the requested MAF,
    carrying **all three** test p-values in parallel columns —
    ``pvalue_burden`` (Burden), ``pvalue_skat`` (SKAT), and ``pvalue``
    (SKAT-O) — plus the burden effect size.

    The API returns rows for every burden mask (``pLoF``, ``missenseLC``,
    ``pLoF;missenseLC``, ``synonymous``) that passed QC at the requested
    MAF. To capture the full variant grid (burden_set × max_maf × test)
    described by the browser UI, call this once per ``max_maf`` in
    :data:`MAF_THRESHOLDS` — :func:`download_all_gene_burden` does that
    iteration for you.

    Parameters
    ----------
    analysis_id : str or int
        Phenotype ID — get one from :func:`list_analyses`.
    ancestry : str, optional
        Server-side ancestry filter. ``None`` returns whatever ancestries
        the backend loaded for the phenotype (typically all 7).
    max_maf : float, default ``0.001``
        Minor-allele-frequency bucket. Must be one of
        :data:`MAF_THRESHOLDS` (``0.01``, ``0.001``, ``0.0001``). ``0.001``
        is the All-by-All default.
    force : bool, default False
        Re-download even if cached.
    session : requests.Session, optional
        Reuse a session for bulk pulls (see
        :func:`download_all_gene_burden`).

    Returns
    -------
    pandas.DataFrame
        Empty if the API has no gene-burden results loaded for that
        (phenotype, ancestry, max_maf).
    """
    aid = str(analysis_id)
    if max_maf not in MAF_THRESHOLDS:
        raise ValueError(f"max_maf={max_maf} not in {MAF_THRESHOLDS}")
    parts = [aid]
    if ancestry is not None:
        parts.append(ancestry)
    # Embed the MAF in the filename so per-MAF pulls don't collide in the cache.
    parts.append(f"maf{max_maf}")
    fname = "_".join(parts) + ".parquet"
    dst = _cache_path(f"gene_burden/{fname}")
    if dst.exists() and not force:
        return pd.read_parquet(dst)
    params: dict[str, str | float] = {"max_maf": max_maf}
    if ancestry is not None:
        params["ancestry_group"] = ancestry
    payload = _request_json(f"/phenotype/{aid}/genes", params=params, session=session)
    df = pd.DataFrame(payload)
    _warn_if_row_limit_hit(df, aid, ancestry=ancestry, max_maf=max_maf)
    df.to_parquet(dst, index=False)
    return df


def _warn_if_row_limit_hit(
    df: pd.DataFrame,
    analysis_id: str,
    *,
    ancestry: str | None,
    max_maf: float,
) -> None:
    """Emit a ``RuntimeWarning`` when a fetch returned exactly :data:`_API_ROW_LIMIT` rows.

    The upstream Rust server caps each ``/phenotype/.../genes`` response
    at 50,000 rows. Phenotypes with more significant gene-burden tests
    are silently truncated. Loud failure here is much cheaper than a
    ranking pipeline that misses ~10% of a high-N phenotype's signal.
    """
    if len(df) == _API_ROW_LIMIT:
        import warnings

        msg = (
            f"AoU All-by-All gene-burden response for analysis_id={analysis_id} "
            f"(ancestry={ancestry!r}, max_maf={max_maf}) returned exactly "
            f"{_API_ROW_LIMIT:,} rows — this matches the upstream server cap "
            f"(`broadinstitute/all-by-all-aou-browser`, "
            f"axaou-server/src/api.rs#list_gene_associations), so the "
            f"response is almost certainly truncated. Downstream gene-vector "
            f"signatures for this phenotype will miss the lowest-ranked "
            f"surviving tests. There is currently no public API path to "
            f"raise the cap; either accept the truncation or use the "
            f"Researcher-Workbench Hail-Table route."
        )
        warnings.warn(msg, RuntimeWarning, stacklevel=3)
        logger.warning(msg)


def download_all_gene_burden(
    ancestry: str = "meta",
    max_mafs: tuple[float, ...] | None = None,
    *,
    analyses: pd.DataFrame | None = None,
    max_workers: int = 8,
    force: bool = False,
    progress: bool = True,
    consolidate: bool = True,
) -> Path:
    """Bulk-pull every (phenotype × max_maf) gene-burden table, concurrently.

    Iterates over the supplied ``max_mafs`` (default: all three —
    :data:`MAF_THRESHOLDS`). Each per-(phenotype, max_maf) JSON is fetched
    once and cached as its own Parquet under
    ``gene_burden/<analysis_id>_<ancestry>_maf<max_maf>.parquet``. If
    ``consolidate=True`` (the default), a single concatenated
    ``gene_burden_all_<ancestry>.parquet`` is also written.

    Total work: ``len(analyses) × len(max_mafs)`` HTTP calls
    (~3,602 × 3 ≈ **10.8k** for the full META track). Per-call payload
    is ~22 MB, so expect ~240 GB of raw JSON, condensed to a ~1–2 GB
    consolidated Parquet after column-projection.

    Parameters
    ----------
    ancestry : str, default ``"meta"``
        Ancestry track to pull.
    max_mafs : tuple of float, optional
        Which MAF buckets to fetch. Defaults to all of
        :data:`MAF_THRESHOLDS` (``(0.01, 0.001, 0.0001)``). Pass a
        smaller tuple (e.g. ``(0.001,)``) to cut wall time by ⅓ or ⅔.
    analyses : pandas.DataFrame, optional
        Pre-fetched :func:`list_analyses` output. Pass an explicit
        (possibly filtered) frame to limit the pull.
    max_workers : int, default 8
        Concurrent HTTP fetches.
    force : bool, default False
        Re-download every per-phenotype shard even if cached.
    progress : bool, default True
        Show a tqdm bar.
    consolidate : bool, default True
        After fetching, write a single concatenated Parquet.

    Returns
    -------
    pathlib.Path
        Path to the consolidated Parquet (or the per-phenotype directory
        if ``consolidate=False``).
    """
    if ancestry not in ANCESTRY_CODES:
        raise ValueError(f"ancestry {ancestry!r} not in {ANCESTRY_CODES}")
    if max_mafs is None:
        max_mafs = MAF_THRESHOLDS
    for m in max_mafs:
        if m not in MAF_THRESHOLDS:
            raise ValueError(f"max_maf={m} not in {MAF_THRESHOLDS}")

    if analyses is None:
        analyses = list_analyses(ancestry=ancestry)
    analysis_ids = analyses["analysis_id"].astype(str).tolist()
    jobs = [(aid, maf) for aid in analysis_ids for maf in max_mafs]
    logger.info(
        "Pulling gene-burden tables: %d phenotypes × %d MAFs = %d HTTP calls "
        "(ancestry=%s, workers=%d)",
        len(analysis_ids),
        len(max_mafs),
        len(jobs),
        ancestry,
        max_workers,
    )

    session = requests.Session()
    session.headers["User-Agent"] = _USER_AGENT

    def _one(args: tuple[str, float]) -> tuple[str, float, int]:
        aid, maf = args
        try:
            df = get_gene_burden(
                aid,
                ancestry=ancestry,
                max_maf=maf,
                force=force,
                session=session,
            )
            return aid, maf, len(df)
        except Exception as exc:  # noqa: BLE001  — record and continue, summarized at end
            logger.warning("gene-burden fetch failed for (%s, maf=%s): %s", aid, maf, exc)
            return aid, maf, -1

    failures: list[tuple[str, float]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_one, job) for job in jobs]
        bar = tqdm(total=len(futures), disable=not progress, desc=f"AoU AllxAll {ancestry}")
        for fut in as_completed(futures):
            aid, maf, n = fut.result()
            if n < 0:
                failures.append((aid, maf))
            bar.update(1)
        bar.close()

    if failures:
        logger.warning("Failed (phenotype, max_maf) pairs (%d): %s", len(failures), failures[:20])

    burden_dir = CACHE_DIR / "gene_burden"
    if not consolidate:
        return burden_dir

    # Consolidate every shard for this ancestry — note shards are named
    # ``<aid>_<ancestry>_maf<maf>.parquet``, so a glob on ``*_<ancestry>_maf*.parquet`` finds them.
    pattern = f"*_{ancestry}_maf*.parquet"
    pq_files = sorted(burden_dir.glob(pattern))
    consolidated = CACHE_DIR / f"gene_burden_all_{ancestry}.parquet"
    logger.info("Consolidating %d per-(phenotype, MAF) shards → %s", len(pq_files), consolidated)
    lf = pl.scan_parquet([str(p) for p in pq_files])
    lf.sink_parquet(str(consolidated))
    return consolidated


def load_gene_burden(
    ancestry: str = "meta",
    *,
    min_neg_log10_p: float | None = None,
    burden_set: str | None = None,
    max_maf: float | None = None,
    test: str | None = None,
    columns: list[str] | None = None,
) -> pl.DataFrame:
    """Read the consolidated gene-burden Parquet as a Polars DataFrame.

    Call :func:`download_all_gene_burden` first to materialize the
    consolidated artifact.

    Parameters
    ----------
    ancestry : str, default ``"meta"``
    min_neg_log10_p : float, optional
        Keep only rows with the relevant ``neg_log10_p`` column ≥ this
        threshold. The exact column depends on ``test`` (defaults to the
        Burden test). ``7.3`` ≈ Bonferroni for 20k genes × 1 test.
    burden_set : str, optional
        Filter to one ``annotation`` value (e.g. ``"pLoF"``,
        ``"missenseLC"``). See :data:`BURDEN_SETS`.
    max_maf : float, optional
        Filter to one MAF bucket. See :data:`MAF_THRESHOLDS`.
    test : str, optional
        One of :data:`BURDEN_TESTS` (``"burden"`` / ``"skat"`` /
        ``"skato"``). Drives which ``neg_log10_p_*`` column is used by
        ``min_neg_log10_p``. The cached rows always carry all three
        p-values; this argument only affects the significance filter,
        not the row schema.
    columns : list of str, optional
        Project a subset of columns to save memory.

    Returns
    -------
    polars.DataFrame
    """
    path = CACHE_DIR / f"gene_burden_all_{ancestry}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Consolidated parquet not found at {path}. "
            "Run download_all_gene_burden(ancestry=...) first."
        )
    lf = pl.scan_parquet(str(path))
    if columns:
        lf = lf.select(columns)
    if burden_set is not None:
        if burden_set not in BURDEN_SETS:
            raise ValueError(f"burden_set={burden_set!r} not in {BURDEN_SETS}")
        lf = lf.filter(pl.col("annotation") == burden_set)
    if max_maf is not None:
        if max_maf not in MAF_THRESHOLDS:
            raise ValueError(f"max_maf={max_maf} not in {MAF_THRESHOLDS}")
        lf = lf.filter(pl.col("max_maf") == max_maf)
    if min_neg_log10_p is not None:
        t = test or "burden"
        if t not in _TEST_TO_NEG_LOG10_P_COL:
            raise ValueError(f"test={t!r} not in {tuple(_TEST_TO_NEG_LOG10_P_COL)}")
        lf = lf.filter(pl.col(_TEST_TO_NEG_LOG10_P_COL[t]) >= min_neg_log10_p)
    return lf.collect()


def melt_gene_burden(
    burden: pl.DataFrame | pd.DataFrame,
    *,
    test: str = "burden",
    beta_signed: bool = True,
    burden_set: str | None = None,
    max_maf: float | None = None,
) -> pd.DataFrame:
    """Reshape gene-burden results into a long ``(sourceId, targetId, score)`` frame.

    The All-by-All gene-burden grid has four faceting axes per row:
    ``(analysis_id, annotation, max_maf, test)``. This function picks
    one ``test`` (collapses the three p-value columns to one) and
    optionally filters to specific ``burden_set`` / ``max_maf`` values,
    then emits one ``(sourceId, targetId, score)`` row per gene per
    surviving facet.

    To enumerate **all 3 × 3 × 4 = 36** variants per phenotype, call
    this once per ``(test, burden_set, max_maf)`` triple — that's
    typically what :mod:`genforge.data.ingest.aou` does. Output schema
    mirrors what
    :func:`biodb.transform.create_gene_association_matrix` expects.

    Parameters
    ----------
    burden : DataFrame
        Output of :func:`load_gene_burden` (Polars) or
        :func:`get_gene_burden` (pandas).
    test : str, default ``"burden"``
        One of :data:`BURDEN_TESTS`. Selects which ``neg_log10_p_*``
        column to use as score magnitude:

        * ``"burden"`` → ``neg_log10_p_burden``
        * ``"skat"``   → ``neg_log10_p_skat``
        * ``"skato"``  → ``neg_log10_p`` (the SKAT-O combined statistic)
    beta_signed : bool, default ``True``
        If True, multiply the score by ``sign(beta_burden)`` to produce
        a **signed** score (negative = protective, positive = risk).
        Set to False for an unsigned ``-log10 p`` magnitude. Note SKAT
        and SKAT-O do not carry their own directional effect — the only
        directional information in a gene-burden row is ``beta_burden``,
        so for ``test`` ∈ {``skat``, ``skato``} the sign is borrowed
        from the burden estimator.
    burden_set : str or None
        Filter to one ``annotation`` value. ``None`` keeps every burden
        mask. See :data:`BURDEN_SETS`.
    max_maf : float or None
        Filter to one MAF bucket. ``None`` keeps every MAF. See
        :data:`MAF_THRESHOLDS`.

    Returns
    -------
    pandas.DataFrame
        Long frame with columns ``sourceId`` (=``analysis_id``),
        ``targetId`` (=``gene_id``), ``score``, plus ``annotation``,
        ``max_maf``, ``test``, ``gene_symbol`` for traceability /
        downstream faceting.
    """
    if isinstance(burden, pl.DataFrame):
        burden = burden.to_pandas()
    df = burden
    if test not in _TEST_TO_NEG_LOG10_P_COL:
        raise ValueError(f"test={test!r} not in {tuple(_TEST_TO_NEG_LOG10_P_COL)}")
    p_col = _TEST_TO_NEG_LOG10_P_COL[test]
    if p_col not in df.columns:
        raise KeyError(f"expected column {p_col!r} not in burden frame")

    if burden_set is not None:
        if burden_set not in BURDEN_SETS:
            raise ValueError(f"burden_set={burden_set!r} not in {BURDEN_SETS}")
        df = df[df["annotation"] == burden_set]
    if max_maf is not None:
        if max_maf not in MAF_THRESHOLDS:
            raise ValueError(f"max_maf={max_maf} not in {MAF_THRESHOLDS}")
        df = df[df["max_maf"] == max_maf]

    score = df[p_col].astype(float)
    if beta_signed and "beta_burden" in df.columns:
        signs = (
            df["beta_burden"].fillna(0).map(lambda b: 1.0 if b > 0 else (-1.0 if b < 0 else 0.0))
        )
        score = score * signs

    out = pd.DataFrame(
        {
            "sourceId": df["analysis_id"].astype(str),
            "targetId": df["gene_id"].astype(str),
            "score": score.values,
            "annotation": df["annotation"].values if "annotation" in df.columns else None,
            "max_maf": df["max_maf"].values if "max_maf" in df.columns else None,
            "test": test,
            "gene_symbol": df["gene_symbol"].values if "gene_symbol" in df.columns else None,
        }
    )
    return out.dropna(subset=["score"]).reset_index(drop=True)


def iter_signature_variants(
    burden: pl.DataFrame | pd.DataFrame,
    *,
    tests: tuple[str, ...] = BURDEN_TESTS,
    burden_sets: tuple[str, ...] | None = None,
    max_mafs: tuple[float, ...] | None = None,
    beta_signed: bool = True,
):
    """Yield one ``(facet_key, long_df)`` pair per signature variant.

    Enumerates the full ``test × burden_set × max_maf`` grid, calling
    :func:`melt_gene_burden` once per cell. Designed to feed
    :class:`genforge.data.schema.SignatureCollection`-style ingesters
    that want one ``Signature`` per phenotype × variant.

    Parameters
    ----------
    burden : DataFrame
        Output of :func:`load_gene_burden` (Polars) or
        :func:`get_gene_burden` (pandas). May span multiple phenotypes,
        annotations, and MAFs.
    tests : tuple of str, default ``BURDEN_TESTS``
        Tests to enumerate.
    burden_sets : tuple of str, optional
        Burden sets to enumerate. Defaults to ``BURDEN_SETS`` (all 4
        annotations including the joint ``pLoF;missenseLC`` mask).
    max_mafs : tuple of float, optional
        MAF buckets to enumerate. Defaults to ``MAF_THRESHOLDS``.
    beta_signed : bool, default True
        Whether the ``score`` should be sign-flipped by ``beta_burden``.

    Yields
    ------
    facet : dict
        ``{"test": ..., "burden_set": ..., "max_maf": ...}``.
    long_df : pandas.DataFrame
        The corresponding melted frame (possibly empty if no rows
        survive the facet filters). Each row carries
        ``(sourceId=analysis_id, targetId=gene_id, score, ...)``.
    """
    if burden_sets is None:
        burden_sets = BURDEN_SETS
    if max_mafs is None:
        max_mafs = MAF_THRESHOLDS
    for test in tests:
        for bs in burden_sets:
            for maf in max_mafs:
                long = melt_gene_burden(
                    burden,
                    test=test,
                    burden_set=bs,
                    max_maf=maf,
                    beta_signed=beta_signed,
                )
                yield {"test": test, "burden_set": bs, "max_maf": maf}, long


# ─── Per-trait targeted-query API ───────────────────────────────────────────


_ANALYSES_CACHE: dict[str, pd.DataFrame] = {}


def _get_analyses_cached(ancestry: str = "meta") -> pd.DataFrame:
    """Lazy-load & memoize the analyses table for one ancestry."""
    if ancestry not in _ANALYSES_CACHE or _ANALYSES_CACHE[ancestry].empty:
        _ANALYSES_CACHE[ancestry] = list_analyses(ancestry=ancestry)
    return _ANALYSES_CACHE[ancestry]


def query_phenotype(
    phenotype: str | int,
    *,
    column: str | None = None,
    ancestry: str = "meta",
) -> pd.DataFrame:
    """Look up phenotype metadata by analysis_id or substring of description.

    Parameters
    ----------
    phenotype : str or int
        * an ``int`` (or all-digit ``str``) is matched against ``analysis_id``.
        * any other string is treated as a case-insensitive substring
          filter on ``description``.
    column : str, optional
        Force the lookup column.
    ancestry : str, default ``"meta"``
    """
    analyses = _get_analyses_cached(ancestry=ancestry)
    if column is not None:
        if column not in analyses.columns:
            raise KeyError(
                f"Column {column!r} not in analyses; available: {list(analyses.columns)[:8]}…"
            )
        col_data = analyses[column].astype(str)
        return analyses[col_data.str.contains(str(phenotype), na=False, case=False)]

    str_val = str(phenotype).strip()
    if str_val.isdigit() and "analysis_id" in analyses.columns:
        hits = analyses[analyses["analysis_id"].astype(str) == str_val]
        if not hits.empty:
            return hits
    if "description" in analyses.columns:
        return analyses[
            analyses["description"].astype(str).str.contains(str_val, na=False, case=False)
        ]
    return analyses.iloc[0:0]


def list_phenotypes(*, ancestry: str = "meta") -> pd.DataFrame:
    """Return the full phenotype metadata table (cached after first call).

    Alias of :func:`list_analyses` for symmetry with
    :func:`biodb.gwas_atlas.list_traits`.
    """
    return _get_analyses_cached(ancestry=ancestry)


__all__ = [
    "ANCESTRY_CODES",
    "API_URL",
    "BASE_URL",
    "BURDEN_SETS",
    "BURDEN_TESTS",
    "CACHE_DIR",
    "DEFAULT_VERSION",
    "MAF_THRESHOLDS",
    "download_all_gene_burden",
    "get_assets_summary",
    "get_config",
    "get_gene_burden",
    "iter_signature_variants",
    "list_analyses",
    "list_assets",
    "list_categories",
    "list_phenotypes",
    "load_gene_burden",
    "melt_gene_burden",
    "query_phenotype",
]
