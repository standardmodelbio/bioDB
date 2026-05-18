"""ClinVar VCF ingestion + simplification helpers.

Adapted from ``bschilder/VEP_protein/src/clinvar.py``, with two changes:

* ``genoray`` and ``pooch`` are lazy-imported inside the helpers that
  need them, so importing :mod:`biodb.clinvar` itself has no extra deps.
* The single dependency on ``src.utils.add_variant_name`` is inlined as
  the private helper :func:`_add_variant_name` below.

Public surface:

* :func:`download_vcf` — fetch the latest ClinVar VCF + tabix index.
* :func:`vcf_to_df` — parse a ClinVar VCF into a Polars DataFrame with
  selected INFO columns, MC split, and a numeric ``CLNREVSTAT_score``.
* :func:`simplify_annotations` — collapse the long-tail ``CLNSIG`` /
  ``CLIN_SIG`` strings into ``benign`` / ``likely_benign`` / ``path`` /
  ``likely_path`` / ``conflicting`` / ``other``.
* :func:`df_to_bed`, :func:`df_to_sites`, :func:`bed_to_sites`,
  :func:`read_bed` — format converters.
* :func:`filter_df`, :func:`count_sites_per_gene` — convenience filters.
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import polars as pl


INFO_COLS_SELECT = [
    "AF_ESP", "AF_EXAC", "AF_TGP", "ALLELEID", "CLNDISDB", "CLNDN",
    "CLNHGVS", "CLNREVSTAT", "CLNSIG", "CLNVC", "CLNVCSO", "CLNSIGCONF",
    "GENEINFO", "MC", "ORIGIN", "RS",
    "ONC", "ONCDN", "ONCDISDB", "ONCREVSTAT", "ONCCONF",
    "SCI", "SCIDN", "SCIDISDB", "SCIREVSTAT",
]


def _pooch_cache(name: str) -> str:
    """Resolve a path inside pooch's default OS cache (lazy import)."""
    import pooch

    return os.path.join(pooch.os_cache("pooch"), name)


def download_vcf(
    vcf_url: str = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz",
    vcf_hash: str | None = None,
    idx_hash: str | None = None,
) -> dict[str, str]:
    """Download the latest ClinVar VCF + index via ``pooch``.

    Hashes are accepted as kwargs so callers can pin a known release;
    pass ``None`` to skip integrity checking entirely (the upstream
    file rotates frequently).
    """
    import pooch

    vcf_file = pooch.retrieve(
        vcf_url,
        fname=os.path.basename(vcf_url),
        known_hash=vcf_hash,
        progressbar=True,
    )
    idx_file = pooch.retrieve(
        vcf_url + ".tbi",
        fname=os.path.basename(vcf_url) + ".tbi",
        known_hash=idx_hash,
        progressbar=True,
    )
    return {"vcf": vcf_file, "idx": idx_file}


def vcf_to_df(
    vcf_file: str | None = None,
    attrs: list[str] | None = None,
    filter: Any = None,
    contig: str | None = None,
    info: list[str] | None = None,
    extract_ids: bool = True,
    progress: bool = True,
    cache: str | None = None,
    force: bool = False,
) -> pl.DataFrame:
    """Parse a ClinVar VCF into a Polars DataFrame.

    Reads attribute columns + selected INFO fields, splits ``MC`` into
    ``MC_id`` / ``MC_term``, and maps ``CLNREVSTAT`` to a numeric
    ``CLNREVSTAT_score``. Caches the resulting parquet at ``cache`` for
    fast reload.
    """
    from genoray import VCF

    if attrs is None:
        attrs = ["CHROM", "POS", "ID", "REF", "ALT"]
    if info is None:
        info = INFO_COLS_SELECT
    if cache is None:
        cache = _pooch_cache("clinvar.parquet")

    if vcf_file is None:
        vcf_file = download_vcf()["vcf"]

    if cache is not None and os.path.exists(cache) and not force:
        print(f"Reading from {cache}")
        return pl.read_parquet(cache)

    vcf = VCF(vcf_file, filter=filter)
    vcf_df = vcf.get_record_info(
        contig=contig, attrs=attrs, info=info, progress=progress
    )

    vcf_df = vcf_df.with_columns(
        vcf_df["MC"].str.split("|").list.first().alias("MC_id"),
        vcf_df["MC"].str.split("|").list.last().alias("MC_term"),
    )

    review_score_map = {
        "practice_guideline": 4,
        "reviewed_by_expert_panel": 3,
        "criteria_provided,_multiple_submitters,_no_conflicts": 2,
        "criteria_provided,_conflicting_classifications": 1,
        "criteria_provided,_single_submitter": 1,
        "no_assertion_criteria_provided": 0,
        "no_classification_provided": 0,
        "no_classification_for_the_single_variant": 0,
        "no_classifications_from_unflagged_records": 0,
    }
    vcf_df = vcf_df.with_columns(
        pl.col("CLNREVSTAT")
        .replace(review_score_map)
        .alias("CLNREVSTAT_score")
        .cast(pl.Int8)
    )

    if extract_ids:
        vcf_df = _extract_id_cols(vcf_df)

    if cache is not None:
        print(f"Caching to {cache}")
        vcf_df.write_parquet(cache)

    return vcf_df


# Long-tail CLNSIG → 6-class map. Kept verbatim from the upstream port so
# downstream collapsing behavior matches the VEP_protein pipeline exactly.
_CLNSIG_MAP: dict[str | None, str] = {
    "Benign": "benign",
    "Likely_benign": "likely_benign",
    "Benign/Likely_benign": "likely_benign",
    "Pathogenic/Likely_pathogenic": "likely_path",
    "Pathogenic": "path",
    "Likely_pathogenic": "likely_path",
    "Pathogenic/Likely_pathogenic/Pathogenic,_low_penetrance": "likely_path",
    "Benign|other": "benign",
    "Benign|confers_sensitivity": "benign",
    "Benign|Affects|association|other": "benign",
    "confers_sensitivity": "other",
    "no_classification_for_the_single_variant": "other",
    "Likely_pathogenic|other": "likely_path",
    "Pathogenic/Likely_risk_allele": "likely_path",
    "Benign/Likely_benign|risk_factor": "likely_benign",
    "Conflicting_classifications_of_pathogenicity|drug_response": "conflicting",
    "Pathogenic|association": "path",
    "Uncertain_significance": "other",
    "Pathogenic|confers_sensitivity": "path",
    "Likely_benign|other": "likely_benign",
    "Affects": "other",
    "Likely_risk_allele": "likely_path",
    "Pathogenic|association|protective": "path",
    "Pathogenic|drug_response": "path",
    "protective|risk_factor": "other",
    "Pathogenic|risk_factor": "path",
    "protective": "other",
    "Conflicting_classifications_of_pathogenicity|drug_response|other": "conflicting",
    "Benign/Likely_benign|drug_response|other": "likely_benign",
    "Conflicting_classifications_of_pathogenicity|Affects": "conflicting",
    "Likely_pathogenic|Affects": "likely_path",
    "Likely_benign|drug_response": "likely_benign",
    "Likely_benign|drug_response|other": "likely_benign",
    "Conflicting_classifications_of_pathogenicity": "conflicting",
    "Benign/Likely_benign|other|risk_factor": "likely_benign",
    "association|risk_factor": "other",
    "Uncertain_significance|association": "other",
    "Benign/Likely_benign|drug_response": "likely_benign",
    "Conflicting_classifications_of_pathogenicity|other|risk_factor": "conflicting",
    "Benign/Likely_benign|association": "likely_benign",
    "Likely_benign|Affects|association": "likely_benign",
    "Pathogenic|other": "path",
    "Uncertain_risk_allele": "other",
    "Benign|association": "benign",
    "Likely_pathogenic|association": "likely_path",
    "not_provided": "other",
    "Pathogenic|Affects": "path",
    "Pathogenic/Likely_pathogenic|risk_factor": "likely_path",
    "drug_response": "other",
    "Conflicting_classifications_of_pathogenicity|protective": "conflicting",
    "Likely_pathogenic/Likely_risk_allele": "likely_path",
    "Benign|Affects": "benign",
    "confers_sensitivity|other": "other",
    "association_not_found": "other",
    "other": "other",
    "Pathogenic/Likely_pathogenic|other": "likely_path",
    "Benign|risk_factor": "benign",
    "Likely_pathogenic|risk_factor": "likely_path",
    "Uncertain_risk_allele|risk_factor": "other",
    "Likely_benign|risk_factor": "likely_benign",
    "Uncertain_significance|drug_response": "other",
    "association|drug_response|risk_factor": "other",
    "Conflicting_classifications_of_pathogenicity|association|risk_factor": "conflicting",
    "Likely_pathogenic,_low_penetrance": "likely_path",
    "risk_factor": "other",
    "association": "other",
    "no_classifications_from_unflagged_records": "other",
    "Uncertain_significance|Affects": "other",
    "Uncertain_significance|other": "other",
    "Pathogenic|protective": "conflicting",
    "Uncertain_significance/Uncertain_risk_allele": "other",
    "Uncertain_significance|risk_factor": "other",
    "Conflicting_classifications_of_pathogenicity|other": "conflicting",
    "Conflicting_classifications_of_pathogenicity|association": "conflicting",
    "Pathogenic/Likely_pathogenic/Pathogenic,_low_penetrance|risk_factor": "likely_path",
    "Pathogenic/Likely_pathogenic/Likely_risk_allele": "likely_path",
    "Benign/Likely_benign|other": "likely_benign",
    "Benign|protective": "benign",
    "Benign|drug_response": "benign",
    "Pathogenic/Pathogenic,_low_penetrance|other|risk_factor": "path",
    "Likely_benign|association": "likely_benign",
    "drug_response|other": "other",
    "Conflicting_classifications_of_pathogenicity|risk_factor": "conflicting",
    "drug_response|risk_factor": "other",
    "Pathogenic/Pathogenic,_low_penetrance|other": "path",
    "other|risk_factor": "other",
    "Likely_pathogenic|drug_response": "likely_path",
    "Pathogenic/Likely_pathogenic|association": "likely_path",
    "Likely_pathogenic|protective": "conflicting",
    "risk_factor,benign": "conflicting",
    "risk_factor,benign,likely_benign": "conflicting",
    None: "other",
}

_CLNSIG_SUPER_SIMPLE_MAP = {
    "benign": "benign",
    "likely_benign": "benign",
    "pathogenic": "path",
    "likely_pathogenic": "path",
    "path": "path",
    "likely_path": "path",
    "conflicting": "conflicting",
    "other": "other",
}


def simplify_annotations(
    bed: pl.DataFrame | pd.DataFrame,
    maps: list[dict[str, Any]] | None = None,
    verbose: bool = True,
) -> pl.DataFrame | pd.DataFrame:
    """Collapse long-tail CLNSIG strings into 6-class + 4-class buckets.

    Adds ``CLNSIG_simple`` (6 buckets) and ``CLNSIG_super_simple``
    (benign / path / conflicting / other). Accepts pandas or polars
    input and returns the same flavor it received.
    """
    if maps is None:
        if verbose:
            print("Using default maps.")
        maps = [
            {"input_col": "CLNSIG", "output_col": "CLNSIG_simple", "map": _CLNSIG_MAP},
            {"input_col": "CLIN_SIG", "output_col": "CLNSIG_simple", "map": _CLNSIG_MAP},
            {
                "input_col": "CLNSIG_simple",
                "output_col": "CLNSIG_super_simple",
                "map": _CLNSIG_SUPER_SIMPLE_MAP,
            },
            {
                "input_col": "CLIN_SIG_simple",
                "output_col": "CLIN_SIG_super_simple",
                "map": _CLNSIG_SUPER_SIMPLE_MAP,
            },
        ]

    was_pandas = isinstance(bed, pd.DataFrame)
    if was_pandas:
        bed = pl.DataFrame(bed)

    if verbose:
        print("Simplifying annotations.")
    for m in maps:
        if m["input_col"] in bed.columns and m["output_col"] not in bed.columns:
            bed = bed.with_columns(
                pl.col(m["input_col"])
                .replace_strict(m["map"], default=pl.lit("other"))
                .alias(m["output_col"])
            )

    if "GENEINFO" in bed.columns and "GENE" not in bed.columns:
        bed = bed.with_columns(
            pl.col("GENEINFO").str.split(":").list.first().alias("GENE")
        )

    if was_pandas:
        bed = bed.to_pandas()
    return bed


def _explode_col(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Explode a column if it is a list-of-strings; otherwise no-op."""
    if df[col].dtype.__str__() == "List(String)":
        df = df.explode(col)
    return df


def _add_variant_name(
    df: pl.DataFrame | pd.DataFrame,
    chrom_col: str = "chrom",
    start_col: str = "chromStart",
    end_col: str | None = "chromEnd",
    ref_col: str = "REF",
    alt_col: str = "ALT",
    alias: str = "name",
    force: bool = False,
) -> pl.DataFrame | pd.DataFrame:
    """Build a canonical variant name (``chrN:start-end_REF_ALT``).

    Inlined from ``bschilder/VEP_protein/src/utils.add_variant_name`` so
    this module has no cross-repo dep. Accepts pandas or polars; returns
    the same flavor.
    """
    if alias in df.columns and not force:
        print(f"Column {alias} already exists in dataframe, skipping")
        return df

    was_pandas = isinstance(df, pd.DataFrame)
    if was_pandas:
        df = pl.DataFrame(df)

    for col in (ref_col, alt_col):
        if col in df.columns and df.schema[col] != pl.String:
            df = df.with_columns(pl.col(col).cast(pl.Utf8))

    if end_col is not None and end_col not in df.columns:
        end_col = None

    if end_col is not None:
        end_expr = pl.col(end_col).cast(pl.Utf8)
    else:
        end_expr = (
            pl.col(start_col).cast(pl.Int32)
            + pl.col(ref_col).str.len_chars().cast(pl.Int32)
        ).cast(pl.Utf8)

    result = df.with_columns(
        pl.concat_str(
            [
                pl.lit("chr"),
                pl.col(chrom_col).cast(pl.Utf8).str.replace("chr", ""),
                pl.lit(":"),
                pl.col(start_col).cast(pl.Utf8),
                pl.lit("-"),
                end_expr,
                pl.lit("_"),
                pl.col(ref_col).cast(pl.Utf8),
                pl.lit("_"),
                pl.col(alt_col).cast(pl.Utf8),
            ]
        ).alias(alias)
    )

    if was_pandas:
        result = result.to_pandas()
    return result


def df_to_bed(
    vcf_df: pl.DataFrame,
    save_path: str | None = None,
    extract_ids: bool = True,
    variant_name_alias: str | None = "name",
    extra_cols: list[str] | None = None,
    simplify: bool = True,
) -> pl.DataFrame:
    """Convert a parsed ClinVar VCF DataFrame to BED-like format."""
    if extra_cols is None:
        extra_cols = []

    vcf_df = _explode_col(vcf_df, "ALT")

    bed = vcf_df.rename(
        {"CHROM": "chrom", "POS": "chromStart"}
    ).with_columns(
        [(pl.col("chromStart") + pl.col("REF").str.len_chars()).alias("chromEnd")]
    )
    if "CLNREVSTAT_score" in vcf_df.columns:
        bed = bed.with_columns(pl.col("CLNREVSTAT_score").alias("score"))

    if variant_name_alias is not None:
        bed = _add_variant_name(bed, alias=variant_name_alias)

    select_cols = [
        "chrom", "chromStart", "chromEnd",
        variant_name_alias, "score",
        "REF", "ALT", "MC_id", "MC_term",
        *INFO_COLS_SELECT,
        "CLNREVSTAT_score",
    ] + extra_cols
    select_cols = [col for col in select_cols if col in bed.columns]
    bed = bed.select(select_cols).filter(
        pl.col("chrom").str.contains("^[0-9]+$|^X$|^Y$")
    )

    bed = bed.with_columns(pl.col("ALT").fill_null("")).drop_nulls(subset=["ALT"])
    bed = bed.with_columns(pl.col("REF").fill_null("")).drop_nulls(subset=["REF"])

    if extract_ids:
        bed = _extract_id_cols(bed)

    if simplify:
        bed = simplify_annotations(bed)

    if save_path:
        bed.to_pandas().to_csv(save_path, sep="\t", index=False)
    return bed


def df_to_sites(vcf_df: pl.DataFrame) -> pl.DataFrame:
    """Convert a parsed ClinVar VCF DataFrame to a sites-format DataFrame."""
    sites = vcf_df.with_columns(
        [
            (pl.col("POS") + pl.col("REF").str.len_chars()).alias("POS_END"),
            (pl.col("ALT").list.join(",").alias("ALT")),
        ]
    )

    sites = _add_variant_name(
        sites, chrom_col="CHROM", start_col="POS", end_col="POS_END"
    )

    sites = sites.select(
        [
            "CHROM", "POS", "POS_END",
            "name",
            "REF", "ALT", "MC_id", "MC_term",
            *INFO_COLS_SELECT,
            "CLNREVSTAT_score",
        ]
    ).filter(pl.col("CHROM").str.contains("^[0-9]+$|^X$|^Y$"))

    sites = sites.with_columns(pl.col("ALT").fill_null("")).drop_nulls(subset=["ALT"])
    sites = sites.with_columns(pl.col("REF").fill_null("")).drop_nulls(subset=["REF"])

    return sites


def bed_to_sites(
    bed: pl.DataFrame | pd.DataFrame,
    chrom_col: str = "chrom",
    start_col: str = "chromStart",
    end_col: str = "chromEnd",
    ref_col: str = "REF",
    alt_col: str = "ALT",
) -> pl.DataFrame:
    """Convert a BED-format DataFrame to a sites-format DataFrame."""
    if isinstance(bed, pd.DataFrame):
        bed = pl.DataFrame(bed.copy())

    sites = bed.rename(
        {
            chrom_col: "CHROM",
            start_col: "POS",
            end_col: "POS_END",
            ref_col: "REF",
            alt_col: "ALT",
        }
    )

    sites = sites.select(
        [
            "CHROM", "POS", "REF", "ALT",
            *[
                col
                for col in sites.columns
                if col not in ["CHROM", "POS", "REF", "ALT"]
            ],
        ]
    )

    return sites


def _extract_id_cols(
    df: pl.DataFrame,
    search_terms: list[str] | None = None,
    add_counts: bool = True,
    verbose: bool = True,
) -> pl.DataFrame:
    """Parse ``CLNDISDB`` into list columns per ontology (MONDO, OMIM, ...)."""
    if search_terms is None:
        search_terms = ["MONDO", "OMIM", "Orphanet", "MedGen", "MeSH"]

    if verbose:
        print("Extracting ID columns.")
    for id_type in search_terms:
        if id_type not in df.columns:
            df = df.with_columns(
                pl.col("CLNDISDB").str.extract_all(f"({id_type}:[^,|]+)").alias(id_type)
            )
        if add_counts and f"{id_type}_n" not in df.columns:
            df = df.with_columns(pl.col(id_type).list.len().alias(f"{id_type}_n"))

    return df


def read_bed(
    path: str,
    schema_overrides: dict[str, Any] | None = None,
    separator: str = "\t",
    simplify: bool = True,
    extract_ids: bool = True,
    as_pandas: bool = False,
    **kwargs: Any,
) -> pl.DataFrame | pd.DataFrame:
    """Read a BED file written by :func:`df_to_bed`.

    Convenience wrapper around :func:`polars.read_csv` with sensible
    schema defaults for the four required BED columns.
    """
    if schema_overrides is None:
        schema_overrides = {
            "chrom": pl.Utf8,
            "chromStart": pl.Int64,
            "chromEnd": pl.Int64,
            "score": pl.Float64,
        }
    bed = pl.read_csv(
        path, schema_overrides=schema_overrides, separator=separator, **kwargs
    ).drop_nulls(subset=["ALT"])

    bed = bed.with_columns(pl.col("name").alias("site"))

    if extract_ids:
        bed = _extract_id_cols(bed)

    if simplify:
        bed = simplify_annotations(bed)

    if as_pandas:
        bed = bed.to_pandas()

    return bed


def filter_df(
    vcf_df: pl.DataFrame,
    filters: dict[str, Any] | None = None,
    verbose: bool = True,
) -> pl.DataFrame:
    """Filter a parsed ClinVar VCF DataFrame.

    ``filters`` accepts ``"COL": value`` pairs. List values become
    OR-substring matches; ``CLNREVSTAT_score`` is compared as ``>=``;
    everything else is an equality check.
    """
    if filters is None:
        filters = {}

    filter_conditions = []
    for key, value in filters.items():
        if key not in vcf_df.columns:
            if verbose:
                print(f"Column {key} not found in DataFrame. Skipping filter.")
            continue
        if isinstance(value, list):
            filter_conditions.append(pl.col(key).str.contains("|".join(value)))
        else:
            filter_conditions.append(
                pl.col(key) >= value
                if key == "CLNREVSTAT_score"
                else pl.col(key) == value
            )

    if not filter_conditions:
        return vcf_df

    combined_condition = filter_conditions[0]
    for condition in filter_conditions[1:]:
        combined_condition = combined_condition & condition

    cv_df = (
        vcf_df.filter(combined_condition)
        .with_columns(pl.col("POS").cast(pl.Int64))
        .drop_nulls(subset=["CLNDN"])
    )

    if verbose:
        print(f"Filtered DataFrame shape: {cv_df.shape}")
        print(f"Variant count: {cv_df.shape[0]}")
        if "GENEINFO" in cv_df.columns:
            print(f"Gene count: {cv_df['GENEINFO'].unique().len()}")

    return cv_df


def count_sites_per_gene(
    vcf_df: pl.DataFrame | None = None,
    groupby_cols: list[str] | None = None,
    sort: bool = True,
) -> pd.DataFrame:
    """Count unique variant sites per gene × disease (MONDO)."""
    if groupby_cols is None:
        groupby_cols = ["MONDO", "GENEINFO"]
    if vcf_df is None:
        vcf_df = vcf_to_df()

    cv_df = _add_variant_name(df_to_bed(vcf_df))
    vpd = (
        cv_df.explode("MONDO")
        .group_by(groupby_cols)
        .agg(pl.col("name").n_unique())
        .sort("MONDO")
        .to_pandas()
        .rename(columns={"name": "sites"})
    )
    vpd = vpd.loc[vpd["MONDO"].notnull()]
    vpd.loc[:, "MONDO"] = vpd["MONDO"].str.replace("MONDO:MONDO:", "MONDO:")
    if sort:
        vpd = vpd.sort_values("sites", ascending=False)
    return vpd


# ─── ClinVar per-variant REST lookup (NCBI E-utilities) ─────────────────────
# Targeted-query "API mode" complementing the bulk-VCF flow above. Wraps
# NCBI's E-utilities at https://eutils.ncbi.nlm.nih.gov/entrez/eutils/ —
# ``esearch`` for resolving terms → ClinVar UIDs and ``esummary`` for
# pulling per-variant records.
#
# Requests are intentionally rate-limit-friendly: each call is a single
# HTTP GET, no pagination by default. Set ``api_key`` to your NCBI E-utils
# key to lift the 3 req/sec limit to 10 req/sec.

import time as _ncbi_time

import requests as _ncbi_requests  # alias avoids the lazy genoray import below

NCBI_EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
"""NCBI E-utilities root for ClinVar lookups."""

_NCBI_RATE_LIMIT_SLEEP_S = 0.34
"""Per-request sleep that keeps us under the no-API-key cap of 3 req/sec."""


def _ncbi_get(path: str, params: dict, *, timeout: int = 30) -> _ncbi_requests.Response:
    """GET an NCBI E-utilities endpoint with polite-rate-limit handling.

    Sleeps ~340 ms between calls so we stay under the 3-req/sec cap that
    NCBI imposes on un-keyed traffic, and retries once on a 429 response.
    """
    url = f"{NCBI_EUTILS_BASE_URL}/{path.lstrip('/')}"
    for attempt in range(2):
        response = _ncbi_requests.get(url, params=params, timeout=timeout)
        if response.status_code != 429:
            response.raise_for_status()
            _ncbi_time.sleep(_NCBI_RATE_LIMIT_SLEEP_S)
            return response
        # Rate-limited: back off and retry once.
        _ncbi_time.sleep(1.0 + attempt)
    response.raise_for_status()  # raises HTTPError on the second 429
    return response  # pragma: no cover  -- unreachable; raise_for_status raises


def _esearch(
    term: str,
    *,
    retmax: int = 20,
    api_key: str | None = None,
    timeout: int = 30,
) -> list[str]:
    """Resolve a ClinVar search term to a list of ClinVar UIDs."""
    params = {"db": "clinvar", "term": term, "retmode": "json", "retmax": retmax}
    if api_key:
        params["api_key"] = api_key
    response = _ncbi_get("esearch.fcgi", params, timeout=timeout)
    return response.json().get("esearchresult", {}).get("idlist", [])


def query_variant(
    variant_id: str | int,
    *,
    api_key: str | None = None,
    timeout: int = 30,
) -> dict:
    """Fetch one ClinVar variant by Entrez UID (or VCV / RCV accession).

    Parameters
    ----------
    variant_id : str or int
        Either:

        * a ClinVar Entrez UID (e.g. ``12345`` or ``"12345"``), or
        * a VCV/RCV accession (e.g. ``"VCV000012345"``) — esearch resolves
          it to a UID first.
    api_key : str, optional
        NCBI E-utilities API key.
    timeout : int, default 30

    Returns
    -------
    dict
        The ESummary record for the variant: ``accession``,
        ``accession_version``, ``title`` (HGVS-like coordinate +
        protein-change), ``obj_type`` (e.g. ``"single nucleotide variant"``),
        ``variation_set`` (cross-references), ``germline_classification``,
        ``clinical_significance``, ``gene_sort`` (gene symbol), and more.

    Raises
    ------
    KeyError
        If the variant_id can't be resolved.
    requests.HTTPError
        On HTTP failure.

    Examples
    --------
    >>> variant = query_variant(12345)  # doctest: +SKIP
    >>> variant["accession"]  # doctest: +SKIP
    'VCV000012345'
    """
    str_id = str(variant_id).strip()
    if not str_id.isdigit():
        # Treat as VCV / RCV accession — resolve to UID first.
        hits = _esearch(str_id, retmax=1, api_key=api_key, timeout=timeout)
        if not hits:
            raise KeyError(f"ClinVar accession {variant_id!r} not found")
        str_id = hits[0]

    params = {"db": "clinvar", "id": str_id, "retmode": "json"}
    if api_key:
        params["api_key"] = api_key
    response = _ncbi_get("esummary.fcgi", params, timeout=timeout)
    result = response.json().get("result", {})
    if str_id not in result:
        raise KeyError(f"ClinVar UID {str_id} not in esummary response")
    return result[str_id]


def query_gene(
    gene_symbol: str,
    *,
    retmax: int = 100,
    api_key: str | None = None,
    timeout: int = 30,
) -> list[str]:
    """Return ClinVar UIDs for variants in a given gene.

    Convenience over :func:`_esearch`. Use the returned UIDs with
    :func:`query_variant` to pull individual records.
    """
    return _esearch(
        f"{gene_symbol}[gene]", retmax=retmax, api_key=api_key, timeout=timeout
    )


__all__ = [
    "INFO_COLS_SELECT",
    "NCBI_EUTILS_BASE_URL",
    "bed_to_sites",
    "count_sites_per_gene",
    "df_to_bed",
    "df_to_sites",
    "download_vcf",
    "filter_df",
    "query_gene",
    "query_variant",
    "read_bed",
    "simplify_annotations",
    "vcf_to_df",
]
