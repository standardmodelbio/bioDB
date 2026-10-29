"""Header-aware raw summary statistics, with explicit coverage and no imputation."""

from __future__ import annotations

import csv
import gzip
import math
from collections.abc import Iterator
from pathlib import Path

import polars as pl

SCHEMA_VERSION = "raw-sumstats-2"
SCHEMA = {
    "source": pl.String,
    "study_id": pl.String,
    "source_file_id": pl.String,
    "source_url": pl.String,
    "source_sha256": pl.String,
    "source_row": pl.Int64,
    "region_sha256": pl.String,
    "region_row": pl.Int64,
    "source_etag": pl.String,
    "chrom": pl.String,
    "pos": pl.Int64,
    "rsid": pl.String,
    "effect_allele": pl.String,
    "other_allele": pl.String,
    "beta": pl.Float64,
    "se": pl.Float64,
    "z": pl.Float64,
    "p": pl.Float64,
    "neg_log_10_p_value": pl.Float64,
    "frq": pl.Float64,
    "n": pl.Float64,
    "raw_fields": pl.List(pl.Struct({"name": pl.String, "value": pl.String})),
}
# Legacy hm_ coordinates/alleles describe the harmonised representation. Numerical
# measurements remain the source values; no genomic reference allele is inferred.
ALIASES = {
    "chrom": ("hm_chrom", "chromosome", "chr", "chrom"),
    "pos": ("hm_pos", "base_pair_location", "bp", "position", "pos"),
    "rsid": ("hm_rsid", "rsid", "variant_id", "snp", "rs_id"),
    "effect_allele": ("hm_effect_allele", "effect_allele", "a2"),
    "other_allele": ("hm_other_allele", "other_allele", "a1", "non_effect_allele"),
    "beta": ("hm_beta", "beta", "effect_size"),
    "se": ("standard_error", "se", "sebeta"),
    "z": ("z", "z_score", "zscore"),
    "p": ("p_value", "p", "pvalue", "pval", "p-value"),
    "neg_log_10_p_value": ("neg_log_10_p_value", "neg_log10_p_value", "neglog10p", "mlogp", "logp"),
    "frq": ("hm_effect_allele_frequency", "effect_allele_frequency", "frq", "eaf"),
    "n": ("n", "n_samples", "sample_size", "n_total"),
}
MISSING = {"", ".", "-", "na", "nan", "none", "null"}


def normalize_regions(scope="genomewide", regions=None):
    """Validate scope and return sorted inclusive intervals for reproducible identity."""
    if scope not in {"genomewide", "regions"}:
        raise ValueError("scope must be 'genomewide' or 'regions'")
    if scope == "genomewide":
        if regions is not None:
            raise ValueError("regions require scope='regions'")
        return None
    if not regions:
        raise ValueError("scope='regions' requires nonempty regions")
    intervals = []
    for region in regions:
        if isinstance(region, str):
            chrom, span = region.split(":")
            start, end = map(int, span.split("-"))
        else:
            chrom, start, end = region
            start, end = int(start), int(end)
        if start < 0 or end < start:
            raise ValueError(f"Invalid region: {region}")
        intervals.append((str(chrom).removeprefix("chr"), start, end))
    return sorted(set(intervals))


def number(value):
    if value.strip().lower() in MISSING:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite numerical measurement: {value!r}")
    return result


def iter_file(
    path,
    *,
    source,
    study_id="",
    file_id="",
    url="",
    sha256="",
    source_metadata=None,
    batch_size=100_000,
    batch_rows=None,
    p_max=None,
    scope="genomewide",
    regions=None,
) -> Iterator[pl.DataFrame]:
    """Read every row to EOF. Partial yielded batches are invalid if iteration raises.

    P is never reconstructed from Z/beta/SE. Log-P is retained separately so
    underflow and P=0 remain distinguishable. Source tokens survive in raw_fields.
    Explicit p_max uses inclusive P <= p_max; missing P is excluded only then.
    Malformed data raises, so a file with dropped rows cannot claim completeness.
    """
    intervals = normalize_regions(scope, regions)
    batch_size = batch_rows if batch_rows is not None else batch_size
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if p_max is not None and (not math.isfinite(p_max) or not 0 < p_max <= 1):
        raise ValueError("p_max must satisfy 0 < p_max <= 1")
    path = Path(path)
    with path.open("rb") as probe:
        compressed = probe.read(2) == b"\x1f\x8b"
    opener = gzip.open if compressed else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        lines = (line for line in handle if line.strip() and not line.startswith("##"))
        reader = csv.reader(lines, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("Empty summary statistics file; missing header") from exc
        header[0] = header[0].lstrip("#")
        lower = {name.lower(): i for i, name in enumerate(header)}
        if len(lower) != len(header):
            raise ValueError("Duplicate summary statistics columns")
        indices = {
            key: next((lower[a] for a in aliases if a in lower), None)
            for key, aliases in ALIASES.items()
        }
        if (
            indices["chrom"] is None
            or indices["pos"] is None
            or all(indices[key] is None for key in ("p", "neg_log_10_p_value"))
        ):
            raise ValueError("Missing columns: chromosome, position, and P or log-P required")
        if source == "gwas_harvest":
            missing = set("SNP CHR BP A1 A2 FRQ BETA SE Z P N".lower().split()) - lower.keys()
            if missing:
                raise ValueError(f"Missing columns in Harvest file: {sorted(missing)}")
        rows = []
        for row_number, fields in enumerate(reader, 1):
            if len(fields) != len(header):
                raise ValueError(
                    f"Row {row_number}: expected {len(header)} columns, got {len(fields)}"
                )

            def get(key, values=fields):
                return values[indices[key]] if indices[key] is not None else ""

            try:
                record = {
                    key: (get(key) or None)
                    for key in ("chrom", "rsid", "effect_allele", "other_allele")
                }
                record["pos"] = int(get("pos"))
                for key in ("beta", "se", "z", "p", "neg_log_10_p_value", "frq", "n"):
                    record[key] = number(get(key))
                if record["p"] is not None and not 0 <= record["p"] <= 1:
                    raise ValueError("P outside [0,1]")
                if record["neg_log_10_p_value"] is not None and record["neg_log_10_p_value"] < 0:
                    raise ValueError("Negative -log10(P)")
            except ValueError as exc:
                raise ValueError(f"Row {row_number}: {exc}") from exc
            if p_max is not None:
                if record["p"] is not None:
                    if record["p"] > p_max:
                        continue
                elif record["neg_log_10_p_value"] is None or record[
                    "neg_log_10_p_value"
                ] < -math.log10(p_max):
                    continue
            if intervals is not None and not any(
                record["chrom"].removeprefix("chr") == chrom and start <= record["pos"] <= end
                for chrom, start, end in intervals
            ):
                continue
            record.update(
                source=source,
                study_id=study_id,
                source_file_id=file_id,
                source_url=url,
                source_sha256=None
                if (source_metadata or {}).get("transport") == "tabix"
                else sha256,
                source_row=None
                if (source_metadata or {}).get("transport") == "tabix"
                else row_number,
                region_sha256=sha256
                if (source_metadata or {}).get("transport") == "tabix"
                else None,
                region_row=row_number
                if (source_metadata or {}).get("transport") == "tabix"
                else None,
                source_etag=(source_metadata or {}).get("source_etag"),
                raw_fields=[
                    {"name": name, "value": value}
                    for name, value in zip(header, fields, strict=True)
                ],
            )
            rows.append(record)
            if len(rows) >= batch_size:
                yield pl.DataFrame(rows, schema=SCHEMA)
                rows = []
        if rows:
            yield pl.DataFrame(rows, schema=SCHEMA)
