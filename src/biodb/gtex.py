"""GTEx transcript expression: sequential targeted API and streaming bulk summaries."""

from __future__ import annotations

import csv
from collections.abc import Iterator, Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import requests

from biodb.transcript_io import download_source, open_text

API = "https://gtexportal.org/api/v2"
EXPRESSION_SUMMARY_SCHEMA = "gtex-distribution-v1"
ANNOTATION_RELEASES = {"gtex_v11": "47", "gtex_v10": "39", "gtex_v8": "26"}


def release_metadata(dataset: str = "gtex_v11") -> dict[str, Any]:
    response = requests.get(f"{API}/metadata/dataset", params={"datasetId": dataset}, timeout=30)
    response.raise_for_status()
    rows = response.json()
    if isinstance(rows, dict):
        rows = rows.get("data", [])
    matches = [r for r in rows if r["datasetId"] == dataset]
    if len(matches) != 1:
        raise ValueError(f"GTEx dataset metadata missing/ambiguous: {dataset}")
    return matches[0]


def list_datasets(dataset: str = "gtex_v11") -> dict[str, dict[str, Any]]:
    """Discover original public bulk-file URLs, rather than guessing bucket paths."""
    response = requests.get(
        f"{API}/dataset/openAccessFilesMetadata", params={"project_id": "adult-gtex"}, timeout=30
    )
    response.raise_for_status()
    version = dataset.removeprefix("gtex_")
    matches: dict[str, dict[str, Any]] = {}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            name = value.get("displayName", "")
            if f"_{version}_" in name and value.get("url"):
                if "RSEM" in name and "transcripts_tpm" in name or "transcript_tpm" in name:
                    matches["transcript_tpm"] = value
                if name.endswith("Annotations_SampleAttributesDS.txt"):
                    matches["samples"] = value
            for child in value.values():
                if isinstance(child, (list, dict)):
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(response.json())
    if not {"transcript_tpm", "samples"} <= matches.keys():
        raise ValueError(f"GTEx public manifest lacks required transcript files for {dataset}")
    return matches


def get_dataset(
    kind: str = "transcript_tpm", *, dataset: str = "gtex_v11", cache_dir: str | Path | None = None
) -> Path:
    item = list_datasets(dataset)[kind]
    root = Path(cache_dir or "~/.cache/biodb/gtex").expanduser() / dataset
    return download_source(
        item["url"],
        root / item["displayName"],
        source="gtex",
        release=dataset,
        terms_url="https://gtexportal.org/home/license",
    )


def query_expression(
    gene_id: str, *, dataset: str = "gtex_v11", tissue: str | None = None, items_per_page: int = 250
) -> list[dict[str, Any]]:
    """Sequentially page transcript medians. GTEx asks clients not to parallelize."""
    rows = []
    page = 0
    while True:
        params: dict[str, Any] = {
            "gencodeId": gene_id,
            "datasetId": dataset,
            "page": page,
            "itemsPerPage": items_per_page,
        }
        if tissue:
            params["tissueSiteDetailId"] = tissue
        response = requests.get(
            f"{API}/expression/medianTranscriptExpression", params=params, timeout=60
        )
        response.raise_for_status()
        payload = response.json()
        for row in payload["data"]:
            if row.get("datasetId") != dataset:
                raise ValueError("GTEx response dataset differs from requested release")
            rows.append(row)
        page += 1
        if page >= payload["paging_info"]["numberOfPages"]:
            return rows


def _matrix_rows(path: str | Path) -> Iterator[tuple[list[str], str, np.ndarray]]:
    with open_text(path) as fh:
        line = next(fh)
        if line.startswith("#1."):
            next(fh)  # GCT dimensions
            line = next(fh)
        header = line.rstrip("\n").split("\t")
        offset = 2 if len(header) > 1 and header[1].lower() in {"description", "gene_id"} else 1
        samples = header[offset:]
        for line in fh:
            fields = line.rstrip("\n").split("\t", offset)
            if len(fields) <= offset:
                raise ValueError("GTEx matrix row is missing sample columns")
            values = np.fromstring(fields[offset], sep="\t", dtype=np.float64)
            if len(values) != len(samples) or not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"invalid GTEx transcript expression row {fields[0]}")
            yield samples, fields[0], values


def select_gene_transcript_matrix(
    tpm_path: str | Path,
    destination: str | Path,
    *,
    transcript_to_gene: Mapping[str, str],
    gene_ids: set[str],
) -> dict[str, Any]:
    """Select complete source genes, verifying matrix gene IDs before filtering.

    Run whole-matrix extraction on a worker. A quantified sibling missing from
    the annotation mapping fails instead of inflating retained siblings' usage.
    The original two-column RSEM identity/header and sample measurements survive.
    """
    from biodb.transcript_io import split_id

    targets = {split_id(gene)[0] for gene in gene_ids}
    mapping = {tx: split_id(gene)[0] for tx, gene in transcript_to_gene.items()}
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    found = set()
    try:
        with open_text(tpm_path) as source, partial.open("w") as dest:
            header = next(source)
            fields = header.rstrip("\n").split("\t")
            if len(fields) < 3 or fields[1].lower() != "gene_id":
                raise ValueError(
                    "complete gene selection requires an explicit matrix gene_id column"
                )
            dest.write(header)
            for line in source:
                fields = line.split("\t", 2)
                if len(fields) != 3:
                    raise ValueError("malformed transcript matrix row")
                tx, matrix_gene = fields[0], split_id(fields[1])[0]
                expected = mapping.get(tx)
                if expected in targets and expected != matrix_gene:
                    raise ValueError(f"transcript/gene mapping conflict: {tx}")
                if matrix_gene not in targets:
                    continue
                if expected is None:
                    raise ValueError(f"quantified sibling missing from source mapping: {tx}")
                if expected != matrix_gene:
                    raise ValueError(f"transcript/gene mapping conflict: {tx}")
                if tx in found:
                    raise ValueError(f"duplicate transcript matrix row: {tx}")
                found.add(tx)
                dest.write(line)
        if not found:
            raise ValueError("no quantified transcripts for selected genes")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
    return {
        "complete_gene_denominators": sorted(targets),
        "quantified_transcripts": sorted(found),
        "source_models_not_in_quantification": sorted(
            tx for tx, gene in mapping.items() if gene in targets and tx not in found
        ),
    }


def iter_expression_summaries(
    tpm_path: str | Path,
    sample_attributes: str | Path,
    *,
    transcript_to_gene: Mapping[str, str],
    work_dir: str | Path,
    dataset: str = "gtex_v11",
    detection_threshold_tpm: float = 1.0,
) -> Iterator[dict[str, Any]]:
    """Two-pass, disk-backed per-sample gene denominators, then tissue summaries.

    Supply the COMPLETE source-annotation transcript mapping (GENCODE47 for V11),
    not a target-release subset. A missing mapping fails to prevent biased usage.
    The denominator matrix can be large and belongs on a RunPod container disk.
    """
    if detection_threshold_tpm < 0:
        raise ValueError("detection threshold cannot be negative")
    with open_text(sample_attributes) as fh:
        sample_tissues = {r["SAMPID"]: r["SMTSD"] for r in csv.DictReader(fh, delimiter="\t")}
    genes = {gene: i for i, gene in enumerate(sorted(set(transcript_to_gene.values())))}
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="gtex-denominators-", dir=work_dir) as scratch:
        denominators = None
        seen: set[str] = set()
        quantified_siblings: dict[str, int] = {}
        sample_names: list[str] | None = None
        for samples, tx, values in _matrix_rows(tpm_path):
            if tx not in transcript_to_gene:
                raise ValueError(f"source transcript mapping missing for {tx}")
            if tx in seen:
                raise ValueError(f"duplicate GTEx transcript row {tx}")
            seen.add(tx)
            gene = transcript_to_gene[tx]
            quantified_siblings[gene] = quantified_siblings.get(gene, 0) + 1
            if denominators is None:
                sample_names = samples
                if any(sample not in sample_tissues for sample in samples):
                    raise ValueError("GTEx sample metadata does not cover every matrix sample")
                denominators = np.memmap(
                    Path(scratch) / "denominators.f64",
                    dtype="float64",
                    mode="w+",
                    shape=(len(genes), len(samples)),
                )
                denominators[:] = 0
            denominators[genes[transcript_to_gene[tx]]] += values
        if denominators is None or sample_names is None:
            raise ValueError("GTEx transcript matrix is empty")
        denominators.flush()
        tissues = {
            tissue: np.array(
                [i for i, sample in enumerate(sample_names) if sample_tissues[sample] == tissue]
            )
            for tissue in sorted({sample_tissues[s] for s in sample_names})
        }
        for _samples, tx, values in _matrix_rows(tpm_path):
            denominator = denominators[genes[transcript_to_gene[tx]]]
            for tissue, positions in tissues.items():
                tpm = values[positions]
                total = denominator[positions]
                valid = total > 0
                usages = tpm[valid] / total[valid]
                yield {
                    "transcript_id": tx,
                    "source_gene_id": transcript_to_gene[tx],
                    "tissue": tissue,
                    "median_tpm": float(np.median(tpm)),
                    "detection_fraction": float(np.mean(tpm >= detection_threshold_tpm)),
                    "n_detected": int(np.sum(tpm >= detection_threshold_tpm)),
                    "tpm_q25": float(np.quantile(tpm, 0.25)),
                    "tpm_q75": float(np.quantile(tpm, 0.75)),
                    "usage_q25": float(np.quantile(usages, 0.25)) if valid.any() else None,
                    "usage_q75": float(np.quantile(usages, 0.75)) if valid.any() else None,
                    "n_gene_detected": int(np.sum(total >= detection_threshold_tpm)),
                    "n_quantified_siblings": quantified_siblings[transcript_to_gene[tx]],
                    "median_usage": float(np.median(usages)) if valid.any() else None,
                    "n_samples": len(positions),
                    "n_usage_samples": int(valid.sum()),
                    "detection_threshold_tpm": detection_threshold_tpm,
                    "unit": "TPM",
                    "source": "gtex",
                    "release": dataset,
                    "annotation_release": ANNOTATION_RELEASES[dataset],
                    "scope": "transcript",
                    "evidence_type": "estimated_expression",
                }
        del denominators
