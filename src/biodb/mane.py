"""MANE exact Ensembl/RefSeq transcript correspondences, preserving versions."""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from biodb.transcript_io import download_source, open_text

BASE_URL = "https://ftp.ncbi.nlm.nih.gov/refseq/MANE/MANE_human"


def get_dataset(
    kind: str = "summary", *, release: str = "1.5", cache_dir: str | Path | None = None
) -> Path:
    """Fetch a versioned MANE summary or RefSeq RNA GenBank file."""
    if release != "1.5":
        raise ValueError("supported MANE release is 1.5; add an explicit archived URL for others")
    suffix = {"summary": "summary.txt.gz", "refseq_rna": "refseq_rna.gbff.gz"}.get(kind)
    if suffix is None:
        raise ValueError(f"unknown MANE dataset {kind!r}")
    filename = f"MANE.GRCh38.v{release}.{suffix}"
    root = Path(cache_dir or "~/.cache/biodb/mane").expanduser() / release
    return download_source(
        f"{BASE_URL}/current/{filename}",
        root / filename,
        source="mane",
        release=release,
        terms_url="https://www.ncbi.nlm.nih.gov/refseq/MANE/",
    )


def iter_mappings(path: str | Path, *, release: str = "1.5") -> Iterator[dict[str, Any]]:
    with open_text(path) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            yield {
                "transcript_id": row["Ensembl_nuc"],
                "protein_id": row.get("Ensembl_prot"),
                "refseq_transcript_id": row["RefSeq_nuc"],
                "refseq_protein_id": row.get("RefSeq_prot"),
                "gene_symbol": row.get("symbol"),
                "ncbi_gene_id": row.get("#NCBI_GeneID"),
                "status": row["MANE_status"],
                "chromosome_accession": row.get("GRCh38_chr"),
                "source": "mane",
                "release": release,
                "scope": "transcript",
                "mapping_basis": "mane_exact",
                "original": row,
            }


def query_transcript(
    identifier: str,
    *,
    release: str = "1.5",
    path: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    return [
        r
        for r in iter_mappings(
            path or get_dataset(release=release, cache_dir=cache_dir), release=release
        )
        if identifier in {r["transcript_id"], r["refseq_transcript_id"]}
    ]
