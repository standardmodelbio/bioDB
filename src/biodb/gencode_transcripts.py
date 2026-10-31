"""Comprehensive GENCODE transcript models, streamed one gene block at a time.

The legacy MANE-select parser in gencode.py remains unchanged. This parser keeps
all biotypes/features, explicit CDS phase, native identifiers and original tags.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from biodb.transcript_io import (
    REFERENCE_CHROMOSOMES,
    SequenceStore,
    download_source,
    iter_gtf_features,
    open_text,
    split_id,
)

BASE_URL = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human"


def download_bundle(
    release: str = "50",
    *,
    cache_dir: str | Path | None = None,
    with_sequences: bool = True,
    with_metadata: bool = True,
) -> dict[str, Path]:
    """Fetch immutable comprehensive CHR annotation, FASTAs and metadata."""
    if not str(release).isdigit():
        raise ValueError("GENCODE release must be numeric")
    root = Path(cache_dir or "~/.cache/biodb/gencode").expanduser() / str(release)
    filenames = {"gtf": f"gencode.v{release}.annotation.gtf.gz"}
    if with_sequences:
        filenames.update(
            rna=f"gencode.v{release}.transcripts.fa.gz",
            protein=f"gencode.v{release}.pc_translations.fa.gz",
        )
    if with_metadata:
        for name in (
            "Annotation_remark",
            "RefSeq",
            "SwissProt",
            "TrEMBL",
            "Transcript_supporting_feature",
        ):
            filenames[name] = f"gencode.v{release}.metadata.{name}.gz"
    return {
        key: download_source(
            f"{BASE_URL}/release_{release}/{filename}",
            root / filename,
            source="gencode",
            release=release,
            terms_url="https://www.gencodegenes.org/pages/data_access.html",
        )
        for key, filename in filenames.items()
    }


def _first(attrs: dict[str, list[str]], name: str, default: Any = None) -> Any:
    return attrs.get(name, [default])[0]


def _finish(record: dict[str, Any], sequences: SequenceStore | None) -> dict[str, Any]:
    reverse = record["strand"] == "-"
    for name in ("exons", "cds", "utrs", "start_codons", "stop_codons"):
        record[name].sort(key=lambda r: r["start"], reverse=reverse)
    for rank, exon in enumerate(record["exons"], 1):
        exon["rank"] = rank
    record["transcript_length"] = sum(e["end"] - e["start"] for e in record["exons"])
    record["cds_length"] = sum(e["end"] - e["start"] for e in record["cds"])
    record["utr5_length"] = record["utr3_length"] = 0
    if record["cds"]:
        left = min(c["start"] for c in record["cds"])
        for utr in record["utrs"]:
            side = "utr5_length" if (utr["end"] <= left) != reverse else "utr3_length"
            record[side] += utr["end"] - utr["start"]
    record["junctions"] = [
        {"start": min(a["end"], b["end"]), "end": max(a["start"], b["start"])}
        for a, b in zip(record["exons"], record["exons"][1:], strict=False)
    ]
    rna = sequences.get(record["transcript_id"]) if sequences else None
    protein = (
        sequences.get(record["protein_id"], kind="protein")
        if sequences and record["protein_id"]
        else None
    )
    if rna is not None and len(rna) != record["transcript_length"]:
        raise ValueError(f"RNA/exon length mismatch for {record['transcript_id']}")
    record.update(
        sequence=rna,
        protein_sequence=protein,
        sequence_hash=hashlib.sha256(rna.encode()).hexdigest() if rna else None,
        protein_sequence_hash=hashlib.sha256(protein.encode()).hexdigest() if protein else None,
    )
    return record


def iter_transcripts(
    path: str | Path,
    *,
    release: str = "50",
    sequences: SequenceStore | None = None,
    chromosomes: Sequence[str] | None = REFERENCE_CHROMOSOMES,
    accounting: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Stream every transcript, requiring GENCODE's contiguous gene ordering.

    The parser never chooses one transcript or restricts by biotype/support tag.
    A repeated gene block fails rather than silently emitting duplicate IDs.
    """
    allowed = set(chromosomes) if chromosomes is not None else None
    counts = accounting if accounting is not None else {}
    counts.update(
        input_transcript_features=0, selected_transcripts=0, excluded_nonreference_transcripts=0
    )
    current: str | None = None
    seen: set[str] = set()
    records: dict[str, dict[str, Any]] = {}
    gene_symbol = ""
    for feature in iter_gtf_features(path):
        attrs = feature["attributes"]
        if feature["feature_type"] == "transcript":
            counts["input_transcript_features"] += 1
            if allowed is not None and feature["chromosome"] not in allowed:
                counts["excluded_nonreference_transcripts"] += 1
            else:
                counts["selected_transcripts"] += 1
        gene = _first(attrs, "gene_id")
        if not gene:
            raise ValueError("GENCODE feature has no gene_id")
        if current != gene:
            for record in records.values():
                yield _finish(record, sequences)
            records = {}
            if gene in seen:
                raise ValueError(f"GENCODE gene blocks must be contiguous: {gene}")
            seen.add(gene)
            current, gene_symbol = gene, _first(attrs, "gene_name", "")
        if allowed is not None and feature["chromosome"] not in allowed:
            continue
        tx = _first(attrs, "transcript_id")
        if not tx:
            continue
        stable, version = split_id(tx)
        if tx not in records:
            records[tx] = {
                "transcript_id": tx,
                "stable_id": stable,
                "transcript_version": version,
                "gene_id": split_id(gene)[0],
                "gene_id_versioned": gene,
                "gene_symbol": _first(attrs, "gene_name", gene_symbol),
                "transcript_name": "",
                "biotype": "",
                "protein_id": None,
                "chromosome": feature["chromosome"],
                "start": feature["start"],
                "end": feature["end"],
                "strand": feature["strand"],
                "tags": [],
                "attributes": {},
                "exons": [],
                "cds": [],
                "utrs": [],
                "start_codons": [],
                "stop_codons": [],
                "source": {
                    "name": "gencode",
                    "release": str(release),
                    "assembly": "GRCh38.p14",
                    "coordinates": "0-based-half-open",
                },
            }
        record = records[tx]
        if feature["feature_type"] == "transcript":
            record.update(
                start=feature["start"],
                end=feature["end"],
                biotype=_first(attrs, "transcript_type", ""),
                transcript_name=_first(attrs, "transcript_name", ""),
                protein_id=_first(attrs, "protein_id"),
                tags=sorted(set(attrs.get("tag", []))),
                attributes=attrs,
            )
        else:
            if _first(attrs, "protein_id"):
                record["protein_id"] = _first(attrs, "protein_id")
            target = {
                "exon": "exons",
                "CDS": "cds",
                "UTR": "utrs",
                "start_codon": "start_codons",
                "stop_codon": "stop_codons",
            }.get(feature["feature_type"])
            if target:
                record[target].append(
                    {
                        "start": feature["start"],
                        "end": feature["end"],
                        "phase": feature["phase"],
                        "exon_id": _first(attrs, "exon_id"),
                        "attributes": attrs,
                    }
                )
    for record in records.values():
        yield _finish(record, sequences)


def iter_metadata(path: str | Path, *, kind: str, release: str = "50") -> Iterator[dict[str, Any]]:
    """Keep original metadata values and their scope; HGNC PubMed links are gene context."""
    with open_text(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            identifier, *values = line.rstrip("\n").split("\t")
            yield {
                "transcript_id": identifier,
                "kind": kind,
                "values": values,
                "source": "gencode",
                "release": str(release),
                "annotation_release": str(release),
                "scope": "gene" if kind == "Pubmed_id" else "transcript",
            }


def query_transcript(
    identifier: str, *, path: str | Path, release: str = "50"
) -> list[dict[str, Any]]:
    """Targeted lookup in a pinned local GTF; stable IDs can return multiple loci."""
    return [
        r
        for r in iter_transcripts(path, release=release)
        if identifier in {r["transcript_id"], r["stable_id"]}
    ]
