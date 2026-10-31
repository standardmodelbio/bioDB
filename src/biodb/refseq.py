"""Curated transcript descriptions from RefSeq GenBank, with gene context separate."""

from __future__ import annotations

import re
from collections.abc import Iterator
from io import StringIO
from pathlib import Path
from typing import Any

import requests

from biodb.transcript_io import download_source, open_text

EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
TRANSCRIPT_BIOLOGY_SCHEMA = "refseq-rich-v1"


def _paragraph(comment: str, label: str) -> str:
    labels = r"Summary|Transcript Variant|Publication Note|Sequence Note|COMPLETENESS|PRIMARY_SPAN"
    match = re.search(
        rf"(?:^|\s){re.escape(label)}:\s*(.*?)(?=\n\s*\n|\s+(?:{labels}):|\Z)",
        comment,
        re.S,
    )
    return " ".join(match[1].split()) if match else ""


def _biological_features(record: Any) -> list[dict[str, Any]]:
    """Expose native-RNA annotations; preserve propagation/fuzzy locations.

    Feature presence is a source annotation, not automatically a measurement.
    Protein features projected onto RNA coordinates retain that distinction.
    """
    from Bio.SeqFeature import ExactPosition, UnknownPosition

    annotations = []
    for feature in record.features:
        qualifiers = dict(feature.qualifiers)
        if feature.type in {"source", "gene", "exon", "mRNA"}:
            continue
        if feature.type == "CDS" and not any(
            key in qualifiers
            for key in ("exception", "transl_except", "ribosomal_slippage", "note")
        ):
            continue
        details = [
            f"{key}: {value}"
            for key in (
                "regulatory_class",
                "ncRNA_class",
                "product",
                "note",
                "function",
                "bound_moiety",
                "exception",
                "transl_except",
                "experiment",
                "inference",
            )
            for value in qualifiers.get(key, [])
        ]
        if not details and feature.type not in {"polyA_site", "polyA_signal", "stem_loop"}:
            continue
        text = "; ".join(details)
        propagated = "propagated from uniprot" in text.casefold()
        kind = (
            "PF"
            if propagated
            else "RG"
            if feature.type
            in {
                "regulatory",
                "polyA_site",
                "polyA_signal",
                "protein_bind",
                "misc_binding",
                "promoter",
                "enhancer",
                "RBS",
            }
            else "CL"
            if feature.type == "variation"
            else "RN"
        )
        evidence = (
            "propagated"
            if propagated
            else "mixed"
            if qualifiers.get("experiment") and qualifiers.get("inference")
            else "experimental"
            if qualifiers.get("experiment")
            else "inferred"
            if qualifiers.get("inference")
            else "source_annotation"
        )
        location: dict[str, Any] = {
            "reference": record.id,
            "coordinate_system": "0-based-half-open RNA",
            "original": str(feature.location),
            "parts": [],
            "operator": getattr(feature.location, "operator", None),
        }
        if feature.location is not None:
            for part in feature.location.parts:
                location["parts"].append(
                    {
                        "start": None
                        if isinstance(part.start, UnknownPosition)
                        else int(part.start),
                        "end": None if isinstance(part.end, UnknownPosition) else int(part.end),
                        "strand": part.strand,
                        "start_exact": isinstance(part.start, ExactPosition),
                        "end_exact": isinstance(part.end, ExactPosition),
                    }
                )
        annotations.append(
            {
                "kind": kind,
                "text": f"{feature.type}: {text or feature.type}",
                "scope": "transcript",
                "evidence_type": evidence,
                "location": location,
                "qualifiers": qualifiers,
                "source_url": f"https://www.ncbi.nlm.nih.gov/nuccore/{record.id}",
                "source_feature_type": feature.type,
            }
        )
    for name, value in (
        record.annotations.get("structured_comment", {}).get("RefSeq-Attributes", {}).items()
    ):
        label = name.casefold()
        if "localized" in label:
            kind = "LO"
        elif "polyadenyl" in label or "regulatory" in label:
            kind = "RG"
        elif "decay" in label:
            kind = "TF"
        else:
            continue  # preserve all other structured attributes in raw custody
        annotations.append(
            {
                "kind": kind,
                "text": f"RefSeq attribute {name}: {value}",
                "scope": "gene" if label.startswith("gene product") else "transcript",
                "evidence_type": "inferred" if "infer" in value.casefold() else "source_annotation",
                "source_url": f"https://www.ncbi.nlm.nih.gov/nuccore/{record.id}",
                "qualifiers": {name: value},
            }
        )
    return annotations


def parse_record(record: Any, *, release: str = "unspecified") -> dict[str, Any]:
    """Separate transcript-variant assertions from broader RefSeq summaries."""
    comment = record.annotations.get("comment", "")
    return {
        "transcript_id": record.id,
        "description": record.description,
        "sequence": str(record.seq),
        "variant_description": _paragraph(comment, "Transcript Variant"),
        "biological_annotations": _biological_features(record),
        "gene_context": _paragraph(comment, "Summary"),
        "comment": comment,
        "structured_comments": {
            key: dict(value)
            for key, value in record.annotations.get("structured_comment", {}).items()
        },
        "source": "refseq",
        "release": release,
        "scope": "transcript",
        "evidence_type": "curated" if "REVIEWED REFSEQ" in comment else "source_annotation",
        "publications": sorted(
            {
                str(ref.pubmed_id)
                for ref in record.annotations.get("references", [])
                if getattr(ref, "pubmed_id", None)
            }
        ),
        "features": [
            {"type": f.type, "location": str(f.location), "qualifiers": dict(f.qualifiers)}
            for f in record.features
        ],
    }


def iter_transcripts(path: str | Path, *, release: str) -> Iterator[dict[str, Any]]:
    from Bio import SeqIO

    with open_text(path) as fh:
        for record in SeqIO.parse(fh, "genbank"):
            yield parse_record(record, release=release)


def query_transcript(identifier: str, *, timeout: float = 30) -> dict[str, Any]:
    """Retrieve an exact versioned RefSeq RNA accession; never silently update it."""
    from Bio import SeqIO

    if not re.fullmatch(r"[NX][MR]_\d+\.\d+", identifier):
        raise ValueError("RefSeq query requires a versioned NM/NR/XM/XR accession")
    response = requests.get(
        EFETCH_URL,
        params={"db": "nuccore", "id": identifier, "rettype": "gb", "retmode": "text"},
        timeout=timeout,
    )
    response.raise_for_status()
    record = SeqIO.read(StringIO(response.text), "genbank")
    if record.id != identifier:
        raise ValueError(f"RefSeq returned {record.id!r} for {identifier!r}")
    return parse_record(record, release=f"accession:{identifier}")


def get_dataset(
    *,
    url: str | None = None,
    release: str = "GCF_000001405.40-RS_2025_08",
    cache_dir: str | Path | None = None,
) -> Path:
    """Cache a release-pinned original RNA GenBank dump from NCBI or MANE.

    The default directory was verified against the public release index.
    Other releases require an explicit URL, retained in the cache manifest.
    """
    from urllib.parse import urlparse

    if url is None:
        if release != "GCF_000001405.40-RS_2025_08":
            raise ValueError("an explicit NCBI bulk URL is required for this release")
        url = (
            "https://ftp.ncbi.nlm.nih.gov/genomes/all/annotation_releases/9606/"
            f"{release}/GCF_000001405.40_GRCh38.p14_rna.gbff.gz"
        )
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "ftp.ncbi.nlm.nih.gov":
        raise ValueError("RefSeq bulk URL must use the official NCBI HTTPS FTP host")
    filename = Path(parsed.path).name
    root = Path(cache_dir or "~/.cache/biodb/refseq").expanduser() / release
    return download_source(
        url,
        root / filename,
        source="refseq",
        release=release,
        terms_url="https://www.ncbi.nlm.nih.gov/home/about/policies/",
    )
