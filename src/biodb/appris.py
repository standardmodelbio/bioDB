"""APPRIS/TRIFID release-pinned bulk annotation and targeted cached lookups."""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from biodb.transcript_io import download_source, iter_gtf_features, open_text

BASE_URL = "https://apprisws.bioinfo.cnio.es/pub/current_release/datafiles/homo_sapiens"
# These are annotation correspondences, not an inference from APPRIS's v suffix.
ANNOTATION_RELEASES = {"e116v50": "50", "e115v50": "49", "e114v50": "48"}
FILES = {
    "scores": "appris_data.appris.txt",
    "trifid": "appris_method.trifid.txt",
    "principal": "appris_data.principal.txt",
    "features": "appris_method.appris.gtf.gz",
    "translations": "appris_data.transl.fa.gz",
    "checksums": "md5checksums.txt",
}
FEATURE_METHODS = (
    "appris",
    "corsair",
    "corsair_alt",
    "crash",
    "firestar",
    "matador3d",
    "matador3d2",
    "proteo",
    "spade",
    "thump",
)
FILES.update({method: f"appris_method.{method}.gtf.gz" for method in FEATURE_METHODS})


def download_features(
    *, dataset: str = "e116v50", cache_dir: str | Path | None = None
) -> dict[str, Path]:
    """Download every method GTF; appris_data.annot is the input annotation."""
    return {
        method: get_dataset(method, dataset=dataset, cache_dir=cache_dir)
        for method in FEATURE_METHODS
    }


def get_dataset(
    kind: str = "scores",
    *,
    dataset: str = "e116v50",
    cache_dir: str | Path | None = None,
) -> Path:
    """Download original annotation bytes with source provenance."""
    if kind not in FILES or dataset not in ANNOTATION_RELEASES:
        raise ValueError(f"unsupported APPRIS kind/dataset: {kind!r}/{dataset!r}")
    root = Path(cache_dir or "~/.cache/biodb/appris").expanduser() / dataset
    filename = FILES[kind]
    return download_source(
        f"{BASE_URL}/{dataset}/{filename}",
        root / filename,
        source="appris",
        release=dataset,
        terms_url="https://appris.bioinfo.cnio.es/#/about",
    )


def _optional(value: str | None) -> str | None:
    return None if value in {None, "", "-", "NA"} else value


def iter_scores(path: str | Path, *, dataset: str = "e116v50") -> Iterator[dict[str, Any]]:
    """Stream all score fields; unknown scores remain null, not zero."""
    with open_text(path) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            scores: dict[str, Any] = {}
            for key, value in row.items():
                value = _optional(value)
                try:
                    scores[key] = float(value) if value is not None else None
                except ValueError:
                    scores[key] = value
            yield {
                "transcript_id": row.get("Transcript ID") or row.get("transcript_id"),
                "gene_id": row.get("Ensembl Gene ID") or row.get("gene_id"),
                "protein_id": _optional(row.get("Translation ID") or row.get("translation_id")),
                "annotation": row.get("APPRIS Annotation"),
                "scores": scores,
                "source": "appris",
                "release": dataset,
                "annotation_release": ANNOTATION_RELEASES[dataset],
                "scope": "protein",
                "evidence_type": "predicted",
            }


def iter_features(path: str | Path, *, dataset: str = "e116v50") -> Iterator[dict[str, Any]]:
    """Stream genome-mapped APPRIS method features (0-based half-open)."""
    for row in iter_gtf_features(path):
        attrs = row["attributes"]
        ids = attrs.get("transcript_id", [])
        if not ids:
            continue
        for identifier in ids:
            yield {
                **row,
                "transcript_id": identifier,
                "method": row["source"],
                "source": "appris",
                "release": dataset,
                "annotation_release": ANNOTATION_RELEASES[dataset],
                "scope": "protein",
                "evidence_type": "measured" if row["source"].lower() == "proteo" else "predicted",
            }


def query_transcript(
    identifier: str,
    *,
    dataset: str = "e116v50",
    path: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Look up the ID in a pinned downloaded release (no live-version drift)."""
    from biodb.transcript_io import split_id

    scores = path or get_dataset(dataset=dataset, cache_dir=cache_dir)
    return [
        r
        for r in iter_scores(scores, dataset=dataset)
        if r["transcript_id"] in {identifier, split_id(identifier)[0]}
    ]
