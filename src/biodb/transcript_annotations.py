"""Validated, provenance-bearing transcript biology from additional datasets.

This interchange complements the original-format adapters. It does not infer
identity: the IsoDocs composer must still establish each row's transcript/RNA
or protein correspondence against the chosen backbone.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from biodb.transcript_io import open_text

BIOLOGY_KINDS = frozenset({"RG", "RN", "PF", "FN", "LO", "IN", "BC", "TF", "CL", "EX", "DI"})
EVIDENCE_TYPES = frozenset(
    {
        "measured",
        "experimental",
        "curated",
        "predicted",
        "inferred",
        "propagated",
        "source_annotation",
        "sequence_derived",
        "mixed",
        "unknown",
    }
)
SCOPES = frozenset({"transcript", "protein", "gene"})


def iter_annotations(path: str | Path, *, release: str) -> Iterator[dict[str, Any]]:
    """Read JSONL with explicit assertion scope, evidence class and provenance.

    Each row needs an identity-bearing transcript_id, a declared release, a
    transcript/protein mapping scope and a nonempty annotations list. Each
    assertion carries its own biological scope and source URL. Optional
    conditions, publications, metrics, sequence/structure and other original
    fields are retained verbatim.
    """
    with open_text(path) as fh:
        for number, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not row.get("transcript_id"):
                raise ValueError(f"annotation row {number} lacks transcript identity")
            if row.get("release") != release:
                raise ValueError(f"annotation row {number} release differs from snapshot")
            if row.get("scope") not in {"transcript", "protein"}:
                raise ValueError(f"annotation row {number} needs transcript/protein mapping scope")
            annotations = row.get("annotations")
            if not isinstance(annotations, list) or not annotations:
                raise ValueError(f"annotation row {number} has no assertions")
            for assertion in annotations:
                if assertion.get("scope") not in SCOPES:
                    raise ValueError(f"annotation row {number} has invalid assertion scope")
                if row["scope"] == "protein" and assertion["scope"] == "transcript":
                    raise ValueError("protein identity cannot establish transcript assertion scope")
                if assertion["scope"] == "protein":
                    if row["scope"] != "protein":
                        raise ValueError(
                            "protein assertions require a separate protein mapping scope"
                        )
                    if (
                        not isinstance(row.get("protein_sequence"), str)
                        or not row["protein_sequence"]
                    ):
                        raise ValueError(
                            "protein assertions require a protein sequence for correspondence"
                        )
                if (
                    assertion.get("kind") not in BIOLOGY_KINDS
                    or not assertion.get("text", "").strip()
                ):
                    raise ValueError(f"annotation row {number} needs a biological kind and text")
                if assertion.get("evidence_type") not in EVIDENCE_TYPES:
                    raise ValueError(f"annotation row {number} has invalid evidence type")
                parsed = urlparse(assertion.get("source_url", ""))
                if parsed.scheme not in {"https", "http"} or not parsed.netloc:
                    raise ValueError(f"annotation row {number} lacks source provenance URL")
            yield row
