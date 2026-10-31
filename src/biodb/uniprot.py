"""UniProt REST client — protein sequences, features, cross-references.

Targeted REST queries return protein records, sequences, features and
cross-references. Bulk FTP helpers download Swiss-Prot or TrEMBL FASTA;
:func:`iter_fasta_records` parses the acquired files with Biopython.

Ported from `VEP_protein <https://github.com/bschilder/VEP_protein>`_'s
``src/unitprot.py`` with the following hygiene fixes:

* The VEP_protein version returned a Biopython ``SeqIO.parse`` iterator,
  which silently exhausts on first iteration — callers had to "reimport
  the records" to use them twice. Here we materialize to a list on
  every call so the return value is freely reusable.
* HTTP failures raise instead of printing and returning ``None``.
* Logging instead of bare ``print`` for verbose mode.
* Filename typo (``unitprot`` → ``uniprot``) fixed.

Endpoint: https://rest.uniprot.org/uniprotkb/

Examples
--------
>>> from biodb.uniprot import query_protein, get_features
>>> records = query_protein("P12345")  # doctest: +SKIP
>>> features = get_features("P12345")  # doctest: +SKIP
"""

from __future__ import annotations

import logging
import re
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

UNIPROT_REST_API = "https://rest.uniprot.org/uniprotkb"
"""UniProt REST root. Append ``/<accession>.<format>``."""

_DEFAULT_TIMEOUT_S: float = 30.0


def query_protein_json(
    uniprot_id: str, *, release: str | None = None, timeout_s: float = _DEFAULT_TIMEOUT_S
) -> dict[str, Any]:
    """Keep UniProt JSON qualifiers/evidence that a generic SeqRecord can lose."""
    response = requests.get(f"{UNIPROT_REST_API}/{uniprot_id}.json", timeout=timeout_s)
    response.raise_for_status()
    if release is not None and response.headers.get("X-UniProt-Release") != release:
        raise ValueError("UniProt live release differs from the requested release")
    return response.json()


def download_human_entries(*, release: str, cache_dir: str | Path | None = None) -> Path:
    """Cache reviewed human JSON entries, requiring the requested live release.

    The API is a live service. A release mismatch fails rather than attributing
    today's annotations to a different release; previously cached snapshots work.
    """
    from urllib.parse import urlencode

    from biodb.transcript_io import download_source

    query = urlencode(
        {"query": "(organism_id:9606) AND (reviewed:true)", "format": "json", "compressed": "true"}
    )
    root = Path(cache_dir or "~/.cache/biodb/uniprot").expanduser() / release
    return download_source(
        f"{UNIPROT_REST_API}/stream?{query}",
        root / "human_reviewed.json.gz",
        source="uniprot",
        release=release,
        expected_headers={"X-UniProt-Release": release},
        terms_url="https://www.uniprot.org/help/license",
    )


def iter_json_entries(path: str | Path):
    """Stream the API's results array without loading the human dump in RAM."""
    import gzip

    import ijson

    path = Path(path)
    with gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb") as fh:
        yield from ijson.items(fh, "results.item", use_float=True)


def _isoform_sequence(entry: dict[str, Any], isoform: dict[str, Any]) -> str | None:
    sequence = entry.get("sequence", {}).get("value")
    if isoform.get("isoformSequenceStatus", isoform.get("sequenceStatus")) == "Displayed":
        return sequence
    ids = set(isoform.get("sequenceIds", []))
    edits = [f for f in entry.get("features", []) if f.get("featureId") in ids]
    if not sequence or not ids or len(edits) != len(ids):
        return None
    for feature in sorted(edits, key=lambda f: f["location"]["start"]["value"], reverse=True):
        start, end = feature["location"]["start"]["value"] - 1, feature["location"]["end"]["value"]
        alternatives = feature.get("alternativeSequence", {}).get("alternativeSequences", [])
        # UniProt represents deleted segments by an empty alternatives list.
        if len(alternatives) > 1:
            return None
        replacement = alternatives[0] if alternatives else ""
        sequence = sequence[:start] + replacement + sequence[end:]
    return sequence


def iter_isoform_annotations(entry: dict[str, Any], *, release: str):
    """Yield isoform-aware annotations for each explicit Ensembl transcript xref.

    Broader comments are kept as gene_context, never copied into isoform
    function claims. Sequence features without a molecule qualifier refer to the
    displayed protein sequence, not automatically to all alternative products.
    """
    products = [
        c for c in entry.get("comments", []) if c.get("commentType") == "ALTERNATIVE PRODUCTS"
    ]
    isoforms = [i for product in products for i in product.get("isoforms", [])]
    by_id = {identifier: iso for iso in isoforms for identifier in iso.get("isoformIds", [])}
    text_labels = {}
    for iso in isoforms:
        names = [iso.get("name", {}).get("value", "")] + [
            s.get("value", "") for s in iso.get("synonyms", [])
        ]
        for name in names:
            if name:
                label = name.casefold()
                identifiers = tuple(iso.get("isoformIds", []))
                if label in text_labels and text_labels[label] != identifiers:
                    text_labels[label] = None
                else:
                    text_labels[label] = identifiers

    def comment_scope(comment):
        if comment.get("molecule"):
            return comment["molecule"]
        if comment.get("commentType") == "TISSUE SPECIFICITY":
            return None  # resolve comparative tissue prose clause by clause
        texts = [t["value"].casefold() for t in comment.get("texts", []) if t.get("value")]
        # Older comments encode a qualifier in prose instead of `molecule`.
        # Accept only an explicit, unambiguous prefix in EVERY text, never a
        # guessed function assignment from a general entry-level comment.
        candidates = [
            label
            for label in text_labels
            if texts and all(t.startswith(f"isoform {label} ") for t in texts)
        ]
        if len(candidates) != 1:
            return None
        label = candidates[0]
        if text_labels[label] is None:
            return None
        if any(
            text_labels[other] != text_labels[label]
            and any(f"isoform {other} " in text for text in texts)
            for other in text_labels
        ):
            return None
        return label

    scoped_comments = []
    for comment in entry.get("comments", []):
        scope = comment_scope(comment)
        scoped_comments.append((comment, scope))
        if scope or comment.get("commentType") != "TISSUE SPECIFICITY":
            continue
        # Some curated tissue paragraphs explicitly compare named forms in
        # separate sentences. Preserve the whole original as context and only
        # promote sentences naming exactly one unambiguous protein form.
        for text in comment.get("texts", []):
            fragments = [
                clause.strip()
                for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.get("value", ""))
                for clause in re.split(r"\b(?:while|whereas)\b", sentence, flags=re.I)
            ]
            for sentence in fragments:
                groups = {}
                ambiguous = False
                for label, identifiers in text_labels.items():
                    distinctive = bool(
                        re.search(r"[-_()]", label)
                        or (
                            len(label) >= 3
                            and re.search(r"[a-z]", label)
                            and re.search(r"\d", label)
                        )
                    )
                    pattern = (
                        (r"(?<!\w)" + re.escape(label) + r"(?!\w)")
                        if distinctive
                        else (r"\bisoform\s+" + re.escape(label) + r"(?!\w)")
                    )
                    if re.search(pattern, sentence, re.I):
                        if identifiers is None:
                            ambiguous = True
                        groups.setdefault(identifiers, label)
                if ambiguous or len(groups) != 1:
                    continue
                label = next(iter(groups.values()))
                scoped_comments.append(
                    (
                        {
                            **comment,
                            "texts": [{**text, "value": sentence}],
                            "molecule": "Isoform " + label,
                            "scope_basis": "explicit_named_isoform_clause"
                            if len(fragments) > 1
                            and re.search(r"\b(?:while|whereas)\b", text.get("value", ""), re.I)
                            else "explicit_named_isoform_sentence",
                            "original_text": text.get("value"),
                        },
                        label,
                    )
                )
    broader = [
        c
        for c, scope in scoped_comments
        if not scope and c.get("commentType") != "ALTERNATIVE PRODUCTS"
    ]
    for xref in entry.get("uniProtKBCrossReferences", []):
        if xref.get("database") != "Ensembl":
            continue
        isoform_id = xref.get("isoformId")
        iso = by_id.get(isoform_id, {})
        names = {str(iso.get("name", {}).get("value", "")), isoform_id or ""}
        names.update(s.get("value", "") for s in iso.get("synonyms", []))
        labels = {name.lower().removeprefix("isoform ") for name in names if name}

        def matches(molecule: str, labels: set[str] = labels) -> bool:
            return molecule.lower().removeprefix("isoform ") in labels

        comments = [
            {
                **c,
                **(
                    {"scope_basis": "explicit_isoform_text_prefix"} if not c.get("molecule") else {}
                ),
            }
            for c, scope in scoped_comments
            if scope and matches(scope)
        ]
        features = [
            f for f in entry.get("features", []) if f.get("molecule") and matches(f["molecule"])
        ]
        displayed = (
            not isoforms
            or iso.get("isoformSequenceStatus", iso.get("sequenceStatus")) == "Displayed"
        )
        if displayed:
            features += [
                {**f, "annotation_scope": "canonical_protein_sequence"}
                for f in entry.get("features", [])
                if not f.get("molecule")
            ]
        # Entries with no alternative products still have a displayed sequence.
        sequence = (
            _isoform_sequence(entry, iso)
            if iso
            else entry.get("sequence", {}).get("value")
            if not isoforms
            else None
        )
        yield {
            "transcript_id": xref["id"],
            "isoform_id": isoform_id,
            "accession": entry["primaryAccession"],
            "name": iso.get("name", {}).get("value"),
            "synonyms": [s["value"] for s in iso.get("synonyms", []) if s.get("value")],
            "comments": comments,
            "features": features,
            "gene_context": broader,
            "protein_sequence": sequence,
            "source": "uniprot",
            "release": release,
            "scope": "protein",
            "evidence_type": "curated",
            "xref": xref,
            "entry_audit": entry.get("entryAudit", {}),
        }


def _fetch_records(
    uniprot_id: str,
    fmt: str = "xml",
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> list[Any]:
    """GET ``{UNIPROT_REST_API}/{uniprot_id}.{fmt}`` and parse to a list
    of Biopython records.

    Raises
    ------
    requests.HTTPError
        On non-2xx response.
    """
    from Bio import SeqIO

    url = f"{UNIPROT_REST_API}/{uniprot_id}.{fmt}"
    response = requests.get(url, timeout=timeout_s)
    response.raise_for_status()
    parser = "uniprot-xml" if fmt == "xml" else fmt
    # Biopython's uniprot-xml reader requires binary mode (it calls
    # ``defusedxml.ElementTree.iterparse`` which expects bytes). Use
    # ``response.content`` + ``BytesIO``, not ``response.text`` + ``StringIO``.
    return list(SeqIO.parse(BytesIO(response.content), parser))


def query_protein(
    uniprot_id: str,
    *,
    fmt: str = "xml",
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    verbose: bool = False,
) -> list[Any]:
    """Fetch one protein from UniProt and return parsed Biopython records.

    Parameters
    ----------
    uniprot_id : str
        UniProt accession (e.g. ``"P12345"``).
    fmt : str, default "xml"
        Response format. ``"xml"`` is parsed with the
        ``uniprot-xml`` Biopython parser.
    timeout_s : float, default 30.0
    verbose : bool, default False
        Log each record's id/name/description.

    Returns
    -------
    list[Bio.SeqRecord.SeqRecord]
        Materialized list (not an iterator), so it can be re-traversed.
    """
    records = _fetch_records(uniprot_id, fmt=fmt, timeout_s=timeout_s)
    if verbose:
        for record in records:
            logger.info("id=%s name=%s desc=%s", record.id, record.name, record.description)
    return records


def get_sequences(uniprot_id: str, **kwargs: Any) -> list[Any]:
    """Sequences for ``uniprot_id`` as a list of ``Bio.Seq`` objects.

    Pass extra arguments through to :func:`query_protein`.
    """
    return [record.seq for record in query_protein(uniprot_id, **kwargs)]


def get_features(uniprot_id: str, **kwargs: Any) -> pd.DataFrame:
    """Protein features as a DataFrame.

    Columns
    -------
    ``id``, ``type``, ``start``, ``end``, ``length``, plus any feature
    qualifier columns (e.g. ``description``, ``evidence``) unpacked from
    the Biopython record.

    ``length`` is computed as ``end - start``.
    """
    records = query_protein(uniprot_id, **kwargs)
    rows: list[dict[str, Any]] = []
    for record in records:
        for feature in record.features:
            row: dict[str, Any] = {
                "id": feature.id,
                "type": feature.type,
                "start": feature.location.start,
                "end": feature.location.end,
            }
            row.update(feature.qualifiers)
            rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["length"] = df["end"] - df["start"]
    return df


def get_dbxrefs(uniprot_id: str, **kwargs: Any) -> pd.DataFrame:
    """Database cross-references for ``uniprot_id`` as a DataFrame.

    Columns
    -------
    dbxref : str  -- original ``"db:id"`` string
    db : str
    id : str
    """
    records = query_protein(uniprot_id, **kwargs)
    xrefs: list[str] = []
    for record in records:
        xrefs.extend(record.dbxrefs)
    df = pd.DataFrame({"dbxref": xrefs})
    if not df.empty:
        df[["db", "id"]] = df["dbxref"].str.split(":", n=1, expand=True)
    return df


# ─── Bulk Swiss-Prot / TrEMBL FASTA download ────────────────────────────────
# Pulls and streams the canonical UniProt Knowledgebase FASTA distributions:
#
#   Swiss-Prot (manually reviewed) — ``uniprot_sprot.fasta.gz`` (~90 MB)
#   TrEMBL (auto-annotated)        — ``uniprot_trembl.fasta.gz`` (~50 GB!)
#
# Use ``iter_fasta_records`` to stream without loading the full file into
# memory (essential for TrEMBL).

UNIPROT_FTP_BASE_URL = (
    "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete"
)
"""UniProt FTP knowledgebase root (HTTPS-served, ``ftp.uniprot.org``)."""

UNIPROT_FASTA_CACHE_DIR = Path("~/.cache/biodb/uniprot").expanduser()
UNIPROT_FASTA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

SWISSPROT_FASTA_FILENAME = "uniprot_sprot.fasta.gz"
TREMBL_FASTA_FILENAME = "uniprot_trembl.fasta.gz"


def download_swissprot_fasta(
    *,
    cache_dir: str | Path | None = None,
    force: bool = False,
    timeout_s: float = 600.0,
    progress: bool = True,
) -> Path:
    """Download the manually reviewed Swiss-Prot FASTA bundle (~90 MB gzipped).

    Cached locally under ``~/.cache/biodb/uniprot/`` by default.

    Parameters
    ----------
    cache_dir : str or Path, optional
    force : bool, default False
        Re-download even if cached.
    timeout_s : float, default 600
        Per-chunk timeout. Swiss-Prot is ~90 MB; allow plenty.
    progress : bool, default True
        Show a tqdm download bar.
    """
    return _download_uniprot_fasta(
        SWISSPROT_FASTA_FILENAME,
        cache_dir=cache_dir,
        force=force,
        timeout_s=timeout_s,
        progress=progress,
    )


def download_trembl_fasta(
    *,
    cache_dir: str | Path | None = None,
    force: bool = False,
    timeout_s: float = 3600.0,
    progress: bool = True,
) -> Path:
    """Download the auto-annotated TrEMBL FASTA bundle.

    .. warning::
       TrEMBL is **~50 GB compressed**. Don't call this on a laptop / CI runner
       without a plan. Use :func:`iter_fasta_records` to stream rather than
       loading the file into memory.
    """
    return _download_uniprot_fasta(
        TREMBL_FASTA_FILENAME,
        cache_dir=cache_dir,
        force=force,
        timeout_s=timeout_s,
        progress=progress,
    )


def _download_uniprot_fasta(
    filename: str,
    *,
    cache_dir: str | Path | None,
    force: bool,
    timeout_s: float,
    progress: bool = True,
) -> Path:
    """Stream ``<UNIPROT_FTP_BASE_URL>/<filename>`` to ``cache_dir/<filename>``."""
    from biodb._downloads import stream_to_file

    root = Path(cache_dir).expanduser() if cache_dir else UNIPROT_FASTA_CACHE_DIR
    root.mkdir(parents=True, exist_ok=True)
    dst = root / filename
    if dst.exists() and not force:
        return dst

    url = f"{UNIPROT_FTP_BASE_URL}/{filename}"
    logger.info("Downloading %s", url)
    return stream_to_file(
        url,
        dst,
        timeout=int(timeout_s),
        progress=progress,
        chunk_size=1 << 20,  # UniProt FASTA bundles are large; 1 MiB chunks reduce syscalls
    )


def iter_fasta_records(
    fasta_path: str | Path | None = None,
    *,
    swissprot: bool = True,
    cache_dir: str | Path | None = None,
):
    """Yield ``Bio.SeqRecord`` objects from a UniProt FASTA bundle.

    Memory-stable iteration: never materializes the full file in RAM, so
    works on TrEMBL too.

    Parameters
    ----------
    fasta_path : str or Path, optional
        Path to a UniProt FASTA file (gzipped or plain). If ``None``,
        downloads Swiss-Prot or TrEMBL according to ``swissprot``.
    swissprot : bool, default True
        When ``fasta_path`` is None, choose Swiss-Prot vs TrEMBL.
    cache_dir : str or Path, optional

    Yields
    ------
    Bio.SeqRecord.SeqRecord
        One record per FASTA entry.
    """
    import gzip

    from Bio import SeqIO

    if fasta_path is None:
        downloader = download_swissprot_fasta if swissprot else download_trembl_fasta
        fasta_path = downloader(cache_dir=cache_dir)
    fasta_path = Path(fasta_path)
    opener = gzip.open if str(fasta_path).endswith(".gz") else open
    with opener(fasta_path, "rt") as handle:
        yield from SeqIO.parse(handle, "fasta")


def count_swissprot_records(
    fasta_path: str | Path | None = None,
    *,
    cache_dir: str | Path | None = None,
) -> int:
    """Cheap streaming count of the records in a UniProt FASTA bundle.

    Convenience for verifying a download succeeded without buffering the
    whole file.
    """
    return sum(1 for _ in iter_fasta_records(fasta_path, cache_dir=cache_dir))
