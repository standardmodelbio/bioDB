"""Rfam client — RNA family models, seed alignments, and genome hits with structure.

`Rfam <https://rfam.org>`_ is the database of RNA families, each a seed
alignment with a consensus secondary structure and a covariance model (CM).
This module exposes the release files and, through Infernal, per-region
annotation of a genome with the structure of every hit:

* :func:`download_models` / :func:`download_seeds` / :func:`download_clanin`
  — ``Rfam.cm.gz`` (every family's covariance model), ``Rfam.seed.gz``
  (every seed alignment with its ``SS_cons``) and ``Rfam.clanin`` (clan
  membership, for overlap resolution), from the EBI FTP at a pinned release.
* :func:`press_models` — ``cmpress`` the models once, so ``cmscan`` can run.
* :func:`scan_fasta` — ``cmscan --cut_ga --rfam`` over a FASTA, returning
  the hits as a table (accession, family, coordinates, strand, score,
  E-value, and the hit's own alignment with structure via ``--fmt 2``
  ``tblout`` plus the alignment output).
* :func:`load_hits` — parse a ``cmscan --tblout --fmt 2`` file to a
  polars table with 0-based half-open coordinates on the sequence scanned.
* :func:`hit_structures` — for each hit, the consensus structure projected
  onto the hit's own bases (WUSS, one symbol per aligned base), parsed
  from the ``cmscan`` alignment output; what a site-level covariation
  test needs.
* :func:`seed_structure` — one family's seed alignment and ``SS_cons``.

Infernal (``cmscan``, ``cmpress``) must be on ``PATH``; it is BSD-licensed
and available from Bioconda and ``brewsci/bio``. Rfam is CC0. Cached files
live at ``~/.cache/biodb/rfam/``.

Examples
--------
>>> from biodb.rfam import download_models, press_models, scan_fasta
>>> press_models(download_models())                       # doctest: +SKIP
>>> hits = scan_fasta("chr19.fa", genome_size_mb=117.2)   # doctest: +SKIP
>>> hits.filter(pl.col("rfam_id") == "mir-290").height    # doctest: +SKIP
"""

from __future__ import annotations

import gzip
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from biodb._downloads import stream_to_file

logger = logging.getLogger(__name__)

RFAM_FTP = "https://ftp.ebi.ac.uk/pub/databases/Rfam"
"""EBI FTP root for Rfam releases."""

DEFAULT_RELEASE = "CURRENT"
"""Release directory to read. ``CURRENT`` follows the latest (15.1 as of
January 2026); pin a version such as ``"15.1"`` for reproducibility. The
downloaded files' SHA-256 is what a downstream record should carry."""

CACHE_DIR = Path("~/.cache/biodb/rfam").expanduser()
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _url(release: str, name: str) -> str:
    return f"{RFAM_FTP}/{release}/{name}"


def _fetch(
    name: str, *, release: str, cache_dir: str | Path | None, force: bool, progress: bool
) -> Path:
    cache = Path(cache_dir or CACHE_DIR).expanduser() / release
    cache.mkdir(parents=True, exist_ok=True)
    dst = cache / name
    if dst.exists() and not force:
        return dst
    return stream_to_file(_url(release, name), dst, progress=progress, desc=f"{release}/{name}")


def download_models(
    *,
    release: str = DEFAULT_RELEASE,
    cache_dir: str | Path | None = None,
    force: bool = False,
    progress: bool = True,
) -> Path:
    """Download ``Rfam.cm.gz`` (all covariance models, ~44 MB)."""
    return _fetch(
        "Rfam.cm.gz", release=release, cache_dir=cache_dir, force=force, progress=progress
    )


def download_seeds(
    *,
    release: str = DEFAULT_RELEASE,
    cache_dir: str | Path | None = None,
    force: bool = False,
    progress: bool = True,
) -> Path:
    """Download ``Rfam.seed.gz`` (all seed alignments with ``SS_cons``, ~6 MB)."""
    return _fetch(
        "Rfam.seed.gz", release=release, cache_dir=cache_dir, force=force, progress=progress
    )


def download_clanin(
    *,
    release: str = DEFAULT_RELEASE,
    cache_dir: str | Path | None = None,
    force: bool = False,
    progress: bool = True,
) -> Path:
    """Download ``Rfam.clanin`` (clan membership for ``cmscan --clanin``)."""
    return _fetch(
        "Rfam.clanin", release=release, cache_dir=cache_dir, force=force, progress=progress
    )


def _require(tool: str) -> str:
    found = shutil.which(tool)
    if found is None:
        raise RuntimeError(
            f"{tool!r} not found on PATH; install Infernal (conda install -c bioconda infernal, "
            "or brew install brewsci/bio/infernal)"
        )
    return found


def press_models(models_gz: str | Path, *, force: bool = False) -> Path:
    """Decompress ``Rfam.cm.gz`` beside itself and ``cmpress`` it; returns ``Rfam.cm``.

    Idempotent: skips when the four ``.i1*`` index files exist unless ``force``.
    """
    models_gz = Path(models_gz)
    models = models_gz.with_suffix("")  # Rfam.cm
    if not models.exists() or force:
        with gzip.open(models_gz, "rb") as src, models.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    indexed = all(
        models.with_name(f"Rfam.cm.{ext}").exists() for ext in ("i1m", "i1i", "i1f", "i1p")
    )
    if not indexed or force:
        subprocess.run([_require("cmpress"), "-F", str(models)], check=True, capture_output=True)
    return models


@dataclass(frozen=True)
class ScanResult:
    """Where ``scan_fasta`` left its outputs."""

    tblout: Path
    alignments: Path
    hits: pl.DataFrame


def scan_fasta(
    fasta: str | Path,
    *,
    models: str | Path | None = None,
    clanin: str | Path | None = None,
    genome_size_mb: float,
    output_dir: str | Path | None = None,
    threads: int = 4,
    release: str = DEFAULT_RELEASE,
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> ScanResult:
    """Annotate a FASTA with every Rfam family (``cmscan --cut_ga --rfam``).

    Parameters
    ----------
    fasta : path
        Sequence(s) to scan. Coordinates in the result are on these records.
    models, clanin : path, optional
        A pressed ``Rfam.cm`` and ``Rfam.clanin``; downloaded and pressed
        from ``release`` when omitted.
    genome_size_mb : float
        The ``-Z`` database size Rfam's gathering thresholds assume: twice
        the genome length in megabases (both strands), e.g. ``117.2`` for
        human chr19 (58.6 Mb) or ``6200`` for a whole human genome.
    output_dir : path, optional
        Where ``<fasta stem>.tblout`` and ``<fasta stem>.cmscan.txt`` go
        (default: beside the FASTA).
    threads : int
        ``--cpu``.

    Returns
    -------
    ScanResult
        The two output paths and the parsed hits (see :func:`load_hits`).
    """
    fasta = Path(fasta)
    if models is None:
        models = press_models(
            download_models(release=release, cache_dir=cache_dir, force=force, progress=False)
        )
    if clanin is None:
        clanin = download_clanin(release=release, cache_dir=cache_dir, force=force, progress=False)
    out = Path(output_dir or fasta.parent)
    out.mkdir(parents=True, exist_ok=True)
    tblout = out / f"{fasta.stem}.tblout"
    alignments = out / f"{fasta.stem}.cmscan.txt"
    if not (tblout.exists() and alignments.exists()) or force:
        command = [
            _require("cmscan"),
            "--cpu",
            str(threads),
            "-Z",
            str(genome_size_mb),
            "--cut_ga",
            "--rfam",
            "--nohmmonly",
            "--tblout",
            str(tblout),
            "--fmt",
            "2",
            "--clanin",
            str(clanin),
            "-o",
            str(alignments),
            str(models),
            str(fasta),
        ]
        logger.info("Running: %s", " ".join(command))
        subprocess.run(command, check=True, capture_output=True)
    return ScanResult(tblout=tblout, alignments=alignments, hits=load_hits(tblout))


_TBLOUT_COLUMNS = (
    "idx",
    "target_name",
    "accession",
    "query_name",
    "query_accession",
    "clan_name",
    "mdl",
    "mdl_from",
    "mdl_to",
    "seq_from",
    "seq_to",
    "strand",
    "trunc",
    "pass",
    "gc",
    "bias",
    "score",
    "evalue",
    "inc",
    "olp",
    "anyidx",
    "afrct1",
    "afrct2",
    "winidx",
    "wfrct1",
    "wfrct2",
)


def load_hits(tblout: str | Path, *, drop_overlapped: bool = True) -> pl.DataFrame:
    """Parse a ``cmscan --tblout --fmt 2`` file.

    Returns
    -------
    polars.DataFrame
        One row per hit: ``rfam_acc``, ``rfam_id``, ``clan``, ``seq_name``,
        ``start``, ``end`` (0-based half-open on the scanned sequence,
        strand-normalised so ``start < end``), ``strand``, ``mdl_from``,
        ``mdl_to``, ``score``, ``evalue``, ``truncated``, ``overlap`` (the
        ``olp`` flag: ``*`` none, ``^`` best of an overlapping clan, ``=``
        overlapped and normally dropped). With ``drop_overlapped`` (default)
        rows marked ``=`` are removed.
    """
    rows: list[dict[str, object]] = []
    with Path(tblout).open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.split()
            if len(fields) < 26:
                continue
            record = dict(zip(_TBLOUT_COLUMNS, fields[:26], strict=True))
            seq_from, seq_to = int(record["seq_from"]), int(record["seq_to"])
            strand = str(record["strand"])
            lo, hi = (seq_from, seq_to) if strand == "+" else (seq_to, seq_from)
            rows.append(
                {
                    "rfam_acc": record["accession"],
                    "rfam_id": record["target_name"],
                    "clan": None if record["clan_name"] == "-" else record["clan_name"],
                    "seq_name": record["query_name"],
                    "start": lo - 1,
                    "end": hi,
                    "strand": strand,
                    "mdl_from": int(record["mdl_from"]),
                    "mdl_to": int(record["mdl_to"]),
                    "score": float(record["score"]),
                    "evalue": float(record["evalue"]),
                    "truncated": record["trunc"] != "no",
                    "overlap": record["olp"],
                }
            )
    frame = pl.DataFrame(
        rows,
        schema={
            "rfam_acc": pl.Utf8,
            "rfam_id": pl.Utf8,
            "clan": pl.Utf8,
            "seq_name": pl.Utf8,
            "start": pl.Int64,
            "end": pl.Int64,
            "strand": pl.Utf8,
            "mdl_from": pl.Int64,
            "mdl_to": pl.Int64,
            "score": pl.Float64,
            "evalue": pl.Float64,
            "truncated": pl.Boolean,
            "overlap": pl.Utf8,
        },
    )
    if drop_overlapped:
        frame = frame.filter(pl.col("overlap") != "=")
    return frame.sort(["seq_name", "start"])


_HIT_HEADER = re.compile(r"^>> (\S+)\s")
_QUERY_HEADER = re.compile(r"^Query:\s+(\S+)")
_RANK_LINE = re.compile(r"^\s+\((\d+)\)\s+[!?]\s")


_SEQ_LINE = re.compile(r"^(\s*\S+\s+\d+\s)(\S.*?)\s+(\d+)\s*$")
_LOCAL_END = re.compile(r"\*\[\s*(\d+)\]\*")


def _sequence_span(line: str) -> tuple[int, int, str, int, int] | None:
    """`(column_start, column_end, sequence, from, to)` of a model or target
    line, found by position so a local-end marker with an internal space
    (``*[ 6]*``) is not split."""
    m = _SEQ_LINE.match(line)
    if not m:
        return None
    prefix, seq, to = m.group(1), m.group(2), int(m.group(3))
    start = len(prefix)
    frm = int(prefix.split()[1])
    return start, start + len(seq), seq, frm, to


def _project(cs: str, model_seq: str, target_seq: str) -> str:
    """The consensus structure over the target's own bases (one symbol per base).

    Column by column: a target gap (``-``) drops the model's symbol; a
    model gap or insert column (``.``, ``-``, ``~``) makes the target base
    an insertion with ``.``; a local-end marker ``*[N]*``, which occupies
    the same columns on every line, stands for ``N`` unaligned target bases
    and yields ``N`` dots.
    """
    structure: list[str] = []
    k = 0
    while k < len(target_seq):
        marker = _LOCAL_END.match(target_seq, k)
        if marker:
            structure.extend("." * int(marker.group(1)))
            k = marker.end()
            continue
        base = target_seq[k]
        if base == "-":
            k += 1
            continue
        model_base = model_seq[k] if k < len(model_seq) else "-"
        symbol = cs[k] if k < len(cs) else "."
        structure.append("." if model_base in ".-~" else symbol)
        k += 1
    return "".join(structure)


def hit_structures(alignments: str | Path) -> pl.DataFrame:
    """Per-hit consensus structure over the hit's own bases, from ``cmscan -o``.

    Infernal prints each hit's alignment in wrapped segments of six lines:
    ``NC``, ``CS`` (WUSS consensus structure), the model consensus, the match
    line, the target sequence and ``PP``. The segments are concatenated per
    hit and the structure projected onto the *target* bases, so the result
    has exactly one symbol per hit base and a bracket at hit index ``i``
    pairs with the bracket at hit index ``j``. Hit index 0 is the first
    printed target base: coordinate ``seq_from`` on the ``+`` strand, which
    on the ``-`` strand is the *higher* coordinate.

    Returns
    -------
    polars.DataFrame
        ``seq_name``, ``rfam_id``, ``rank`` (within the family's hit list),
        ``seq_from``, ``seq_to`` (1-based, as printed, strand-ordered),
        ``strand``, ``structure``.
    """
    rows: list[dict[str, object]] = []
    query: str | None = None
    family: str | None = None
    rank = 0
    cs_parts: list[str] = []
    model_parts: list[str] = []
    target_parts: list[str] = []
    seq_from: int | None = None
    seq_to: int | None = None

    def flush() -> None:
        nonlocal cs_parts, model_parts, target_parts, seq_from, seq_to
        if family and query and target_parts and seq_from is not None and seq_to is not None:
            rows.append(
                {
                    "seq_name": query,
                    "rfam_id": family,
                    "rank": rank,
                    "seq_from": seq_from,
                    "seq_to": seq_to,
                    "strand": "+" if seq_to >= seq_from else "-",
                    "structure": _project(
                        "".join(cs_parts), "".join(model_parts), "".join(target_parts)
                    ),
                }
            )
        cs_parts, model_parts, target_parts = [], [], []
        seq_from, seq_to = None, None

    lines = Path(alignments).read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        q = _QUERY_HEADER.match(line)
        if q:
            flush()
            query = q.group(1)
        h = _HIT_HEADER.match(line)
        if h:
            flush()
            family = h.group(1)
            rank = 0
        r = _RANK_LINE.match(line)
        if r and family is not None:
            flush()
            rank = int(r.group(1))
        if line.rstrip().endswith(" CS") and family is not None and i + 3 < len(lines):
            # CS line -> model line -> match line -> target line, all sharing
            # one column layout: slice the CS at the model's sequence columns.
            model = _sequence_span(lines[i + 1])
            target = _sequence_span(lines[i + 3])
            if model and target:
                col_start, col_end, model_seq, _, _ = model
                _, _, target_seq, frm, to = target
                cs_parts.append(line[col_start:col_end])
                model_parts.append(model_seq)
                target_parts.append(target_seq)
                if seq_from is None:
                    seq_from = frm
                seq_to = to
            i += 4
            continue
        i += 1
    flush()
    return pl.DataFrame(
        rows,
        schema={
            "seq_name": pl.Utf8,
            "rfam_id": pl.Utf8,
            "rank": pl.Int64,
            "seq_from": pl.Int64,
            "seq_to": pl.Int64,
            "strand": pl.Utf8,
            "structure": pl.Utf8,
        },
    )


def seed_structure(
    accession: str,
    *,
    release: str = DEFAULT_RELEASE,
    cache_dir: str | Path | None = None,
    force: bool = False,
    progress: bool = True,
) -> tuple[dict[str, str], str]:
    """One family's seed alignment (name -> aligned sequence) and its ``SS_cons``."""
    seeds = download_seeds(release=release, cache_dir=cache_dir, force=force, progress=progress)
    sequences: dict[str, str] = {}
    structure: list[str] = []
    inside = False
    with gzip.open(seeds, "rt", encoding="latin-1") as handle:
        for line in handle:
            if line.startswith("#=GF AC"):
                inside = line.split()[2] == accession
                sequences, structure = {}, []
                continue
            if not inside:
                continue
            if line.startswith("#=GC SS_cons"):
                structure.append(line.split(None, 2)[2].strip())
            elif line.startswith("//"):
                return sequences, "".join(structure)
            elif line.startswith("#") or not line.strip():
                continue
            else:
                name, _, chunk = line.partition(" ")
                sequences[name] = sequences.get(name, "") + chunk.strip()
    raise KeyError(f"{accession} not found in {seeds.name}")
