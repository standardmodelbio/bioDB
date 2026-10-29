"""Per-base constraint tracks: UCSC phyloP / phastCons bigWigs and GPN-Star scores.

Two kinds of resource, one module, because a validation covariate is "how
constrained is this base" and a caller should not care which model said so.

**UCSC conservation bigWigs** (:data:`TRACKS`): phyloP and phastCons over the
hg38 alignments UCSC distributes -- the Zoonomia 241-way (``cactus241way``),
the 100-way vertebrate multiz, the 447-way mammal+primate alignment (whole
and primates-only) and the 470-way. A bigWig is random-access by design, so
:func:`open_track` reads a track **in place over HTTPS** and
:func:`score_intervals` / :func:`score_positions` pull only the bases asked
for; :func:`download_track` fetches the whole file (5-12 GB each) for callers
who will hit it hard. Requires the ``[bigwig]`` extra (``pyBigWig``).

**GPN-Star** (`Benegas et al. <https://doi.org/10.1101/2025.09.21.677619>`_,
Song lab): genome-wide, mutation-rate-calibrated constraint from a
phylogeny-informed genomic language model, published on the Hub at
``songlab/gpn-star-scores`` as chromosome-sharded Parquet (canonical) and
bigWigs (display). :func:`download_gpn_star` / :func:`load_gpn_star` read a
chromosome shard at a pinned revision: ``entropy`` is one row per position
(``entropy_calibrated``, ~1 neutral, lower = more constrained) and ``llr`` one
row per position x alternate allele (``llr_calibrated``, ``abs_llr_calibrated``).
Three hg38 models exist: ``m447`` (447 mammals, the default), ``v100``
(100 vertebrates) and ``p243`` (243 primates).

Coordinates: bigWig queries and every ``start``/``end`` column here are
0-based half-open (BED); GPN-Star's own ``pos`` is 1-based and is kept as
``pos`` beside a derived 0-based ``start``.

Examples
--------
>>> from biodb.constraint_tracks import open_track, score_intervals
>>> bw = open_track("phylop241")                                    # doctest: +SKIP
>>> score_intervals(bw, exons, stat="mean")                         # doctest: +SKIP
>>> from biodb.constraint_tracks import load_gpn_star
>>> load_gpn_star("chr19", positions=[1_000_001, 1_000_002])         # doctest: +SKIP
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from biodb._downloads import stream_to_file

logger = logging.getLogger(__name__)

UCSC_BASE = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38"


@dataclass(frozen=True)
class Track:
    """One published bigWig: where it is, what it scores, how big it is."""

    name: str
    url: str
    description: str
    assembly: str
    size_bytes: int
    source: str
    license: str


_UCSC_TERMS = (
    "UCSC Genome Browser downloads: free for all uses; conservation tracks are "
    "computed by UCSC from the named alignment (see each directory's README.txt)"
)
_GPN_STAR_TERMS = "Apache-2.0 (dataset card of songlab/gpn-star-scores)"

GPN_STAR_DATASET = "songlab/gpn-star-scores"
"""The Hugging Face dataset holding the GPN-Star genome-wide scores."""

GPN_STAR_REVISION = "a7b13bbf0d2338d74a7e5f0f8466e41ac0722f50"
"""Dataset revision (commit) read by default. Pin a full SHA for reproducibility."""

GPN_STAR_MODELS = {
    "m447": "gpn-star-hg38-m447-200m",
    "v100": "gpn-star-hg38-v100-200m",
    "p243": "gpn-star-hg38-p243-200m",
}
"""Short model keys to the score-set directories published for hg38."""

GPN_STAR_KINDS = ("entropy", "llr")


def _gpn_star_url(path: str, revision: str) -> str:
    return f"https://huggingface.co/datasets/{GPN_STAR_DATASET}/resolve/{revision}/{path}"


TRACKS: dict[str, Track] = {
    "phylop241": Track(
        "phylop241",
        f"{UCSC_BASE}/cactus241way/cactus241way.phyloP.bw",
        "phyloP over the Zoonomia 241-way cactus alignment",
        "hg38",
        9_644_660_543,
        "UCSC cactus241way",
        _UCSC_TERMS,
    ),
    "phylop100": Track(
        "phylop100",
        f"{UCSC_BASE}/phyloP100way/hg38.phyloP100way.bw",
        "phyloP over the 100-way vertebrate multiz alignment",
        "hg38",
        9_870_053_206,
        "UCSC phyloP100way",
        _UCSC_TERMS,
    ),
    "phylop447": Track(
        "phylop447",
        f"{UCSC_BASE}/phyloP447way/hg38.phyloP447way.bw",
        "phyloP over the 447-way mammal + expanded primate cactus alignment",
        "hg38",
        10_022_801_463,
        "UCSC phyloP447way",
        _UCSC_TERMS,
    ),
    "phylop447_primates": Track(
        "phylop447_primates",
        f"{UCSC_BASE}/phyloP447way/hg38.phyloP447wayPrimates.bw",
        "phyloP over the primate subtree of the 447-way alignment",
        "hg38",
        0,
        "UCSC phyloP447way",
        _UCSC_TERMS,
    ),
    "phylop470": Track(
        "phylop470",
        f"{UCSC_BASE}/phyloP470way/hg38.phyloP470way.bw",
        "phyloP over the 470-way alignment",
        "hg38",
        11_511_784_023,
        "UCSC phyloP470way",
        _UCSC_TERMS,
    ),
    "phastcons100": Track(
        "phastcons100",
        f"{UCSC_BASE}/phastCons100way/hg38.phastCons100way.bw",
        "phastCons over the 100-way vertebrate multiz alignment",
        "hg38",
        5_886_377_734,
        "UCSC phastCons100way",
        _UCSC_TERMS,
    ),
    "phastcons470": Track(
        "phastcons470",
        f"{UCSC_BASE}/phastCons470way/hg38.phastCons470way.bw",
        "phastCons over the 470-way alignment",
        "hg38",
        0,
        "UCSC phastCons470way",
        _UCSC_TERMS,
    ),
    **{
        f"gpn_star_{key}_entropy": Track(
            f"gpn_star_{key}_entropy",
            _gpn_star_url(f"bigwig/{directory}/entropy.bw", GPN_STAR_REVISION),
            f"GPN-Star calibrated positional entropy, {directory} (display bigWig; "
            "the Parquet shards are canonical)",
            "hg38",
            0,
            "songlab/gpn-star-scores",
            _GPN_STAR_TERMS,
        )
        for key, directory in GPN_STAR_MODELS.items()
    },
}
"""Published tracks by short name. ``size_bytes`` is 0 where it was not recorded."""

CACHE_DIR = Path("~/.cache/biodb/constraint_tracks").expanduser()
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def track(name: str) -> Track:
    """The :class:`Track` registered as ``name``."""
    try:
        return TRACKS[name]
    except KeyError:
        raise ValueError(f"unknown track {name!r}; expected one of {sorted(TRACKS)}") from None


def _pybigwig() -> Any:
    try:
        import pyBigWig
    except ImportError as error:  # pragma: no cover - exercised only without the extra
        raise ImportError("reading bigWigs needs pyBigWig: pip install 'biodb[bigwig]'") from error
    return pyBigWig


def open_track(name_or_path: str | Path) -> Any:
    """Open a bigWig for random access: a registered track name (read over
    HTTPS in place), a URL, or a local path. Returns a ``pyBigWig`` handle."""
    pybigwig = _pybigwig()
    target = str(name_or_path)
    if target in TRACKS:
        target = TRACKS[target].url
    handle = pybigwig.open(target)
    if handle is None:  # pragma: no cover - pyBigWig returns None on failure in some builds
        raise OSError(f"could not open bigWig {target}")
    return handle


def download_track(
    name: str,
    *,
    cache_dir: str | Path | None = None,
    force: bool = False,
    progress: bool = True,
) -> Path:
    """Fetch a whole track (5-12 GB) into the cache; returns the local path.

    Prefer :func:`open_track` for anything but repeated genome-wide reads --
    a bigWig is indexed for range reads and the Hub and UCSC both serve
    byte ranges.
    """
    item = track(name)
    root = Path(cache_dir).expanduser() if cache_dir else CACHE_DIR
    dst = root / f"{item.name}.bw"
    if dst.exists() and not force:
        return dst
    logger.info("Downloading %s (%s, %.1f GB)", item.name, item.url, item.size_bytes / 1e9)
    return stream_to_file(item.url, dst, progress=progress, desc=item.name)


def score_intervals(
    handle: Any,
    frame: pl.DataFrame,
    *,
    stat: str = "mean",
    chrom: str = "chrom",
    start: str = "start",
    end: str = "end",
    column: str | None = None,
) -> pl.DataFrame:
    """One summary per interval, appended as a column.

    ``stat`` is a pyBigWig summary: ``mean``, ``max``, ``min``, ``coverage``
    (fraction of bases with a value) or ``std``. Intervals are 0-based
    half-open; an interval with no scored base gets ``null``. A contig the
    track does not carry yields ``null`` rather than an error, so a frame
    mixing assemblies fails loudly downstream instead of silently here.
    """
    if stat not in ("mean", "max", "min", "coverage", "std"):
        raise ValueError("stat must be one of mean, max, min, coverage, std")
    name = column or f"{stat}"
    chroms = handle.chroms()
    values: list[float | None] = []
    for c, s, e in zip(
        frame[chrom].to_list(), frame[start].to_list(), frame[end].to_list(), strict=True
    ):
        if c not in chroms or e <= s:
            values.append(None)
            continue
        length = chroms[c]
        got = handle.stats(c, int(max(0, s)), int(min(e, length)), type=stat, exact=True)
        value = got[0] if got else None
        values.append(None if value is None else float(value))
    return frame.with_columns(pl.Series(name, values, dtype=pl.Float64))


def score_positions(
    handle: Any,
    frame: pl.DataFrame,
    *,
    chrom: str = "chrom",
    position: str = "start",
    column: str = "score",
) -> pl.DataFrame:
    """The track's value at each single base (``position`` 0-based), appended.

    Positions are grouped by contig and read in sorted runs, so a table of a
    few hundred thousand bases costs a few hundred range requests, not one
    per row.
    """
    import numpy as np

    chroms = handle.chroms()
    out = np.full(frame.height, np.nan, dtype=np.float64)
    positions = frame[position].to_numpy()
    contigs = frame[chrom].to_numpy()
    for c in np.unique(contigs):
        if c not in chroms:
            continue
        rows = np.flatnonzero(contigs == c)
        wanted = positions[rows]
        order = np.argsort(wanted, kind="stable")
        sorted_positions = wanted[order]
        # Read each dense run of bases in one call.
        breaks = np.flatnonzero(np.diff(sorted_positions) > 1024) + 1
        for chunk in np.split(np.arange(sorted_positions.size), breaks):
            lo = int(sorted_positions[chunk[0]])
            hi = int(sorted_positions[chunk[-1]]) + 1
            if lo < 0 or hi > chroms[c]:
                continue
            block = np.asarray(handle.values(c, lo, hi), dtype=np.float64)
            out[rows[order[chunk]]] = block[sorted_positions[chunk] - lo]
    return frame.with_columns(pl.Series(column, out, dtype=pl.Float64).fill_nan(None))


def _normalise_chrom(chrom: str) -> str:
    return chrom if chrom.startswith("chr") else f"chr{chrom}"


def download_gpn_star(
    chrom: str,
    *,
    model: str = "m447",
    kind: str = "entropy",
    revision: str = GPN_STAR_REVISION,
    cache_dir: str | Path | None = None,
    force: bool = False,
    progress: bool = True,
) -> Path:
    """Fetch one chromosome's GPN-Star Parquet shard; returns the local path.

    Shards are 0.2-1.2 GB (``entropy``) and 0.8-4.6 GB (``llr``) each; the
    cache is keyed by revision, model and kind.
    """
    if model not in GPN_STAR_MODELS:
        raise ValueError(f"unknown model {model!r}; expected one of {sorted(GPN_STAR_MODELS)}")
    if kind not in GPN_STAR_KINDS:
        raise ValueError(f"kind must be one of {GPN_STAR_KINDS}")
    directory = GPN_STAR_MODELS[model]
    name = _normalise_chrom(chrom)
    path = f"data/{directory}/{kind}/{kind}_{name}.parquet"
    root = Path(cache_dir).expanduser() if cache_dir else CACHE_DIR
    dst = root / "gpn-star" / revision / path
    if dst.exists() and not force:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    return stream_to_file(_gpn_star_url(path, revision), dst, progress=progress, desc=path)


def load_gpn_star(
    chrom: str,
    *,
    model: str = "m447",
    kind: str = "entropy",
    positions: Sequence[int] | None = None,
    revision: str = GPN_STAR_REVISION,
    cache_dir: str | Path | None = None,
    force: bool = False,
    progress: bool = True,
) -> pl.DataFrame:
    """One chromosome's GPN-Star scores, optionally at given 1-based positions.

    Returns
    -------
    polars.DataFrame
        ``chrom`` (``chr``-prefixed), ``pos`` (1-based, as published),
        ``start`` (0-based, derived), ``ref``, and ``entropy_calibrated``
        (``kind="entropy"``) or ``alt``, ``llr_calibrated``,
        ``abs_llr_calibrated`` (``kind="llr"``; note the absolute score is
        supplied separately and is not ``abs(llr_calibrated)``).
    """
    path = download_gpn_star(
        chrom,
        model=model,
        kind=kind,
        revision=revision,
        cache_dir=cache_dir,
        force=force,
        progress=progress,
    )
    lazy = pl.scan_parquet(path)
    if positions is not None:
        lazy = lazy.filter(pl.col("pos").is_in(list(positions)))
    frame = lazy.collect()
    if frame.schema["chrom"] != pl.Utf8:
        frame = frame.with_columns(pl.col("chrom").cast(pl.Utf8))
    return frame.with_columns(
        pl.when(pl.col("chrom").str.starts_with("chr"))
        .then(pl.col("chrom"))
        .otherwise(pl.lit("chr") + pl.col("chrom"))
        .alias("chrom"),
        (pl.col("pos") - 1).alias("start"),
    )


__all__ = [
    "CACHE_DIR",
    "GPN_STAR_DATASET",
    "GPN_STAR_KINDS",
    "GPN_STAR_MODELS",
    "GPN_STAR_REVISION",
    "TRACKS",
    "Track",
    "download_gpn_star",
    "download_track",
    "load_gpn_star",
    "open_track",
    "score_intervals",
    "score_positions",
    "track",
]
