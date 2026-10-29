"""Ensembl Compara homologies — orthologues and, above all, within-species paralogues.

Ensembl publishes every homology its gene trees infer as one TSV per release
under ``pub/release-<N>/tsv/ensembl-compara/homologies/<species>/``: one row
per gene pair with the homology type (``ortholog_one2one``,
``within_species_paralog``, ``other_paralog``, ``gene_split``, ...), the
percent identity seen from each side, and for orthologues the gene-order
conservation and whole-genome-alignment coverage that make a call
*high-confidence*. The human file is ~110 MB gzipped and is the complete set
of homologies involving a human gene, so the paralogue rows are the full
human paralogue catalogue.

* :func:`download_homologies` / :func:`load_homologies` — the raw table.
* :func:`paralog_pairs` — the within-species paralogue rows as an
  unordered, deduplicated gene-pair table (``gene_a < gene_b``).
* :func:`paralog_families` — connected components of the paralogue graph:
  ``gene_stable_id`` -> ``family`` (0-based, families numbered by their
  lexically smallest member), so any two genes in one family are paralogues
  by some path -- the exclusion set a coevolution benchmark needs.

Identifiers are versionless Ensembl gene IDs (``ENSG...``); map them to
symbols with :func:`biodb.gencode.fetch_mane_select` or
:func:`biodb.mapping.map_gene_ids`. Licence: Ensembl data are released under
the Apache-2.0 licence (https://www.ensembl.org/info/about/legal/code_licence.html).

Examples
--------
>>> from biodb.ensembl_compara import paralog_families
>>> fam = paralog_families()                                        # doctest: +SKIP
>>> fam.filter(pl.col("gene_stable_id") == "ENSG00000012048")        # doctest: +SKIP
"""

from __future__ import annotations

import gzip
import logging
from pathlib import Path

import numpy as np
import polars as pl

from biodb._downloads import stream_to_file

logger = logging.getLogger(__name__)

ENSEMBL_FTP = "https://ftp.ensembl.org/pub"

DEFAULT_RELEASE = "116"
"""Ensembl release read by default. Bump after checking the dump's columns."""

DEFAULT_SPECIES = "homo_sapiens"

PARALOG_TYPES = ("within_species_paralog", "other_paralog")
"""Homology types that relate two genes of one genome. ``gene_split`` (one
gene annotated as two) is deliberately not a paralogue here."""

CACHE_DIR = Path("~/.cache/biodb/ensembl_compara").expanduser()
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_COLUMNS = {
    "gene_stable_id": pl.Utf8,
    "protein_stable_id": pl.Utf8,
    "species": pl.Utf8,
    "identity": pl.Float64,
    "homology_type": pl.Utf8,
    "homology_gene_stable_id": pl.Utf8,
    "homology_protein_stable_id": pl.Utf8,
    "homology_species": pl.Utf8,
    "homology_identity": pl.Float64,
    "dn": pl.Float64,
    "ds": pl.Float64,
    "goc_score": pl.Float64,
    "wga_coverage": pl.Float64,
    "is_high_confidence": pl.Utf8,
    "homology_id": pl.Int64,
}


def _homologies_url(release: str, species: str, kind: str) -> str:
    return (
        f"{ENSEMBL_FTP}/release-{release}/tsv/ensembl-compara/homologies/{species}/"
        f"Compara.{release}.{kind}_default.homologies.tsv.gz"
    )


def download_homologies(
    *,
    release: str = DEFAULT_RELEASE,
    species: str = DEFAULT_SPECIES,
    kind: str = "protein",
    cache_dir: str | Path | None = None,
    force: bool = False,
    progress: bool = True,
) -> Path:
    """Fetch one species' homology dump (``protein`` or ``ncrna`` gene trees)."""
    if kind not in ("protein", "ncrna"):
        raise ValueError("kind must be 'protein' or 'ncrna'")
    root = Path(cache_dir).expanduser() if cache_dir else CACHE_DIR
    dst = root / release / species / f"Compara.{release}.{kind}_default.homologies.tsv.gz"
    if dst.exists() and not force:
        return dst
    return stream_to_file(
        _homologies_url(release, species, kind), dst, progress=progress, desc=dst.name
    )


def load_homologies(
    *,
    release: str = DEFAULT_RELEASE,
    species: str = DEFAULT_SPECIES,
    kind: str = "protein",
    homology_types: tuple[str, ...] | None = None,
    cache_dir: str | Path | None = None,
    force: bool = False,
    progress: bool = True,
) -> pl.DataFrame:
    """The homology table, optionally restricted to some ``homology_type`` values.

    Returns
    -------
    polars.DataFrame
        The dump's 15 columns with ``NULL`` read as null and
        ``is_high_confidence`` as a nullable boolean.
    """
    path = download_homologies(
        release=release,
        species=species,
        kind=kind,
        cache_dir=cache_dir,
        force=force,
        progress=progress,
    )
    with gzip.open(path, "rb") as handle:
        frame = pl.read_csv(
            handle.read(),
            separator="\t",
            schema_overrides=_COLUMNS,
            null_values=["NULL"],
        )
    frame = frame.with_columns(
        pl.when(pl.col("is_high_confidence").is_null())
        .then(None)
        .otherwise(pl.col("is_high_confidence") == "1")
        .alias("is_high_confidence")
    )
    if homology_types is not None:
        frame = frame.filter(pl.col("homology_type").is_in(list(homology_types)))
    logger.info("Loaded %d homologies from %s", frame.height, path.name)
    return frame


def paralog_pairs(
    *,
    release: str = DEFAULT_RELEASE,
    species: str = DEFAULT_SPECIES,
    kind: str = "protein",
    homology_types: tuple[str, ...] = PARALOG_TYPES,
    min_identity: float | None = None,
    cache_dir: str | Path | None = None,
    force: bool = False,
    progress: bool = True,
) -> pl.DataFrame:
    """Within-species paralogue pairs, unordered and deduplicated.

    ``min_identity`` keeps a pair only when the identity seen from *both*
    genes reaches it (percent of each gene's representative sequence).

    Returns
    -------
    polars.DataFrame
        ``gene_a``, ``gene_b`` (``gene_a < gene_b``), ``homology_type``,
        ``identity_a``, ``identity_b`` (percent), ``homology_id``.
    """
    frame = load_homologies(
        release=release,
        species=species,
        kind=kind,
        homology_types=homology_types,
        cache_dir=cache_dir,
        force=force,
        progress=progress,
    ).filter(pl.col("species") == pl.col("homology_species"))
    if min_identity is not None:
        frame = frame.filter(
            (pl.col("identity") >= min_identity) & (pl.col("homology_identity") >= min_identity)
        )
    first = pl.col("gene_stable_id") < pl.col("homology_gene_stable_id")
    out = frame.select(
        pl.when(first)
        .then(pl.col("gene_stable_id"))
        .otherwise(pl.col("homology_gene_stable_id"))
        .alias("gene_a"),
        pl.when(first)
        .then(pl.col("homology_gene_stable_id"))
        .otherwise(pl.col("gene_stable_id"))
        .alias("gene_b"),
        pl.col("homology_type"),
        pl.when(first)
        .then(pl.col("identity"))
        .otherwise(pl.col("homology_identity"))
        .alias("identity_a"),
        pl.when(first)
        .then(pl.col("homology_identity"))
        .otherwise(pl.col("identity"))
        .alias("identity_b"),
        pl.col("homology_id"),
    )
    return out.filter(pl.col("gene_a") != pl.col("gene_b")).unique(
        subset=["gene_a", "gene_b"], keep="first", maintain_order=True
    )


def paralog_families(
    pairs: pl.DataFrame | None = None,
    **kwargs: object,
) -> pl.DataFrame:
    """Connected components of the paralogue graph.

    ``pairs`` is a :func:`paralog_pairs` table; when omitted it is built with
    ``**kwargs`` forwarded to :func:`paralog_pairs` (``min_identity`` is the
    knob that keeps distant ``other_paralog`` links from fusing families).

    Returns
    -------
    polars.DataFrame
        ``gene_stable_id``, ``family`` (0-based; numbered by each family's
        lexically smallest gene), ``family_size``. Genes with no paralogue
        are absent: a missing gene is its own family.
    """
    if pairs is None:
        pairs = paralog_pairs(**kwargs)  # type: ignore[arg-type]
    genes = sorted(set(pairs["gene_a"].to_list()) | set(pairs["gene_b"].to_list()))
    index = {gene: i for i, gene in enumerate(genes)}
    parent = np.arange(len(genes), dtype=np.int64)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    for a, b in zip(pairs["gene_a"].to_list(), pairs["gene_b"].to_list(), strict=True):
        ra, rb = find(index[a]), find(index[b])
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = np.asarray([find(i) for i in range(len(genes))], dtype=np.int64)
    _, family = np.unique(roots, return_inverse=True)
    sizes = np.bincount(family)
    return pl.DataFrame(
        {
            "gene_stable_id": genes,
            "family": family.astype(np.int64),
            "family_size": sizes[family].astype(np.int64),
        }
    )


__all__ = [
    "CACHE_DIR",
    "DEFAULT_RELEASE",
    "DEFAULT_SPECIES",
    "PARALOG_TYPES",
    "download_homologies",
    "load_homologies",
    "paralog_families",
    "paralog_pairs",
]
