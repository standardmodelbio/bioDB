"""biodb: biomedical knowledge graph helpers, ported from ``AoU.phenome``.

Module map:

* :mod:`biodb.opentargets` -- Open Targets Platform bulk downloaders
  and parsers (disease/drug/PGx/expression/essentiality/pathways) +
  gene-association matrix builders.
* :mod:`biodb.opentargets_graphql` -- Targeted Open Targets GraphQL
  queries (``query_target``, ``query_disease``, ``query_drug``,
  ``query_variant``).
* :mod:`biodb.monarch` -- Monarch Initiative association readers
  (causal gene-to-disease and friends).
* :mod:`biodb.ontology` -- OBO / OWL / Mondo loaders, N-hop keyword
  set expansion, hierarchical keyword set generation, attention
  analysis, gene-phenotype matrix construction.
* :mod:`biodb.ontology_owl` -- Generic owlready2-based primitives that
  work for any OBO Foundry OWL file: SO, HPO, EFO, GO, ChEBI, … (load,
  descendants/ancestors walk, most-recent common ancestor).
* :mod:`biodb.uniprot` -- UniProt REST client (protein sequences,
  features, cross-references).
* :mod:`biodb.harmonizome` -- Maayan-Lab Harmonizome client (~114
  curated gene-attribute datasets; ``list_datasets``, ``download_datasets``,
  ``get_gmt``, ``load_gene_attribute_matrix``, ``get_dataset_metadata``).
* :mod:`biodb.clinvar` -- ClinVar VCF download + parsing, CLNSIG
  long-tail → 6-class simplification, BED / sites format converters.
* :mod:`biodb.aou_allxall` -- All of Us *All-by-All* PheWAS atlas
  (~3,600 phenotypes × ~414k WGS participants); public-API client for
  per-phenotype gene-burden tables (pLoF/missenseLC/synonymous) with
  concurrent bulk download + consolidated Parquet output.
* :mod:`biodb.tads` -- topologically associating domains from ENCODE Hi-C
  contact-domain calls (``list_domain_files``, ``load_domains``,
  ``boundaries``, ``same_domain``). Two positions in one TAD are in contact
  and co-regulated, so domain co-membership is the control any paired-position
  analysis owes its reader.
* :mod:`biodb.coessentiality` -- the Wainberg et al. 2021 co-essential
  network from DepMap CRISPR screens (GLS p-value matrices, memory-mapped):
  ``coessential_pairs`` at a BH FDR, ``pair_p_values`` for explicit pairs.
* :mod:`biodb.ensembl_compara` -- Ensembl Compara homology dumps: the
  human paralogue catalogue as deduplicated gene pairs and as connected
  families (``paralog_pairs``, ``paralog_families``), plus the raw table.
* :mod:`biodb.constraint_tracks` -- per-base constraint: UCSC phyloP /
  phastCons bigWigs (241-, 100-, 447-, 470-way) read in place over HTTPS,
  and GPN-Star calibrated entropy / LLR Parquet shards at a pinned Hub
  revision (``open_track``, ``score_intervals``, ``score_positions``,
  ``load_gpn_star``).
* :mod:`biodb.traitgym` -- TraitGym causal-variant benchmarks (Mendelian and
  complex-trait positives with matched negatives) and their precomputed
  feature tables, at a pinned Hub revision.
* :mod:`biodb.corum` -- CORUM curated protein complexes as co-complex
  gene pairs, current or by archived release.
* :mod:`biodb.encode_re2g` -- ENCODE-rE2G CRISPRi enhancer-gene benchmark
  (tested pairs with matched negatives) as a labelled pair table.
* :mod:`biodb.rfam` -- Rfam RNA family models, seed alignments, and
  Infernal-based genome annotation with per-hit consensus structures.
* :mod:`biodb.orphanet` -- Orphanet rare-disease epidemiology
  (Orphadata ``en_product9_prev``, CC-BY-4.0): per-ORPHAcode prevalence
  band → per-individual K for liability-threshold effect modeling.
* :mod:`biodb.string` -- STRING database physical PPI edges with
  continuous combined-score weights (``download_physical_links``,
  ``load_physical_links``, ``physical_ppi_edges``). The physical
  sub-network is direct binding evidence only — closer to "PPI"
  than the full functional-coupling network in ``protein.links``.
* :mod:`biodb.celltaxonomy` / :mod:`biodb.cellmarker` /
  :mod:`biodb.cellxgene` -- cell-type marker genes ranked per Cell
  Ontology (CL) term. Cell Taxonomy + CellMarker are curated bulk
  flat files with native CL ids (``get_markers`` / ``query_markers`` /
  ``to_gmt``); CELLxGENE fetches CZI's precomputed Marker Score via the
  WMG REST API (``query_markers`` / ``get_tissue_markers``). All three emit
  the shared *(species, tissue, cell type, CL id, gene, score, rank)*
  schema from :mod:`biodb._celltype`.

Genomic-interval helpers (BED I/O, overlap joins, nearest) live in
:mod:`biodb.intervals`. Shared utilities live in :mod:`biodb.utils` (random seeding,
similarity helpers, token counting, ``create_gene_association_matrix``,
``filter_adaptive``).
"""

from biodb import (
    aou_allxall,
    cellmarker,
    celltaxonomy,
    cellxgene,
    cellxgene_dataset,
    clinvar,
    coessentiality,
    constraint_tracks,
    corum,
    encode_re2g,
    ensembl_compara,
    europepmc,
    gencode,
    gprofiler,
    gtr,
    gwas_atlas,
    gwas_catalog,
    harmonizome,
    hf_datasets,
    intervals,
    mapping,
    markers,
    monarch,
    msigdb,
    ols,
    omicspred,
    ontology,
    ontology_owl,
    opentargets,
    opentargets_graphql,
    orphanet,
    panukbb,
    protein_structure,
    pubmed,
    rfam,
    snomed,
    string,
    tads,
    traitgym,
    transform,
    uniprot,
    utils,
)
from biodb.cellmarker import (
    get_markers as cellmarker_get_markers,
)
from biodb.cellmarker import (
    query_markers as cellmarker_query_markers,
)

# A small slice of high-frequency public symbols is re-exported at the
# top level for convenience. The full APIs live on the submodules.
from biodb.celltaxonomy import (
    get_markers as celltaxonomy_get_markers,
)
from biodb.celltaxonomy import (
    query_markers as celltaxonomy_query_markers,
)
from biodb.cellxgene import (
    cellguide_markers as cellxgene_cellguide_markers,
)
from biodb.cellxgene import (
    disease_vs_normal as cellxgene_disease_vs_normal,
)
from biodb.cellxgene import (
    get_all_markers as cellxgene_get_all_markers,
)
from biodb.cellxgene import (
    get_tissue_markers as cellxgene_get_tissue_markers,
)
from biodb.cellxgene import (
    query_markers as cellxgene_query_markers,
)
from biodb.cellxgene_dataset import (
    build_dataset as cellxgene_build_dataset,
)
from biodb.clinvar import (
    bed_to_sites,
    df_to_bed,
    df_to_sites,
    download_vcf,
    simplify_annotations,
    vcf_to_df,
)
from biodb.gtr import (
    aggregate_gene_sets as gtr_aggregate_gene_sets,
)
from biodb.gtr import (
    gene_sets as gtr_gene_sets,
)
from biodb.gtr import (
    panel_text as gtr_panel_text,
)
from biodb.gtr import (
    query_test as gtr_query_test,
)
from biodb.gtr import (
    search_tests as gtr_search_tests,
)
from biodb.gtr import (
    to_gmt as gtr_to_gmt,
)
from biodb.harmonizome import (
    download_datasets as harmonizome_download_datasets,
)
from biodb.harmonizome import (
    get_dataset_metadata as harmonizome_get_dataset_metadata,
)
from biodb.harmonizome import (
    get_gmt,
    load_gene_attribute_matrix,
)
from biodb.harmonizome import (
    list_datasets as harmonizome_list_datasets,
)
from biodb.mapping import map_gene_ids
from biodb.markers import load_markers
from biodb.monarch import (
    get_gene_associations as monarch_get_gene_associations,
)
from biodb.monarch import (
    read_causal_gene_to_disease_association,
)
from biodb.ontology import (
    get_ancestors,
    get_descendants,
    get_mrca,
    get_ontology,
)
from biodb.opentargets import (
    ensure_cached_shards,
    get_dataset,
    get_gene_associations,
    get_pathways,
    get_targets,
    list_available_versions,
    list_datasets,
)
from biodb.opentargets_graphql import (
    query_disease,
    query_drug,
    query_target,
    query_variant,
)
from biodb.string import (
    physical_ppi_edges as string_physical_ppi_edges,
)
from biodb.uniprot import (
    get_dbxrefs,
    get_features,
    get_sequences,
    query_protein,
)
from biodb.utils import (
    RANDOM_SEED,
    cosine_similarity,
    count_tokens,
    create_gene_association_matrix,
    dot_product_similarity,
    euclidean_similarity,
    filter_adaptive,
    l2_normalize,
    set_random_seed,
)

__version__ = "0.1.0"

__all__ = [
    "RANDOM_SEED",
    "aou_allxall",
    "bed_to_sites",
    "celltaxonomy",
    "celltaxonomy_get_markers",
    "celltaxonomy_query_markers",
    "cellmarker",
    "cellmarker_get_markers",
    "cellmarker_query_markers",
    "cellxgene",
    "cellxgene_build_dataset",
    "cellxgene_cellguide_markers",
    "cellxgene_dataset",
    "cellxgene_disease_vs_normal",
    "cellxgene_get_all_markers",
    "cellxgene_get_tissue_markers",
    "cellxgene_query_markers",
    "clinvar",
    "corum",
    "cosine_similarity",
    "encode_re2g",
    "count_tokens",
    "create_gene_association_matrix",
    "df_to_bed",
    "df_to_sites",
    "dot_product_similarity",
    "download_vcf",
    "ensure_cached_shards",
    "euclidean_similarity",
    "filter_adaptive",
    "get_ancestors",
    "get_dataset",
    "get_dbxrefs",
    "get_descendants",
    "get_features",
    "get_gene_associations",
    "get_gmt",
    "get_mrca",
    "get_ontology",
    "get_pathways",
    "get_sequences",
    "get_targets",
    "gprofiler",
    "gtr",
    "gtr_aggregate_gene_sets",
    "gtr_gene_sets",
    "gtr_panel_text",
    "gtr_query_test",
    "gtr_search_tests",
    "gtr_to_gmt",
    "gwas_atlas",
    "gwas_catalog",
    "gencode",
    "europepmc",
    "protein_structure",
    "harmonizome",
    "intervals",
    "harmonizome_download_datasets",
    "harmonizome_get_dataset_metadata",
    "harmonizome_list_datasets",
    "hf_datasets",
    "l2_normalize",
    "list_available_versions",
    "list_datasets",
    "load_gene_attribute_matrix",
    "load_markers",
    "map_gene_ids",
    "mapping",
    "markers",
    "monarch",
    "monarch_get_gene_associations",
    "msigdb",
    "ols",
    "omicspred",
    "ontology",
    "ontology_owl",
    "opentargets",
    "opentargets_graphql",
    "orphanet",
    "panukbb",
    "pubmed",
    "rfam",
    "query_disease",
    "query_drug",
    "query_protein",
    "query_target",
    "query_variant",
    "read_causal_gene_to_disease_association",
    "set_random_seed",
    "simplify_annotations",
    "snomed",
    "coessentiality",
    "ensembl_compara",
    "constraint_tracks",
    "string",
    "tads",
    "string_physical_ppi_edges",
    "traitgym",
    "transform",
    "uniprot",
    "utils",
    "vcf_to_df",
]
