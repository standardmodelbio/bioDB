API reference
=============

The full APIs live on the modules below. Only a curated subset is re-exported
from ``biodb``. See the :doc:`source catalog <sources>` for access modes and
the :doc:`quickstart <quickstart>` for optional dependencies.

Open Targets
------------

.. autosummary::
   :toctree: _generated

   biodb.opentargets
   biodb.opentargets.readers
   biodb.opentargets.variants
   biodb.opentargets.studies
   biodb.opentargets.literature
   biodb.opentargets.constants
   biodb.opentargets_graphql

GWAS and molecular traits
-------------------------

.. autosummary::
   :toctree: _generated

   biodb.gwas_catalog
   biodb.gwas_harvest
   biodb.gwas_atlas
   biodb.panukbb
   biodb.aou_allxall
   biodb.omicspred

Ontologies, diseases, and literature
------------------------------------

.. autosummary::
   :toctree: _generated

   biodb.ols
   biodb.ols_text
   biodb.snomed
   biodb.ontology
   biodb.ontology_owl
   biodb.monarch
   biodb.orphanet
   biodb.gtr
   biodb.pubmed
   biodb.europepmc

Cell types
----------

.. autosummary::
   :toctree: _generated

   biodb.cellmarker
   biodb.celltaxonomy
   biodb.cellxgene
   biodb.markers
   biodb.cellxgene_dataset
   biodb.hf_datasets

Genes, proteins, and gene sets
------------------------------

.. autosummary::
   :toctree: _generated

   biodb.gencode
   biodb.uniprot
   biodb.protein_structure
   biodb.harmonizome
   biodb.gprofiler
   biodb.msigdb
   biodb.string
   biodb.corum
   biodb.coessentiality
   biodb.ensembl_compara

Variants, intervals, and tracks
-------------------------------

.. autosummary::
   :toctree: _generated

   biodb.clinvar
   biodb.traitgym
   biodb.constraint_tracks
   biodb.encode_re2g
   biodb.tads
   biodb.rfam
   biodb.intervals

Shared helpers
--------------

.. autosummary::
   :toctree: _generated

   biodb.mapping
   biodb.transform
   biodb.utils
