# bioDB tutorials

All 17 notebooks contain executed **offline synthetic examples**. Upstream
requests, large downloads, optional native tools, and publishing calls appear
as explicitly labeled **live recipes in Markdown**, so re-execution does not
contact upstream services. HTTP fixtures replace only the transport boundary;
readers, public API calls, real Parquet/gzip files, manifests and completion
receipts execute normally.

Start with [the source catalog](17_source_catalog.ipynb) to find every public
source module and Open Targets submodule. A linked recipe is not evidence that
a live service or a whole-corpus download was validated. Record releases,
assemblies, manifest membership and provenance for production analyses.

## Index

| Notebook | Covered APIs and contracts |
|---|---|
| [01 · Utils quickstart](01_utils_quickstart.ipynb) | `utils`, `transform`: similarity, adaptive filtering, matrices; optional Torch normalization recipe |
| [02 · Open Targets](02_opentargets.ipynb) | Bulk/readers/variants/studies: raw rows, lead versus member PIP, study joins, lazy scans, manifest completeness; GraphQL recipe |
| [03 · Monarch](03_monarch.ipynb) | BioLink REST, pagination; Neo4j Cypher and bulk TSV recipes |
| [04 · Ontology and OLS](04_ontology.ipynb) | Keyword expansion; OLS hierarchy/search and canonical `ontology` OWL recipes; deprecated `ontology_owl` migration |
| [05 · UniProt](05_uniprot.ipynb) | Streaming gzip FASTA; targeted REST and Swiss-Prot/TrEMBL recipes |
| [06 · Harmonizome](06_harmonizome.ipynb) | Gzip gene-attribute matrix, merged Parquet; dataset catalog/GMT recipes |
| [07 · ClinVar](07_clinvar.ipynb) | Clinical-significance buckets; E-utils and VCF-to-Polars/BED recipes |
| [08 · GWAS Atlas](08_gwas_atlas.ipynb) | MAGMA gene-score matrix to long table/matrix; metadata trait lookup recipes |
| [09 · gProfiler](09_gprofiler.ipynb) | Cached organism GMT; live functional enrichment recipe |
| [10 · MSigDB](10_msigdb.ipynb) | Versioned collection GMT; per-set JSON lookup recipes |
| [11 · PubMed](11_pubmed.ipynb) | Local XML parser; E-utils, baseline/update listing/download recipes |
| [12 · SNOMED CT](12_snomed.ipynb) | Local Athena TSV/ZIP parsing; graph versus legacy/v2 OLS text bulk recipes |
| [13 · Raw GWAS](13_raw_gwas.ipynb) | GWAS Catalog + GWAS Harvest: exact manifests, provenance, missing/zero P, significance, inclusive regions, error handling and resumable verified artifacts |
| [14 · GENCODE and enrichment](14_gencode_enrichment.ipynb) | All-biotype genes versus MANE, raw OT literature/constants/SO, EuropePMC, structure-provider receipts, audited OLS v2 text |
| [15 · Cell markers](15_cell_markers.ipynb) | Cell Taxonomy, CellMarker 2.0, CellGuide, `markers`, local `hf_datasets`/`cellxgene_dataset` exports; WMG disease/rollup recipes |
| [16 · Genomic resources](16_genomic_resources.ipynb) | Intervals/TADs, Compara, coessentiality, STRING, CORUM, GPN-Star, Rfam; TraitGym, ENCODE-rE2G, Orphanet and native-tool recipes |
| [17 · Source catalog](17_source_catalog.ipynb) | Every public source module; All-by-All burden transforms, Pan-UKBB selection, OmicsPred scoring parser, GTR panel sets/text; gene-ID mapping recipe |

## Re-execute and validate

From a source checkout, install the small optional dependencies used by the
fixtures (no Torch, model weights, Infernal or tabix required):

```bash
pip install -e '.[docs]'
python scripts/validate_notebooks.py --execute
```

The validator checks notebook schema/syntax/saved errors, executes each complete
notebook in an isolated temporary directory, and blocks unexpected network
connections. It checks temporary execution copies without overwriting reviewed
outputs. `metadata.biodb.offline = true` declares the whole-notebook policy;
individual executable cells also carry the `offline` tag.

To refresh saved outputs deliberately after changing examples:

```bash
jupyter nbconvert --to notebook --execute --inplace tutorials/*.ipynb
python scripts/validate_notebooks.py --execute
```

The docs build renders the saved notebooks. Source-specific extras needed only
for live recipes are called out in the notebooks and API documentation. Examples
use synthetic data in private temporary directories and never seed a real user
cache with synthetic source releases.
