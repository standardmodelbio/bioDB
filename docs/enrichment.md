# Raw annotations and enrichment

These clients acquire source records for downstream annotation. They preserve
provider fields; document rendering, ranking, and application-specific
summaries belong to callers. Examples below make remote requests unless
their inputs are already cached.

## Publication records

```python
from biodb.europepmc import fetch_metadata

records = fetch_metadata(["36991119", "30700171"], cache_dir="./annotations")
for pmid, record in records.items():
    print(pmid, record.get("title"))
```

`fetch_metadata` returns original EuropePMC MED/core records keyed by PMID.
Validated successful batches immediately commit per-PMID receipts. A
successful no-match is cached as completed absence and is omitted from the
returned dictionary. Failed/truncated requests do not establish absence or
completion, so retries request only unfinished IDs. The defaults are
`batch_size=80`, `max_retries=3`, and strict failures. `allow_partial=True`
returns successes and warns about failed batches; `force=True` refreshes IDs.

For NCBI E-utilities and baseline/update XML instead, use `biodb.pubmed`:

```python
from biodb import pubmed

hits = pubmed.search("BRCA1", retmax=3)
abstracts = pubmed.query_abstracts(hits["pmids"])
```

The PubMed bulk path exposes `list_baseline_files`, `list_update_files`,
`download_baseline_file`, `download_update_file`, and `parse_pubmed_xml`.
Downloads verify companion MD5 files by default; baseline acquisition is a
large corpus operation. These literature clients use core dependencies.

## Protein structure metadata

```python
from biodb.protein_structure import fetch_structure_for_accession

record = fetch_structure_for_accession("P38398", cache_dir="./annotations")
print(record.keys())  # uniprot_accession, alphafold, rcsb
```

`alphafold` and `rcsb` contain complete provider JSON, not a chosen or ranked
structure. Each provider/accession has an independent validated cache entry.
A 404 is a successful empty result (`[]`); a failed request is retried on the
next call. `fetch_provider(accession, "alphafold")` or `"rcsb"` queries one
provider. `force=True` refreshes responses.

`build_structure_cache(accessions, "structures.parquet", concurrency=8)`
persists successes immediately and exports a Parquet snapshot. Completion is
established by per-accession receipts, so an old export alone does not mark
failed inputs complete. Strict failure is the default. `allow_partial=True`
returns successes and warns about failures. An injected `fetch_fn` can provide
caller-owned normalization; the default keeps raw provider data.

## Open Targets literature and constants

```python
from biodb.opentargets.literature import fetch_literature_matches

matches = fetch_literature_matches(
    "25.12",
    wanted_pairs={("ENSG00000012048", "30700171")},
    entity_type="GP",
    limit_files=1,
    cache_dir="./ot-cache",
)
```

This returns complete matching intermediate rows for `(keywordId, pmid)`
pairs. Choose the source `entity_type` for your IDs (`"CD"` is the default).
It uses the public Open Targets manifest/cache path for intermediate shards.
`limit_files=1` is an explicit preview and can omit matches in later shards;
omit it for full membership. Acquisition/read failures raise unless
`allow_partial=True`, which warns and returns available rows.

```python
from biodb.opentargets.constants import fetch_constants, fetch_sequence_ontology_labels

constants = fetch_constants(ref="main", cache_dir="./annotations")
print(constants["variant_ts_sha"])
labels = fetch_sequence_ontology_labels(version="25.12", cache_dir="./ot-cache")
```

`fetch_constants` returns source-derived variant consequences, effect-method
names, population labels, ClinVar star mappings, credible-set confidence
mappings, and source Git blob SHAs. Supply a fixed Git `ref` for reproducible
acquisition; `main` is a moving ref. `fetch_source` exposes raw text and its
blob SHA. Sequence Ontology labels come from the complete release-pinned
Open Targets `so` dataset. Both label maps normalize SO IDs to underscore
form (`SO_0001583`) for OT variant annotations.

## GENCODE genes and MANE Select

GENCODE requires `pooch` for download, in addition to core dependencies:

```bash
pip install pooch
```

```python
from biodb.gencode import get_genes, fetch_mane_select

genes = get_genes(release="47", cache_dir="./gencode-cache")
print(genes.select("gene_id", "gene_symbol", "biotype").head())

mane = fetch_mane_select(release="47", cache_dir="./gencode-cache")
print(mane.select("gene_id", "mane_select_tx", "aa_length").head())
```

Both calls reuse the human release GTF fetched by `download_gencode_gtf`.
`get_genes` keeps all gene biotypes. `fetch_mane_select` restricts to
protein-coding genes with a source-tagged MANE Select transcript and reports
the implementation's CDS-derived `aa_length`. Genes without that tag are
omitted. Gene IDs omit Ensembl version suffixes; the selected transcript ID
retains its version. Coordinates are 0-based half-open (`start`, `end`),
converted from the source GTF. Pin the release explicitly; no liftover occurs.

## Gene sets and functional enrichment

```python
from biodb import gprofiler, msigdb

hits = gprofiler.gost(["BRCA1", "TP53", "EGFR"], sources=["GO:BP"])
gene_set = msigdb.query_gene_set("HALLMARK_APOPTOSIS")
```

g:Profiler also exposes `download_gmt` / `load_gmt` for bulk annotations.
`gost` uses the optional `gprofiler-official` client when installed and falls
back to core `requests` otherwise. MSigDB offers `query_genes` for targeted
membership and `download_gmt` / `load_gmt` for release/collection-pinned GMT.
Harmonizome offers dataset metadata, GMT, and gene–attribute matrices.
`biodb.utils.read_gmt` is the shared local GMT reader.

See the [source catalog](sources.md) for gene-set, interaction, and genomic
clients, the [GENCODE/enrichment notebook](_generated/tutorials/14_gencode_enrichment.ipynb)
for local fixtures, and the [API reference](api.rst) for complete signatures.
