# Raw GWAS acquisition

`biodb.gwas_catalog` and `biodb.gwas_harvest` acquire exact manifest files and
yield raw Polars batches. The defaults are `scope="genomewide"` and
`p_max=None`: every source row is retained, including non-significant variants,
P=0, and rows with missing effects. Downloads and parsers validate complete
input; parsing failures propagate.

## Select and save a manifest

Catalog accepts the official harmonised manifest by default, or a local
manifest/URL supplied by the caller. One study can have multiple files; all
listed aliases and paths are retained.

```python
from biodb import gwas_catalog

files = gwas_catalog.list_datasets(study_ids=["GCST90027158"])
files.write_parquet("catalog-selection.parquet")
print(files.select("study_id", "file_id", "url"))
```

This fetches manifest metadata only. Save the selected table to pin membership
before acquisition. The entire manifest is validated before filtering;
malformed entries and temporary error pages cannot become a valid manifest.

Harvest requires your actual newline-delimited list of absolute source URLs;
there is no invented public default endpoint:

```python
from biodb import gwas_harvest

harvest_files = gwas_harvest.list_datasets("harvest-urls.txt")
harvest_files.write_parquet("harvest-selection.parquet")
```

Harvest `study_id` is the complete filename stem, including ontology/build
suffixes. `gcst_id` is an optional extracted Catalog accession. `study_ids`
selection accepts either identity and keeps every matching file. Filename
suffixes do not establish a verified reference build.

## Stream or durably ingest

For one source file, pass an exact manifest row or local TSV/TSV.gz path:

```python
for batch in gwas_catalog.iter_sumstats(files.row(0, named=True), batch_size=100_000):
    print(batch.height)
```

Exhaust the iterator to validate EOF, including gzip CRC/trailer checks. If
iteration raises after yielding batches, the partial file output cannot claim
complete coverage. `download_dataset(row)` separately returns a validated
local source path. Whole-file streaming needs no optional extra.

For durable output, use the batch API:

```python
from biodb import gwas_catalog

try:
    result = gwas_catalog.ingest_sumstats(
        files,
        "catalog-run",
        workers=4,
        batch_size=100_000,
        required_study_ids=["GCST90027158"],
    )
except gwas_catalog.IngestError as error:
    result = error.result
    print(result.status, result.failed)
    print([item for item in result.files if item["status"] == "failed"])
    raise

print(result.rows, result.succeeded, result.failed)
raw = result.scan()  # LazyFrame over committed successes only
result.consolidate("catalog-raw.parquet")
```

`required_study_ids` makes a requested-but-absent study an explicit failed
entry. Without it, ingest covers exactly the supplied table. `workers=4`,
`batch_size=100_000`, `resume=True`, and `allow_partial=False` are defaults.

Each file commits immutable Parquet artifacts and a completion receipt.
Successful zero-row files are complete successes. An interrupted retry checks
artifact hashes, reuses valid successes, and retries failed files. Membership,
source identity, schema, backend, scope, regions, and filters are frozen in the run;
changing them requires a new output directory. Worker count and batch size may
change on resume. `resume=False` requires a fresh directory.

To read a prior run without acquiring any data:

```python
verified = gwas_catalog.load_ingest_result("catalog-run")
verified.consolidate("catalog-verified.parquet")
```

This verifies receipts and artifacts. Missing/corrupt/uncommitted files are
failures; stray Parquet files are ignored. `allow_partial=True` is an explicit
opt-in on ingestion or loading and returns committed successes with
`status="partial"` and per-file failures. It does not turn failures into
successful empty files.

## Regions and significance filters

Regional coverage must be requested explicitly:

```python
regional = gwas_catalog.ingest_sumstats(
    files,
    "catalog-regions-run",
    scope="regions",
    regions=["chr1:100000-200000"],
    backend="auto",
)
```

Regions use the source's coordinates with inclusive endpoints. `chr` prefixes
are accepted for matching; output coordinates, chromosome tokens, and alleles
are preserved. No liftover occurs. A regional run is complete for its selected
regions, not for the whole genome. Batch ingestion also accepts a
`{study_id: [regions...]}` mapping with an entry for every selected study.

For remote Catalog files, `backend="auto"` uses indexed range acquisition
when a `.tbi` is available. This requires `pip install "biodb[gwas]"`
(`pysam`) and stable source/index byte identities. A known missing index can
fall back to full-file parsing, recorded in the receipt. `backend="tabix"`
requires the index; `backend="full"` explicitly downloads/parses the whole
file and then applies the regional filter. An indexed fetch or identity error
fails explicitly. Harvest and local inputs use full-file parsing.

An optional `p_max=5e-8` retains P ≤ 5e-8 inclusively. If P is missing but
`neg_log_10_p_value` is available, the corresponding log threshold is used.
Rows missing both are excluded only when this filter is requested. P=0 is
retained. Neither P nor effects are reconstructed from other measurements.

For processing beside an existing source archive, pass `local_root=...` to
the Harvest or Catalog download/iterator/ingestion calls. Exact flat or nested
source paths are resolved locally without transferring the archive.

## Output and provenance

| Columns | Meaning |
|---|---|
| `source`, `study_id`, `source_file_id`, `source_url` | Exact source membership |
| `chrom`, `pos`, `rsid`, `effect_allele`, `other_allele` | Header-mapped source identifiers and alleles |
| `beta`, `se`, `z`, `p`, `neg_log_10_p_value`, `frq`, `n` | Float64 measurements; missing remains null |
| `raw_fields` | Every original header/value text token, including underflow-scale P strings |
| `source_sha256`, `source_row` | Whole-file hash and source row number for full/local reads |
| `region_sha256`, `region_row`, `source_etag` | Regional-view hash/row and remote byte identity for indexed reads |

Indexed reads leave `source_sha256` and `source_row` null because only a
regional view was acquired. Receipts also retain transport, source/index
identity, requested scope/regions, filter, and failure details.

Catalog harmonised headers (`hm_*`) and GWAS-SSF names are matched by header.
Harvest requires `SNP CHR BP A1 A2 FRQ BETA SE Z P N`; A2 is its effect allele
and A1 its other allele. These are not asserted to be genomic ALT/REF. No
allele flipping, imputation, coordinate normalization, or effect reconstruction
is applied. `p` and `neg_log_10_p_value` remain separate so a supplied P=0 and
an independently supplied log-P are distinguishable.

## Related GWAS clients

`biodb.panukbb` provides dense regional Pan-UKBB statistics in GRCh37/hg19.
Use `phenotype_manifest`, `select_traits`, and `sumstats_region` /
`sumstats_regions`; the latter use `pysam` or the `tabix` executable. Its beta
is oriented to ALT, and frequency is cohort/ancestry-specific. This contract
differs from the raw Catalog/Harvest allele contract above.

`biodb.gwas_atlas` provides gene-level MAGMA results; `biodb.aou_allxall`
provides public gene-burden statistics; `biodb.omicspred` provides molecular
trait polygenic-score metadata/files. See the [source catalog](sources.md)
and [API reference](api.rst).

The [raw GWAS notebook](_generated/tutorials/13_raw_gwas.ipynb) demonstrates
local input parsing, durable ingestion, and resume without live downloads.
