# Querying genetic tests with GTR

The [`biodb.gtr`](api.rst) module is a client for NCBI's
[Genetic Testing Registry (GTR)](https://www.ncbi.nlm.nih.gov/gtr/) — a
catalog of ~64k genetic **tests**, where each record is *one test offered by
one lab*. A record carries the lab, the **gene list / panel** (each gene with
an Entrez ID + cytoband), the **conditions** (MedGen/UMLS CUIs + OMIM), a
3-level **methodology** taxonomy, a clinical-vs-research flag, and clinical
validity/utility free-text.

`biodb.gtr` exposes both bioDB modes:

| You want… | Use |
|---|---|
| Tests for one gene / condition / accession | `query_gene`, `query_condition`, `query_test` |
| A raw test-id search scoped to a GTR index | `search_tests(term, field=...)` |
| All test↔gene↔condition mappings (fast) | `download` + `load_test_condition_gene` (daily TSV) |
| The full records incl. descriptions/methodology | `download(full_xml=True)` + `iter_full_records` (~224 MB) |
| Curated gene sets for GenForge | `gene_sets`, `aggregate_gene_sets`, `to_gmt` |
| Embeddable panel text for HaploForge weighting | `panel_text` (+ the `support_count` prior) |

## Targeted lookups (API mode)

```python
from biodb import gtr

# Every test that targets BRCA1 (gene-symbol index).
tests = gtr.query_gene("BRCA1", retmax=50)
print(tests[0].name, tests[0].lab, [g["entrez"] for g in tests[0].genes])

# One record by accession (or bare uid).
brca = gtr.query_test("GTR000509983")
print(brca.clinical_validity, brca.methods)

# Tests for a condition by name or MedGen CUI.
hboc = gtr.query_condition("C0677776")      # CUI  -> DCUI index
hboc = gtr.query_condition("breast cancer") # text -> DISNAME index
```

GTR rides NCBI E-utilities, so the 3 req/sec un-keyed cap applies; pass
`api_key="..."` to any query to lift it to 10 req/sec. The numeric Entrez UID
is **not** the accession — `accession_from_uid(509983) == "GTR000509983"`.

## Bulk gene sets for GenForge

```python
from biodb import gtr

# Raw per-test gene sets (one row per test×gene), highly redundant.
raw = gtr.gene_sets()

# Deduplicated sets with a source-grounded importance prior: support_count
# is the number of distinct labs/tests whose panel includes each gene.
panels = gtr.aggregate_gene_sets(by="condition")

# Export to GMT for GenForge (set -> gene vector signature).
gtr.to_gmt("gtr_condition_panels.gmt", by="condition")
```

`download()` always fetches the light daily `test_condition_gene.txt` TSV;
the gene-set builders call it for you and cache under `~/.cache/biodb/gtr/`.

## Panel text for HaploForge weighting

bioDB produces the clean inputs; the embedding + cosine-similarity weighting
lives downstream in HaploForge/biodocs.

```python
from biodb import gtr

brca = gtr.query_test("GTR000509983")
text = gtr.panel_text(brca)   # name + conditions + clinical text + methods
# -> feed `text` to your embedding backend; weight each gene in brca.genes
#    by cosine similarity to the panel embedding. The aggregate_gene_sets
#    support_count is a complementary, embedding-free prior.
```

For descriptions across *all* panels in one pass, stream the full dump
(downloads `gtr_ftp.xml.gz` once, then parses it with flat memory use):

```python
for rec in gtr.iter_full_records():
    blob = gtr.panel_text(rec)
    ...   # embed / index downstream
```
