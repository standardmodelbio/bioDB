# Searching ontologies with OLS

The [`biodb.ols`](api.rst) module is a REST client for [EMBL-EBI's OLS4](https://www.ebi.ac.uk/ols4/) (Ontology Lookup Service), the public index of ~280 biomedical ontologies — Mondo, EFO, HPO, GO, SO, ChEBI, SNOMED CT, and many more. Unlike [`biodb.ontology`](api.rst), which requires a local OWL file walked via `owlready2`, `biodb.ols` hits the OLS REST API and lets you look up terms, traverse hierarchies, and full-text-search **without** downloading multi-hundred-MB OWL files. The natural fit when you only need a handful of terms, or in environments where you can't or don't want to materialise the whole ontology locally.

```{contents}
:local:
:depth: 2
```

## When to use OLS vs. local OWL

| You want… | Use |
|---|---|
| A handful of terms / occasional lookups | `biodb.ols` (REST, no setup) |
| Full ontology in memory for repeated traversal | `biodb.ontology` (local OWL via `owlready2`) |
| A reproducible offline pipeline pinned to one ontology version | `biodb.ols.list_terms` (caches to parquet, version-aware) |
| Cross-ontology lookup ("which ontology owns this term?") | `biodb.ols.search` / `find_terms` (scope-less mode) |

OLS is the right starting point for almost everything ID-resolution-related. Drop to local OWL only when you've measured that REST round-trips dominate your workload (rare — even a 5,000-term walk completes in under a minute).

## Finding a term by name

The bread-and-butter use case: you know the disease / phenotype / anatomical part by name, and you need its canonical ID.

```python
from biodb.ols import find_terms, find_term

# Ranked DataFrame (top-K)
find_terms("breast carcinoma", ontology="efo", top_k=5)
#    obo_id        label                            match_quality
# 0  EFO:0000305   breast carcinoma                 4   (exact label)
# 1  EFO:1000307   inflammatory breast carcinoma    2   (prefix match)
# 2  EFO:0009547   breast carcinoma in situ         2   (prefix match)
# 3  ...

# One-shot best ID
find_term("breast carcinoma", ontology="efo")["obo_id"]
# 'EFO:0000305'

# Cross-ontology — handy when you don't know which vocabulary owns the concept
find_terms("Alzheimer disease")  # no ontology= scope
```

### The `match_quality` ranking

`find_terms` wraps the underlying Solr-backed `search` endpoint with a deterministic re-ranker so callers can filter or threshold by match tier without depending on opaque Solr boost configs that shift between OLS releases:

| Quality | Meaning |
|---|---|
| **4** | Exact case-insensitive label match. |
| **3** | Exact case-insensitive synonym match. |
| **2** | Label prefix match (label starts with the query). |
| **1** | Regex substring match in either label or any synonym. |
| **0** | Solr surfaced it but no surface-form overlap (usually a description hit). |

Stable sort preserves OLS's Solr order within each tier. To keep only exact-or-near-exact matches:

```python
df = find_terms("hypertension", ontology="mondo", top_k=20)
df[df["match_quality"] >= 3]  # exact label or exact synonym only
```

### Solr, not RAG

OLS4 does **not** currently expose a vector / RAG / embedding-based search endpoint — every hit goes through Solr's lexical (TF-IDF + field-boost) ranker. For paraphrase-aware lookup ("uncontrolled high blood pressure" → "hypertensive disorder") you would need to pair OLS output with an external embedding index — e.g. embed `list_terms(ontology)` once with a sentence-transformer and query by cosine similarity. EBI's experimental [Talk-to-EBI](https://www.ebi.ac.uk/about/news/updates-from-data-resources/talk-to-ebi-prototype) and Monarch's [OntoGPT](https://github.com/monarch-initiative/ontogpt) prove the pattern works, but neither is a stable API. For the >90% case where you know the canonical name and just need the ID, the Solr + exact-match re-ranker in `find_terms` is the pragmatic answer.

## Walking a known term

Once you have a CURIE (`MONDO:0004975`, `EFO:0000311`, …) you can pull the term record and traverse its hierarchy with five helpers:

```python
from biodb.ols import (
    get_term, get_descendants, get_ancestors, get_children, get_parents,
)

# Fetch the term itself (label, synonyms, description, IRI, …)
ad = get_term("mondo", "MONDO:0004975")
ad["label"]      # 'Alzheimer disease'
ad["synonyms"]   # ['Alzheimer dementia', 'AD', ...]

# All transitive descendants — DataFrame with obo_id / label / iri / …
get_descendants("mondo", "MONDO:0004975")
# 21 rows: Familial Alzheimer disease, early-onset AD, late-onset AD, ...

# All transitive ancestors
get_ancestors("mondo", "MONDO:0004975")
# 6 rows: dementia, neurodegenerative disease, nervous-system disease, ...

# Direct (one-hop) children / parents only
get_children("mondo", "MONDO:0004975")
get_parents("mondo", "MONDO:0004975")
```

All five helpers handle OLS's HAL-style pagination internally and return a DataFrame with the canonical columns `obo_id`, `label`, `iri`, `description`, `synonyms`, `is_obsolete`.

## Dumping a whole ontology

For workflows that need every term in memory — building a local Solr-free index, computing all-pairs similarity, training a sentence-transformer — use `list_terms`. It paginates `/ontologies/{slug}/terms` and caches the result to `~/.cache/biodb/ols/{slug}/{version}.parquet`, so subsequent calls in the same or another Python session re-read from disk without hitting OLS:

```python
from biodb.ols import list_terms

mondo = list_terms("mondo")        # first call: walks OLS (~1 min)
mondo_again = list_terms("mondo")  # second call: ~ms from parquet

# Versioned cache — a new OLS release writes alongside the old one,
# so you can rebuild from any historical version for reproducibility.
list_terms("snomed")               # ~5–10 min first run for 376k terms
```

Pass `include_obsolete=True` to keep deprecated terms; pass `refresh=True` to force a fresh walk past the cache. See the [API reference](api.rst) for the full kwarg list.

## CURIE / IRI conversion

OLS internally identifies terms by IRI, not CURIE — but users and external systems virtually always carry the CURIE form. Two helpers bridge the gap:

```python
from biodb.ols import curie_to_iri, ontology_id_from_curie

curie_to_iri("MONDO:0004975")
# 'http://purl.obolibrary.org/obo/MONDO_0004975'

curie_to_iri("EFO:0000400")
# 'http://www.ebi.ac.uk/efo/EFO_0000400'  (non-OBO scheme)

curie_to_iri("SNOMED:38341003")
# 'http://snomed.info/id/38341003'        (non-OBO scheme)

ontology_id_from_curie("MONDO:0004975")  # → "mondo"
ontology_id_from_curie("EFO_0000311")     # → "efo"
ontology_id_from_curie("SCTID:38341003")  # → "snomed"  (alias)
ontology_id_from_curie("ORPHA:733")       # → "ordo"    (alias)
```

`ontology_id_from_curie` is what you reach for when you have a CURIE but don't yet know which OLS slug to pass to `get_descendants` etc. — it centralises the prefix → slug mapping including the non-OBO aliases (`SCTID` → `snomed`, `ORPHA`/`ORPHANET` → `ordo`).

## Worked example: expanding a disease group

A common downstream pattern (e.g. for [`seqlab`](https://github.com/standardmodelbio/seqlab)'s variant-injection cohort builder) is "give me every cancer-related disease ID I can use to filter Open Targets." That's a 4-line composition:

```python
from biodb.ols import find_term, get_descendants, ontology_id_from_curie

# 1. Resolve "neoplasm" to its EFO ID — the canonical "all cancers" umbrella.
root = find_term("neoplasm", ontology="efo")["obo_id"]   # 'EFO:0000311'

# 2. Get every transitive descendant (carcinoma, sarcoma, leukaemia, ...).
ont = ontology_id_from_curie(root)                       # 'efo'
descendants = get_descendants(ont, root)
print(f"{len(descendants)} cancer-related EFO terms")    # ~5,000

# 3. Pass the list downstream to whatever filter wants disease IDs.
disease_ids = [root, *descendants["obo_id"]]
```

The same pattern works for any high-level grouping — pass a Mondo therapeutic-area root, an HPO phenotype family, an EFO trait subtree, and so on.

## Reliability

Every `_get(url)` call inside `biodb.ols` is wrapped in exponential-backoff retries with a 5-attempt default — OLS occasionally drops connections under load, and naive single-shot requests fail noticeably across long paginated walks (at 753 pages even a 99% success rate gives only ~0.05% chance of completing). 4xx errors short-circuit immediately (no point retrying a malformed request); 5xx and connection-level errors are retried with doubling backoff starting at 1s.

If you're hitting OLS hard in CI and want to suppress the retry chatter, raise `logging.getLogger("biodb.ols").setLevel(logging.ERROR)` — the warning logs are otherwise per-retry, which can flood test output.
