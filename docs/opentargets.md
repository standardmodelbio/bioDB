# Open Targets

Use `biodb.opentargets_graphql` for targeted GraphQL records. Use
`biodb.opentargets` and its submodules for release-pinned bulk data. Both use
core dependencies; the optional `[gget]` extra is needed only by helpers that
explicitly use that backend.

## Full acquisition and previews

```python
from biodb.opentargets import DEFAULT_VERSION, ensure_cached_shards

# Explicit preview: one entire shard, not one complete dataset.
preview_paths = ensure_cached_shards(
    "target", version=DEFAULT_VERSION, limit_files=1, cache_dir="./ot-cache"
)
```

The pinned default release is `25.12`. A full call omits `limit_files`:

```python
all_paths = ensure_cached_shards(
    "target", version=DEFAULT_VERSION, cache_dir="./ot-cache"
)
```

The cache records the entire authoritative shard listing even when the first
call fetches only a preview. The second call validates cached members and
acquires missing shards. A directory containing Parquet files alone does not
prove that a dataset is complete. Cached membership and file fingerprints are
recorded under `<cache_dir>/<version>/<dataset>/`; custom `remote_url` sources
are isolated by source URL. A dataset lock protects interrupted/concurrent
acquisition. `force=True` refreshes the listing and selected files.

`get_dataset(..., output_format="polars", parse=False)` eagerly concatenates
the acquired raw shards. The default output is pandas. `limit_files` limits
source files; the separate `limit` argument limits returned rows. Neither is
a random or representative sample.

## Lazy projections and raw nested columns

```python
import polars as pl
from biodb.opentargets import scan_dataset, get_diseases, get_traits, get_drugs

targets = scan_dataset(
    "target", limit_files=1, columns=["id", "approvedSymbol"]
).filter(pl.col("approvedSymbol") == "BRCA1").collect()

diseases = get_diseases(limit_files=1, columns=["id", "name"])
traits = get_traits(limit_files=1, columns=["id", "name"])  # disease alias
drugs = get_drugs(limit_files=1, columns=["id", "name"])
```

`scan_dataset` returns a Polars `LazyFrame`; the convenience `get_*` readers
return a Polars `DataFrame`. Without `columns`, raw readers retain nested
fields. Local scans download each selected whole shard first; projection and
predicate pushdown reduce Parquet decoding, not the download size.

For narrow remote reads, pass `remote=True` to `scan_dataset`,
`scan_credible_sets`, or `scan_variant_annotations`. This scans the explicit
remote shard listing using HTTP range reads. Collection requires network
access, and it does not create or certify a complete local cache.
`force=True` applies only to local acquisition.

## Credible-set leads and members

Choose the representation that matches the unit of analysis:

| Reader | Unit / retained data |
|---|---|
| `scan_credible_sets` / `get_credible_sets` | Raw credible-set row, including nested `locus` members |
| `summarize_credible_sets(frame)` | Raw row plus matched lead `pip`, `credibleSetSize`, and `pipStatus` |
| `get_credible_set` | Flat lead row; scalar source fields retained |
| `get_credible_set_members` / `get_credible_set_variants` | One row per non-null member; member fields retained |
| `scan_variant_annotations` / `get_variant_annotations` | Raw `variant` table, including nested effects |
| `get_variant_effects` | Derived `(variantId, vep_score)` aggregation |

Lead PIP is the posterior of the member whose `variantId` matches the
top-level lead `variantId`, searching the whole locus. It is not the maximum
posterior and does not depend on member order. `pipStatus="matched"`
identifies a valid match. Missing leads, conflicting duplicate probabilities,
missing probabilities, or missing loci produce null `pip` with an explicit
status. `credibleSetSize` counts the source locus entries.

This example is entirely offline:

```python
import polars as pl
from biodb.opentargets.variants import summarize_credible_sets

raw = pl.DataFrame({
    "variantId": ["1_100_A_G"],
    "locus": [[
        {"variantId": "1_200_C_T", "posteriorProbability": 0.8},
        {"variantId": "1_100_A_G", "posteriorProbability": 0.2},
    ]],
})
summary = summarize_credible_sets(raw)
assert summary["pip"][0] == 0.2
assert summary["pipStatus"][0] == "matched"
```

Member-level acquisition:

```python
from biodb.opentargets.variants import get_credible_set_members

members = get_credible_set_members(limit_files=1)
print(members.select("studyId", "variantId", "pip"))
```

Each member's `pip` aliases its own `posteriorProbability`. Member `beta`,
`standardError`, `zScore`, and p-value fields remain the member's values when
present. Missing member statistics stay missing; top-level lead effects are
preserved with `lead*` names. The `is95CredibleSet` and `is99CredibleSet` flags
are retained, or null if unavailable. Empty/null loci contribute no member
rows.

`study_type="gwas"` (or a list of types) filters the flat lead/member readers.
An existing `studyType` is used directly; otherwise it is joined from
`biodb.opentargets.studies.get_study`. Raw study metadata is also available
through `scan_dataset("study")`.

See [raw annotations and enrichment](enrichment.md) for literature matches and
canonical constants, the [Open Targets notebook](_generated/tutorials/02_opentargets.ipynb)
for offline fixtures, and the [API reference](api.rst) for signatures.
