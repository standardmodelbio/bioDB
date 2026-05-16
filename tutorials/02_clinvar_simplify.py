# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ClinVar: collapse long-tail CLNSIG strings
#
# ClinVar `CLNSIG` values are notoriously long-tail — the same biological
# interpretation gets encoded in dozens of slightly different
# combinations. `biodb.clinvar.simplify_annotations` collapses that
# vocabulary into 6 (and then 4) buckets:
#
# - **6-class** (`CLNSIG_simple`): benign / likely_benign / path /
#   likely_path / conflicting / other
# - **4-class** (`CLNSIG_super_simple`): benign / path / conflicting /
#   other
#
# This notebook uses a synthetic mini-table — no NCBI download needed.

# %%
import polars as pl

from biodb import clinvar

# A handful of real CLNSIG strings observed in ClinVar.
df = pl.DataFrame(
    {
        "CLNSIG": [
            "Benign",
            "Likely_benign",
            "Pathogenic",
            "Likely_pathogenic",
            "Pathogenic/Likely_pathogenic",
            "Conflicting_classifications_of_pathogenicity",
            "Uncertain_significance",
            "Pathogenic|drug_response",
            "Benign|risk_factor",
        ],
        "GENEINFO": [
            "BRCA1:672",
            "TP53:7157",
            "EGFR:1956",
            "MYC:4609",
            "PTEN:5728",
            "KRAS:3845",
            "APOE:348",
            "CYP2C19:1557",
            "MTHFR:4524",
        ],
    }
)
df

# %%
out = clinvar.simplify_annotations(df, verbose=True)
out.select(["CLNSIG", "CLNSIG_simple", "CLNSIG_super_simple", "GENE"])

# %% [markdown]
# Unknown strings fall through to ``other`` rather than raising — useful
# for ad-hoc filtering and group-by operations on raw ClinVar exports.

# %%
unknown = pl.DataFrame({"CLNSIG": ["mystery_string_not_in_map"]})
clinvar.simplify_annotations(unknown, verbose=False)

# %% [markdown]
# ## Filtering a parsed ClinVar table
#
# `filter_df` accepts column → value pairs. List values become
# OR-substring matches; the special `CLNREVSTAT_score` key is compared
# as `>=`; everything else is equality.

# %%
sample = pl.DataFrame(
    {
        "CHROM": ["1", "1", "2", "3"],
        "POS": [10, 20, 30, 40],
        "CLNDN": ["a", "b", "c", "d"],
        "CLNREVSTAT_score": [0, 1, 2, 4],
        "CLNVC": [
            "single_nucleotide_variant",
            "single_nucleotide_variant",
            "Deletion",
            "single_nucleotide_variant",
        ],
    }
)
sample

# %%
clinvar.filter_df(
    sample,
    filters={
        "CLNREVSTAT_score": 2,  # >= 2 (multi-submitter consensus or stronger)
        "CLNVC": "single_nucleotide_variant",
    },
)
