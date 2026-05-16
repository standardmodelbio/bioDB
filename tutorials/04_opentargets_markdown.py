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
# # Open Targets: markdown summaries
#
# Most Open Targets helpers in `biodb.opentargets` need network access
# (parquet downloads + GraphQL queries). The markdown renderers,
# however, operate on a single in-memory row and are very useful in
# isolation — e.g. when you've already pulled the parquet by other
# means and just want a tidy disease / drug / target summary.

# %%
from biodb.opentargets import df_to_markdown

# A toy target row that mirrors the Open Targets `target` parquet shape.
target_row = {
    "approvedSymbol": "BRCA1",
    "approvedName": "BRCA1 DNA repair associated",
    "id": "ENSG00000012048",
    "biotype": "protein_coding",
    "functionDescriptions": [
        "This gene encodes a 190 kD nuclear phosphoprotein that plays "
        "a role in maintaining genomic stability, and it also acts as "
        "a tumor suppressor."
    ],
}
md = df_to_markdown(target_row)
print(md)

# %% [markdown]
# ## Listing available parquet datasets
#
# `list_datasets()` enumerates Open Targets' parquet dump paths. Off-line
# the registry index is cached, so the call below is cheap once
# you've run it once.

# %%
from biodb.opentargets import list_datasets

datasets = list_datasets()
print(f"{len(datasets)} datasets registered")
# ``list_datasets`` returns a dict (name → URL); show the first five keys.
sample = list(datasets)[:5] if datasets else []
print("Sample (first 5):", sample)
