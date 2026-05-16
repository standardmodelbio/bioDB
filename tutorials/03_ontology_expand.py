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
# # Ontology: N-hop keyword expansion
#
# `biodb.ontology.expand_keyword_sets_from_ontology` takes a tiny seed
# set and walks an in-memory ontology graph N hops out, returning the
# expanded keyword set per seed-bucket.
#
# We use a synthetic four-term dementia fragment so the notebook stays
# offline — in production you'd pass a real Mondo / HPO graph loaded via
# `biodb.ontology` helpers.

# %%
from biodb.ontology import expand_keyword_sets_from_ontology

ontology = {
    "dementia": ["alzheimer's disease", "vascular dementia"],
    "alzheimer's disease": ["early onset alzheimer's"],
    "vascular dementia": [],
    "early onset alzheimer's": [],
}
seeds = {"dementia": ["dementia"]}

# %% [markdown]
# ### 1-hop expansion: direct children only

# %%
expanded_1 = expand_keyword_sets_from_ontology(
    seed_keywords=seeds,
    ontology_dict=ontology,
    n_hops=1,
)
expanded_1

# %% [markdown]
# ### 2-hop expansion: includes grandchildren

# %%
expanded_2 = expand_keyword_sets_from_ontology(
    seed_keywords=seeds,
    ontology_dict=ontology,
    n_hops=2,
)
expanded_2
