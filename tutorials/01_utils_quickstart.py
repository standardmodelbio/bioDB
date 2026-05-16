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
# # bioDB quickstart: utils
#
# `biodb.utils` contains the small surface of helpers shared across the
# four data modules — random seeding, similarity functions, adaptive
# row-filtering, and gene-association-matrix construction.
#
# Everything in this notebook runs offline against synthetic data.

# %%
import numpy as np
import pandas as pd

from biodb import utils

utils.set_random_seed(42)
print("RANDOM_SEED:", utils.RANDOM_SEED)

# %% [markdown]
# ## Similarity helpers
#
# All similarity functions take row-wise (n, d) matrices and return an
# (n1, n2) pairwise matrix.

# %%
rng = np.random.default_rng(0)
a = rng.normal(size=(4, 8)).astype(np.float32)
b = rng.normal(size=(6, 8)).astype(np.float32)

a_norm = utils.l2_normalize(a)
b_norm = utils.l2_normalize(b)

print("cosine     :", utils.cosine_similarity(a_norm, b_norm).shape)
print("euclidean  :", utils.euclidean_similarity(a, b).shape)
print("dot product:", utils.dot_product_similarity(a, b).shape)

# %% [markdown]
# ## Adaptive filtering
#
# `filter_adaptive` keeps the top-K per source by score, with optional
# pval / FDR gates and min/max-genes clamps. Typical use: thinning a
# long-tailed gene-association table before building a matrix.

# %%
np.random.seed(0)
rows = []
for source in ["DIS:1", "DIS:2", "DIS:3"]:
    for gene in ["BRCA1", "TP53", "EGFR", "MYC", "PTEN", "KRAS"]:
        rows.append(
            {
                "sourceId": source,
                "HGNC": gene,
                "score": float(np.random.rand()),
                "pval": float(np.random.rand()) * 0.01,
            }
        )
assoc = pd.DataFrame(rows)
assoc.head()

# %%
top = utils.filter_adaptive(
    assoc,
    source_id_col="sourceId",
    target_id_col="HGNC",
    percentile=0.5,
    verbose=False,
)
top.groupby("sourceId").size()

# %% [markdown]
# ## Gene-association matrix
#
# `create_gene_association_matrix` aggregates a long-form
# (source, gene, score) DataFrame into a (samples × genes) matrix with
# AnnData-style `obs` / `var` metadata.

# %%
X, meta = utils.create_gene_association_matrix(
    top,
    source_id_col="sourceId",
    target_id_col="HGNC",
    convert_to_dense=True,
    verbose=False,
)
print("matrix shape :", X.shape)
print("dtype        :", X.dtype)
print("obs columns  :", list(meta["obs"].columns))
print("var columns  :", list(meta["var"].columns))
meta["obs"].head()
