# bioDB tutorials

Rendered `.ipynb` notebooks (with executed outputs — open on
GitHub / nbviewer / Colab).

To re-execute against the current `biodb` install:

```bash
pip install "biodb[clinvar,test]" nbconvert ipykernel
jupyter nbconvert --to notebook --execute --inplace tutorials/*.ipynb
```

## Index

| # | Notebook | Demonstrates |
|---|----------|--------------|
| 01 | [Utils quickstart](01_utils_quickstart.ipynb) | Random seeding, similarity helpers, `filter_adaptive`, `create_gene_association_matrix` |
| 02 | [ClinVar simplification](02_clinvar_simplify.ipynb) | `CLNSIG` long-tail → 6 / 4-class buckets, `filter_df` |
| 03 | [Ontology expansion](03_ontology_expand.ipynb) | N-hop keyword expansion over a Mondo-like graph |
| 04 | [Open Targets markdown](04_opentargets_markdown.ipynb) | Markdown rendering + dataset registry |

All four tutorials run offline against synthetic data — they don't hit
Open Targets / NCBI / Monarch endpoints.
