"""Shared pytest fixtures + tiny synthetic data builders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def small_associations() -> pd.DataFrame:
    """Tiny (12-row) gene-association DataFrame for matrix-builder tests."""
    rng = np.random.default_rng(42)
    rows = []
    for source in ["DIS:1", "DIS:2", "DIS:3"]:
        for gene in ["BRCA1", "TP53", "EGFR", "MYC"]:
            rows.append(
                {
                    "sourceId": source,
                    "HGNC": gene,
                    "score": float(rng.random()),
                    "pval": float(rng.random()) * 0.01,
                    "group": "test_group",
                    "label": f"{source} label",
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def small_embeddings() -> tuple[np.ndarray, np.ndarray]:
    """A (10, 8) + (5, 8) pair of float32 embedding matrices, L2-ish scaled."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=(10, 8)).astype(np.float32)
    b = rng.normal(size=(5, 8)).astype(np.float32)
    return a, b


@pytest.fixture
def tiny_ontology_dict() -> dict[str, list[str]]:
    """Tiny 4-term ontology fragment used by ontology tests."""
    return {
        "dementia": ["alzheimer's disease", "vascular dementia"],
        "alzheimer's disease": ["early onset alzheimer's"],
        "vascular dementia": [],
        "early onset alzheimer's": [],
    }
