"""Shared pytest fixtures + tiny synthetic data builders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import requests

# HTTP statuses that mean "the upstream service is unavailable" — a gateway
# couldn't reach its backend (502), the service is down (503), or a gateway
# timed out (504). These are infrastructure/availability failures, NOT a
# response the API actually produced about our request.
_UPSTREAM_OUTAGE_STATUS = frozenset({502, 503, 504})


def is_upstream_outage(exc: BaseException) -> bool:
    """True iff ``exc`` reflects a *transient upstream outage*, narrowly.

    Returns True only for genuine availability failures:

    * a connection error or read/connect timeout (the server never answered), or
    * an HTTP ``502`` / ``503`` / ``504`` (gateway / service-unavailable).

    Returns False for everything else — ``4xx`` (our request is wrong / a
    contract changed), ``500`` (the server ran and errored, possibly because of
    *us*), JSON/parse errors, and assertion failures. Those indicate a real API
    change or a bug on our side and MUST fail loudly, never be skipped. Use this
    to guard live-API integration tests so a third party's downtime doesn't gate
    CI, without masking real regressions.
    """
    if isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
        return True
    if isinstance(exc, requests.exceptions.HTTPError):
        response = getattr(exc, "response", None)
        return response is not None and response.status_code in _UPSTREAM_OUTAGE_STATUS
    return False


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
